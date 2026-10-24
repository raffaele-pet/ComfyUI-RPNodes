import math

import torch
import torch.nn.functional as F

import comfy.utils
from comfy_api.latest import io
from comfy_extras.nodes_post_processing import ColorTransfer


TILE_SIZE = 1024
TILE_COUNTS = ["4", "6", "8", "9", "10", "12", "15", "16", "18", "20", "24", "25"]
TileContext = io.Custom("RP_TILE_CONTEXT")


def tile_layout(width, height, count):
    # More seams get a wider transition, from 192 px (4) to 288 px (25).
    overlap = round((128 + 32 * math.sqrt(count)) / 16) * 16
    stride = TILE_SIZE - overlap
    candidates = []
    for rows in range(1, count + 1):
        if count % rows:
            continue
        columns = count // rows
        canvas_width = TILE_SIZE + (columns - 1) * stride
        canvas_height = TILE_SIZE + (rows - 1) * stride
        error = abs(math.log((canvas_width / canvas_height) / (width / height)))
        candidates.append((error, abs(rows - columns), rows, columns, canvas_width, canvas_height))
    _, _, rows, columns, canvas_width, canvas_height = min(candidates)
    scale = min(canvas_width / width, canvas_height / height)
    scaled_width = min(canvas_width, max(1, round(width * scale)))
    scaled_height = min(canvas_height, max(1, round(height * scale)))
    left = (canvas_width - scaled_width) // 2
    top = (canvas_height - scaled_height) // 2
    return {
        "rows": rows, "columns": columns, "overlap": overlap,
        "canvas_size": (canvas_width, canvas_height),
        "content_size": (scaled_width, scaled_height),
        "padding": (left, top, canvas_width - scaled_width - left, canvas_height - scaled_height - top),
        "positions": [(column * stride, row * stride) for row in range(rows) for column in range(columns)],
    }


def _preview(tiles, rows, columns, batch_size):
    previews = []
    count = rows * columns
    for batch_index in range(batch_size):
        thumbs = [F.interpolate(tile.movedim(-1, 1), size=(192, 192), mode="area").movedim(1, -1)
                  for tile in tiles[batch_index * count:(batch_index + 1) * count]]
        previews.append(torch.cat([torch.cat(thumbs[row * columns:(row + 1) * columns], dim=2)
                                   for row in range(rows)], dim=1))
    return torch.cat(previews)


def split_tiles(image, count, upscale_method):
    height, width = image.shape[1:3]
    layout = tile_layout(width, height, count)
    scaled_width, scaled_height = layout["content_size"]
    reference = comfy.utils.common_upscale(
        image[..., :3].movedim(-1, 1), scaled_width, scaled_height, upscale_method, "disabled"
    ).movedim(1, -1)
    left, top, right, bottom = layout["padding"]
    padded = F.pad(reference.movedim(-1, 1), (left, right, top, bottom), mode="replicate").movedim(1, -1)
    tiles = [padded[b:b + 1, y:y + TILE_SIZE, x:x + TILE_SIZE].clone()
             for b in range(image.shape[0]) for x, y in layout["positions"]]
    context = {**layout, "reference": reference, "original_size": (width, height), "batch_size": image.shape[0]}
    preview = _preview(tiles, layout["rows"], layout["columns"], image.shape[0])
    description = (
        f'{layout["columns"]} columns x {layout["rows"]} rows; {count} tiles/image, 1024 x 1024; '
        f'overlap {layout["overlap"]} px; output {scaled_width} x {scaled_height}; '
        f'padding L/T/R/B: {left}/{top}/{right}/{bottom}'
    )
    return tiles, context, reference, preview, description


def _feather(row, column, rows, columns, overlap, device):
    ramp = (1 - torch.cos(math.pi * (torch.arange(overlap, device=device) + 0.5) / overlap)) * 0.5
    wx = torch.ones(TILE_SIZE, device=device)
    wy = torch.ones(TILE_SIZE, device=device)
    if column > 0:
        wx[:overlap] = ramp
    if column < columns - 1:
        wx[-overlap:] = ramp.flip(0)
    if row > 0:
        wy[:overlap] = ramp
    if row < rows - 1:
        wy[-overlap:] = ramp.flip(0)
    return (wy[:, None] * wx[None, :])[None, :, :, None]


def merge_tiles(tiles, context, color_method, tile_color_strength, global_color_strength):
    tiles = [tile[b:b + 1] for tile in tiles for b in range(tile.shape[0])]
    count = len(context["positions"])
    expected = context["batch_size"] * count
    if len(tiles) != expected:
        raise ValueError(f"RP Tile Merge: expected {expected} tiles in splitter order, received {len(tiles)}.")
    reference = context["reference"]
    device = reference.device
    width, height = context["content_size"]
    left, top, _, _ = context["padding"]
    rows, columns = context["rows"], context["columns"]
    outputs = []
    for batch_index in range(context["batch_size"]):
        canvas = torch.zeros((1, height, width, 3), dtype=torch.float32, device=device)
        weights = torch.zeros((1, height, width, 1), dtype=torch.float32, device=device)
        for index, (x, y) in enumerate(context["positions"]):
            tile = tiles[batch_index * count + index]
            if tuple(tile.shape[1:]) != (TILE_SIZE, TILE_SIZE, 3):
                raise ValueError(f"RP Tile Merge: tile {batch_index * count + index + 1} must be 1024 x 1024 RGB; got {tuple(tile.shape)}.")
            # Work only on real image pixels. Padding never enters color statistics or blending.
            x0, y0 = max(x, left), max(y, top)
            x1, y1 = min(x + TILE_SIZE, left + width), min(y + TILE_SIZE, top + height)
            if x0 >= x1 or y0 >= y1:
                continue
            target = tile[:, y0 - y:y1 - y, x0 - x:x1 - x, :].to(device=device, dtype=torch.float32)
            ref = reference[batch_index:batch_index + 1, y0 - top:y1 - top, x0 - left:x1 - left, :]
            if tile_color_strength:
                target = ColorTransfer.execute(target, ref, color_method, {"source_stats": "per_frame"}, tile_color_strength)[0].to(device=device, dtype=torch.float32)
            row, column = divmod(index, columns)
            weight = _feather(row, column, rows, columns, context["overlap"], device)[:, y0 - y:y1 - y, x0 - x:x1 - x, :]
            canvas[:, y0 - top:y1 - top, x0 - left:x1 - left, :].add_(target * weight)
            weights[:, y0 - top:y1 - top, x0 - left:x1 - left, :].add_(weight)
        result = canvas / weights.clamp_min_(1e-8)
        if global_color_strength:
            result = ColorTransfer.execute(result, reference[batch_index:batch_index + 1], color_method,
                                           {"source_stats": "per_frame"}, global_color_strength)[0].to(device=device)
        outputs.append(result.clamp_(0, 1).to(dtype=reference.dtype))
    return torch.cat(outputs)


class RPTileSplit(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="RPTileSplit", display_name="RP Tile Split", category="RP/image_sizing_and_resizing",
            description="Resize once without cropping, then split into exact 1024 x 1024 tiles. Edge padding and positions are saved for RP Tile Merge.",
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input("tile_count", options=TILE_COUNTS, default="12"),
                io.Combo.Input("upscale_method", options=["lanczos", "bicubic", "bilinear", "area", "nearest-exact"], default="lanczos"),
            ],
            outputs=[
                io.Image.Output(display_name="tiles", is_output_list=True),
                TileContext.Output(display_name="tile_context"),
                io.Image.Output(display_name="scaled_image"),
                io.Image.Output(display_name="tile_preview"),
                io.String.Output(display_name="layout"),
            ],
        )

    @classmethod
    def execute(cls, image, tile_count="12", upscale_method="lanczos"):
        return io.NodeOutput(*split_tiles(image, int(tile_count), upscale_method))


class RPTileMerge(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="RPTileMerge", display_name="RP Tile Merge", category="RP/image_sizing_and_resizing",
            description="Reconstruct processed tiles in splitter order, remove padding, and match per-tile and global colors to the saved source.",
            is_input_list=True,
            inputs=[
                io.Image.Input("tiles"),
                TileContext.Input("tile_context"),
                io.Combo.Input("color_method", options=["reinhard_lab", "mkl_lab", "histogram"], default="reinhard_lab"),
                io.Float.Input("tile_color_strength", default=0.75, min=0.0, max=1.0, step=0.01),
                io.Float.Input("global_color_strength", default=0.25, min=0.0, max=1.0, step=0.01),
            ],
            outputs=[io.Image.Output(display_name="image")],
        )

    @classmethod
    def execute(cls, tiles, tile_context, color_method, tile_color_strength, global_color_strength):
        if len(tile_context) != 1:
            raise ValueError("RP Tile Merge: connect one splitter context; use an IMAGE batch to process multiple images together.")
        return io.NodeOutput(merge_tiles(tiles, tile_context[0], color_method[0], tile_color_strength[0], global_color_strength[0]))


NODE_CLASS_MAPPINGS = {"RPTileSplit": RPTileSplit, "RPTileMerge": RPTileMerge}
NODE_DISPLAY_NAME_MAPPINGS = {"RPTileSplit": "RP Tile Split", "RPTileMerge": "RP Tile Merge"}
