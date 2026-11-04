import numpy as np
import torch

from comfy.utils import ProgressBar
from comfy_api.latest import io
from comfy_extras.nodes_post_processing import ColorTransfer

from .dense_grid_align import cv2, estimate_deformation, grid_preview, warp_image, _structure, _correlation
from .tiled_image import TILE_SIZE, TileContext


def minimum_seam(cost):
    height, width = cost.shape
    cumulative = cost[0].copy()
    parents = np.zeros((height, width), np.int8)
    for row in range(1, height):
        options = np.stack([np.r_[np.inf, cumulative[:-1]] + 0.002,
                            cumulative, np.r_[cumulative[1:], np.inf] + 0.002])
        choice = options.argmin(axis=0)
        parents[row] = choice - 1
        cumulative = cost[row] + options[choice, np.arange(width)]
    seam = np.empty(height, np.int32)
    seam[-1] = cumulative.argmin()
    for row in range(height - 1, 0, -1):
        seam[row - 1] = seam[row] + parents[row, seam[row]]
    return seam


def _seam_cost(existing, incoming, reference):
    old = cv2.GaussianBlur(existing, (0, 0), 0.8)
    new = cv2.GaussianBlur(incoming, (0, 0), 0.8)
    difference = np.abs(old - new).mean(axis=-1)
    gray_old, gray_new = cv2.cvtColor(old, cv2.COLOR_RGB2GRAY), cv2.cvtColor(new, cv2.COLOR_RGB2GRAY)
    gradient_old = np.hypot(cv2.Sobel(gray_old, cv2.CV_32F, 1, 0), cv2.Sobel(gray_old, cv2.CV_32F, 0, 1))
    gradient_new = np.hypot(cv2.Sobel(gray_new, cv2.CV_32F, 1, 0), cv2.Sobel(gray_new, cv2.CV_32F, 0, 1))
    boundary = difference + 0.15 * (gradient_old + gradient_new) + 0.1 * np.abs(gradient_old - gradient_new)
    structure_ref = _structure(reference).astype(np.float32) / 255
    score_old, _ = _correlation(structure_ref, _structure(existing).astype(np.float32) / 255)
    score_new, _ = _correlation(structure_ref, _structure(incoming).astype(np.float32) / 255)
    return boundary, (1 - score_old) * 0.5, (1 - score_new) * 0.5


def _cut_cost(boundary, old_error, new_error):
    width = boundary.shape[1]
    old_prefix = np.cumsum(old_error, axis=1)
    new_suffix = np.cumsum(new_error[:, ::-1], axis=1)[:, ::-1] - new_error
    preference = (old_prefix + new_suffix) / max(1, width)
    center = ((np.arange(width, dtype=np.float32) - (width - 1) * 0.5) / max(1, width)) ** 2
    return boundary + 0.35 * preference + 0.005 * center[None, :]


def multiband_seam(existing, incoming, choose_new, covered, feather):
    inside = cv2.distanceTransform(choose_new.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    outside = cv2.distanceTransform((~choose_new).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    distance = inside - outside
    alpha = np.clip(0.5 + distance / (2 * feather), 0, 1)
    alpha = alpha * alpha * (3 - 2 * alpha)
    base = np.where(covered[..., None], existing, incoming)
    difference = incoming - base
    detail_alpha = np.clip(0.5 + distance / (2 * min(feather, 4)), 0, 1)
    detail_alpha = detail_alpha * detail_alpha * (3 - 2 * detail_alpha)
    gradients = []
    for image in (base, incoming):
        gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), (0, 0), 1.0)
        gradients.append(np.stack([cv2.Sobel(gray, cv2.CV_32F, 1, 0),
                                   cv2.Sobel(gray, cv2.CV_32F, 0, 1)], axis=-1))
    old_edge, new_edge = [np.linalg.norm(gradient, axis=-1) for gradient in gradients]
    neighborhood = np.ones((25, 25), np.uint8)
    both_edges = np.minimum(cv2.dilate(old_edge, neighborhood), cv2.dilate(new_edge, neighborhood))
    disagreement = np.linalg.norm(gradients[0] - gradients[1], axis=-1)
    guard = np.clip((both_edges - 0.08) / 0.16, 0, 1) * np.clip(disagreement / 0.12, 0, 1)
    guard = cv2.GaussianBlur(cv2.dilate(guard, np.ones((5, 5), np.uint8)), (0, 0), 1.0)
    # Broaden texture/focus transitions; keep conflicting strong contours narrowly blended.
    detail_alpha = alpha * (1 - guard) + detail_alpha * guard
    differences, masks = [difference], [detail_alpha]
    scale = 1
    while scale < feather and min(differences[-1].shape[:2]) > 2:
        differences.append(cv2.pyrDown(differences[-1]))
        masks.append(cv2.pyrDown(masks[-1]))
        scale *= 2
    blended = differences[-1] * masks[-1][..., None]
    for level in range(len(differences) - 2, -1, -1):
        size = (differences[level].shape[1], differences[level].shape[0])
        detail = differences[level] - cv2.pyrUp(differences[level + 1], dstsize=size)
        blended = cv2.pyrUp(blended, dstsize=size) + detail * masks[level][..., None]
    hard = difference * choose_new[..., None]
    # Bound the pyramid's support so padding/uncovered pixels and distant detail stay exact.
    support = np.clip((feather - np.abs(distance)) / max(1, feather * 0.25), 0, 1)
    support = support * support * (3 - 2 * support)
    result = np.where(choose_new[..., None], incoming, base) + (blended - hard) * support[..., None]
    result[~covered] = incoming[~covered]
    alpha[~covered] = 1
    return result, alpha


def seam_composite(existing, incoming, reference, covered, left_overlap, top_overlap, feather):
    height, width = incoming.shape[:2]
    choose_new = np.ones((height, width), bool)
    if covered.any() and (left_overlap or top_overlap):
        boundary, old_error, new_error = _seam_cost(existing, incoming, reference)
        if left_overlap:
            cost = _cut_cost(boundary[:, :left_overlap], old_error[:, :left_overlap], new_error[:, :left_overlap])
            seam = minimum_seam(cost)
            choose_new &= np.arange(width)[None, :] > seam[:, None]
        if top_overlap:
            cost = _cut_cost(boundary[:top_overlap].T, old_error[:top_overlap].T, new_error[:top_overlap].T)
            seam = minimum_seam(cost)
            choose_new &= np.arange(height)[:, None] > seam[None, :]
    choose_new[~covered] = True
    alpha = choose_new.astype(np.float32)
    if feather > 0 and choose_new.any() and not choose_new.all():
        return multiband_seam(existing, incoming, choose_new, covered, feather)
    result = existing * (1 - alpha[..., None]) + incoming * alpha[..., None]
    return result, alpha


def merge_advanced(tiles, context, color_method='reinhard_lab', tile_color_strength=0.75,
                   global_color_strength=0.25, alignment_strength=1.0, grid_spacing=16,
                   max_shift=96.0, anchor_tolerance=1.0, seam_feather=64,
                   rigidity=0.5, minimum_match=0.55):
    if cv2 is None:
        raise RuntimeError('RP Tile Merge Advanced requires OpenCV in the ComfyUI Python environment.')
    tiles = [tile[b:b + 1] for tile in tiles for b in range(tile.shape[0])]
    count = len(context['positions'])
    expected = context['batch_size'] * count
    if len(tiles) != expected:
        raise ValueError(f'RP Tile Merge Advanced: expected {expected} tiles in splitter order, received {len(tiles)}.')
    reference = context['reference']
    reference_np = reference.detach().to(device='cpu', dtype=torch.float32).numpy()
    width, height = context['content_size']
    left, top, right, bottom = context['padding']
    rows, columns = context['rows'], context['columns']
    images, aligned_tiles, previews, motion_masks, seams, reports = [], [], [], [], [], []
    progress = ProgressBar(expected)
    for batch in range(context['batch_size']):
        ref = reference_np[batch]
        padded = cv2.copyMakeBorder(ref, top, bottom, left, right, cv2.BORDER_REPLICATE)
        canvas = np.zeros_like(ref)
        covered = np.zeros((height, width), bool)
        owners = np.full((height, width), -1, np.int16)
        motion_mask = np.zeros((height, width), np.float32)
        grid_thumbs, corrections = [], []
        anchors, total_points = 0, 0
        for index, (x, y) in enumerate(context['positions']):
            tile = tiles[batch * count + index]
            if tuple(tile.shape[1:]) not in ((TILE_SIZE, TILE_SIZE, 3), (TILE_SIZE, TILE_SIZE, 4)):
                raise ValueError(f'RP Tile Merge Advanced: tile {batch * count + index + 1} must be 1024 x 1024 RGB or RGBA; got {tuple(tile.shape)}.')
            target = tile[0].detach().to(device='cpu', dtype=torch.float32).numpy()
            ref_tile = padded[y:y + TILE_SIZE, x:x + TILE_SIZE]
            x0, y0 = max(x, left), max(y, top)
            x1, y1 = min(x + TILE_SIZE, left + width), min(y + TILE_SIZE, top + height)
            protect = np.ones((TILE_SIZE, TILE_SIZE), np.float32)
            protect[max(0, y0 - y):max(0, y1 - y), max(0, x0 - x):max(0, x1 - x)] = 0
            field, mesh, pinned, stats = estimate_deformation(
                ref_tile, target[..., :3] if alignment_strength else ref_tile,
                max_shift=max_shift, grid_spacing=grid_spacing, anchor_tolerance=anchor_tolerance,
                rigidity=rigidity, analysis_max_side=TILE_SIZE, minimum_match=minimum_match, protect_mask=protect)
            aligned = warp_image(target, field, alignment_strength) if alignment_strength else target.copy()
            motion = np.linalg.norm(field * alignment_strength, axis=-1) / max(max_shift, 1)
            preview = grid_preview(ref_tile, target[..., :3], mesh, pinned, alignment_strength)
            thumb = np.round(cv2.resize(preview, (320, 166), interpolation=cv2.INTER_AREA) * 255).astype(np.uint8)
            cv2.putText(thumb, f'Tile {index + 1}: {stats["max_displacement"] * alignment_strength:.1f}px',
                        (5, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            grid_thumbs.append(thumb)
            corrections.append(stats['max_displacement'] * alignment_strength)
            anchors += stats['anchored_points']
            total_points += stats['grid_points']
            if x0 < x1 and y0 < y1:
                sy, sx = slice(y0 - y, y1 - y), slice(x0 - x, x1 - x)
                dy, dx = slice(y0 - top, y1 - top), slice(x0 - left, x1 - left)
                incoming = aligned[sy, sx, :3]
                ref_valid = ref[dy, dx]
                if tile_color_strength:
                    matched = ColorTransfer.execute(torch.from_numpy(incoming)[None], torch.from_numpy(ref_valid)[None],
                                                    color_method, {'source_stats': 'per_frame'}, tile_color_strength)[0]
                    incoming = matched[0].detach().to(device='cpu', dtype=torch.float32).numpy()
                    aligned[sy, sx, :3] = incoming
                row, column = divmod(index, columns)
                overlap_x = max(0, min(x + context['overlap'], x1) - x0) if column else 0
                overlap_y = max(0, min(y + context['overlap'], y1) - y0) if row else 0
                canvas[dy, dx], alpha = seam_composite(canvas[dy, dx], incoming, ref_valid, covered[dy, dx],
                                                      overlap_x, overlap_y, seam_feather)
                motion_mask[dy, dx] = motion_mask[dy, dx] * (1 - alpha) + motion[sy, sx] * alpha
                owners[dy, dx][alpha >= 0.5] = index
                covered[dy, dx] = True
            aligned_tiles.append(torch.from_numpy(aligned.copy())[None].to(device=reference.device, dtype=reference.dtype))
            progress.update(1)
        if not covered.all():
            raise ValueError('RP Tile Merge Advanced: tile_context does not cover the complete source image.')
        result = torch.from_numpy(canvas.copy())[None].to(device=reference.device, dtype=reference.dtype)
        if global_color_strength:
            result = ColorTransfer.execute(result, reference[batch:batch + 1], color_method,
                                           {'source_stats': 'per_frame'}, global_color_strength)[0]
        images.append(result.clamp(0, 1).to(device=reference.device, dtype=reference.dtype))
        atlas = np.concatenate([np.concatenate(grid_thumbs[r * columns:(r + 1) * columns], axis=1) for r in range(rows)], axis=0)
        previews.append(torch.from_numpy(atlas.astype(np.float32) / 255))
        boundary = np.zeros((height, width), np.float32)
        boundary[:, 1:] = owners[:, 1:] != owners[:, :-1]
        boundary[1:] = np.maximum(boundary[1:], owners[1:] != owners[:-1])
        seams.append(torch.from_numpy(boundary))
        motion_masks.append(torch.from_numpy(np.clip(motion_mask, 0, 1)))
        reports.append(f'Image {batch + 1}: {columns} x {rows} tiles; {width} x {height}; '
                       f'{total_points} grid points; {anchors} fixed anchors; max correction {max(corrections):.2f} px; '
                       f'structure-guided seams, multiband feather {seam_feather}px; alignment strength {alignment_strength:.2f}.\n'
                       'Tile corrections (px): ' + ', '.join(f'{v:.2f}' for v in corrections))
    return (torch.cat(images), aligned_tiles,
            torch.stack(previews).to(device=reference.device, dtype=reference.dtype),
            torch.stack(motion_masks).to(device=reference.device, dtype=reference.dtype),
            torch.stack(seams).to(device=reference.device, dtype=reference.dtype), '\n'.join(reports))


class RPTileMergeAdvanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='RPTileMergeAdvanced', display_name='RP Tile Merge Advanced', category='RP/image_sizing_and_resizing',
            description='Align decoded tiles to their saved original geometry before merging. Use elastic grids, fixed anchors and structure-guided seams to reduce shifts and double contours. Removes padding and integrates per-tile/global color matching.',
            is_input_list=True,
            inputs=[
                io.Image.Input('tiles'), TileContext.Input('tile_context'),
                io.Combo.Input('color_method', options=['reinhard_lab', 'mkl_lab', 'histogram'], default='reinhard_lab'),
                io.Float.Input('tile_color_strength', default=0.75, min=0.0, max=1.0, step=0.01),
                io.Float.Input('global_color_strength', default=0.25, min=0.0, max=1.0, step=0.01),
                io.Float.Input('alignment_strength', default=1.0, min=0.0, max=1.0, step=0.05),
                io.Int.Input('grid_spacing', default=16, min=4, max=128, step=4),
                io.Float.Input('max_shift', default=96, min=1.0, max=256.0, step=1),
                io.Float.Input('anchor_tolerance', default=1.0, min=0.0, max=8.0, step=0.25, advanced=True),
                io.Int.Input('seam_feather', default=64, min=0, max=128, step=1, tooltip='Multiband seam width in pixels: broad color/lighting transitions, narrow fine-detail transitions. 64 is the default. 0 keeps the hard cut. Saved workflows retain their current value.'),
                io.Float.Input('rigidity', default=0.5, min=0.05, max=4.0, step=0.05, advanced=True),
                io.Float.Input('minimum_match', default=0.55, min=0.1, max=0.95, step=0.05, advanced=True),
            ],
            outputs=[
                io.Image.Output(display_name='image'), io.Image.Output(display_name='aligned_tiles', is_output_list=True),
                io.Image.Output(display_name='deformation_grids'), io.Mask.Output(display_name='correction_mask'),
                io.Mask.Output(display_name='seam_mask'), io.String.Output(display_name='report'),
            ],
        )

    @classmethod
    def execute(cls, tiles, tile_context, color_method, tile_color_strength, global_color_strength,
                alignment_strength, grid_spacing, max_shift, anchor_tolerance, seam_feather, rigidity, minimum_match):
        if len(tile_context) != 1:
            raise ValueError('RP Tile Merge Advanced: connect one splitter context; use an IMAGE batch for multiple images.')
        return io.NodeOutput(*merge_advanced(
            tiles, tile_context[0], color_method[0], tile_color_strength[0], global_color_strength[0],
            alignment_strength[0], grid_spacing[0], max_shift[0], anchor_tolerance[0], seam_feather[0],
            rigidity[0], minimum_match[0]))


NODE_CLASS_MAPPINGS = {'RPTileMergeAdvanced': RPTileMergeAdvanced}
NODE_DISPLAY_NAME_MAPPINGS = {'RPTileMergeAdvanced': 'RP Tile Merge Advanced'}
