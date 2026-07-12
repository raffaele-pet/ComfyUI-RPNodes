import math
from fractions import Fraction

import torch
import torch.nn.functional as F

from .nodes import RESOLUTIONS, dimension_text, resolution_output, unique_dimensions, unique_resolutions


try:
    from comfy.utils import common_upscale
except ImportError:
    common_upscale = None

UPSCALE_METHODS = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
RESIZE_MODES = [
    "stretch",
    "resize",
    "pad",
    "pad_edge",
    "pad_edge_pixel",
    "crop",
    "pillarbox_blur",
    "total_pixels",
]
CROP_POSITIONS = ["center", "top", "bottom", "left", "right"]


def _find_dimensions(model, resolution, dimensions):
    if model not in RESOLUTIONS:
        model = next(iter(RESOLUTIONS))
    model_resolutions = RESOLUTIONS[model]
    if resolution not in model_resolutions:
        resolution = next(iter(model_resolutions))
    items = model_resolutions[resolution]
    selected = next((item for item in items if dimension_text(item) == dimensions), items[0])
    return model, resolution, items, selected


def _closest_dimensions(items, width, height):
    source_ratio = width / max(1, height)
    return min(
        items,
        key=lambda item: abs(math.log(source_ratio / (item["width"] / item["height"]))),
    )


def _dimensions_from_longer_side(item, longer_side):
    longer_side = max(1, int(longer_side))
    item_width = int(item["width"])
    item_height = int(item["height"])
    if item_width >= item_height:
        return longer_side, max(1, round(longer_side * item_height / item_width))
    return max(1, round(longer_side * item_width / item_height)), longer_side


def _parse_color(text, channels, dtype, device):
    try:
        values = [float(value.strip()) for value in text.split(",") if value.strip()]
    except ValueError:
        values = []
    if not values:
        values = [0.0, 0.0, 0.0]
    if max(values) > 1.0:
        values = [value / 255.0 for value in values]
    while len(values) < channels:
        values.append(values[-1])
    return torch.tensor(values[:channels], dtype=dtype, device=device).clamp(0.0, 1.0)


def _resize_nchw(tensor, width, height, method):
    width = max(1, int(width))
    height = max(1, int(height))
    if common_upscale is not None:
        return common_upscale(tensor, width, height, method, crop="disabled")
    fallback = "nearest" if method == "nearest-exact" else "bicubic" if method == "lanczos" else method
    kwargs = {"size": (height, width), "mode": fallback}
    if fallback in ("bilinear", "bicubic"):
        kwargs["align_corners"] = False
        kwargs["antialias"] = True
    return F.interpolate(tensor, **kwargs)


def _resize_image(image, width, height, method):
    return _resize_nchw(image.movedim(-1, 1), width, height, method).movedim(1, -1)


def _resize_mask(mask, width, height):
    return F.interpolate(mask.unsqueeze(1), size=(height, width), mode="nearest-exact").squeeze(1)


def _crop_box(width, height, target_ratio, position):
    source_ratio = width / height
    if source_ratio > target_ratio:
        crop_width, crop_height = max(1, round(height * target_ratio)), height
    else:
        crop_width, crop_height = width, max(1, round(width / target_ratio))

    if position == "left":
        x = 0
    elif position == "right":
        x = width - crop_width
    else:
        x = (width - crop_width) // 2

    if position == "top":
        y = 0
    elif position == "bottom":
        y = height - crop_height
    else:
        y = (height - crop_height) // 2
    return x, y, crop_width, crop_height


def _padding(target_width, target_height, width, height, position):
    extra_width = max(0, target_width - width)
    extra_height = max(0, target_height - height)
    if position == "left":
        left, right = 0, extra_width
    elif position == "right":
        left, right = extra_width, 0
    else:
        left, right = extra_width // 2, extra_width - extra_width // 2
    if position == "top":
        top, bottom = 0, extra_height
    elif position == "bottom":
        top, bottom = extra_height, 0
    else:
        top, bottom = extra_height // 2, extra_height - extra_height // 2
    return left, right, top, bottom


def _blur(image, sigma):
    radius = max(1, int(3.0 * sigma))
    coords = torch.arange(-radius, radius + 1, device=image.device, dtype=image.dtype)
    kernel = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum()
    channels = image.shape[1]
    horizontal = kernel.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
    vertical = kernel.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)
    image = F.conv2d(image, horizontal, padding=(0, radius), groups=channels)
    return F.conv2d(image, vertical, padding=(radius, 0), groups=channels)


def _pad_image(image, target_width, target_height, position, mode, color):
    batch, height, width, channels = image.shape
    left, right, top, bottom = _padding(target_width, target_height, width, height, position)

    if mode == "pillarbox_blur":
        scale = max(target_width / width, target_height / height)
        bg_width, bg_height = max(1, round(width * scale)), max(1, round(height * scale))
        background = _resize_image(image, bg_width, bg_height, "bilinear")
        x = max(0, (bg_width - target_width) // 2)
        y = max(0, (bg_height - target_height) // 2)
        background = background[:, y : y + target_height, x : x + target_width, :]
        background = _blur(background.movedim(-1, 1), max(1.0, 0.006 * min(target_width, target_height)))
        background = (background * 0.35).clamp(0.0, 1.0).movedim(1, -1)
        background[:, top : top + height, left : left + width, :] = image
        return background, (left, right, top, bottom)

    if mode == "pad_edge_pixel":
        output = F.pad(image.movedim(-1, 1), (left, right, top, bottom), mode="replicate").movedim(1, -1)
        return output, (left, right, top, bottom)

    if mode == "pad_edge":
        output = torch.empty(
            (batch, target_height, target_width, channels), dtype=image.dtype, device=image.device
        )
        output[:, :, :, :] = image.mean(dim=(1, 2), keepdim=True)
        if top:
            output[:, :top, :, :] = image[:, 0, :, :].mean(dim=1, keepdim=True).unsqueeze(1)
        if bottom:
            output[:, top + height :, :, :] = image[:, -1, :, :].mean(dim=1, keepdim=True).unsqueeze(1)
        if left:
            output[:, :, :left, :] = image[:, :, 0, :].mean(dim=1, keepdim=True).unsqueeze(1)
        if right:
            output[:, :, left + width :, :] = image[:, :, -1, :].mean(dim=1, keepdim=True).unsqueeze(1)
        output[:, top : top + height, left : left + width, :] = image
        return output, (left, right, top, bottom)

    fill = _parse_color(color, channels, image.dtype, image.device)
    output = fill.view(1, 1, 1, channels).repeat(batch, target_height, target_width, 1)
    output[:, top : top + height, left : left + width, :] = image
    return output, (left, right, top, bottom)


def _pad_mask(mask, target_width, target_height, pads, fill=1.0):
    left, right, top, bottom = pads
    output = torch.full(
        (mask.shape[0], target_height, target_width), fill, dtype=mask.dtype, device=mask.device
    )
    output[:, top : top + mask.shape[1], left : left + mask.shape[2]] = mask
    return output


def _actual_ratio(width, height):
    ratio = Fraction(int(width), int(height))
    return f"{ratio.numerator}:{ratio.denominator}"


class SmartImageResize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (list(RESOLUTIONS.keys()), {"default": "FLUX.2 Klein"}),
                "resolution_preset": (unique_resolutions(),),
                "selection_mode": (["automatic", "manual"],),
                "dimensions": (unique_dimensions(),),
                "width": ("INT", {"default": 1024, "min": 1, "max": 16384, "step": 1}),
                "height": ("INT", {"default": 1024, "min": 1, "max": 16384, "step": 1}),
                "upscale_method": (UPSCALE_METHODS, {"default": "lanczos"}),
                "keep_proportion": (RESIZE_MODES, {"default": "pad_edge_pixel"}),
                "pad_color": ("STRING", {"default": "0, 0, 0"}),
                "crop_position": (CROP_POSITIONS,),
            },
            "optional": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "resolution": (
                    "INT",
                    {
                        "forceInput": True,
                        "tooltip": "Optional longer-side resolution override.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "STRING", "STRING", "MASK")
    RETURN_NAMES = ("IMAGE", "width", "height", "aspect_ratio", "resolution", "mask")
    FUNCTION = "resize"
    CATEGORY = "image/resolution"

    def resize(
        self,
        model,
        resolution_preset,
        selection_mode,
        dimensions,
        width,
        height,
        upscale_method,
        keep_proportion,
        pad_color,
        crop_position,
        image=None,
        mask=None,
        resolution=None,
    ):
        if image is None and mask is None:
            raise ValueError("Connect an IMAGE or a MASK to Smart Image Resize.")

        if image is not None:
            source_height, source_width = int(image.shape[1]), int(image.shape[2])
        else:
            source_height, source_width = int(mask.shape[1]), int(mask.shape[2])

        model, resolution_preset, items, selected = _find_dimensions(
            model, resolution_preset, dimensions
        )
        if selection_mode == "automatic":
            selected = _closest_dimensions(items, source_width, source_height)
            width, height = int(selected["width"]), int(selected["height"])
        else:
            width, height = max(1, int(width)), max(1, int(height))

        if resolution is not None and int(resolution) > 0:
            width, height = _dimensions_from_longer_side(selected, resolution)
            output_resolution = str(int(resolution))
        else:
            output_resolution = resolution_output(resolution_preset)

        target_device = torch.device("cpu")

        if image is None:
            base_mask = mask.to(target_device)
            image = base_mask.unsqueeze(-1).repeat(1, 1, 1, 3)
        else:
            image = image.to(target_device)
            base_mask = mask.to(target_device) if mask is not None else torch.zeros(
                (image.shape[0], source_height, source_width), dtype=image.dtype, device=target_device
            )

        if base_mask.shape[0] == 1 and image.shape[0] > 1:
            base_mask = base_mask.repeat(image.shape[0], 1, 1)
        if base_mask.shape[1:3] != image.shape[1:3]:
            base_mask = _resize_mask(base_mask, image.shape[2], image.shape[1])

        if keep_proportion == "stretch":
            output_image = _resize_image(image, width, height, upscale_method)
            output_mask = _resize_mask(base_mask, width, height)
        elif keep_proportion == "crop":
            x, y, crop_width, crop_height = _crop_box(source_width, source_height, width / height, crop_position)
            cropped_image = image[:, y : y + crop_height, x : x + crop_width, :]
            cropped_mask = base_mask[:, y : y + crop_height, x : x + crop_width]
            output_image = _resize_image(cropped_image, width, height, upscale_method)
            output_mask = _resize_mask(cropped_mask, width, height)
        elif keep_proportion == "total_pixels":
            total_pixels = width * height
            source_ratio = source_width / source_height
            output_width = max(1, round(math.sqrt(total_pixels * source_ratio)))
            output_height = max(1, round(math.sqrt(total_pixels / source_ratio)))
            output_image = _resize_image(image, output_width, output_height, upscale_method)
            output_mask = _resize_mask(base_mask, output_width, output_height)
        else:
            scale = min(width / source_width, height / source_height)
            resized_width = max(1, round(source_width * scale))
            resized_height = max(1, round(source_height * scale))
            resized_image = _resize_image(image, resized_width, resized_height, upscale_method)
            resized_mask = _resize_mask(base_mask, resized_width, resized_height)

            if keep_proportion == "resize":
                output_image, output_mask = resized_image, resized_mask
            else:
                output_image, pads = _pad_image(
                    resized_image, width, height, crop_position, keep_proportion, pad_color
                )
                output_mask = _pad_mask(resized_mask, width, height, pads, fill=1.0)

        output_width = int(output_image.shape[2])
        output_height = int(output_image.shape[1])
        return (
            output_image.cpu(),
            output_width,
            output_height,
            _actual_ratio(output_width, output_height),
            output_resolution,
            output_mask.cpu(),
        )


NODE_CLASS_MAPPINGS = {
    "SmartImageResize": SmartImageResize,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SmartImageResize": "Smart Image Resize",
}
