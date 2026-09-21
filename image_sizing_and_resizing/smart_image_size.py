import json
import re
from pathlib import Path


DATA_FILE = Path(__file__).with_name("resolutions.json")


with DATA_FILE.open("r", encoding="utf-8") as file:
    RESOLUTIONS = json.load(file)


def dimension_text(item):
    return f"{item['ratio']} ({item['label']}) - {item['width']} x {item['height']}"


def unique_resolutions():
    result = []
    for model_resolutions in RESOLUTIONS.values():
        for resolution in model_resolutions:
            if resolution not in result:
                result.append(resolution)
    return result


def unique_dimensions():
    result = []
    for model_resolutions in RESOLUTIONS.values():
        for dimensions in model_resolutions.values():
            for item in dimensions:
                text = dimension_text(item)
                if text not in result:
                    result.append(text)
    return result


def resolution_side(resolution):
    if not isinstance(resolution, str):
        return None

    square_size = re.search(r"\((\d+)\s*[x×]\s*\1\)", resolution)
    if square_size:
        return int(square_size.group(1))

    k_size = re.fullmatch(r"(\d+(?:\.\d+)?)K", resolution, re.IGNORECASE)
    if k_size:
        return round(float(k_size.group(1)) * 1024)

    if resolution.isdigit():
        return int(resolution)
    return None


def resolve_resolution(model_resolutions, resolution):
    if resolution in model_resolutions:
        return resolution

    requested_side = resolution_side(resolution)
    if requested_side is not None:
        for available_resolution in model_resolutions:
            if resolution_side(available_resolution) == requested_side:
                return available_resolution
    return next(iter(model_resolutions))


def resolution_output(resolution):
    side = resolution_side(resolution)
    if side is not None:
        return side
    return int(resolution)


class SmartImageSize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (list(RESOLUTIONS.keys()), {"default": "FLUX.2 Klein"}),
                "resolution": (unique_resolutions(),),
                "dimensions": (unique_dimensions(),),
            }
        }

    RETURN_TYPES = ("INT", "INT", "STRING", "INT")
    RETURN_NAMES = ("width", "height", "aspect_ratio", "resolution")
    FUNCTION = "get_resolution"
    CATEGORY = "image/resolution"

    def get_resolution(self, model, resolution, dimensions):
        if model not in RESOLUTIONS:
            model = next(iter(RESOLUTIONS))

        available_resolutions = RESOLUTIONS[model]
        resolution = resolve_resolution(available_resolutions, resolution)

        available_dimensions = available_resolutions[resolution]
        selected = next(
            (item for item in available_dimensions if dimension_text(item) == dimensions),
            available_dimensions[0],
        )

        return (
            int(selected["width"]),
            int(selected["height"]),
            selected["ratio"],
            resolution_output(resolution),
        )


try:
    from aiohttp import web
    from server import PromptServer

    @PromptServer.instance.routes.get("/image-model-resolution-selector/data")
    async def get_resolution_data(_request):
        return web.json_response(RESOLUTIONS)
except (ImportError, AttributeError):
    pass


NODE_CLASS_MAPPINGS = {
    "SmartImageSize": SmartImageSize,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SmartImageSize": "RP Smart Image Size",
}
