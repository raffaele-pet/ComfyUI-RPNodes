import json
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


def resolution_output(resolution):
    if isinstance(resolution, str) and resolution.upper().endswith("K"):
        try:
            return str(round(float(resolution[:-1]) * 1024))
        except ValueError:
            pass
    return str(resolution)


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

    RETURN_TYPES = ("INT", "INT", "STRING", "STRING")
    RETURN_NAMES = ("width", "height", "aspect_ratio", "resolution")
    FUNCTION = "get_resolution"
    CATEGORY = "image/resolution"

    def get_resolution(self, model, resolution, dimensions):
        if model not in RESOLUTIONS:
            model = next(iter(RESOLUTIONS))

        available_resolutions = RESOLUTIONS[model]
        if resolution not in available_resolutions:
            resolution = next(iter(available_resolutions))

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
    "SmartImageSize": "Smart Image Size",
}
