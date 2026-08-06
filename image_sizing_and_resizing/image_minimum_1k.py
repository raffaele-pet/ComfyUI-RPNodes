import torch.nn.functional as F


try:
    from comfy.utils import common_upscale
except ImportError:
    common_upscale = None


MINIMUM_LONGER_SIDE = 1024


class RPImageMinimum1K:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "upscale_if_needed"
    CATEGORY = "image/resolution"

    def upscale_if_needed(self, image):
        height = int(image.shape[1])
        width = int(image.shape[2])
        longer_side = max(width, height)

        if longer_side >= MINIMUM_LONGER_SIDE:
            return (image,)

        scale = MINIMUM_LONGER_SIDE / longer_side
        target_width = max(1, round(width * scale))
        target_height = max(1, round(height * scale))
        image_nchw = image.movedim(-1, 1)

        if common_upscale is not None:
            output = common_upscale(
                image_nchw,
                target_width,
                target_height,
                "lanczos",
                crop="disabled",
            )
        else:
            output = F.interpolate(
                image_nchw,
                size=(target_height, target_width),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )

        return (output.movedim(1, -1),)


NODE_CLASS_MAPPINGS = {
    "RPImageMinimum1K": RPImageMinimum1K,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RPImageMinimum1K": "RP Image Minimum 1K",
}
