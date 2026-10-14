import nodes
import torch


DEFAULT_MODEL = "bbox/face_yolov8m.pt"
DEPENDENCIES = {
    "UltralyticsDetectorProvider": "ComfyUI Impact Subpack",
    "BboxDetectorSEGS": "ComfyUI Impact Pack",
    "ImpactSEGSToMaskList": "ComfyUI Impact Pack",
    "BatchCropFromMask": "ComfyUI KJNodes",
}


def resize_bbox_mask(mask, width=0, height=0, scale=1.0, square=False):
    if width == 0 and height == 0 and scale == 1.0 and not square:
        return mask

    points = torch.nonzero(mask[0] > 0, as_tuple=False)
    if len(points) == 0:
        return mask
    min_y, min_x = points.amin(dim=0).tolist()
    max_y, max_x = points.amax(dim=0).tolist()
    original_width = max_x - min_x + 1
    original_height = max_y - min_y + 1
    if square:
        side = max(original_width, original_height)
        target_width = target_height = max(width, height) or side
    else:
        target_width = width or original_width
        target_height = height or original_height

    image_height, image_width = mask.shape[-2:]
    target_width = max(1, round(target_width * scale))
    target_height = max(1, round(target_height * scale))
    if square:
        target_width = target_height = min(target_width, image_width, image_height)
    else:
        target_width = min(target_width, image_width)
        target_height = min(target_height, image_height)

    left = round((min_x + max_x + 1 - target_width) / 2)
    top = round((min_y + max_y + 1 - target_height) / 2)
    left = min(max(0, left), image_width - target_width)
    top = min(max(0, top), image_height - target_height)
    result = torch.zeros_like(mask)
    result[:, top:top + target_height, left:left + target_width] = 1.0
    return result


def crop_face(mask, frame, crop_size_mult, bbox_smooth_alpha):
    points = torch.nonzero(mask[0] > 0, as_tuple=False)
    span_y, span_x = (points.amax(0) - points.amin(0)).tolist()
    crop_width = round(round(span_x * bbox_smooth_alpha) * crop_size_mult)
    crop_height = round(round(span_y * bbox_smooth_alpha) * crop_size_mult)
    if crop_width >= 2 and crop_height >= 2:
        # The upstream mask list executes a fresh crop node once per face.
        crop_node = nodes.NODE_CLASS_MAPPINGS["BatchCropFromMask"]()
        return crop_node.crop(mask, frame, crop_size_mult, bbox_smooth_alpha)

    # KJNodes rounds a one-pixel crop axis down to zero at its center.
    crop_width, crop_height = max(1, crop_width), max(1, crop_height)
    center_y, center_x = points.float().mean(0).tolist()
    image_height, image_width = frame.shape[1:3]
    box_width, box_height = min(crop_width, image_width), min(crop_height, image_height)
    left = min(max(0, round(center_x) - round(box_width / 2)), image_width - box_width)
    top = min(max(0, round(center_y) - round(box_height / 2)), image_height - box_height)
    cropped = frame[:, top:top + box_height, left:left + box_width, :]
    if (box_width, box_height) != (crop_width, crop_height):
        cropped = torch.nn.functional.interpolate(
            cropped.movedim(-1, 1), size=(crop_height, crop_width), mode="bilinear", align_corners=False
        ).movedim(1, -1)
    return frame, cropped, [(left, top, box_width, box_height)], crop_width, crop_height


class RPFaceDetection:
    @classmethod
    def INPUT_TYPES(cls):
        provider = nodes.NODE_CLASS_MAPPINGS.get("UltralyticsDetectorProvider")
        models = provider.INPUT_TYPES()["required"]["model_name"][0] if provider else []
        models = list(models) or [DEFAULT_MODEL]
        default_model = DEFAULT_MODEL if DEFAULT_MODEL in models else models[0]
        return {
            "required": {
                "image": ("IMAGE",),
                "model_name": (models, {"default": default_model}),
                "width": ("INT", {"default": 0, "min": 0, "max": 16384,
                                  "tooltip": "Mask rectangle width in pixels. 0 keeps the detected width."}),
                "height": ("INT", {"default": 0, "min": 0, "max": 16384,
                                   "tooltip": "Mask rectangle height in pixels. 0 keeps the detected height."}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01,
                                    "tooltip": "Scale the mask rectangle after width, height and square."}),
                "square": ("BOOLEAN", {"default": False,
                                       "tooltip": "Use the longer detected side for both sides, then apply size overrides and scale."}),
                "show_advanced": ("BOOLEAN", {"default": False}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "dilation": ("INT", {"default": 10, "min": -512, "max": 512}),
                "crop_factor": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 100.0, "step": 0.1}),
                "drop_size": ("INT", {"default": 10, "min": 1, "max": 16384}),
                "crop_size_mult": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.001}),
                "bbox_smooth_alpha": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "BBOX", "INT", "INT", "MASK")
    RETURN_NAMES = ("original_images", "cropped_images", "bboxes", "width", "height", "bbox_masks")
    OUTPUT_IS_LIST = (True, True, True, True, True, True)
    FUNCTION = "detect"
    CATEGORY = "RP/face_detection"
    DESCRIPTION = "Face detection, mask resizing and cropping in one node. BBOX coordinates connect to BatchUncrop; bbox_masks connects to MaskPreview."

    def detect(self, image, model_name=DEFAULT_MODEL, width=0, height=0, scale=1.0,
               square=False, show_advanced=False, threshold=0.5, dilation=10,
               crop_factor=1.5, drop_size=10, crop_size_mult=3.0, bbox_smooth_alpha=0.5):
        for name, package in DEPENDENCIES.items():
            if name not in nodes.NODE_CLASS_MAPPINGS:
                raise RuntimeError(f"RP Face Detection requires {package} ({name}). Install it and restart ComfyUI.")

        provider = nodes.NODE_CLASS_MAPPINGS["UltralyticsDetectorProvider"]()
        detector_node = nodes.NODE_CLASS_MAPPINGS["BboxDetectorSEGS"]()
        mask_node = nodes.NODE_CLASS_MAPPINGS["ImpactSEGSToMaskList"]()
        detector, _ = provider.doit(model_name)
        outputs = tuple([] for _ in self.RETURN_TYPES)
        for frame_index in range(len(image)):
            frame = image[frame_index:frame_index + 1]
            segs, = detector_node.doit(detector, frame, threshold, dilation, crop_factor, drop_size, labels="all")
            masks, = mask_node.doit(segs)
            for mask in masks:
                if not torch.any(mask > 0):
                    raise ValueError(f"RP Face Detection: no face detected in image {frame_index + 1}. Try a lower threshold or another detector model.")
                mask = resize_bbox_mask(mask, width, height, scale, square)
                cropped = crop_face(mask, frame, crop_size_mult, bbox_smooth_alpha)
                for output, value in zip(outputs, (*cropped, mask)):
                    output.append(value)
        return outputs


NODE_CLASS_MAPPINGS = {"RPFaceDetection": RPFaceDetection}
NODE_DISPLAY_NAME_MAPPINGS = {"RPFaceDetection": "RP Face Detection"}
