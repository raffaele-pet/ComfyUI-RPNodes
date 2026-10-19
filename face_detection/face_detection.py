import nodes
import torch


DEFAULT_MODEL = "bbox/face_yolov8m.pt"
DEPENDENCIES = {
    "UltralyticsDetectorProvider": "ComfyUI Impact Subpack",
    "BboxDetectorSEGS": "ComfyUI Impact Pack",
    "ImpactSEGSToMaskList": "ComfyUI Impact Pack",
}


def bbox_geometry(mask, width=0, height=0, scale=1.0, square=False):
    points = torch.nonzero(mask[0] > 0, as_tuple=False)
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

    target_width = max(1, round(target_width * scale))
    target_height = max(1, round(target_height * scale))
    center_y, center_x = points.double().mean(0).tolist()
    return center_x, center_y, target_width, target_height


def resize_bbox_mask(mask, width=0, height=0, scale=1.0, square=False, padding=(0, 0, 0, 0)):
    if (width == 0 and height == 0 and scale == 1.0 and not square) or not torch.any(mask > 0):
        return torch.nn.functional.pad(mask, padding) if any(padding) else mask
    center_x, center_y, target_width, target_height = bbox_geometry(mask, width, height, scale, square)
    pad_left, pad_right, pad_top, pad_bottom = padding
    image_height = mask.shape[-2] + pad_top + pad_bottom
    image_width = mask.shape[-1] + pad_left + pad_right
    left = round(center_x + 0.5 - target_width / 2) + pad_left
    top = round(center_y + 0.5 - target_height / 2) + pad_top
    result = mask.new_zeros((mask.shape[0], image_height, image_width))
    result[:, max(0, top):min(image_height, top + target_height),
           max(0, left):min(image_width, left + target_width)] = 1.0
    return result


def crop_geometry(mask, crop_size_mult, bbox_smooth_alpha, width=0, height=0, scale=1.0, square=False):
    center_x, center_y, mask_width, mask_height = bbox_geometry(mask, width, height, square=square)
    base_width = round(round((mask_width - 1) * bbox_smooth_alpha) * crop_size_mult)
    base_height = round(round((mask_height - 1) * bbox_smooth_alpha) * crop_size_mult)
    crop_width = max(2, round(base_width * scale))
    crop_height = max(2, round(base_height * scale))
    left = round(center_x) - round(crop_width / 2)
    top = round(center_y) - round(crop_height / 2)
    return left, top, crop_width, crop_height


def face_padding(mask, crop_size_mult, bbox_smooth_alpha, width=0, height=0, scale=1.0, square=False):
    left, top, crop_width, crop_height = crop_geometry(mask, crop_size_mult, bbox_smooth_alpha, width, height, scale, square)
    center_x, center_y, mask_width, mask_height = bbox_geometry(mask, width, height, scale, square)
    mask_left = round(center_x + 0.5 - mask_width / 2)
    mask_top = round(center_y + 0.5 - mask_height / 2)
    image_height, image_width = mask.shape[-2:]
    return (
        max(0, -left, -mask_left),
        max(0, left + crop_width - image_width, mask_left + mask_width - image_width),
        max(0, -top, -mask_top),
        max(0, top + crop_height - image_height, mask_top + mask_height - image_height),
    )


def crop_face(mask, frame, crop_size_mult, bbox_smooth_alpha,
              width=0, height=0, scale=1.0, square=False, padding=(0, 0, 0, 0)):
    left, top, crop_width, crop_height = crop_geometry(mask, crop_size_mult, bbox_smooth_alpha, width, height, scale, square)
    if any(padding):
        pad_left, pad_right, pad_top, pad_bottom = padding
        frame = torch.nn.functional.pad(frame, (0, 0, pad_left, pad_right, pad_top, pad_bottom))
        left += pad_left
        top += pad_top
    image_height, image_width = frame.shape[1:3]
    right, bottom = min(image_width, left + crop_width), min(image_height, top + crop_height)
    left, top = max(0, left), max(0, top)
    # KJNodes' image conversion squeezes one-pixel axes, losing RGB channels.
    if right - left < 2:
        left = max(0, min(left, image_width - 2))
        right = min(image_width, left + 2)
    if bottom - top < 2:
        top = max(0, min(top, image_height - 2))
        bottom = min(image_height, top + 2)
    box_width, box_height = right - left, bottom - top
    # Crop pixels and BBOX must describe exactly the same source region for BatchUncrop.
    cropped = frame[:, top:bottom, left:right, :].clone()
    return frame, cropped, [(left, top, box_width, box_height)], box_width, box_height


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
            "optional": {
                "pad_outside_image": ("BOOLEAN", {"default": False,
                    "tooltip": "Add black padding to keep the entire mask and crop outside image borders. original_images, bboxes and bbox_masks use the expanded canvas."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "BBOX", "INT", "INT", "MASK", "MASK", "MASK")
    RETURN_NAMES = ("original_images", "cropped_images", "bboxes", "width", "height", "bbox_masks", "image_area_mask", "cropped_image_mask")
    OUTPUT_IS_LIST = (True, True, True, True, True, True, True, True)
    FUNCTION = "detect"
    CATEGORY = "RP/face_detection"
    DESCRIPTION = "Face detection, mask resizing and cropping in one node. BBOX coordinates connect to BatchUncrop; bbox_masks connects to MaskPreview."

    def detect(self, image, model_name=DEFAULT_MODEL, width=0, height=0, scale=1.0,
               square=False, show_advanced=False, threshold=0.5, dilation=10,
               crop_factor=1.5, drop_size=10, crop_size_mult=3.0, bbox_smooth_alpha=0.5,
               pad_outside_image=False):
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
                padding = face_padding(mask, crop_size_mult, bbox_smooth_alpha, width, height, scale, square) if pad_outside_image else (0, 0, 0, 0)
                cropped = crop_face(mask, frame, crop_size_mult, bbox_smooth_alpha, width, height, scale, square, padding)
                mask = resize_bbox_mask(mask, width, height, scale, square, padding)
                image_area_mask = frame.new_ones(frame.shape[:3], dtype=torch.float32)
                if any(padding):
                    image_area_mask = torch.nn.functional.pad(image_area_mask, padding)
                x, y, crop_width, crop_height = cropped[2][0]
                cropped_image_mask = image_area_mask[:, y:y + crop_height, x:x + crop_width].clone()
                for output, value in zip(outputs, (*cropped, mask, image_area_mask, cropped_image_mask)):
                    output.append(value)
        return outputs


NODE_CLASS_MAPPINGS = {"RPFaceDetection": RPFaceDetection}
NODE_DISPLAY_NAME_MAPPINGS = {"RPFaceDetection": "RP Face Detection"}
