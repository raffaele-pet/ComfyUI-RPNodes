import math
from contextlib import contextmanager
from functools import lru_cache

import numpy as np
from PIL import Image

import comfy.model_management as model_management
import torch


GUIDE_MODES = ("pose_guide", "face_guide", "rectangle_guide")
POSE_GUIDES = frozenset(GUIDE_MODES[:2])
GUIDE_SCALE_MIN = 0.1
GUIDE_SCALE_MAX = 4.0


def guide_scale(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2 or any(
        type(scale) not in (int, float) or not GUIDE_SCALE_MIN <= scale <= GUIDE_SCALE_MAX
        for scale in value
    ):
        raise ValueError("Guide scale must be between 10% and 400% on each axis.")
    return tuple(value)


@lru_cache(maxsize=1)
def load_pose_detector():
    try:
        from custom_controlnet_aux.open_pose import OpenposeDetector
    except ImportError as exc:
        raise RuntimeError("Install comfyui_controlnet_aux to use the OpenPose guides.") from exc
    return OpenposeDetector.from_pretrained()


@contextmanager
def guide_detector():
    detector = load_pose_detector()
    try:
        yield detector.to(model_management.get_torch_device())
    finally:
        detector.to(torch.device("cpu"))


def pose_geometry(detector, rgb, include_hands):
    from custom_controlnet_aux.util import resize_image_with_pad

    prepared, unpad = resize_image_with_pad(rgb, 512, "INTER_CUBIC")
    # The annotator pads only the right and bottom, so rendered coordinates
    # and matching coordinates share the same origin.
    poses = detector.detect_poses(prepared, include_hand=include_hands, include_face=True)
    return poses, prepared.shape[:2], unpad(prepared).shape[:2]


def mask_bounds(mask):
    rows = np.flatnonzero(mask.any(axis=1))
    if not rows.size:
        return None
    columns = np.flatnonzero(mask.any(axis=0))
    return int(columns[0]), int(rows[0]), int(columns[-1] + 1), int(rows[-1] + 1)


def keypoints_overlap_mask(points, mask, padded, unpadded):
    height, width = mask.shape
    scale_x = padded[1] * width / unpadded[1]
    scale_y = padded[0] * height / unpadded[0]
    hits = visible = 0
    for point in points:
        if point is None:
            continue
        x, y = math.floor(point.x * scale_x), math.floor(point.y * scale_y)
        if 0 <= x < width and 0 <= y < height:
            visible += 1
            hits += bool(mask[y, x])
    return hits > 0 and hits * 2 >= visible


def guide_artwork(kind, mask, color, geometry=None):
    bounds = mask_bounds(mask)
    if bounds is None:
        return None
    if kind == "rectangle_guide":
        left, top, right, bottom = bounds
        return Image.new("RGBA", (right - left, bottom - top), color), bounds

    from custom_controlnet_aux.open_pose import draw_poses

    poses, padded, unpadded = geometry
    body = kind == "pose_guide"
    matched = []
    for pose in poses:
        points = pose.body.keypoints if body else pose.face
        if points and keypoints_overlap_mask(points, mask, padded, unpadded):
            matched.append(pose)
    if not matched:
        return None
    rgb = draw_poses(matched, *padded, draw_body=body, draw_hand=body, draw_face=True)
    rgb = rgb[:unpadded[0], :unpadded[1]]
    # The annotation renderer draws on black. Give its painted pixels real
    # alpha before resampling so the canvas does not create a black plate.
    alpha = (rgb.any(axis=2) * 255).astype(np.uint8)
    height, width = mask.shape
    rgba = Image.fromarray(np.dstack((rgb, alpha))).resize((width, height), Image.Resampling.BILINEAR)
    bounds = rgba.getbbox(alpha_only=True)
    return (rgba.crop(bounds), bounds) if bounds else None


def composite_guide(canvas, artwork, scale):
    if artwork is None:
        return
    image, (left, top, right, bottom) = artwork
    width = max(1, round(image.width * scale[0]))
    height = max(1, round(image.height * scale[1]))
    x = round((left + right - width) / 2)
    y = round((top + bottom - height) / 2)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(canvas.width, x + width), min(canvas.height, y + height)
    if x0 >= x1 or y0 >= y1:
        return
    box = (
        (x0 - x) * image.width / width, (y0 - y) * image.height / height,
        (x1 - x) * image.width / width, (y1 - y) * image.height / height,
    )
    visible = image.resize((x1 - x0, y1 - y0), Image.Resampling.BILINEAR, box=box)
    canvas.alpha_composite(visible, (x0, y0))


def render_guides(output, items, rgb, detector):
    height, width = output.shape[1:3]
    for index in range(len(output)):
        visible = [
            (group, masks[index - section.start]) for group, section, masks in items
            if section.start <= index < section.stop and masks[index - section.start].any()
        ]
        if not visible:
            continue
        kinds = {group["preprocessor"] for group, _ in visible}
        geometry = pose_geometry(detector, rgb[index], "pose_guide" in kinds) if kinds & POSE_GUIDES else None
        canvas = Image.new("RGBA", (width, height))
        for group, mask in visible:
            artwork = guide_artwork(group["preprocessor"], mask, group["color"], geometry)
            composite_guide(canvas, artwork, group["guide_scale"])
        output[index] = torch.from_numpy(np.array(canvas)).to(output.dtype).div_(255)
