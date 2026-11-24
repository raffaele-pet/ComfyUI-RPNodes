import math
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PIL import Image

import comfy.model_management as model_management
import torch


GUIDE_MODES = ("pose_guide", "face_guide", "rectangle_guide")
POSE_GUIDES = frozenset(GUIDE_MODES[:2])
GUIDE_SCALE_MIN = 0.1
GUIDE_SCALE_MAX = 4.0


@dataclass(frozen=True)
class GuideFrame:
    center: tuple[float, float]
    size: tuple[float, float]
    angle: float = 0.0


def guide_scale(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2 or any(
        type(scale) not in (int, float) or not GUIDE_SCALE_MIN <= scale <= GUIDE_SCALE_MAX
        for scale in value
    ):
        raise ValueError("Guide scale must be between 10% and 400% on each axis.")
    return tuple(value)


def guide_placement(offset, rotation):
    if not isinstance(offset, (list, tuple)) or len(offset) != 2 or any(
        type(value) not in (int, float) or not -1 <= value <= 1 for value in offset
    ):
        raise ValueError("Guide offset must be within one frame width and height.")
    if type(rotation) not in (int, float) or not -180 <= rotation <= 180:
        raise ValueError("Guide rotation must be between -180 and 180 degrees.")
    return tuple(offset), rotation


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


def rectangle_frame(mask):
    rows, columns = np.nonzero(mask)
    if not rows.size:
        return None
    cx, cy = columns.mean() + 0.5, rows.mean() + 0.5
    x, y = columns + 0.5 - cx, rows + 0.5 - cy
    xx, xy, yy = (x * x).sum(), (x * y).sum(), (y * y).sum()
    angle = 0.0
    if math.hypot(xx - yy, 2 * xy) > 0.03 * (xx + yy):
        major = math.degrees(math.atan2(2 * xy, xx - yy)) / 2
        angle = major % 180 - 90
    cos, sin = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    across = cos * x + sin * y
    along = -sin * x + cos * y
    # Fit the length to the silhouette. Use the central half of its area
    # across the axis so extended arms and other side details don't set the width.
    low, high = np.percentile(along, (1, 99))
    near, far = np.percentile(across, (25, 75))
    length = high - low + 1
    width = max(1, min(far - near + 1, length * 0.45))
    a, b = (near + far) / 2, (low + high) / 2
    center = float(cx + cos * a - sin * b), float(cy + sin * a + cos * b)
    return GuideFrame(center, (float(width), float(length)), angle)


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
    if kind == "rectangle_guide":
        frame = rectangle_frame(mask)
        return (Image.new("RGBA", (1, 1), color), frame) if frame else None
    if geometry is None:
        return None

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
    if not bounds:
        return None
    left, top, right, bottom = bounds
    frame = GuideFrame(((left + right) / 2, (top + bottom) / 2), (right - left, bottom - top))
    return rgba.crop(bounds), frame


def composite_guide(canvas, artwork, scale, offset, rotation):
    if artwork is None:
        return
    image, frame = artwork
    width, height = (max(1, size * stretch) for size, stretch in zip(frame.size, scale))
    cx = frame.center[0] + offset[0] * canvas.width
    cy = frame.center[1] + offset[1] * canvas.height
    angle = math.radians(frame.angle + rotation)
    cos, sin = math.cos(angle), math.sin(angle)
    extent_x = (abs(cos) * width + abs(sin) * height) / 2
    extent_y = (abs(sin) * width + abs(cos) * height) / 2
    x0, y0 = max(0, math.floor(cx - extent_x)), max(0, math.floor(cy - extent_y))
    x1, y1 = min(canvas.width, math.ceil(cx + extent_x)), min(canvas.height, math.ceil(cy + extent_y))
    if x0 >= x1 or y0 >= y1:
        return
    # Pillow expects the inverse map from the clipped destination to the artwork.
    sx, sy = image.width / width, image.height / height
    dx, dy = x0 - cx, y0 - cy
    affine = (
        sx * cos, sx * sin, image.width / 2 + sx * (cos * dx + sin * dy),
        -sy * sin, sy * cos, image.height / 2 + sy * (-sin * dx + cos * dy),
    )
    visible = image.transform((x1 - x0, y1 - y0), Image.Transform.AFFINE, affine, Image.Resampling.BILINEAR)
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
            composite_guide(canvas, artwork, group["guide_scale"], group["guide_offset"], group["guide_rotation"])
        output[index] = torch.from_numpy(np.array(canvas)).to(output.dtype).div_(255)
