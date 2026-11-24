import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter

from .video_guides import GuideFrame, rectangle_measurement


try:
    import cv2
except ImportError:
    cv2 = None


@dataclass(frozen=True)
class RectangleSample:
    placement: GuideFrame | None
    confidence: float
    motion: np.ndarray | None = None


def point_motion(previous, image, previous_mask, mask):
    """Estimate the shared translation, rotation and zoom of visible features."""
    if cv2 is None or not previous_mask.any() or not mask.any():
        return None
    # Keep corners away from mask boundaries, where the background can win.
    interior = cv2.erode(previous_mask.astype(np.uint8), np.ones((5, 5), np.uint8))
    points = cv2.goodFeaturesToTrack(previous, 48, 0.015, 6, mask=interior)
    if points is None or len(points) < 4:
        return None
    moved, forward, _ = cv2.calcOpticalFlowPyrLK(previous, image, points, None, winSize=(21, 21), maxLevel=2)
    if moved is None:
        return None
    returned, backward, _ = cv2.calcOpticalFlowPyrLK(image, previous, moved, None, winSize=(21, 21), maxLevel=2)
    if returned is None:
        return None
    source = points.reshape(-1, 2)
    target = moved.reshape(-1, 2)
    height, width = mask.shape
    x, y = np.rint(target).astype(np.int32).T
    valid = (
        forward.ravel().astype(bool) & backward.ravel().astype(bool)
        & (np.linalg.norm(source - returned.reshape(-1, 2), axis=1) <= 1.5)
        & (x >= 0) & (x < width) & (y >= 0) & (y < height)
    )
    valid &= mask[np.clip(y, 0, height - 1), np.clip(x, 0, width - 1)]
    if valid.sum() < 4:
        return None
    source, target = source[valid], target[valid]
    model, inliers = cv2.estimateAffinePartial2D(
        source, target, method=cv2.RANSAC, ransacReprojThreshold=2.0, maxIters=100,
    )
    if model is None or inliers is None:
        return None
    kept = inliers.ravel().astype(bool)
    if kept.sum() < max(4, len(source) * 0.5) or np.linalg.norm(np.ptp(source[kept], axis=0)) < 12:
        return None
    scale = math.hypot(model[0, 0], model[1, 0])
    rotation = math.degrees(math.atan2(model[1, 0], model[0, 0]))
    return model if 0.78 <= scale <= 1.28 and abs(rotation) <= 26 else None


def sample_rectangle(image, mask, previous, previous_mask):
    placement, confidence = rectangle_measurement(mask)
    motion = None
    if placement is not None and previous is not None and previous_mask is not None:
        motion = point_motion(previous, image, previous_mask, mask)
    return RectangleSample(placement, confidence, motion)


def is_shot_cut(previous, current):
    left, right = previous[::8, ::8].astype(np.int16), current[::8, ::8].astype(np.int16)
    return np.mean(np.abs(left - right)) > 49


def axis_delta(target, current):
    return (target - current + 90) % 180 - 90


def smooth_measurements(samples, fps):
    frames = [sample.placement for sample in samples]
    radius = max(1, round(fps * 0.10))

    def smooth(values):
        return gaussian_filter1d(
            median_filter(values, size=2 * radius + 1, mode="nearest"), max(1, fps * 0.065), mode="nearest",
        )

    centers = np.array([frame.center for frame in frames])
    sizes = np.log([frame.size for frame in frames])
    angles = np.radians([frame.angle for frame in frames]) * 2
    confidence = np.array([sample.confidence for sample in samples])
    # The axis points both ways. Averaging doubled angles treats +89° and
    # -89° as neighboring orientations and downweights almost round masks.
    weight = np.maximum(0, confidence - 0.08)
    along = smooth(np.cos(angles) * weight)
    across = smooth(np.sin(angles) * weight)
    strength = np.hypot(along, across)
    return np.column_stack((
        smooth(centers[:, 0]), smooth(centers[:, 1]), smooth(sizes[:, 0]), smooth(sizes[:, 1]),
        np.degrees(np.arctan2(across, along)) / 2, strength,
    ))


def predicted_step(previous, measured, motion, fps):
    # Follow image motion first, then ease toward the mask to correct drift.
    cx, cy, log_w, log_h, angle = previous
    if motion is not None:
        cx, cy = motion @ (cx, cy, 1)
        magnification = math.log(math.hypot(motion[0, 0], motion[1, 0]))
        log_w += magnification * 0.65
        log_h += magnification * 0.65
        angle += math.degrees(math.atan2(motion[1, 0], motion[0, 0]))

    def gain(seconds):
        return 1 - math.exp(-1 / (fps * seconds))

    size = math.exp(log_h)
    xy = np.array((cx, cy))
    error = measured[:2] - xy
    limit = max(2, size * (0.12 if motion is not None else 0.25))
    error *= min(1.0, limit / max(1e-6, np.linalg.norm(error)))
    center = xy + error * gain(0.23 if motion is not None else 0.095)
    dims = np.array((log_w, log_h))
    dims += np.clip(measured[2:4] - dims, -0.4, 0.4) * gain(0.28)
    if measured[5] > 0.035:
        correction = np.clip(axis_delta(measured[4], angle), -42, 42)
        reliability = min(1.0, measured[5] / 0.40)
        angle += correction * reliability * gain(0.48 if motion is not None else 0.19)
    return np.array((*center, *dims, angle))


def stabilize_span(samples, anchors, fps):
    measured = smooth_measurements(samples, fps)
    count = len(samples)
    if not anchors:
        anchors = [int(np.argmax(measured[:, 5]))]

    def walk(start, end):
        direction = 1 if start <= end else -1
        seed = measured[start, :5].copy()
        if measured[start, 5] < 0.035:
            seed[4] = samples[start].placement.angle
        yield seed
        for current in range(start + direction, end + direction, direction):
            motion = samples[max(current, current - direction)].motion
            if motion is not None and direction < 0:
                motion = cv2.invertAffineTransform(motion)
            seed = predicted_step(seed, measured[current], motion, fps)
            yield seed

    frames = np.empty((count, 5))
    frames[:anchors[0] + 1] = list(walk(anchors[0], 0))[::-1]
    frames[anchors[-1]:] = list(walk(anchors[-1], count - 1))
    for left, right in zip(anchors, anchors[1:]):
        forward = np.array(list(walk(left, right)))
        backward = np.array(list(walk(right, left))[::-1])
        backward[:, 4] = forward[:, 4] + axis_delta(backward[:, 4], forward[:, 4])
        mix = np.linspace(0, 1, right - left + 1)[:, None]
        frames[left:right + 1] = forward * (1 - mix) + backward * mix
    return [
        GuideFrame((float(x), float(y)), (math.exp(w), math.exp(h)), float(axis_delta(angle, 0)))
        for x, y, w, h, angle in frames
    ]


def stabilize_rectangles(samples, anchors, cuts, fps):
    """Stabilize each visible span without mixing motion across cuts or gaps."""
    result = {}
    start = None
    for index, sample in enumerate([*samples, RectangleSample(None, 0.0)]):
        if start is not None and (sample.placement is None or index in cuts):
            local_anchors = [frame - start for frame in anchors if start <= frame < index]
            stabilized = stabilize_span(samples[start:index], local_anchors, fps)
            result.update((start + offset, frame) for offset, frame in enumerate(stabilized))
            start = None
        if start is None and sample.placement is not None:
            start = index
    return result
