import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter

from .video_guides import GuideFrame, rectangle_frame


try:
    import cv2
except ImportError:
    cv2 = None


@dataclass(frozen=True)
class RectangleSample:
    placement: GuideFrame | None
    motion: np.ndarray | None = None


def point_motion(previous, image, previous_mask, mask):
    """Estimate the shared translation of visible features."""
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
    if 0.78 <= scale <= 1.28 and abs(rotation) <= 26:
        return np.median(target[kept] - source[kept], axis=0)


def sample_rectangle(image, mask, previous, previous_mask):
    placement = rectangle_frame(mask)
    motion = None
    if placement is not None and previous is not None and previous_mask is not None:
        motion = point_motion(previous, image, previous_mask, mask)
    return RectangleSample(placement, motion)


def is_shot_cut(previous, current):
    left, right = previous[::8, ::8].astype(np.int16), current[::8, ::8].astype(np.int16)
    return np.mean(np.abs(left - right)) > 49


def stabilize_span(samples, anchors, fps):
    placements = [sample.placement for sample in samples]
    count = len(samples)
    window = 2 * max(1, round(fps * 0.12)) + 1
    measured = median_filter(
        np.array([frame.center for frame in placements]), size=(window, 1), mode="nearest",
    )
    widths, heights = np.array([frame.size for frame in placements]).T
    # Keep one aspect ratio per visible span. Filter apparent zoom in log space
    # so approaching and receding are smoothed by the same relative amount.
    aspect = np.median(widths / heights)
    log_heights = median_filter(np.log(heights), size=window, mode="nearest")
    heights = np.exp(gaussian_filter1d(log_heights, fps * 0.22, mode="nearest"))
    correction = 1 - math.exp(-1 / (fps * 0.18))
    if not anchors:
        anchors = [count // 2]

    def walk(start, end):
        direction = 1 if start <= end else -1
        center = measured[start]
        yield center
        for current in range(start + direction, end + direction, direction):
            motion = samples[max(current, current - direction)].motion
            if motion is not None and direction < 0:
                motion = -motion
            if motion is not None:
                center = center + motion
            error = measured[current] - center
            limit = max(2, heights[current] * 0.16)
            center = center + error * (correction * min(1.0, limit / max(1e-6, np.linalg.norm(error))))
            yield center

    centers = np.empty((count, 2))
    centers[:anchors[0] + 1] = list(walk(anchors[0], 0))[::-1]
    centers[anchors[-1]:] = list(walk(anchors[-1], count - 1))
    for left, right in zip(anchors, anchors[1:]):
        forward = np.array(list(walk(left, right)))
        backward = np.array(list(walk(right, left))[::-1])
        mix = np.linspace(0, 1, right - left + 1)[:, None]
        centers[left:right + 1] = forward * (1 - mix) + backward * mix
    centers = gaussian_filter1d(centers, fps * 0.15, axis=0, mode="nearest")
    return [
        GuideFrame((float(x), float(y)), (float(height * aspect), float(height)))
        for (x, y), height in zip(centers, heights)
    ]


def stabilize_rectangles(samples, anchors, cuts, fps):
    """Stabilize each visible span without mixing motion across cuts or gaps."""
    result = {}
    start = None
    for index, sample in enumerate([*samples, RectangleSample(None)]):
        if start is not None and (sample.placement is None or index in cuts):
            local_anchors = [frame - start for frame in anchors if start <= frame < index]
            stabilized = stabilize_span(samples[start:index], local_anchors, fps)
            result.update((start + offset, frame) for offset, frame in enumerate(stabilized))
            start = None
        if start is None and sample.placement is not None:
            start = index
    return result
