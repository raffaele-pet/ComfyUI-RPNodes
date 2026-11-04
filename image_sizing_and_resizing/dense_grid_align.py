import math

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


def _visible_rgb(image):
    if image.shape[-1] == 4:
        alpha = np.clip(image[..., 3:4], 0, 1)
        return image[..., :3] * alpha + 0.5 * (1 - alpha)
    return image[..., :3]


def _structure(image):
    gray = cv2.cvtColor(_visible_rgb(image), cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (0, 0), 1.2)
    mean = cv2.GaussianBlur(gray, (0, 0), 8)
    variance = cv2.GaussianBlur(gray * gray, (0, 0), 8) - mean * mean
    normalized = (gray - mean) / np.sqrt(np.maximum(variance, 0) + 0.001)
    return np.clip(normalized * 42 + 128, 0, 255).astype(np.uint8)


def _correlation(a, b, size=19):
    mean_a = cv2.boxFilter(a, -1, (size, size))
    mean_b = cv2.boxFilter(b, -1, (size, size))
    var_a = np.maximum(cv2.boxFilter(a * a, -1, (size, size)) - mean_a * mean_a, 0)
    var_b = np.maximum(cv2.boxFilter(b * b, -1, (size, size)) - mean_b * mean_b, 0)
    covariance = cv2.boxFilter(a * b, -1, (size, size)) - mean_a * mean_b
    return np.clip(covariance / np.sqrt(var_a * var_b + 1e-7), -1, 1), var_a


def _optical_flow(a, b):
    solver = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    solver.setFinestScale(0)
    solver.setGradientDescentIterations(40)
    solver.setVariationalRefinementIterations(5)
    return solver.calc(a, b, None)


def _regularize(data, weight, pinned, rigidity):
    displacement = data.copy()
    kernel = np.array([[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]], dtype=np.float32)
    denominator = weight[..., None] + rigidity + 0.015
    for _ in range(160):
        neighbors = cv2.filter2D(displacement, -1, kernel, borderType=cv2.BORDER_REPLICATE)
        displacement = (weight[..., None] * data + rigidity * neighbors) / denominator
        displacement[pinned] = 0
    return displacement


def _constrain_mesh(mesh, pinned, step_x, step_y):
    for _ in range(100):
        for axis, step in [(1, step_x), (0, step_y)]:
            first = (slice(None), slice(None, -1)) if axis == 1 else (slice(None, -1), slice(None))
            second = (slice(None), slice(1, None)) if axis == 1 else (slice(1, None), slice(None))
            delta = mesh[second] - mesh[first]
            norm = np.linalg.norm(delta, axis=-1, keepdims=True)
            excess = delta * np.maximum(0, 1 - 0.45 * step / np.maximum(norm, 1e-7))
            share_first = np.where(pinned[first], 0, np.where(pinned[second], 1, 0.5))[..., None]
            share_second = np.where(pinned[second], 0, np.where(pinned[first], 1, 0.5))[..., None]
            mesh[first] += 0.8 * excess * share_first
            mesh[second] -= 0.8 * excess * share_second
        mesh[pinned] = 0
    maximum = max(np.linalg.norm(np.diff(mesh, axis=1), axis=-1).max() / step_x,
                  np.linalg.norm(np.diff(mesh, axis=0), axis=-1).max() / step_y)
    safety = min(1.0, 0.45 / max(float(maximum), 1e-7))
    return mesh * safety, safety


def _interpolate_mesh(mesh, width, height):
    ny, nx = mesh.shape[:2]
    xs = np.linspace(0, nx - 1, width, dtype=np.float32)
    ys = np.linspace(0, ny - 1, height, dtype=np.float32)
    x0, y0 = xs.astype(np.int32), ys.astype(np.int32)
    tx, ty = (xs - x0.astype(np.float32))[None, :, None], (ys - y0.astype(np.float32))[:, None, None]
    horizontal = mesh[:, x0] * (1 - tx) + mesh[:, np.minimum(x0 + 1, nx - 1)] * tx
    return (horizontal[y0] * (1 - ty) + horizontal[np.minimum(y0 + 1, ny - 1)] * ty).astype(np.float32)


def estimate_deformation(reference, image, max_shift=96, grid_spacing=16, anchor_tolerance=1.5,
                         rigidity=0.5, analysis_max_side=2048, minimum_match=0.55, protect_mask=None):
    if cv2 is None:
        raise RuntimeError("RP tile grid alignment requires OpenCV. Install opencv-python in the ComfyUI Python environment.")
    height, width = image.shape[:2]
    if reference.shape[:2] != (height, width):
        reference = cv2.resize(reference, (width, height), interpolation=cv2.INTER_CUBIC)
    ny = max(2, math.ceil((height - 1) / grid_spacing) + 1)
    nx = max(2, math.ceil((width - 1) / grid_spacing) + 1)
    if min(width, height) < 12 or np.array_equal(_visible_rgb(reference), _visible_rgb(image)):
        return np.zeros((height, width, 2), np.float32), np.zeros((ny, nx, 2), np.float32), np.ones((ny, nx), bool), {
            "grid_columns": nx, "grid_rows": ny, "grid_points": nx * ny, "anchored_points": nx * ny,
            "reliable_moving_fraction": 0.0, "mean_correlation_before": 1.0,
            "mesh_safety_scale": 1.0, "analysis_size": [width, height], "max_displacement": 0.0,
        }
    scale = min(1.0, analysis_max_side / max(width, height))
    aw, ah = max(12, round(width * scale)), max(12, round(height * scale))
    reference_small = cv2.resize(reference, (aw, ah), interpolation=cv2.INTER_AREA)
    image_small = cv2.resize(image, (aw, ah), interpolation=cv2.INTER_AREA)
    a, b = _structure(reference_small), _structure(image_small)
    forward, backward = _optical_flow(a, b), _optical_flow(b, a)
    yy, xx = np.mgrid[:ah, :aw].astype(np.float32)
    mapped_x, mapped_y = xx + forward[..., 0], yy + forward[..., 1]
    inverse = cv2.remap(backward, mapped_x, mapped_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    consistency = np.linalg.norm(forward + inverse, axis=-1)
    gray_a, gray_b = a.astype(np.float32) / 255, b.astype(np.float32) / 255
    warped_gray = cv2.remap(gray_b, mapped_x, mapped_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    score, texture = _correlation(gray_a, warped_gray)
    baseline, _ = _correlation(gray_a, gray_b)
    improvement = score - baseline
    motion = np.linalg.norm(forward * np.array([width / aw, height / ah], np.float32), axis=-1)
    limit = max_shift
    tolerance = anchor_tolerance
    inside = (mapped_x >= 1) & (mapped_x < aw - 2) & (mapped_y >= 1) & (mapped_y < ah - 2)
    reliable = inside & (score >= minimum_match) & (consistency <= 2.0) & (texture >= 0.0003) & (motion <= limit)
    moving = reliable & (motion > tolerance) & (improvement >= 0.025)
    confidence = np.clip((score - minimum_match) / max(0.01, 1 - minimum_match), 0, 1)
    confidence *= np.exp(-0.5 * (consistency / 1.0) ** 2) * moving
    stable = (baseline >= 0.8) & ((motion <= tolerance) | (improvement < 0.025))
    gx, gy = np.meshgrid(np.linspace(0, aw - 1, nx, dtype=np.float32), np.linspace(0, ah - 1, ny, dtype=np.float32))
    window = max(3, round(grid_spacing * scale) | 1)
    smooth_weight = cv2.boxFilter(confidence, -1, (window, window))
    weighted_flow = cv2.boxFilter(forward * confidence[..., None], -1, (window, window))
    data = cv2.remap(weighted_flow, gx, gy, cv2.INTER_LINEAR) / np.maximum(cv2.remap(smooth_weight, gx, gy, cv2.INTER_LINEAR)[..., None], 1e-7)
    weight = cv2.remap(smooth_weight, gx, gy, cv2.INTER_LINEAR)
    stable_fraction = cv2.remap(cv2.boxFilter(stable.astype(np.float32), -1, (window, window)), gx, gy, cv2.INTER_LINEAR)
    pinned = (stable_fraction >= 0.65) & (weight < 0.15)
    if protect_mask is not None:
        protected = cv2.resize(protect_mask, (width, height), interpolation=cv2.INTER_NEAREST) >= 0.5
        protected = cv2.dilate(protected.astype(np.uint8), np.ones((2 * grid_spacing + 1, 2 * grid_spacing + 1), np.uint8))
        px, py = np.meshgrid(np.linspace(0, width - 1, nx, dtype=np.float32), np.linspace(0, height - 1, ny, dtype=np.float32))
        pinned |= cv2.remap(protected, px, py, cv2.INTER_NEAREST).astype(bool)
    data[pinned] = 0
    mesh = _regularize(data, weight, pinned, max(0.01, rigidity))
    mesh[..., 0] *= width / aw
    mesh[..., 1] *= height / ah
    magnitude = np.linalg.norm(mesh, axis=-1)
    transition = np.clip((magnitude - anchor_tolerance) / max(anchor_tolerance, 0.25), 0, 1)
    transition = transition * transition * (3 - 2 * transition)
    mesh *= transition[..., None]
    mesh, safety = _constrain_mesh(mesh, pinned, (width - 1) / (nx - 1), (height - 1) / (ny - 1))
    mesh *= min(1.0, max_shift / max(float(np.linalg.norm(mesh, axis=-1).max()), 1e-7))
    displacement = _interpolate_mesh(mesh, width, height)
    return displacement, mesh, pinned, {
        "grid_columns": nx, "grid_rows": ny, "grid_points": nx * ny,
        "anchored_points": int(pinned.sum()), "reliable_moving_fraction": float(moving.mean()),
        "mean_correlation_before": float(baseline.mean()), "mesh_safety_scale": safety,
        "analysis_size": [aw, ah], "max_displacement": float(np.linalg.norm(displacement, axis=-1).max()),
    }


def warp_image(image, displacement, strength=1.0):
    yy, xx = np.mgrid[:image.shape[0], :image.shape[1]].astype(np.float32)
    field = displacement * strength
    result = np.clip(cv2.remap(image, xx + field[..., 0], yy + field[..., 1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE), 0, 1)
    result[np.all(field == 0, axis=-1)] = image[np.all(field == 0, axis=-1)]
    return result


def grid_preview(reference, image, mesh, pinned, strength):
    height, width = image.shape[:2]
    preview_scale = min(1.0, 720 / max(width, height))
    pw, ph = max(1, round(width * preview_scale)), max(1, round(height * preview_scale))
    preview = np.zeros((ph + 28, pw * 2, 3), np.uint8)
    preview[28:, :pw] = np.clip(cv2.resize(_visible_rgb(reference), (pw, ph)) * 255, 0, 255).astype(np.uint8)
    preview[28:, pw:] = np.clip(cv2.resize(_visible_rgb(image), (pw, ph)) * 255, 0, 255).astype(np.uint8)
    cv2.putText(preview, 'REFERENCE: fixed grid', (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1)
    cv2.putText(preview, 'EDITED: orange = motion, green = anchors', (pw + 8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1)
    ny, nx = mesh.shape[:2]
    xx, yy = np.meshgrid(np.linspace(0, width - 1, nx), np.linspace(0, height - 1, ny))
    fixed = np.stack([xx * pw / width, yy * ph / height + 28], axis=-1)
    moved = fixed + mesh * strength * np.array([pw / width, ph / height])
    moved[..., 0] += pw
    step = max(1, math.ceil(nx / 60))
    rows, columns = list(range(0, ny, step)), list(range(0, nx, step))
    for grid in [fixed, moved]:
        for row in rows:
            cv2.polylines(preview, [np.round(grid[row]).astype(np.int32)], False, (60, 180, 220), 1, cv2.LINE_AA)
        for column in columns:
            cv2.polylines(preview, [np.round(grid[:, column]).astype(np.int32)], False, (60, 180, 220), 1, cv2.LINE_AA)
    for row in rows:
        for column in columns:
            color = ((65, 235, 100) if pinned[row, column] else
                     (255, 155, 40) if np.linalg.norm(mesh[row, column]) * strength > 0.25 else (160, 160, 160))
            cv2.circle(preview, tuple(np.round(moved[row, column]).astype(int)), 1, color, -1)
    return preview.astype(np.float32) / 255
