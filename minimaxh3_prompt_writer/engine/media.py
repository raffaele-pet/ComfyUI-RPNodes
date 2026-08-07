"""Safe media preparation for the public Gemma 4 tokenizer API."""

from __future__ import annotations

import math
from typing import Any, Iterable


def first_image(image: Any) -> Any:
    """Mirror native H3 keyframe/reference behavior: use batch element zero."""

    if image is None:
        return None
    if getattr(image, "ndim", None) != 4 or image.shape[0] < 1:
        raise ValueError("IMAGE inputs must have ComfyUI shape [B, H, W, C].")
    return image[:1, :, :, :3]


def letterbox_image_batch(images: Iterable[Any], long_edge: int = 1024) -> Any:
    """Create one aspect-preserving batch for Gemma's multi-image call.

    Each source uses its first batch element. Gray padding makes differently
    sized/aspected references batchable without stretching identities.
    """

    import torch
    import torch.nn.functional as functional

    prepared = [first_image(image) for image in images]
    prepared = [image for image in prepared if image is not None]
    if not prepared:
        return None

    scaled_shapes: list[tuple[int, int]] = []
    for image in prepared:
        height, width = int(image.shape[1]), int(image.shape[2])
        scale = min(1.0, float(long_edge) / max(height, width))
        scaled_shapes.append((max(1, round(height * scale)), max(1, round(width * scale))))

    canvas_h = max(height for height, _ in scaled_shapes)
    canvas_w = max(width for _, width in scaled_shapes)
    canvas_h = max(16, int(math.ceil(canvas_h / 16.0) * 16))
    canvas_w = max(16, int(math.ceil(canvas_w / 16.0) * 16))

    batch = []
    for image in prepared:
        source = image[0].movedim(-1, 0).unsqueeze(0).float()
        height, width = int(source.shape[2]), int(source.shape[3])
        scale = min(canvas_h / height, canvas_w / width, 1.0)
        target_h = max(1, round(height * scale))
        target_w = max(1, round(width * scale))
        if target_h != height or target_w != width:
            source = functional.interpolate(
                source,
                size=(target_h, target_w),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
        canvas = torch.full(
            (1, 3, canvas_h, canvas_w),
            0.5,
            dtype=source.dtype,
            device=source.device,
        )
        top = (canvas_h - target_h) // 2
        left = (canvas_w - target_w) // 2
        canvas[:, :, top : top + target_h, left : left + target_w] = source[:, :3]
        batch.append(canvas.movedim(1, -1))
    return torch.cat(batch, dim=0).clamp(0.0, 1.0)


def prepare_reference_video(
    video: Any,
    *,
    target_frame_count: int,
) -> Any:
    """Mirror the native H3 reference-video frame selection.

    The native node first caps to the target frame count, rejects fewer than
    five frames, then trims downward to the nearest ``17k + 5`` count. Matching
    that order is important: a long source is valid when the target consumes
    only a shorter prefix.
    """

    if video is None:
        return None
    if getattr(video, "ndim", None) != 4 or video.shape[0] < 1:
        raise ValueError("Reference videos must be IMAGE frame batches [F, H, W, C].")
    source_frames = int(video.shape[0])
    frame_count = min(source_frames, int(target_frame_count))
    if frame_count < 5:
        raise ValueError("MiniMax H3 reference videos need at least 5 frames.")
    while frame_count % 17 != 5:
        frame_count -= 1
    return video[:frame_count, :, :, :3]


def trim_audio(audio: Any, max_seconds: float | None = None) -> Any:
    """Select ComfyUI batch zero and optionally cap duration without mixing channels."""

    if audio is None:
        return None
    # Native LoadAudio returns a dict, while VideoHelperSuite can return a
    # LazyAudioMap that materializes waveform/sample_rate through __getitem__.
    # Duck-typing both keeps the public ComfyUI AUDIO contract intact without
    # importing or depending on another custom node's private class.
    try:
        waveform = audio["waveform"]
    except (KeyError, TypeError, AttributeError):
        raise ValueError("AUDIO inputs must contain waveform and sample_rate.") from None
    getter = getattr(audio, "get", None)
    if callable(getter):
        sample_rate_value = getter("sample_rate", 16000)
    else:
        try:
            sample_rate_value = audio["sample_rate"]
        except (KeyError, TypeError, AttributeError):
            sample_rate_value = 16000
    sample_rate = int(sample_rate_value)
    if sample_rate <= 0:
        raise ValueError("AUDIO sample_rate must be a positive integer.")
    if getattr(waveform, "ndim", None) != 3 or waveform.shape[0] < 1:
        raise ValueError("AUDIO waveform must have ComfyUI shape [B, C, T].")
    max_samples = (
        waveform.shape[-1]
        if max_seconds is None
        else max(1, int(round(max_seconds * sample_rate)))
    )
    trimmed = {"waveform": waveform, "sample_rate": sample_rate}
    # Match native H3's singleton-batch behavior. Gemma's tokenizer squeezes
    # this first dimension before mixing channels and extracting mel features.
    trimmed["waveform"] = waveform[:1, :, :max_samples]
    return trimmed


def media_shape_summary(value: Any, kind: str) -> str:
    if value is None:
        return "not connected"
    if kind in ("image", "video"):
        shape = tuple(int(part) for part in value.shape)
        return f"tensor shape {shape}"
    if kind == "audio":
        waveform = value.get("waveform")
        sample_rate = int(value.get("sample_rate", 0))
        samples = int(waveform.shape[-1])
        duration = samples / sample_rate if sample_rate else 0.0
        return f"waveform shape {tuple(int(part) for part in waveform.shape)}, {sample_rate} Hz, {duration:.3f}s"
    return type(value).__name__
