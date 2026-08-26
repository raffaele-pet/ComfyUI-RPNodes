# Copyright (C) 2026 NikoDemon80 and contributors
# Modified for ComfyUI-RPNodes in 2026. Licensed under GPL-3.0-or-later.

"""Behavioral checks for the native MiniMax H3 keyframe guide path."""

from __future__ import annotations

import torch


def _conditioning_ref_requirements(conditioning) -> tuple[bool, bool]:
    need_video = False
    need_audio = False
    for item in conditioning or []:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        metadata = item[1]
        if not isinstance(metadata, dict):
            continue
        for reference in metadata.get("minimax_refs", []) or []:
            if not isinstance(reference, dict):
                continue
            if reference.get("latent") is not None:
                need_video = True
            if reference.get("audio_latent") is not None:
                need_audio = True
    return need_video, need_audio


def native_keyframe_status() -> dict[str, object]:
    """Probe arbitrary guide placement and coexistence with Ref2VA payloads."""
    import comfy.ldm.minimax.model as minimax_model
    import comfy.model_base as model_base

    status: dict[str, object] = {
        "arbitrary_guides": False,
        "guide_audio_segment": False,
        "keyframe_ref_merge": False,
        "keyframe_ref_audio_merge": False,
    }

    layout_class = getattr(minimax_model, "PackedLayout", None)
    if layout_class is not None:
        try:
            keyframe = {
                "resolved_frame_index": 3,
                "latent": torch.zeros((1, 24, 1, 2, 2)),
                "audio_latent": torch.zeros((1, 32, 2, 2)),
            }
            references = [{"kind": "image", "latent_h": 2, "latent_w": 2}]
            layout = layout_class(
                7, 7, 2, 2, 16, keyframes=[keyframe], refs=references
            )
            video_spans = [
                (start, end)
                for start, end, kind in layout.segments
                if kind == "cond"
            ]
            audio_spans = [
                (start, end)
                for start, end, kind in layout.segments
                if kind == "cond_audio"
            ]
            expected = 8.0 + float(minimax_model.FRAME_RESCALE) * 3.0
            if video_spans:
                start, end = video_spans[0]
                values = layout.position_ids[start:end, 0]
                status["arbitrary_guides"] = bool(
                    torch.allclose(
                        values,
                        torch.full_like(values, expected),
                        atol=1e-9,
                        rtol=0.0,
                    )
                )
            if audio_spans:
                start, end = audio_spans[0]
                values = layout.position_ids[start:end, 0]
                status["guide_audio_segment"] = bool(
                    torch.allclose(
                        values[:2],
                        torch.tensor([expected, expected + 1.0], dtype=values.dtype),
                        atol=1e-9,
                        rtol=0.0,
                    )
                )
        except Exception as exc:  # pragma: no cover - runtime capability report
            status["layout_error"] = repr(exc)

    h3_class = getattr(model_base, "MiniMaxH3", None)
    extra_conds = getattr(h3_class, "extra_conds", None) if h3_class else None
    if extra_conds is not None:
        try:
            probe = h3_class.__new__(h3_class)
            probe.concat_keys = ()
            probe.latent_shapes = None
            keyframe_video = torch.zeros((1, 24, 1, 2, 2))
            reference_video = torch.ones((1, 24, 1, 2, 2))
            keyframe_audio = torch.zeros((1, 32, 2, 2))
            reference_audio = torch.ones((1, 32, 2, 2))
            result = extra_conds(
                probe,
                minimax_keyframes=[
                    {
                        "resolved_frame_index": 0,
                        "latent": keyframe_video,
                        "audio_latent": keyframe_audio,
                    }
                ],
                minimax_refs=[
                    {
                        "kind": "image",
                        "latent_h": 2,
                        "latent_w": 2,
                        "latent": reference_video,
                    },
                    {
                        "kind": "audio",
                        "ref_audio_t": 2,
                        "audio_latent": reference_audio,
                    },
                ],
            )
            wrapped = result.get("minimax_payload") if isinstance(result, dict) else None
            payload = getattr(wrapped, "cond", wrapped)
            if isinstance(payload, dict):
                videos = payload.get("cond_video_latents", [])
                audios = payload.get("cond_audio_latents", [])
                status["keyframe_ref_merge"] = (
                    len(videos) == 2
                    and videos[0] is keyframe_video
                    and videos[1] is reference_video
                )
                status["keyframe_ref_audio_merge"] = (
                    len(audios) == 2
                    and audios[0] is keyframe_audio
                    and audios[1] is reference_audio
                )
        except Exception as exc:  # pragma: no cover - runtime capability report
            status["extra_conds_error"] = repr(exc)
    return status


def ensure_h3_keyframe_support(conditioning) -> None:
    status = native_keyframe_status()
    need_video_merge, need_audio_merge = _conditioning_ref_requirements(conditioning)
    missing = []
    if not status.get("arbitrary_guides"):
        missing.append("arbitrary guide placement")
    if not status.get("guide_audio_segment"):
        missing.append("guide audio placement")
    if need_video_merge and not status.get("keyframe_ref_merge"):
        missing.append("keyframe/reference video merge")
    if need_audio_merge and not status.get("keyframe_ref_audio_merge"):
        missing.append("keyframe/reference audio merge")
    if missing:
        raise RuntimeError(
            "RP H3-Keyframes requires current native MiniMax H3 guide support; "
            f"missing {', '.join(missing)}. Capability report: {status!r}"
        )
