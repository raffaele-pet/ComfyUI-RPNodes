"""Multistage media analysis that respects Gemma 4's one-video limitation."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from .constants import FPS
from .gemma import GemmaRunner
from .manifests import ReferenceManifest
from .media import letterbox_image_batch, prepare_reference_video, trim_audio
from .prompts import (
    keyframes_analysis_prompt,
    reference_audio_analysis_prompt,
    reference_images_analysis_prompt,
    reference_video_analysis_prompt,
)


def _noop() -> None:
    return None


def clean_analysis_text(text: str) -> str:
    value = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    value = re.sub(r"<think>.*?</think>", "", value, flags=re.DOTALL | re.IGNORECASE).strip()
    orphan_closes = list(re.finditer(r"</think>", value, flags=re.IGNORECASE))
    if orphan_closes:
        value = value[orphan_closes[-1].end() :].strip()
    if value.lower().startswith("<think>") and "</think>" not in value.lower():
        return ""
    value = re.sub(r"^```(?:text|markdown)?\s*\n?", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\n?```\s*$", "", value).strip()
    first_record = re.search(r"(?m)^<(?:Picture|Video|Audio) [1-9]\d*>:", value)
    if first_record:
        value = value[first_record.start() :]
    elif re.search(
        r"(?i)\b(?:the user wants me|i need to|instructions? for (?:the )?output|"
        r"i must (?:not )?restate the task)\b",
        value,
    ):
        return ""
    return value


def ensure_analysis_records(text: str, labels: list[str]) -> str:
    """Guarantee traceable, non-hallucinated records for every analyzed label."""

    value = clean_analysis_text(text)
    missing = [label for label in labels if label not in value]
    if missing:
        fallback = (
            "No reliable observation was produced within the compact analysis "
            "budget; use only the connected source role and explicit user request."
        )
        additions = [f"{label}: {fallback}" for label in missing]
        value = "\n\n".join(part for part in (value, *additions) if part.strip())
    return value


def analyze_base_media(
    runner: GemmaRunner,
    *,
    mode: str,
    first_frame: Any = None,
    last_frame: Any = None,
    max_new_tokens: int = 512,
    seed: int = 0,
    after_call: Callable[[], None] = _noop,
) -> dict[str, str]:
    observations: dict[str, str] = {}
    entries: list[tuple[str, str, str, Any]] = []
    if first_frame is not None:
        entries.append(
            (
                "<Picture 1>",
                "first_frame",
                "literal first frame at 0.00 seconds",
                first_frame,
            )
        )
    if last_frame is not None:
        entries.append(
            (
                "<Picture 2>" if mode == "FL2VA" else "<Picture 1>",
                "last_frame",
                "literal final frame at the effective end time",
                last_frame,
            )
        )

    if entries:
        batch = letterbox_image_batch(entry[3] for entry in entries)
        result = runner.generate_media_analysis(
            keyframes_analysis_prompt(
                [(label, socket, temporal_role) for label, socket, temporal_role, _ in entries]
            ),
            image=batch,
            max_new_tokens=max_new_tokens,
            seed=seed,
        )
        observations["keyframes"] = clean_analysis_text(result)
        after_call()
    return observations


def analyze_reference_media(
    runner: GemmaRunner,
    *,
    manifest: ReferenceManifest,
    ref_image_0: Any = None,
    ref_image_1: Any = None,
    ref_image_2: Any = None,
    ref_video_0: Any = None,
    ref_video_1: Any = None,
    ref_video_audio_0: Any = None,
    ref_audio_0: Any = None,
    target_frame_count: int,
    max_new_tokens: int = 512,
    seed: int = 0,
    after_call: Callable[[], None] = _noop,
) -> dict[str, str]:
    observations: dict[str, str] = {}
    call_index = 0
    analysis_seconds = float(target_frame_count) / FPS
    image_values = {
        "ref_image_0": ref_image_0,
        "ref_image_1": ref_image_1,
        "ref_image_2": ref_image_2,
    }
    image_entries = [
        (asset.label, asset.socket)
        for asset in manifest.pictures
        if image_values.get(asset.socket) is not None
    ]
    if image_entries:
        batch = letterbox_image_batch(image_values[socket] for _, socket in image_entries)
        # The Gemma E4B decoder can spend a small fixed prefix on channel
        # transition before the direct answer. Add only 64 tokens for each
        # additional Picture so every item in the shared batch gets a record,
        # without adding another green-bar generation pass.
        image_token_budget = max_new_tokens + 64 * (len(image_entries) - 1)
        result = runner.generate_media_analysis(
            reference_images_analysis_prompt(image_entries),
            image=batch,
            max_new_tokens=image_token_budget,
            seed=seed + call_index,
        )
        observations["reference_images"] = ensure_analysis_records(
            result,
            [label for label, _ in image_entries],
        )
        call_index += 1
        after_call()

    video_values = {
        "ref_video_0": ref_video_0,
        "ref_video_1": ref_video_1,
    }
    for video_asset in manifest.videos:
        video = video_values[video_asset.socket]
        audio_asset = None
        audio = None
        if video_asset.socket == "ref_video_0" and ref_video_audio_0 is not None:
            audio_asset = manifest.asset_for_socket("ref_video_audio_0")
            audio = trim_audio(ref_video_audio_0, max_seconds=analysis_seconds)
        result = runner.generate_media_analysis(
            reference_video_analysis_prompt(
                video_asset.label,
                video_asset.socket,
                audio_asset.label if audio_asset else None,
            ),
            video=prepare_reference_video(
                video,
                target_frame_count=target_frame_count,
            ),
            audio=audio,
            max_new_tokens=max_new_tokens + (64 if audio_asset else 0),
            seed=seed + call_index,
        )
        expected_labels = [video_asset.label]
        if audio_asset is not None:
            expected_labels.append(audio_asset.label)
        observations[video_asset.socket] = ensure_analysis_records(
            result,
            expected_labels,
        )
        call_index += 1
        after_call()

    if ref_audio_0 is not None:
        audio_asset = manifest.asset_for_socket("ref_audio_0")
        if audio_asset is None:
            raise RuntimeError("Reference manifest lost the ref_audio_0 mapping.")
        result = runner.generate_media_analysis(
            reference_audio_analysis_prompt(audio_asset.label, audio_asset.socket),
            audio=trim_audio(ref_audio_0, max_seconds=analysis_seconds),
            max_new_tokens=max_new_tokens,
            seed=seed + call_index,
        )
        observations["ref_audio_0"] = ensure_analysis_records(
            result,
            [audio_asset.label],
        )
        after_call()

    return observations
