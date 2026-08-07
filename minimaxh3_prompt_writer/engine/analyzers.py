"""Multistage media analysis that respects Gemma 4's one-video limitation."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from .constants import FPS
from .gemma import GemmaRunner
from .manifests import ReferenceManifest
from .media import first_image, prepare_reference_video, trim_audio
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


_FALLBACK_OBSERVATION = (
    "No reliable observation was produced within the compact analysis "
    "budget; use only the connected source role and explicit user request."
)


def _recover_single_source_analysis(text: str) -> str:
    """Recover visible/audible facts from Gemma's occasional meta preamble.

    Gemma 4 E4B can describe the media accurately under an ``Image Analysis``
    or similar heading before it emits the requested labelled summary. A short
    generation may end during that useful block. This extractor drops the task
    paraphrase and keeps only the descriptive portion; callers add the trusted
    H3 label themselves.
    """

    value = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    value = re.sub(r"<think>.*?</think>", "", value, flags=re.DOTALL | re.IGNORECASE).strip()
    heading = re.search(
        r"(?im)^\s*(?:#{1,6}\s*)?(?:\*\*)?"
        r"(?:(?:image|visual|video|audio|media)\s+)?analysis\s*:\s*(?:\*\*)?\s*$",
        value,
    )
    if not heading:
        return ""
    value = value[heading.end() :].strip()
    lines: list[str] = []
    for raw_line in value.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^(?:[-*+]\s+|\d+[.)]\s+)", "", line)
        line = re.sub(r"\*\*([^*]+):\*\*", r"\1:", line)
        line = line.replace("**", "").strip()
        if line:
            lines.append(line)
    return "\n".join(lines).strip()


def _has_labeled_record(text: str, label: str) -> bool:
    match = re.search(
        rf"(?ms)^{re.escape(label)}:[ \t]*(.*?)"
        r"(?=^<(?:Picture|Video|Audio) [1-9]\d*>:|\Z)",
        text,
    )
    if not match:
        return False
    body = match.group(1).strip()
    return bool(body and not body.startswith(_FALLBACK_OBSERVATION))


def ensure_analysis_records(text: str, labels: list[str]) -> str:
    """Guarantee traceable, non-hallucinated records for every analyzed label."""

    value = clean_analysis_text(text)
    if len(labels) == 1 and not _has_labeled_record(value, labels[0]):
        recovered = _recover_single_source_analysis(text)
        if recovered:
            value = f"{labels[0]}: {recovered}"
    missing = [label for label in labels if not _has_labeled_record(value, label)]
    if missing:
        additions = [f"{label}: {_FALLBACK_OBSERVATION}" for label in missing]
        value = "\n\n".join(part for part in (value, *additions) if part.strip())
    return value


def _generate_complete_record(
    runner: GemmaRunner,
    *,
    prompt: str,
    label: str,
    max_new_tokens: int,
    seed: int,
    after_call: Callable[[], None],
    image: Any = None,
    video: Any = None,
    audio: Any = None,
) -> str:
    """Analyze exactly one labelled asset, retrying only an empty observation."""

    first_budget = max(256, int(max_new_tokens))
    budgets = (first_budget, max(512, first_budget * 2))
    for attempt, budget in enumerate(budgets):
        result = runner.generate_media_analysis(
            prompt,
            image=image,
            video=video,
            audio=audio,
            max_new_tokens=budget,
            seed=seed + attempt * 10_000,
        )
        after_call()
        value = ensure_analysis_records(result, [label])
        if _has_labeled_record(value, label):
            return value
    raise ValueError(
        f"Gemma could not produce a usable media observation for {label} "
        f"after {len(budgets)} attempts (up to {budgets[-1]} tokens). "
        "Check the connected media and Gemma 4 CLIP."
    )


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

    for call_index, (label, socket, temporal_role, image) in enumerate(entries):
        observations[socket] = _generate_complete_record(
            runner,
            prompt=keyframes_analysis_prompt([(label, socket, temporal_role)]),
            label=label,
            image=first_image(image),
            max_new_tokens=max_new_tokens,
            seed=seed + call_index,
            after_call=after_call,
        )
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
    for label, socket in image_entries:
        observations[socket] = _generate_complete_record(
            runner,
            prompt=reference_images_analysis_prompt([(label, socket)]),
            label=label,
            image=first_image(image_values[socket]),
            max_new_tokens=max_new_tokens,
            seed=seed + call_index,
            after_call=after_call,
        )
        call_index += 1

    video_values = {
        "ref_video_0": ref_video_0,
        "ref_video_1": ref_video_1,
    }
    for video_asset in manifest.videos:
        video = video_values[video_asset.socket]
        observations[video_asset.socket] = _generate_complete_record(
            runner,
            prompt=reference_video_analysis_prompt(
                video_asset.label, video_asset.socket, None
            ),
            label=video_asset.label,
            video=prepare_reference_video(
                video,
                target_frame_count=target_frame_count,
            ),
            max_new_tokens=max_new_tokens,
            seed=seed + call_index,
            after_call=after_call,
        )
        call_index += 1

    if ref_video_audio_0 is not None:
        audio_asset = manifest.asset_for_socket("ref_video_audio_0")
        if audio_asset is None:
            raise RuntimeError("Reference manifest lost the ref_video_audio_0 mapping.")
        observations["ref_video_audio_0"] = _generate_complete_record(
            runner,
            prompt=reference_audio_analysis_prompt(
                audio_asset.label, audio_asset.socket
            ),
            label=audio_asset.label,
            audio=trim_audio(ref_video_audio_0, max_seconds=analysis_seconds),
            max_new_tokens=max_new_tokens,
            seed=seed + call_index,
            after_call=after_call,
        )
        call_index += 1

    if ref_audio_0 is not None:
        audio_asset = manifest.asset_for_socket("ref_audio_0")
        if audio_asset is None:
            raise RuntimeError("Reference manifest lost the ref_audio_0 mapping.")
        observations["ref_audio_0"] = _generate_complete_record(
            runner,
            prompt=reference_audio_analysis_prompt(
                audio_asset.label, audio_asset.socket
            ),
            label=audio_asset.label,
            audio=trim_audio(ref_audio_0, max_seconds=analysis_seconds),
            max_new_tokens=max_new_tokens,
            seed=seed + call_index,
            after_call=after_call,
        )

    return observations
