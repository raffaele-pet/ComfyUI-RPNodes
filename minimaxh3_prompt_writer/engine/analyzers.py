"""Multistage media analysis that respects Gemma 4's one-video limitation."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from .constants import FPS
from .gemma import GemmaRunner
from .manifests import ReferenceManifest, collect_reference_inputs
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

_META_PHRASES = (
    "internal compact keyframe-analysis task",
    "attached image is evidence",
    "temporal role is authoritative",
    "return one concise english block",
    "spend at most about",
    "first output characters",
    "restate the task",
    "discuss instructions",
    "one compact english block",
    "write one compact english block",
    "record only prompt-useful",
    "starts with `<",
    "review against constraints",
)

_LEADING_META_PHRASES = (
    "first output characters",
    "restate the task",
    "discuss instructions",
    "one compact english block",
    "write one compact english block",
    "record only prompt-useful",
    "start with `<",
)


def _usable_observation_body(body: str) -> bool:
    lowered = body.lower()
    if not body.strip() or body.startswith(_FALLBACK_OBSERVATION):
        return False
    if any(phrase in lowered for phrase in _META_PHRASES):
        return False
    words = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+", body)
    if len(words) < 4:
        return False
    if lowered.count("[unclear]") >= 2:
        return False
    return True


def _normalize_single_labeled_record(text: str, label: str) -> str:
    match = re.search(
        rf"(?ms)^{re.escape(label)}:[ \t]*(.*?)"
        r"(?=^<(?:Picture|Video|Audio) [1-9]\d*>:|\Z)",
        text,
    )
    if not match:
        return ""
    body = match.group(1).strip()
    body = re.sub(rf"^`?{re.escape(label)}`?:\s*", "", body).strip()

    # Gemma occasionally repeats compact task constraints after the correct
    # label and only then starts describing the asset. Keep the facts after
    # its explicit analysis marker, never the repeated instructions.
    if any(phrase in body[:160].lower() for phrase in _LEADING_META_PHRASES):
        analysis_markers = list(
            re.finditer(
                r"(?is)\b(?:analy(?:zing|sis)(?:\s+of)?(?:\s+the)?"
                r"(?:\s+(?:attached\s+)?(?:picture|image|video|audio))?"
                r"(?:\s*\([^)]*\))?)\s*:\s*",
                body,
            )
        )
        if not analysis_markers:
            return ""
        body = body[analysis_markers[-1].end() :].strip()

    # Conversely, stop when the model switches from visible/audible facts to
    # drafting or self-review commentary.
    trailing_meta = re.search(
        r"(?is)\b(?:drafting (?:the )?description|review against constraints|"
        r"checking (?:the )?(?:answer|constraints)|final answer)\s*:",
        body,
    )
    if trailing_meta:
        body = body[: trailing_meta.start()].strip()

    body = re.sub(r"\s+", " ", body).strip(" `*\n\t")
    if not _usable_observation_body(body):
        return ""
    return f"{label}: {body}"


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
        r"(?im)^\s*(?:#{1,6}\s*)?(?:\*\*)?(?:"
        r"(?:(?:image|visual|keyframe|picture|video|audio|media)\s+)?analysis|"
        r"analysis\s+of\s+(?:(?:the|attached)\s+)?"
        r"(?:picture|image|keyframe|video|audio|media)(?:\s+\d+)?|"
        r"(?:keyframe|image|picture|visual|media)\s+"
        r"(?:details|description|observations?)"
        r")\s*:\s*(?:\*\*)?\s*$",
        value,
    )
    if heading:
        value = value[heading.end() :].strip()
    else:
        # Keyframe responses are less consistent than reference-image
        # responses: Gemma may call the section "Keyframe Details" or omit a
        # heading entirely while still returning a useful structured list.
        # A field-style bullet is descriptive content, not task paraphrase, so
        # it is a safe deterministic recovery boundary for a single asset.
        first_field = re.search(
            r"(?m)^\s*(?:[-*+]\s+|\d+[.)]\s+)"
            r"(?:\*\*)?[^:\n]{1,80}:\s*(?:\*\*)?",
            value,
        )
        if first_field:
            value = value[first_field.start() :].strip()
        else:
            # A compact single-image answer is often plain prose rather than
            # the requested labelled record, for example "The image shows a
            # woman...".  It is still attributable because this call contains
            # exactly one media source.  Prefer an explicit descriptive-sentence
            # boundary when Gemma emitted a short planning preamble.
            descriptive = re.search(
                r"(?im)^\s*(?:the|this|attached)\s+"
                r"(?:image|picture|keyframe|frame|visual|scene)\s+"
                r"(?:shows|depicts|features|presents|contains|captures|is)\b",
                value,
            )
            if descriptive:
                value = value[descriptive.start() :].strip()
            elif not _usable_observation_body(value):
                return ""
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
    return _usable_observation_body(body)


def ensure_analysis_records(text: str, labels: list[str]) -> str:
    """Guarantee traceable, non-hallucinated records for every analyzed label."""

    value = clean_analysis_text(text)
    if len(labels) == 1:
        normalized = _normalize_single_labeled_record(value, labels[0])
        if normalized:
            value = normalized
        else:
            recovered = _recover_single_source_analysis(text)
            if recovered:
                value = _normalize_single_labeled_record(
                    f"{labels[0]}: {recovered}", labels[0]
                )
            else:
                value = ""
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
    # Media observations guide the prompt writer but are not the actual H3
    # image/video inputs.  A conservative labelled fallback is safer than
    # aborting the whole workflow or inventing visual facts when Gemma twice
    # returns only meta commentary.
    return ensure_analysis_records("", [label])


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


def analyze_frames_media(
    runner: GemmaRunner,
    *,
    frames: list[tuple[int, Any]],
    max_new_tokens: int = 512,
    seed: int = 0,
    after_call: Callable[[], None] = _noop,
) -> dict[str, str]:
    """Analyze an ordered sequence of I2V keyframes independently."""

    observations: dict[str, str] = {}
    frame_count = len(frames)
    for call_index, (slot, image) in enumerate(frames):
        label = f"<Picture {slot}>"
        socket = f"frame_{slot}"
        if frame_count == 1:
            temporal_role = "literal first frame at 0.00 seconds"
        elif slot == 1:
            temporal_role = f"opening keyframe 1 of {frame_count}"
        elif slot == frame_count:
            temporal_role = f"ending keyframe {slot} of {frame_count}"
        else:
            temporal_role = f"intermediate keyframe {slot} of {frame_count}"
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
    ref_images: dict[str, Any] | None = None,
    ref_videos: dict[str, Any] | None = None,
    ref_video_audios: dict[str, Any] | None = None,
    ref_audios: dict[str, Any] | None = None,
    target_frame_count: int,
    max_new_tokens: int = 512,
    seed: int = 0,
    after_call: Callable[[], None] = _noop,
    **individual_inputs: Any,
) -> dict[str, str]:
    observations: dict[str, str] = {}
    call_index = 0
    analysis_seconds = float(target_frame_count) / FPS
    values = collect_reference_inputs(
        ref_images=ref_images,
        ref_videos=ref_videos,
        ref_video_audios=ref_video_audios,
        ref_audios=ref_audios,
        **individual_inputs,
    )
    image_entries = [
        (asset.label, asset.socket)
        for asset in manifest.pictures
        if values.get(asset.socket) is not None
    ]
    for label, socket in image_entries:
        observations[socket] = _generate_complete_record(
            runner,
            prompt=reference_images_analysis_prompt([(label, socket)]),
            label=label,
            image=first_image(values[socket]),
            max_new_tokens=max_new_tokens,
            seed=seed + call_index,
            after_call=after_call,
        )
        call_index += 1

    for video_asset in manifest.videos:
        video = values[video_asset.socket]
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

    for audio_asset in manifest.audios:
        audio = values.get(audio_asset.socket)
        if audio is None:
            raise RuntimeError(
                f"Reference manifest lost the {audio_asset.socket} mapping."
            )
        observations[audio_asset.socket] = _generate_complete_record(
            runner,
            prompt=reference_audio_analysis_prompt(
                audio_asset.label, audio_asset.socket
            ),
            label=audio_asset.label,
            audio=trim_audio(audio, max_seconds=analysis_seconds),
            max_new_tokens=max_new_tokens,
            seed=seed + call_index,
            after_call=after_call,
        )
        call_index += 1

    return observations
