"""Skill selection, final synthesis, validation, and one repair pass."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass

from .constants import (
    BUNDLE_VERSION,
    SKILL_AUTO,
    SKILL_BY_ID,
    SKILL_CORE,
    SkillProfile,
    get_skill_profile,
)
from .chronology import ordered_event_ledger, validate_dialogue_event_ownership
from .gemma import GemmaRunner, SamplingConfig
from .manifests import ReferenceManifest, effective_duration
from .prompts import (
    auto_skill_system_prompt,
    auto_skill_user_payload,
    base_system_prompt,
    base_user_payload,
    ref_system_prompt_with_i2v_description,
    ref_system_prompt,
    ref_user_payload,
    repair_system_prompt,
    repair_user_payload,
    t2v_system_prompt,
    t2v_user_payload,
)
from .validation import (
    ValidationResult,
    canonicalize_base_structure,
    canonicalize_ref_structure,
    canonicalize_t2v_structure,
    validate_base_prompt,
    validate_ref_prompt,
    validate_t2v_prompt,
)


def _noop() -> None:
    return None


def _dialogue_language(content: str, raw_prompt: str) -> str:
    """Infer a conservative language tag without altering dialogue text."""

    if re.search(r"[\u3040-\u30ff]", content):
        return "Japanese"
    if re.search(r"[\uac00-\ud7af]", content):
        return "Korean"
    if re.search(r"[\u4e00-\u9fff]", content):
        return "Chinese"
    if re.search(r"[\u0400-\u04ff]", content):
        return "Russian"
    if re.search(r"[\u0600-\u06ff]", content):
        return "Arabic"
    if re.search(r"[\u0900-\u097f]", content):
        return "Hindi"

    lowered = f"{raw_prompt}\n{content}".lower()
    explicit = re.search(
        r"\b(?:language|in|lingua|in lingua)\s*[:=]?\s*"
        r"(english|italian|spanish|french|german|portuguese)\b",
        lowered,
    )
    if explicit:
        return explicit.group(1).capitalize()
    spoken = content.lower()
    if re.search(
        r"\b(?:ciao|grazie|sono|questo|questa|perché|allora|buongiorno)\b",
        spoken,
    ):
        return "Italian"
    if re.search(
        r"\b(?:hola|gracias|porque|estoy|buenos|buenas|quiero)\b",
        spoken,
    ):
        return "Spanish"
    if re.search(
        r"\b(?:bonjour|merci|parce que|suis|avec|voilà)\b",
        spoken,
    ):
        return "French"
    return "English"


def _canonicalize_dialogue_languages(text: str, raw_prompt: str) -> str:
    """Add only a missing H3 language tag; preserve the spoken words exactly."""

    def replace(match: re.Match[str]) -> str:
        content = match.group(1)
        if re.match(r"\s*\[[^\]\n]*\S[^\]\n]*\]\s+\S", content):
            return match.group(0)
        spoken = content.strip()
        if not spoken:
            return match.group(0)
        language = _dialogue_language(spoken, raw_prompt)
        return f"<d>[{language}] {spoken}</d>"

    return re.sub(r"<d>(.*?)</d>", replace, text, flags=re.DOTALL)


def _ref_detail_bounds(text: str) -> tuple[int, int] | None:
    start = re.search(r"(?m)^detailed_description:(?=$|[ \t])", text)
    end = re.search(r"(?m)^overall_soundscape:(?=$|[ \t])", text)
    if start is None or end is None or start.end() >= end.start():
        return None
    return start.end(), end.start()


def _requested_picture_location(
    label: str,
    raw_prompt: str,
) -> tuple[str | None, str | None]:
    occurrence = raw_prompt.find(label)
    if occurrence < 0:
        return None, None
    prefix = raw_prompt[:occurrence]
    shots = list(
        re.finditer(
            r"\[Shot\s+([1-9]\d*)\](?:\s+At\s+(\d{2}:\d{2}\.\d{3}))?",
            prefix,
            flags=re.IGNORECASE,
        )
    )
    if not shots:
        return None, None
    shot = shots[-1]
    return shot.group(1), shot.group(2)


def _requested_picture_insertion(
    body: str,
    label: str,
    raw_prompt: str,
) -> int | None:
    shot_id, timestamp = _requested_picture_location(label, raw_prompt)
    if shot_id is None:
        return None
    if timestamp:
        target = re.search(
            rf"\[Shot\s+[1-9]\d*\]\s+At\s+{re.escape(timestamp)},",
            body,
            flags=re.IGNORECASE,
        )
        if target is None:
            target = re.search(
                rf"\[Shot\s+{re.escape(shot_id)}\]",
                body,
                flags=re.IGNORECASE,
            )
    else:
        target = re.search(
            rf"\[Shot\s+{re.escape(shot_id)}\]",
            body,
            flags=re.IGNORECASE,
        )
    if target is None:
        return None
    next_shot = re.search(r"\[Shot\s+[1-9]\d*\]", body[target.end() :])
    return target.end() + next_shot.start() if next_shot else len(body)


_TRANSITION_STOPWORDS = {
    "about",
    "after",
    "against",
    "also",
    "before",
    "from",
    "into",
    "picture",
    "reference",
    "referencing",
    "shot",
    "subject",
    "that",
    "their",
    "then",
    "there",
    "these",
    "this",
    "through",
    "towards",
    "with",
}

_FRAME_CITATION_STOPWORDS = _TRANSITION_STOPWORDS | {
    "background",
    "character",
    "depicted",
    "depicts",
    "frame",
    "image",
    "observation",
    "rendered",
    "scene",
    "shown",
    "shows",
    "state",
    "style",
    "visible",
}


def _transition_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'-]{2,}", text.lower())
        if token not in _TRANSITION_STOPWORDS
    }


def _frame_citation_tokens(text: str) -> set[str]:
    """Return conservative word stems for matching a frame to action prose."""

    result: set[str] = set()
    for token in re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'-]{2,}", text.lower()):
        if token in _FRAME_CITATION_STOPWORDS:
            continue
        stem = token
        if len(stem) > 5 and stem.endswith("ing"):
            stem = stem[:-3]
            if stem.endswith(("clos", "giv", "mov", "smil", "wav")):
                stem += "e"
        elif len(stem) > 4 and stem.endswith("ed"):
            stem = stem[:-2]
        elif len(stem) > 4 and stem.endswith(
            ("ses", "xes", "zes", "ches", "shes", "oes")
        ):
            stem = stem[:-2]
        elif len(stem) > 4 and stem.endswith("s"):
            stem = stem[:-1]
        if len(stem) >= 3:
            result.add(stem)
    return result


def _raw_picture_context(label: str, raw_prompt: str) -> str:
    occurrence = raw_prompt.find(label)
    if occurrence < 0:
        return ""
    shot_headers = list(
        re.finditer(r"\[Shot\s+[1-9]\d*\]", raw_prompt, flags=re.IGNORECASE)
    )
    current_shot = [match for match in shot_headers if match.start() < occurrence]
    if current_shot:
        shot_start = current_shot[-1].start()
        following = [match for match in shot_headers if match.start() > occurrence]
        shot_end = following[0].start() if following else len(raw_prompt)
        sentence_breaks = list(
            re.finditer(
                r"[.!?](?=\s|$)",
                raw_prompt[shot_start:shot_end],
            )
        )
        relative_occurrence = occurrence - shot_start
        previous = [
            match for match in sentence_breaks if match.end() <= relative_occurrence
        ]
        following_break = [
            match for match in sentence_breaks if match.start() > relative_occurrence
        ]
        start = shot_start + (previous[-1].end() if previous else 0)
        end = shot_start + (
            following_break[0].end() if following_break else shot_end - shot_start
        )
        return raw_prompt[start:end]
    start = max(
        raw_prompt.rfind(".", 0, occurrence),
        raw_prompt.rfind("!", 0, occurrence),
        raw_prompt.rfind("?", 0, occurrence),
        raw_prompt.rfind("\n", 0, occurrence),
    )
    ends = [
        position
        for delimiter in (".", "!", "?", "\n")
        if (position := raw_prompt.find(delimiter, occurrence + len(label))) >= 0
    ]
    end = min(ends) + 1 if ends else len(raw_prompt)
    return raw_prompt[start + 1 : end]


def _action_sentence_spans(body: str) -> list[tuple[int, int]]:
    masked = re.sub(
        r"<d>.*?</d>",
        lambda match: " " * len(match.group(0)),
        body,
        flags=re.DOTALL,
    )
    return [
        (match.start(), match.end())
        for match in re.finditer(
            r".+?(?:[.!?](?=\s|\[Shot|$)|$)",
            masked,
            flags=re.DOTALL,
        )
        if masked[match.start() : match.end()].strip()
    ]


def _best_picture_action_span(
    body: str,
    asset_label: str,
    asset_socket: str,
    raw_prompt: str,
    observations: dict[str, str],
) -> tuple[int, int] | None:
    raw_context = _raw_picture_context(asset_label, raw_prompt)
    if not raw_context:
        return None
    raw_tokens = _transition_tokens(raw_context)
    observation_tokens = _transition_tokens(
        str(observations.get(asset_socket, "") or "")
    )
    best: tuple[int, int] | None = None
    best_score = 0
    for start, end in _action_sentence_spans(body):
        sentence_tokens = _transition_tokens(body[start:end])
        score = 3 * len(raw_tokens & sentence_tokens) + len(
            observation_tokens & sentence_tokens
        )
        if score > best_score:
            best = (start, end)
            best_score = score
    return best


def _best_ordered_frame_action_span(
    body: str,
    picture_index: int,
    manifest: ReferenceManifest,
    observations: dict[str, str],
    *,
    require_semantic_match: bool,
) -> tuple[int, int] | None:
    """Choose an in-order action sentence for one omitted frame citation.

    The first pass requires lexical evidence from that frame's own observation.
    A post-repair fallback may use relative timeline position, but it is still
    constrained between the nearest cited earlier and later Pictures so a
    missing label cannot be moved across an established keyframe boundary.
    """

    spans = _action_sentence_spans(body)
    if not spans:
        return None

    assets = manifest.pictures
    asset = assets[picture_index]
    earlier_positions = [
        body.rfind(item.label)
        for item in assets[:picture_index]
        if item.label in body
    ]
    later_positions = [
        body.find(item.label)
        for item in assets[picture_index + 1 :]
        if item.label in body
    ]
    lower_bound = max(earlier_positions, default=-1)
    upper_bound = min(later_positions, default=len(body))
    eligible = [
        (span_index, left, right)
        for span_index, (left, right) in enumerate(spans)
        if right > lower_bound and left < upper_bound
    ]
    if not eligible:
        return None

    observation_tokens = {
        item.label: _frame_citation_tokens(
            str(observations.get(item.socket, "") or "")
        )
        for item in assets
    }
    token_frequency: dict[str, int] = {}
    for tokens in observation_tokens.values():
        for token in tokens:
            token_frequency[token] = token_frequency.get(token, 0) + 1
    maximum_frequency = max(1, len(assets) // 3)
    target_tokens = observation_tokens[asset.label]
    distinctive_tokens = {
        token
        for token in target_tokens
        if token_frequency.get(token, 0) <= maximum_frequency
    }
    target_span = (
        0.0
        if len(assets) == 1
        else picture_index * (len(spans) - 1) / (len(assets) - 1)
    )

    ranked: list[tuple[int, float, int, int]] = []
    for span_index, left, right in eligible:
        sentence_tokens = _frame_citation_tokens(body[left:right])
        semantic_score = (
            4 * len(distinctive_tokens & sentence_tokens)
            + len(target_tokens & sentence_tokens)
        )
        ranked.append(
            (
                semantic_score,
                -abs(span_index - target_span),
                left,
                right,
            )
        )
    best_score, _, left, right = max(ranked)
    if require_semantic_match and best_score < 2:
        return None
    return left, right


def _restore_ordered_frame_picture_coverage(
    text: str,
    manifest: ReferenceManifest,
    observations: dict[str, str],
    *,
    require_semantic_match: bool,
) -> tuple[str, tuple[str, ...]]:
    """Attach omitted Picture labels without rewriting frame semantics."""

    bounds = _ref_detail_bounds(text)
    if bounds is None:
        return text, ()
    start, end = bounds
    body = text[start:end].strip()
    restored: list[str] = []
    for picture_index, asset in enumerate(manifest.pictures):
        if asset.label in body:
            continue
        span = _best_ordered_frame_action_span(
            body,
            picture_index,
            manifest,
            observations,
            require_semantic_match=require_semantic_match,
        )
        if span is None:
            continue
        body = _attach_picture_to_action(body, span, asset.label)
        restored.append(asset.label)
    value = text[:start].rstrip() + "\n" + body + "\n\n" + text[end:].lstrip()
    return value, tuple(restored)


def _attach_picture_to_action(
    body: str,
    span: tuple[int, int],
    label: str,
) -> str:
    _, end = span
    insertion = end
    while insertion > 0 and body[insertion - 1].isspace():
        insertion -= 1
    if insertion > 0 and body[insertion - 1] in ".!?":
        insertion -= 1
    return body[:insertion].rstrip() + f" ({label})" + body[insertion:]


def _continuous_picture_transition(
    previous_label: str | None,
    current_label: str,
) -> str:
    if previous_label is None:
        return (
            f"The continuous opening composition is anchored by {current_label}, "
            "with stable subject, environment, and camera state."
        )
    return (
        f"Without a cut, the visible action moves smoothly from {previous_label} "
        f"into {current_label}, preserving continuous subject, object, and camera "
        "motion through physically coherent intermediate states."
    )


def _remove_static_picture_insertions(body: str) -> str:
    """Remove legacy validation prose so labels can be bound to real actions."""

    value = re.sub(
        r"(?is)\bAt this point,\s*<Picture\s+[1-9]\d*>\s+contributes\s+"
        r"this concrete visible state\s*:\s*.*?"
        r"(?=\bAt this point,\s*<Picture\s+[1-9]\d*>|"
        r"<Subject\s+[1-9]\d*>|\[Shot\s+[1-9]\d*\]|"
        r"\bThe (?:camera|final state)\b|$)",
        " ",
        body,
    )
    return re.sub(r"[ \t]{2,}", " ", value).strip()


def _ensure_ref_picture_coverage(
    text: str,
    manifest: ReferenceManifest,
    observations: dict[str, str],
    raw_prompt: str,
    *,
    semantic_only: bool = False,
) -> str:
    """Ground any omitted Picture label in its own analyzed visual evidence."""

    # Ordered frames are semantic target states.  Rewriting, moving, or adding
    # their labels can make an incorrect description look formally valid while
    # leaving the visible actions attached to the wrong images.  Frames mode
    # therefore validates model output and repairs it through Gemma instead of
    # mutating citation meaning in Python.
    if semantic_only:
        return text

    bounds = _ref_detail_bounds(text)
    if bounds is None:
        return text
    start, end = bounds
    body = _remove_static_picture_insertions(text[start:end].strip())
    for index, asset in enumerate(manifest.pictures):
        if asset.label in body:
            continue
        action_span = _best_picture_action_span(
            body,
            asset.label,
            asset.socket,
            raw_prompt,
            observations,
        )
        if action_span is not None:
            body = _attach_picture_to_action(body, action_span, asset.label)
            continue

        previous_label = (
            manifest.pictures[index - 1].label if index > 0 else None
        )
        sentence = _continuous_picture_transition(previous_label, asset.label)
        requested_insertion = _requested_picture_insertion(
            body,
            asset.label,
            raw_prompt,
        )
        insertion = requested_insertion
        if requested_insertion is None:
            insertion = len(body)
            previous_labels = [item.label for item in manifest.pictures[:index]]
            previous_positions = [body.rfind(label) for label in previous_labels]
            previous_position = max(previous_positions, default=-1)
            if previous_position >= 0:
                sentence_end = re.search(r"[.!?](?=\s|$)", body[previous_position:])
                if sentence_end:
                    insertion = previous_position + sentence_end.end()
            elif shot_open := re.search(
                r"\[Shot 1\](?:\s+At\s+[^,]+,)?\s*",
                body,
            ):
                insertion = shot_open.end()
        body = (
            body[:insertion].rstrip()
            + " "
            + sentence
            + " "
            + body[insertion:].lstrip()
        ).strip()
    return text[:start].rstrip() + "\n" + body + "\n\n" + text[end:].lstrip()


def _canonicalize_ref_generated_content(
    text: str,
    *,
    raw_prompt: str,
    observations: dict[str, str],
    manifest: ReferenceManifest,
    semantic_picture_binding: bool = False,
) -> str:
    value = _canonicalize_dialogue_languages(text, raw_prompt)
    value = re.sub(
        r"(?im)^(\[(?:reference generation|keyframe completion)\])\s*"
        r"(?:reference generation|keyframe completion)\b\s*",
        r"\1 ",
        value,
    )
    value = re.sub(
        r"(?im)^(\[(?:reference generation|keyframe completion)\])\s*\.\s*",
        r"\1 ",
        value,
    )
    return _ensure_ref_picture_coverage(
        value,
        manifest,
        observations,
        raw_prompt,
        semantic_only=semantic_picture_binding,
    )


def _remove_frame_picture_definitions(text: str) -> str:
    start = re.search(r"(?m)^subject_definitions:(?=$|[ \t])", text)
    end = re.search(r"(?m)^summary:(?=$|[ \t])", text)
    if start is None or end is None or start.end() >= end.start():
        return text
    body = text[start.end() : end.start()]
    kept = [
        line
        for line in body.splitlines()
        if not re.match(r"^\s*<Picture\s+[1-9]\d*>\s*:", line)
    ]
    normalized = "\n".join(kept).strip() or "N/A"
    return text[: start.end()] + "\n" + normalized + "\n\n" + text[end.start() :]


_FRAME_EVENT_MARKER_RE = re.compile(
    r"\[(?:\[\s*)?(?:EVENT|E)[ _-]?([1-9]\d*)\s*\](?:\])?",
    flags=re.IGNORECASE,
)


def _remove_frame_non_picture_labels(text: str) -> str:
    """Frames mode has image evidence only; discard hallucinated media labels."""

    value = re.sub(
        r"(?m)^\s*<(?:Audio|Video)\s+[1-9]\d*>\s*:.*(?:\n|$)",
        "",
        text,
    )
    value = re.sub(r"<(?:Audio|Video)\s+[1-9]\d*>", "", value)
    return re.sub(r"[ \t]+(?=[,.;:])", "", value)


def _frame_dialogue_spans(body: str) -> list[tuple[int, int]]:
    """Locate outer dialogue regions even when Gemma nests H3 tags."""

    tokens = list(re.finditer(r"</?d>", body, flags=re.IGNORECASE))
    spans: list[tuple[int, int]] = []
    depth = 0
    opening = 0
    opening_end = 0
    for token in tokens:
        closing = token.group(0).lower().startswith("</")
        if not closing:
            if depth == 0:
                opening = token.start()
                opening_end = token.end()
            depth += 1
            continue
        if depth == 0:
            spans.append((token.start(), token.end()))
            continue
        depth -= 1
        if depth == 0:
            spans.append((opening, token.end()))
    if depth:
        spans.append((opening, opening_end))
    return spans


def _mask_frame_dialogue_blocks(body: str) -> str:
    value = list(body)
    for start, end in _frame_dialogue_spans(body):
        value[start:end] = " " * (end - start)
    return "".join(value)


def _strip_frame_dialogue_blocks(body: str) -> str:
    """Remove model-authored dialogue, including malformed nested H3 tags."""

    spans = _frame_dialogue_spans(body)
    if not spans:
        return body
    pieces: list[str] = []
    cursor = 0
    for start, end in spans:
        pieces.append(body[cursor:start])
        cursor = end
    pieces.append(body[cursor:])
    value = "".join(pieces)
    value = re.sub(r"[\"“”]\s*[\"“”]", "", value)
    value = _remove_dangling_speech_carriers(value)
    value = re.sub(r",\s*,+", ",", value)
    value = re.sub(r",\s*\.", ".", value)
    value = re.sub(r"\s+,", ",", value)
    value = re.sub(r"\s+\.", ".", value)
    return re.sub(r"[ \t]{2,}", " ", value)


def _remove_dangling_speech_carriers(body: str) -> str:
    """Remove a speech verb only after its dialogue content was removed."""

    return re.sub(
        r"\b(?:and\s+)?(?:says?|saying|asks?|asking|shouts?|shouting|"
        r"whispers?|whispering|replies|replying|calls out)\s*,?\s*"
        r"(?=[,.;:]|\b(?:matching|then|while|as)\b|$)",
        "",
        body,
        flags=re.IGNORECASE,
    )


def _materialize_frame_event_dialogue(text: str, raw_prompt: str) -> str:
    """Anchor exact speech to its chronological action, then erase event markers."""

    bounds = _ref_detail_bounds(text)
    ledger = ordered_event_ledger(raw_prompt)
    if bounds is None or not ledger:
        return _FRAME_EVENT_MARKER_RE.sub("", text)
    start, end = bounds
    body = text[start:end]
    markers = list(_FRAME_EVENT_MARKER_RE.finditer(body))
    found = [int(match.group(1)) for match in markers]
    expected = list(range(1, len(ledger) + 1))

    spoken_values = [
        str(value)
        for event in ledger
        for value in event["spoken_verbatim_strings"]
    ]
    original_sentence_targets: dict[str, int] = {}
    if spoken_values:
        masked_body = _mask_frame_dialogue_blocks(body)
        original_spans = _action_sentence_spans(masked_body)
        for spoken in spoken_values:
            occurrence = body.casefold().find(spoken.casefold())
            if occurrence < 0:
                continue
            sentence_index = next(
                (
                    index
                    for index, (left, right) in enumerate(original_spans)
                    if left <= occurrence < right
                ),
                -1,
            )
            if sentence_index >= 0:
                original_sentence_targets[spoken.casefold()] = sentence_index
        target_counts: dict[int, int] = {}
        for sentence_index in original_sentence_targets.values():
            target_counts[sentence_index] = target_counts.get(sentence_index, 0) + 1
        original_sentence_targets = {
            spoken: sentence_index
            for spoken, sentence_index in original_sentence_targets.items()
            if target_counts[sentence_index] == 1
        }
        body = _strip_frame_dialogue_blocks(body)
    for spoken in spoken_values:
        escaped = re.escape(spoken)
        body = re.sub(
            rf"<d>\s*(?:\[[^\]\n]+\]\s*)?{escaped}\s*</d>",
            "",
            body,
            flags=re.IGNORECASE,
        )
        body = re.sub(
            rf"[\"“”]\s*{escaped}\s*[\"“”]",
            "",
            body,
            flags=re.IGNORECASE,
        )
        body = re.sub(escaped, "", body, flags=re.IGNORECASE)
    if spoken_values:
        body = _remove_dangling_speech_carriers(body)
        body = re.sub(r"\[\s*[A-Za-zÀ-ÿ -]+\s*\](?=\s*[,.;:]|\s*$)", "", body)

    def add_spoken(segment: str, spoken: list[str]) -> str:
        if spoken:
            segment = re.sub(
                r"\b(?:and\s+)?(?:says?|asks?|shouts?|whispers?|replies|calls out)"
                r"\s*,?\s*(?=[.!?])",
                "",
                segment,
                flags=re.IGNORECASE,
            )
            blocks = [
                f'"<d>[{_dialogue_language(value, raw_prompt)}] {value}</d>"'
                for value in spoken
            ]
            speech = " and then says ".join(blocks)
            sentence_end = re.search(r"[.!?](?=\s|$)", segment)
            if sentence_end:
                position = sentence_end.start()
                segment = (
                    segment[:position].rstrip(" ,")
                    + f", and says {speech}"
                    + segment[position:]
                )
            else:
                segment = segment.rstrip(" ,") + f', and says {speech}.'
        return segment

    if found == expected:
        markers = list(_FRAME_EVENT_MARKER_RE.finditer(body))
        rebuilt: list[str] = []
        cursor = 0
        for index, marker in enumerate(markers):
            rebuilt.append(body[cursor : marker.start()])
            segment_end = (
                markers[index + 1].start()
                if index + 1 < len(markers)
                else len(body)
            )
            segment = body[marker.end() : segment_end]
            spoken = [
                str(value) for value in ledger[index]["spoken_verbatim_strings"]
            ]
            rebuilt.append(add_spoken(segment, spoken))
            cursor = segment_end
        rebuilt.append(body[cursor:])
        materialized = "".join(rebuilt)
    else:
        body = _FRAME_EVENT_MARKER_RE.sub("", body)
        sentence_ends = list(re.finditer(r"[.!?](?=\s|$)", body))
        spans: list[tuple[int, int]] = []
        cursor = 0
        for sentence_end in sentence_ends:
            spans.append((cursor, sentence_end.end()))
            cursor = sentence_end.end()
        if body[cursor:].strip():
            spans.append((cursor, len(body)))
        if not spans:
            materialized = body
        else:
            sentences = [body[left:right] for left, right in spans]
            tail = body[spans[-1][1] :]
            spoken_targets: set[int] = set()
            for index, event in enumerate(ledger):
                spoken = [
                    str(value) for value in event["spoken_verbatim_strings"]
                ]
                if spoken:
                    preserved_targets = [
                        original_sentence_targets[value.casefold()]
                        for value in spoken
                        if value.casefold() in original_sentence_targets
                    ]
                    if preserved_targets:
                        target = min(preserved_targets[0], len(sentences) - 1)
                    elif len(ledger) == 1 or len(sentences) == 1:
                        target = 0
                    else:
                        target = round(
                            index * (len(sentences) - 1) / (len(ledger) - 1)
                        )
                    available = [
                        candidate
                        for candidate in range(len(sentences))
                        if candidate not in spoken_targets
                    ]
                    if available:
                        target = min(available, key=lambda candidate: abs(candidate - target))
                    spoken_targets.add(target)
                    sentences[target] = add_spoken(sentences[target], spoken)
            materialized = "".join(sentences) + tail
    materialized = re.sub(r"[ \t]{2,}", " ", materialized)
    return text[:start] + materialized + text[end:]


def _ensure_frame_event_dialogue(text: str, raw_prompt: str) -> str:
    """Apply chronology, then deterministically restore any omitted speech.

    Small models can return valid event prose while dropping the first dialogue
    block during a repair.  This final pass does not ask the model again: it
    audits exact normalized dialogue values and attaches each missing value to
    a distinct action sentence near its source event.
    """

    value = _materialize_frame_event_dialogue(text, raw_prompt)
    ledger = ordered_event_ledger(raw_prompt)
    bounds = _ref_detail_bounds(value)
    if bounds is None or not ledger:
        return value

    def normalized(content: str) -> str:
        content = re.sub(r"^\s*\[[^\]\n]+\]\s*", "", content.strip())
        content = content.replace("’", "'").replace("‘", "'")
        return re.sub(r"\s+", " ", content).strip().casefold()

    existing = [
        normalized(match.group(1))
        for match in re.finditer(
            r"<d>(.*?)</d>",
            value,
            flags=re.DOTALL | re.IGNORECASE,
        )
    ]
    missing: list[tuple[int, str]] = []
    for index, event in enumerate(ledger):
        for spoken in event["spoken_verbatim_strings"]:
            spoken_text = str(spoken)
            if existing.count(normalized(spoken_text)) != 1:
                missing.append((index, spoken_text))
    if not missing:
        return value

    missing_keys = {normalized(spoken) for _, spoken in missing}

    def remove_non_unique(match: re.Match[str]) -> str:
        return "" if normalized(match.group(1)) in missing_keys else match.group(0)

    value = re.sub(
        r"<d>(.*?)</d>",
        remove_non_unique,
        value,
        flags=re.DOTALL | re.IGNORECASE,
    )
    bounds = _ref_detail_bounds(value)
    if bounds is None:
        return value

    start, end = bounds
    body = value[start:end]
    occupied: set[int] = set()
    for event_index, spoken in missing:
        spans = _action_sentence_spans(body)
        if not spans:
            break
        preferred = (
            0
            if len(ledger) == 1 or len(spans) == 1
            else round(event_index * (len(spans) - 1) / (len(ledger) - 1))
        )
        available = [
            sentence_index
            for sentence_index, (left, right) in enumerate(spans)
            if sentence_index not in occupied
            and "<d>" not in body[left:right].casefold()
        ]
        target = (
            min(available, key=lambda item: abs(item - preferred))
            if available
            else preferred
        )
        occupied.add(target)
        left, right = spans[target]
        sentence = body[left:right]
        block = (
            f'"<d>[{_dialogue_language(spoken, raw_prompt)}] '
            f'{spoken}</d>"'
        )
        if sentence and sentence[-1] in ".!?":
            insertion = right - 1
            prefix = body[:insertion].rstrip(" ,")
            suffix = body[insertion:]
            body = prefix + f", and says {block}" + suffix
        else:
            insertion = right
            body = (
                body[:insertion].rstrip(" ,")
                + f", and says {block}."
                + body[insertion:]
            )
    return value[:start] + body + value[end:]


def _validate_frame_picture_order(
    text: str,
    manifest: ReferenceManifest,
) -> list[str]:
    bounds = _ref_detail_bounds(text)
    if bounds is None:
        return []
    start, end = bounds
    body = text[start:end]
    allowed = {asset.label: index for index, asset in enumerate(manifest.pictures, 1)}
    matches = [
        match
        for match in re.finditer(r"<Picture\s+[1-9]\d*>", body)
        if match.group(0) in allowed
    ]
    first_occurrences: list[int] = []
    seen: set[int] = set()
    for match in matches:
        number = allowed[match.group(0)]
        if number not in seen:
            seen.add(number)
            first_occurrences.append(number)
    if first_occurrences != sorted(first_occurrences):
        return [
            "The first citation of each Picture does not follow connected frame "
            "order. Review where each target state is reached."
        ]
    for current, following in zip(matches, matches[1:]):
        if allowed[following.group(0)] <= allowed[current.group(0)]:
            continue
        between = body[current.end() : following.start()]
        if not re.search(r"[A-Za-zÀ-ÿ0-9]", between):
            return [
                "Each adjacent Picture pair needs narrative action between its "
                "citation slots; bare clusters of Picture labels are invalid."
            ]
    return []


def _frame_quality_warnings(
    text: str,
    manifest: ReferenceManifest,
    raw_prompt: str,
    frame_prompts: dict[int, str] | None = None,
) -> list[str]:
    """Report review hints that must never trigger generation or failure."""

    warnings = validate_dialogue_event_ownership(text, raw_prompt)
    warnings.extend(_validate_frame_picture_order(text, manifest))
    warnings.extend(
        _frame_prompt_binding_warnings(text, frame_prompts or {}, manifest)
    )
    bounds = _ref_detail_bounds(text)
    if bounds is not None:
        start, end = bounds
        if re.search(
            r"\bcontributes\s+this\s+concrete\s+visible\s+state\b",
            text[start:end],
            flags=re.IGNORECASE,
        ):
            warnings.append(
                "Static reference-analysis wording remains in detailed_description; "
                "review it as direct target-video action if desired."
            )
    return list(dict.fromkeys(warnings))


_EXPLICIT_DISCONTINUITY = re.compile(
    r"\b(?:hard\s+cut|cut\s+to|scene\s+change|location\s+change|time\s+jump|"
    r"stacco|taglio\s+netto|cambio\s+(?:di\s+)?scena|cambio\s+(?:di\s+)?luogo|"
    r"salto\s+temporale)\b",
    flags=re.IGNORECASE,
)


def _normalized_frame_string(text: str) -> str:
    value = re.sub(r"^\s*\[[^\]\n]+\]\s*", "", str(text or "").strip())
    value = value.replace("’", "'").replace("‘", "'")
    return re.sub(r"\s+", " ", value).strip().casefold()


def _frame_prompt_binding_issues(
    text: str,
    frame_prompts: dict[int, str],
    manifest: ReferenceManifest,
) -> list[str]:
    """Validate continuity and exact prompt_N ownership near Picture anchors.

    A prompt_N event may occur immediately before its Picture citation while
    motion converges to that keyframe, or immediately after the citation while
    the anchored state acts. Its valid neighborhood is therefore bounded by
    the preceding and following Picture citations, not only by Picture N and
    Picture N+1. This also gives the final connected Picture a usable left-hand
    transition when no later input exists.
    """

    if not frame_prompts:
        return []
    bounds = _ref_detail_bounds(text)
    if bounds is None:
        return []
    start, end = bounds
    body = text[start:end]
    issues: list[str] = []

    prompt_text = "\n".join(frame_prompts.values())
    if not _EXPLICIT_DISCONTINUITY.search(prompt_text) and re.search(
        r"\[Shot\s+(?:[2-9]|[1-9]\d+)\]", body, flags=re.IGNORECASE
    ):
        issues.append(
            "Ordered prompt_N inputs request one continuous Shot: remove every "
            "[Shot 2+] marker and express keyframe changes as smooth motion."
        )

    positions: dict[int, int] = {}
    for index, asset in enumerate(manifest.pictures, 1):
        match = re.search(re.escape(asset.label), body)
        if match is not None:
            positions[index] = match.start()
    ordered_positions = [positions.get(index, -1) for index in range(1, len(manifest.pictures) + 1)]
    if any(position < 0 for position in ordered_positions) or ordered_positions != sorted(
        ordered_positions
    ):
        issues.append(
            "Picture citations must occur once in prompt order before dialogue/text "
            "ownership can be verified."
        )
        return list(dict.fromkeys(issues))

    dialogue_matches = list(
        re.finditer(r"<d>(.*?)</d>", body, flags=re.DOTALL | re.IGNORECASE)
    )
    for index, prompt in sorted(frame_prompts.items()):
        if index not in positions:
            continue
        segment_start = positions.get(index - 1, 0)
        segment_end = positions.get(index + 1, len(body))
        segment = body[segment_start:segment_end]
        for event in ordered_event_ledger(prompt):
            for spoken in event["spoken_verbatim_strings"]:
                normalized = _normalized_frame_string(str(spoken))
                occurrences = [
                    match
                    for match in dialogue_matches
                    if _normalized_frame_string(match.group(1)) == normalized
                ]
                if len(occurrences) != 1:
                    issues.append(
                        f"prompt_{index} dialogue `{spoken}` must appear exactly "
                        f"once inside the <Picture {index}> action span."
                    )
                elif not (
                    segment_start <= occurrences[0].start() < segment_end
                ):
                    issues.append(
                        f"prompt_{index} dialogue `{spoken}` was assigned to the "
                        f"wrong Picture neighborhood; keep it between the "
                        f"adjacent anchors around <Picture {index}>."
                    )
            for visible in event.get("visible_verbatim_strings", []):
                if visible in event.get(
                    "ambiguous_visible_verbatim_strings", []
                ):
                    continue
                if str(visible) not in segment:
                    issues.append(
                        f"prompt_{index} visible text `{visible}` must remain "
                        f"verbatim between the adjacent anchors around "
                        f"<Picture {index}>."
                    )

    subject_start = re.search(r"(?m)^subject_definitions:(?=$|[ \t])", text)
    subject_end = re.search(r"(?m)^summary:(?=$|[ \t])", text)
    if subject_start is not None and subject_end is not None:
        subject_lines = re.findall(
            r"(?m)^\s*<Subject\s+[1-9]\d*>\s+.*$",
            text[subject_start.end() : subject_end.start()],
        )
        singleton_sources = [
            re.findall(r"<Picture\s+([1-9]\d*)>", line)
            for line in subject_lines
        ]
        threshold = max(3, (7 * len(manifest.pictures) + 9) // 10)
        if (
            len(subject_lines) >= threshold
            and all(len(sources) == 1 for sources in singleton_sources)
            and len({sources[0] for sources in singleton_sources})
            == len(singleton_sources)
        ):
            issues.append(
                "The recurring character was split into one Subject per Picture. "
                "Define it once as a continuous Subject with multi-Picture provenance."
            )
    return list(dict.fromkeys(issues))


def _frame_prompt_binding_warnings(
    text: str,
    frame_prompts: dict[int, str],
    manifest: ReferenceManifest,
) -> list[str]:
    """Report uncertain prompt_N bindings without invalidating the H3 prompt."""

    if not frame_prompts:
        return []
    bounds = _ref_detail_bounds(text)
    if bounds is None:
        return []
    start, end = bounds
    body = text[start:end]
    positions: dict[int, int] = {}
    for index, asset in enumerate(manifest.pictures, 1):
        match = re.search(re.escape(asset.label), body)
        if match is not None:
            positions[index] = match.start()

    warnings: list[str] = []
    for index, prompt in sorted(frame_prompts.items()):
        if index not in positions:
            continue
        segment_start = positions.get(index - 1, 0)
        segment_end = positions.get(index + 1, len(body))
        segment = body[segment_start:segment_end]
        for event in ordered_event_ledger(prompt):
            for visible in event.get("ambiguous_visible_verbatim_strings", []):
                preservation = (
                    "was preserved"
                    if str(visible) in segment
                    else "was not preserved exactly"
                )
                warnings.append(
                    f"prompt_{index} has an ambiguous visible-text boundary near "
                    f"`{visible}`; it {preservation} around <Picture {index}>, "
                    "but this heuristic interpretation did not block execution."
                )
    return list(dict.fromkeys(warnings))


def _extend_validation(
    validation: ValidationResult,
    issues: list[str],
) -> ValidationResult:
    return ValidationResult(
        validation.text,
        tuple(dict.fromkeys((*validation.issues, *issues))),
    )


@dataclass(frozen=True)
class ComposeResult:
    prompt: str
    selected_skill: SkillProfile
    initial_validation: ValidationResult
    final_validation: ValidationResult
    repaired: bool
    quality_warnings: tuple[str, ...] = ()

    def analysis_report(
        self,
        *,
        mode: str,
        length: int,
        requested_duration_seconds: float | None = None,
        observations: dict[str, str],
        manifest: ReferenceManifest | None = None,
    ) -> str:
        payload = {
            "contract_version": BUNDLE_VERSION,
            "mode": mode,
            "requested_duration_seconds": (
                float(requested_duration_seconds)
                if requested_duration_seconds is not None
                else effective_duration(length)
            ),
            "aligned_length_frames": int(length),
            "effective_duration_seconds": effective_duration(length),
            "selected_skill": {
                "id": self.selected_skill.identifier,
                "label": self.selected_skill.label,
            },
            "reference_manifest": manifest.to_dict() if manifest else None,
            "media_observations": observations,
            "repair_performed": self.repaired,
            "initial_validation_issues": list(self.initial_validation.issues),
            "final_validation_issues": list(self.final_validation.issues),
            "quality_warnings": list(self.quality_warnings),
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)


def resolve_skill(
    runner: GemmaRunner,
    selected_label: str,
    *,
    raw_prompt: str,
    observations: dict[str, str],
    seed: int,
    after_call: Callable[[], None] = _noop,
) -> SkillProfile:
    if selected_label != SKILL_AUTO:
        return get_skill_profile(selected_label)

    response = runner.generate_chat(
        auto_skill_system_prompt(),
        auto_skill_user_payload(raw_prompt, observations),
        max_new_tokens=32,
        sampling=SamplingConfig(do_sample=False, seed=seed),
    )
    after_call()
    # Classify from the answer, not from optional reasoning residue that may
    # mention several candidate profiles.
    cleaned = re.sub(
        r"<think>.*?</think>",
        " ",
        str(response),
        flags=re.DOTALL | re.IGNORECASE,
    ).strip(" `\n\t")
    lowered = cleaned.lower()
    if lowered in SKILL_BY_ID:
        return SKILL_BY_ID[lowered]
    # Prefer exact identifier matches; choose the longest first to avoid a
    # hypothetical shorter ID being a substring of another.
    for identifier in sorted(SKILL_BY_ID, key=len, reverse=True):
        if re.search(rf"(?<![a-z0-9-]){re.escape(identifier)}(?![a-z0-9-])", lowered):
            return SKILL_BY_ID[identifier]
    return get_skill_profile(SKILL_CORE)


def _repair(
    runner: GemmaRunner,
    *,
    mode: str,
    length: int,
    authoritative_system_prompt: str,
    original_task_payload: str,
    candidate: str,
    validation: ValidationResult,
    max_new_tokens: int,
    seed: int,
    manifest: ReferenceManifest | None,
) -> str:
    return runner.generate_chat(
        authoritative_system_prompt + "\n\n" + repair_system_prompt(mode),
        repair_user_payload(
            mode=mode,
            length=length,
            original_task_payload=original_task_payload,
            candidate=candidate,
            issues=list(validation.issues),
            manifest=manifest,
        ),
        max_new_tokens=max_new_tokens,
        sampling=SamplingConfig(do_sample=False, seed=seed),
        assistant_prefix=(
            "subject_definitions:\n"
            if mode in ("Ref2VA", "Frames2VA")
            else ""
        ),
    )


def compose_base_prompt(
    runner: GemmaRunner,
    *,
    raw_prompt: str,
    mode: str,
    length: int,
    selected_skill_label: str,
    observations: dict[str, str],
    max_new_tokens: int,
    sampling: SamplingConfig,
    requested_duration_seconds: float | None = None,
    picture_count: int = 0,
    strict_validation: bool = True,
    after_call: Callable[[], None] = _noop,
) -> ComposeResult:
    skill = resolve_skill(
        runner,
        selected_skill_label,
        raw_prompt=raw_prompt,
        observations=observations,
        seed=sampling.seed,
        after_call=after_call,
    )
    system_prompt = base_system_prompt(
        mode,
        length,
        skill,
        requested_duration_seconds=requested_duration_seconds,
        picture_count=picture_count,
    )
    task_payload = base_user_payload(
        raw_prompt=raw_prompt,
        mode=mode,
        length=length,
        skill=skill,
        media_observations=observations,
        requested_duration_seconds=requested_duration_seconds,
        picture_count=picture_count,
    )
    candidate = runner.generate_chat(
        system_prompt,
        task_payload,
        max_new_tokens=max_new_tokens,
        sampling=sampling,
        assistant_prefix=(
            "subject_definitions:\n" if mode == "Frames2VA" else ""
        ),
    )
    after_call()
    candidate = canonicalize_base_structure(candidate, mode, length)
    initial = validate_base_prompt(
        candidate, mode, length, picture_count=picture_count
    )
    final = initial
    repaired = False
    if not initial.valid:
        repaired_text = _repair(
            runner,
            mode=mode,
            length=length,
            authoritative_system_prompt=system_prompt,
            original_task_payload=task_payload,
            candidate=initial.text,
            validation=initial,
            max_new_tokens=max_new_tokens,
            seed=sampling.seed + 10_000,
            manifest=None,
        )
        after_call()
        repaired_text = canonicalize_base_structure(repaired_text, mode, length)
        final = validate_base_prompt(
            repaired_text, mode, length, picture_count=picture_count
        )
        repaired = True
    if strict_validation and not final.valid:
        raise ValueError(
            "Gemma could not produce a structurally valid H3 prompt after one repair pass:\n- "
            + "\n- ".join(final.issues)
        )
    return ComposeResult(final.text, skill, initial, final, repaired)


def compose_ref_prompt(
    runner: GemmaRunner,
    *,
    raw_prompt: str,
    length: int,
    selected_skill_label: str,
    observations: dict[str, str],
    manifest: ReferenceManifest,
    max_new_tokens: int,
    sampling: SamplingConfig,
    requested_duration_seconds: float | None = None,
    strict_validation: bool = True,
    after_call: Callable[[], None] = _noop,
    i2v_detailed_description: bool = False,
    frame_prompts: dict[int, str] | None = None,
) -> ComposeResult:
    frame_prompts = {
        int(index): str(value or "").strip()
        for index, value in (frame_prompts or {}).items()
        if str(value or "").strip()
    }
    intent_parts = [raw_prompt.strip()]
    intent_parts.extend(
        f"<Picture {index}> prompt_{index}: {value}"
        for index, value in sorted(frame_prompts.items())
    )
    complete_user_intent = "\n\n".join(part for part in intent_parts if part)
    skill = resolve_skill(
        runner,
        selected_skill_label,
        raw_prompt=complete_user_intent,
        observations=observations,
        seed=sampling.seed,
        after_call=after_call,
    )
    system_prompt_factory = (
        ref_system_prompt_with_i2v_description
        if i2v_detailed_description
        else ref_system_prompt
    )
    system_prompt = system_prompt_factory(
        length,
        skill,
        manifest,
        requested_duration_seconds=requested_duration_seconds,
    )
    task_payload = ref_user_payload(
        raw_prompt=raw_prompt,
        length=length,
        skill=skill,
        manifest=manifest,
        media_observations=observations,
        requested_duration_seconds=requested_duration_seconds,
        ordered_frames=i2v_detailed_description,
        frame_prompts=frame_prompts,
    )
    candidate = runner.generate_chat(
        system_prompt,
        task_payload,
        max_new_tokens=max_new_tokens,
        sampling=sampling,
        assistant_prefix="subject_definitions:\n",
    )
    after_call()
    candidate = canonicalize_ref_structure(
        candidate,
        length,
        manifest,
        requested_duration_seconds=requested_duration_seconds,
        raw_user_request=complete_user_intent,
    )
    if i2v_detailed_description:
        candidate = _remove_frame_picture_definitions(candidate)
        candidate = _remove_frame_non_picture_labels(candidate)
    candidate = _canonicalize_ref_generated_content(
        candidate,
        raw_prompt=complete_user_intent,
        observations=observations,
        manifest=manifest,
        semantic_picture_binding=i2v_detailed_description,
    )
    if i2v_detailed_description:
        if not frame_prompts:
            candidate = _ensure_frame_event_dialogue(candidate, raw_prompt)
        candidate, _ = _restore_ordered_frame_picture_coverage(
            candidate,
            manifest,
            observations,
            require_semantic_match=True,
        )
    initial = validate_ref_prompt(candidate, length, manifest)
    if i2v_detailed_description:
        initial = _extend_validation(
            initial,
            _frame_prompt_binding_issues(initial.text, frame_prompts, manifest),
        )
    final = initial
    repaired = False
    positional_frame_restorations: tuple[str, ...] = ()
    if not initial.valid:
        repaired_text = _repair(
            runner,
            mode="Ref2VA",
            length=length,
            authoritative_system_prompt=system_prompt,
            original_task_payload=task_payload,
            candidate=initial.text,
            validation=initial,
            max_new_tokens=max_new_tokens,
            seed=sampling.seed + 10_000,
            manifest=manifest,
        )
        after_call()
        repaired_text = canonicalize_ref_structure(
            repaired_text,
            length,
            manifest,
            requested_duration_seconds=requested_duration_seconds,
            raw_user_request=complete_user_intent,
        )
        if i2v_detailed_description:
            repaired_text = _remove_frame_picture_definitions(repaired_text)
            repaired_text = _remove_frame_non_picture_labels(repaired_text)
        repaired_text = _canonicalize_ref_generated_content(
            repaired_text,
            raw_prompt=complete_user_intent,
            observations=observations,
            manifest=manifest,
            semantic_picture_binding=i2v_detailed_description,
        )
        if i2v_detailed_description:
            if not frame_prompts:
                repaired_text = _ensure_frame_event_dialogue(
                    repaired_text,
                    raw_prompt,
                )
            repaired_text, positional_frame_restorations = (
                _restore_ordered_frame_picture_coverage(
                    repaired_text,
                    manifest,
                    observations,
                    require_semantic_match=False,
                )
            )
        final = validate_ref_prompt(repaired_text, length, manifest)
        if i2v_detailed_description:
            final = _extend_validation(
                final,
                _frame_prompt_binding_issues(
                    final.text, frame_prompts, manifest
                ),
            )
        repaired = True
    if strict_validation and not final.valid:
        raise ValueError(
            "Gemma could not produce a structurally valid H3 Ref2VA prompt after one repair pass:\n- "
            + "\n- ".join(final.issues)
        )
    quality_warnings: tuple[str, ...] = ()
    if i2v_detailed_description:
        warnings = _frame_quality_warnings(
            final.text,
            manifest,
            raw_prompt,
            frame_prompts=frame_prompts,
        )
        if positional_frame_restorations:
            warnings.append(
                "After Gemma's repair, omitted frame citations were restored "
                "by their constrained timeline positions: "
                + ", ".join(positional_frame_restorations)
                + ". Review those citation locations if the source states are "
                "highly ambiguous."
            )
        quality_warnings = tuple(dict.fromkeys(warnings))
    return ComposeResult(
        final.text,
        skill,
        initial,
        final,
        repaired,
        quality_warnings,
    )


def compose_t2v_prompt(
    runner: GemmaRunner,
    *,
    raw_prompt: str,
    length: int,
    selected_skill_label: str,
    observations: dict[str, str],
    manifest: ReferenceManifest,
    max_new_tokens: int,
    sampling: SamplingConfig,
    requested_duration_seconds: float | None = None,
    strict_validation: bool = True,
    after_call: Callable[[], None] = _noop,
) -> ComposeResult:
    """Compose a standalone T2V prompt from text plus optional media evidence."""

    skill = resolve_skill(
        runner,
        selected_skill_label,
        raw_prompt=raw_prompt,
        observations=observations,
        seed=sampling.seed,
        after_call=after_call,
    )
    system_prompt = t2v_system_prompt(
        length,
        skill,
        requested_duration_seconds=requested_duration_seconds,
    )
    task_payload = t2v_user_payload(
        raw_prompt=raw_prompt,
        length=length,
        skill=skill,
        manifest=manifest,
        media_observations=observations,
        requested_duration_seconds=requested_duration_seconds,
    )
    candidate = runner.generate_chat(
        system_prompt,
        task_payload,
        max_new_tokens=max_new_tokens,
        sampling=sampling,
    )
    after_call()
    requested_duration = (
        effective_duration(length)
        if requested_duration_seconds is None
        else float(requested_duration_seconds)
    )
    minimum_visual_shots = max(1, len(manifest.pictures) + len(manifest.videos))
    candidate = canonicalize_t2v_structure(candidate)
    initial = validate_t2v_prompt(
        candidate,
        requested_duration,
        minimum_storyboard_shots=minimum_visual_shots,
    )
    final = initial
    repaired = False
    if not initial.valid:
        repaired_text = _repair(
            runner,
            mode="T2V",
            length=length,
            authoritative_system_prompt=system_prompt,
            original_task_payload=task_payload,
            candidate=initial.text,
            validation=initial,
            max_new_tokens=max_new_tokens,
            seed=sampling.seed + 10_000,
            manifest=manifest,
        )
        after_call()
        repaired_text = canonicalize_t2v_structure(repaired_text)
        final = validate_t2v_prompt(
            repaired_text,
            requested_duration,
            minimum_storyboard_shots=minimum_visual_shots,
        )
        repaired = True
    if strict_validation and not final.valid:
        raise ValueError(
            "Gemma could not produce a valid standalone H3 T2V prompt after one "
            "repair pass:\n- "
            + "\n- ".join(final.issues)
        )
    return ComposeResult(final.text, skill, initial, final, repaired)
