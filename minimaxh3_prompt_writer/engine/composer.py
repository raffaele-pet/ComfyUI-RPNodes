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


def _transition_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'-]{2,}", text.lower())
        if token not in _TRANSITION_STOPWORDS
    }


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
) -> str:
    """Ground any omitted Picture label in its own analyzed visual evidence."""

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
) -> str:
    value = _canonicalize_dialogue_languages(text, raw_prompt)
    return _ensure_ref_picture_coverage(
        value,
        manifest,
        observations,
        raw_prompt,
    )


@dataclass(frozen=True)
class ComposeResult:
    prompt: str
    selected_skill: SkillProfile
    initial_validation: ValidationResult
    final_validation: ValidationResult
    repaired: bool

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
) -> ComposeResult:
    skill = resolve_skill(
        runner,
        selected_skill_label,
        raw_prompt=raw_prompt,
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
        raw_user_request=raw_prompt,
    )
    candidate = _canonicalize_ref_generated_content(
        candidate,
        raw_prompt=raw_prompt,
        observations=observations,
        manifest=manifest,
    )
    initial = validate_ref_prompt(candidate, length, manifest)
    final = initial
    repaired = False
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
            raw_user_request=raw_prompt,
        )
        repaired_text = _canonicalize_ref_generated_content(
            repaired_text,
            raw_prompt=raw_prompt,
            observations=observations,
            manifest=manifest,
        )
        final = validate_ref_prompt(repaired_text, length, manifest)
        repaired = True
    if strict_validation and not final.valid:
        raise ValueError(
            "Gemma could not produce a structurally valid H3 Ref2VA prompt after one repair pass:\n- "
            + "\n- ".join(final.issues)
        )
    return ComposeResult(final.text, skill, initial, final, repaired)


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
