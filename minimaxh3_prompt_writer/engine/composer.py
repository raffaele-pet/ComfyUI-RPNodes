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
