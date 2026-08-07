"""Deterministic structural validation for generated H3 prompts."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .constants import MAX_H3_PROMPT_CHARS
from .manifests import ReferenceManifest, duration_2dp, effective_duration


BASE_FIELDS = (
    "integrated_multimodal_description:",
    "overall_soundscape:",
    "non_diegetic_music:",
)
REF_FIELDS = (
    "subject_definitions:",
    "summary:",
    "retention_analysis:",
    "detailed_description:",
    "overall_soundscape:",
    "non_diegetic_music:",
)


def _contains_schema_anchor(text: str, mode: str) -> bool:
    """Return whether text already contains the beginning of an H3 draft."""

    lowered = text.lower()
    if mode == "T2V":
        anchors = ("scene overview:", "storyboard:", "camera:", "audio:")
    elif mode == "Ref2VA":
        anchors = REF_FIELDS
    elif mode == "I2VA":
        anchors = ("for the target video,",) + BASE_FIELDS
    elif mode in ("FL2VA", "L2VA"):
        anchors = ("how the reference pictures align with the target video",) + BASE_FIELDS
    else:
        anchors = BASE_FIELDS
    return any(anchor in lowered for anchor in anchors)


def _strip_orphan_think_closures(value: str, mode: str) -> str:
    """Remove stray decoder closers without deleting an H3 draft before them.

    Gemma 4 occasionally emits a final ``<channel|>`` token even though its
    non-thinking chat prefix already closed the thought channel. ComfyUI decodes
    that token as ``</think>``. The old cleanup always kept only text *after*
    the closer, which turned an otherwise complete answer into an empty string.
    Prefer the newest side containing a recognizable schema; otherwise retain
    the historical post-closer behavior for genuine reasoning residue.
    """

    result = value
    while True:
        closes = list(re.finditer(r"</think>", result, flags=re.IGNORECASE))
        if not closes:
            return result.strip()
        close = closes[-1]
        before = result[: close.start()].strip()
        after = result[close.end() :].strip()
        if _contains_schema_anchor(after, mode):
            result = after
        elif _contains_schema_anchor(before, mode):
            result = before
        else:
            result = after


@dataclass(frozen=True)
class ValidationResult:
    text: str
    issues: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.issues


def sanitize_generated_text(text: str, mode: str) -> str:
    """Remove model-channel residue and isolate the structured candidate."""

    value = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    value = re.sub(r"<think>.*?</think>", "", value, flags=re.DOTALL | re.IGNORECASE).strip()
    value = _strip_orphan_think_closures(value, mode)
    if value.lower().startswith("<think>") and "</think>" not in value.lower():
        value = ""

    # Strip a surrounding Markdown fence without touching literal backticks in
    # dialogue or scene text.
    value = re.sub(r"^```(?:text|markdown)?\s*\n?", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\n?```\s*$", "", value).strip()

    if mode == "Ref2VA":
        starts = [value.find("subject_definitions:")]
    elif mode == "I2VA":
        starts = [value.find("For the target video,"), value.find(BASE_FIELDS[0])]
    elif mode in ("FL2VA", "L2VA"):
        starts = [
            value.find("How the reference pictures align with the target video"),
            value.find(BASE_FIELDS[0]),
        ]
    else:
        starts = [value.find(BASE_FIELDS[0])]
    starts = [index for index in starts if index >= 0]
    if starts:
        value = value[min(starts) :]
    return value.strip()


def _header_matches(text: str, field: str) -> list[re.Match[str]]:
    return list(re.finditer(rf"(?m)^{re.escape(field)}(?=$|[ \t])", text))


def _validate_common(text: str, fields: tuple[str, ...]) -> list[str]:
    issues: list[str] = []
    if not text:
        return ["The model returned an empty prompt."]
    if len(text) > MAX_H3_PROMPT_CHARS:
        issues.append(
            f"Prompt has {len(text)} characters; maximum is {MAX_H3_PROMPT_CHARS}."
        )
    if "```" in text:
        issues.append("Markdown code fences are not allowed.")
    if re.search(r"</?think>|<\|channel>", text, flags=re.IGNORECASE):
        issues.append("Model reasoning-channel residue is not allowed.")
    structural_scan = re.sub(r'"[^"\n]*"', "", text)
    structural_scan = re.sub(r"<d>.*?</d>", "", structural_scan, flags=re.DOTALL)
    if re.search(r"(?i)(?<![\w\"'])negative_prompt\s*:", structural_scan):
        issues.append("A separate negative_prompt field is not allowed.")

    positions: list[int] = []
    for field in fields:
        matches = _header_matches(text, field)
        if len(matches) != 1:
            issues.append(
                f"Field header `{field}` must occur exactly once (found {len(matches)})."
            )
        positions.append(matches[0].start() if len(matches) == 1 else -1)

    allowed_headers = {field[:-1] for field in fields}
    for match in re.finditer(
        r"(?m)^([A-Za-z][A-Za-z0-9_ -]{0,40}):(?=$|[ \t])",
        text,
    ):
        if match.group(1) not in allowed_headers:
            issues.append(f"Unknown top-level field `{match.group(1)}:` is not allowed.")

    if any(position < 0 for position in positions) or positions != sorted(positions):
        issues.append("Required fields are missing or out of order.")
    else:
        for index, field in enumerate(fields):
            body_start = positions[index] + len(field)
            body_end = positions[index + 1] if index + 1 < len(fields) else len(text)
            if not text[body_start:body_end].strip():
                issues.append(f"Field `{field}` must have a non-empty body.")
            if index > 0:
                separator = text[positions[index - 1] + len(fields[index - 1]) : positions[index]]
                trailing_newlines = re.search(r"\n+$", separator)
                if trailing_newlines is None or len(trailing_newlines.group()) != 2:
                    issues.append(
                        f"Field `{field}` must be separated from the previous field by exactly one blank line."
                    )

    placeholders = (
        r"\[Shot\s+[NX]\]",
        r"<Subject\s+[NX]>",
        r"<Picture\s+[NX]>",
        r"<Video\s+[NX]>",
        r"<Audio\s+[NX]>",
        r"\bAt\s+S\.SS\b",
        r"\bAt\s+MM:SS(?:\.mmm)?\b",
    )
    if any(
        re.search(pattern, structural_scan, flags=re.IGNORECASE)
        for pattern in placeholders
    ):
        issues.append("Unresolved timing or reference placeholders remain in the prompt.")

    for token in re.findall(
        r"<\s*(?:Subject|Picture|Video|Audio)(?=\s|\d|>)[^<>]*>",
        structural_scan,
        flags=re.IGNORECASE,
    ):
        if not re.fullmatch(r"<(?:Subject|Picture|Video|Audio) [1-9]\d*>", token):
            issues.append(f"Malformed structured reference tag `{token}`.")

    speaker_groups = list(re.finditer(
        r"\(\s*[sS]\s*(?:\d+|[xXnN])(?:\s*,\s*[sS]\s*(?:\d+|[xXnN]))*\s*\)",
        structural_scan,
    ))
    for group in speaker_groups:
        if not re.fullmatch(r"\(S[1-9]\d*(?:,S[1-9]\d*)*\)", group.group()):
            issues.append(f"Malformed Speaker group `{group.group()}`.")
    speaker_context_scan = re.sub(r'"[^"\n]*"', "", text)
    for group in reversed(list(re.finditer(r"\(S[1-9]\d*(?:,S[1-9]\d*)*\)", speaker_context_scan))):
        speaker_context_scan = (
            speaker_context_scan[: group.start()]
            + " " * len(group.group())
            + speaker_context_scan[group.end() :]
        )
    if re.search(
        r"(?i)\bS\s*\d+\b\s*(?::|says?\b|speaks?\b|sings?\b|"
        r"whispers?\b|shouts?\b|asks?\b|replies?\b|narrates?\b)[^<\n]{0,80}<d>",
        speaker_context_scan,
    ):
        issues.append("A Speaker identifier before <d> must use canonical parenthesized form.")

    if text.count("<d>") != text.count("</d>"):
        issues.append("Dialogue tags are unbalanced.")
    for dialogue in re.findall(r"<d>(.*?)</d>", text, flags=re.DOTALL):
        if not re.match(r"\s*\[[^\]\n]*\S[^\]\n]*\]\s+\S", dialogue):
            issues.append("Every <d> block must begin with a non-empty [Language] tag.")
            break
    return issues


def canonicalize_t2v_structure(text: str) -> str:
    """Normalize harmless heading variants in a standalone T2V prompt."""

    value = sanitize_generated_text(text, "T2V")
    aliases = {
        "Scene overview:": r"scene[\s_-]*overview",
        "Storyboard:": r"storyboard(?:[ \t]*\([^\n]*\))?",
        "Camera:": r"camera",
        "Audio:": r"audio",
    }
    for canonical, alias in aliases.items():
        value = re.sub(
            rf"(?im)^[ \t]*(?:\#{{1,6}}[ \t]+)?(?:\*\*|__)?"
            rf"{alias}(?:[ \t]*:(?:\*\*|__)?|(?:\*\*|__)[ \t]*:)",
            canonical,
            value,
        )
    return value.strip()


def validate_t2v_prompt(
    text: str,
    requested_duration_seconds: float,
) -> ValidationResult:
    """Validate a self-contained narrative prompt for native H3 T2V."""

    candidate = canonicalize_t2v_structure(text)
    issues: list[str] = []
    if not candidate:
        return ValidationResult(candidate, ("The model returned an empty prompt.",))
    if len(candidate) > MAX_H3_PROMPT_CHARS:
        issues.append(
            f"Prompt has {len(candidate)} characters; maximum is {MAX_H3_PROMPT_CHARS}."
        )
    if "```" in candidate:
        issues.append("Markdown code fences are not allowed.")
    if re.search(r"</?think>|<\|channel>", candidate, flags=re.IGNORECASE):
        issues.append("Model reasoning-channel residue is not allowed.")
    if re.search(r"(?i)(?<![\w\"'])negative_prompt\s*:", candidate):
        issues.append("A separate negative_prompt field is not allowed.")

    headings = ("Scene overview:", "Storyboard:", "Camera:", "Audio:")
    matches = {heading: _header_matches(candidate, heading) for heading in headings}
    for heading in headings:
        count = len(matches[heading])
        if count != 1:
            issues.append(
                f"Heading `{heading}` must occur exactly once (found {count})."
            )

    if all(len(matches[heading]) == 1 for heading in headings):
        positions = [matches[heading][0].start() for heading in headings]
        if positions != sorted(positions):
            issues.append("T2V headings are out of order.")
        style = candidate[: positions[0]].strip()
        if not style:
            issues.append("T2V must begin with an unlabelled visual-style paragraph.")
        elif re.match(
            r"(?i)^(?:here(?:'s| is)|sure\b|certainly\b|the user wants|final prompt)",
            style,
        ):
            issues.append("T2V must not include an assistant preface.")

        for index, heading in enumerate(headings):
            body_start = matches[heading][0].end()
            body_end = positions[index + 1] if index + 1 < len(headings) else len(candidate)
            if not candidate[body_start:body_end].strip():
                issues.append(f"Section `{heading}` must have a non-empty body.")

        storyboard_start = matches["Storyboard:"][0].end()
        storyboard_end = matches["Camera:"][0].start()
        storyboard = candidate[storyboard_start:storyboard_end].strip()
        beat_pattern = re.compile(
            r"^\[(\d+(?:\.\d+)?)s-(\d+(?:\.\d+)?)s\] "
            r"Shot ([1-9]\d*):\s*(\S.*)$"
        )
        beat_lines = [line.strip() for line in storyboard.splitlines() if line.strip()]
        beats: list[tuple[float, float, int]] = []
        for line in beat_lines:
            match = beat_pattern.fullmatch(line)
            if match is None:
                issues.append(
                    "Every Storyboard line must use `[Xs-Ys] Shot N: description`."
                )
                continue
            start, end, shot = match.groups()[:3]
            beats.append((float(start), float(end), int(shot)))

        if not beats:
            issues.append("Storyboard must contain at least one timed shot.")
        else:
            if beats[0][0] != 0.0:
                issues.append("The first Storyboard shot must start at 0 seconds.")
            shot_ids = [shot for _, _, shot in beats]
            if shot_ids != list(range(1, len(shot_ids) + 1)):
                issues.append("Storyboard Shot numbers must start at 1 without gaps.")
            for index, (start, end, _) in enumerate(beats):
                if end <= start:
                    issues.append("Every Storyboard range must have positive duration.")
                if index and abs(start - beats[index - 1][1]) > 1e-6:
                    issues.append(
                        "Storyboard ranges must be contiguous without gaps or overlaps."
                    )
            requested = float(requested_duration_seconds)
            if abs(beats[-1][1] - requested) > 1e-6:
                issues.append(
                    "The final Storyboard range must end at the requested duration "
                    f"of {requested:g} seconds."
                )

    structural_scan = re.sub(r'"[^"\n]*"', "", candidate)
    if re.search(
        r"<\s*(?:Picture|Video|Audio|Subject)\b[^<>]*>",
        structural_scan,
        flags=re.IGNORECASE,
    ):
        issues.append("T2V output must not contain structured media or Subject tags.")
    if re.search(
        r"\bref_(?:image|video|audio)(?:_audio)?_\d+\b",
        structural_scan,
        flags=re.IGNORECASE,
    ):
        issues.append("T2V output must not expose internal media socket names.")
    if re.search(
        r"\b(?:connected|provided|input|reference|source)\s+"
        r"(?:image|video|audio|media|clip|attachment)s?\b",
        structural_scan,
        flags=re.IGNORECASE,
    ):
        issues.append("T2V output must describe content without source attribution.")

    return ValidationResult(candidate, tuple(dict.fromkeys(issues)))


def _field_body(text: str, field: str, next_field: str) -> str:
    starts = _header_matches(text, field)
    ends = _header_matches(text, next_field)
    if len(starts) != 1 or len(ends) != 1 or starts[0].start() >= ends[0].start():
        return ""
    return text[starts[0].end() : ends[0].start()].strip()


def _validate_shot_timeline(
    body: str,
    duration: float,
    *,
    shot_one_must_start_body: bool,
) -> list[str]:
    """Validate only the chronological field, not cross-section Shot citations."""

    issues: list[str] = []
    matches = list(re.finditer(r"\[Shot\s+(\d+)\]", body))
    for token in re.findall(r"\[Shot\s*\d+\]", body):
        if not re.fullmatch(r"\[Shot [1-9]\d*\]", token):
            issues.append(f"Malformed shot header `{token}`.")
    if not matches:
        issues.append("The chronological body must contain [Shot 1].")
        return issues

    raw_ids = [match.group(1) for match in matches]
    ids = [int(value) for value in raw_ids]
    if any(raw != str(value) or value < 1 for raw, value in zip(raw_ids, ids)):
        issues.append("Shot headers must use canonical non-zero-padded integers.")
    if ids != list(range(1, len(ids) + 1)):
        issues.append(
            "Shot headers must occur exactly once each in sequential order starting at 1."
        )
    if shot_one_must_start_body and not body.startswith("[Shot 1]"):
        issues.append("The chronological body must begin directly with [Shot 1].")
    elif ids[0] != 1:
        issues.append("The first chronological shot must be [Shot 1].")

    first_tail = body[matches[0].end() :]
    if re.match(r"\s*(?i:at)\s+\d{2}:\d{2}\.\d{3}\b", first_tail):
        issues.append("[Shot 1] must not have a timestamp.")

    timestamps: list[float] = []
    for index, match in enumerate(matches):
        segment_end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        segment = body[match.end() : segment_end]
        if index == 0:
            content = segment
            timestamp_match = None
        else:
            ordinal = index + 1
            timestamp_match = re.match(r" At (\d{2}):(\d{2})\.(\d{3}),", segment)
            content = segment[timestamp_match.end() :] if timestamp_match else ""
        if not re.search(r"[A-Za-z0-9<\"']", content):
            issues.append(f"[Shot {index + 1}] must contain a concrete description.")
        if index == 0:
            continue
        if timestamp_match is None:
            issues.append(
                f"[Shot {ordinal}] must begin exactly `[Shot {ordinal}] At MM:SS.mmm,`."
            )
            continue
        minutes, seconds, millis = (int(value) for value in timestamp_match.groups())
        if seconds >= 60:
            issues.append(f"[Shot {ordinal}] has an invalid seconds component.")
            continue
        timestamp = minutes * 60 + seconds + millis / 1000.0
        timestamps.append(timestamp)
        if timestamp <= 0:
            issues.append("Later-shot cut times must be positive.")
        if timestamp >= duration + 1e-6:
            issues.append(
                f"Cut timestamp {timestamp:.3f}s is not before duration {duration:.6f}s."
            )
    if timestamps != sorted(timestamps) or len(timestamps) != len(set(timestamps)):
        issues.append("Later-shot timestamps must be strictly increasing.")
    return issues


def _last_shot_id(text: str) -> int | None:
    ids = [int(value) for value in re.findall(r"\[Shot\s+(\d+)\]", text)]
    return max(ids) if ids else None


def _format_cut_timestamp(seconds: float) -> str:
    total_millis = max(0, int(round(float(seconds) * 1000.0)))
    minutes, remainder = divmod(total_millis, 60_000)
    whole_seconds, millis = divmod(remainder, 1_000)
    return f"{minutes:02d}:{whole_seconds:02d}.{millis:03d}"


def _canonicalize_timeline_cut_markers(body: str) -> str:
    """Put later Shot markers at a beat's start with an exact cut timestamp.

    Gemma often writes an otherwise useful beat as ``[2.5s-5s] ... [Shot 2]``
    or omits the redundant timestamp after the marker. The beat boundary already
    determines the cut time, so this normalization does not invent content.
    """

    beat_pattern = re.compile(
        r"^(?P<indent>\s*)(?P<beat>\[(?P<start>\d+(?:\.\d+)?)s-"
        r"\d+(?:\.\d+)?s\])(?P<rest>.*)$"
    )
    shot_pattern = re.compile(r"\[Shot (?P<shot>[2-9]\d*)\]")
    implicit_cut_pattern = re.compile(
        r"^\s*At\s+(?:\d{2}:\d{2}\.\d{3}|\d+(?:\.\d+)?s),\s*"
        r"(?:(?:the\s+)?scene\s+cuts?\b|hard\s+cut\b|cut\s+to\b)",
        flags=re.IGNORECASE,
    )
    leading_time_pattern = re.compile(
        r"^\s*(?i:at)\s+(?:\d{1,3}:\d{2}(?:\.\d{1,6})?"
        r"|\d+(?:\.\d+)?\s*(?:s|seconds?))\s*,?\s*"
    )
    lines: list[str] = []
    trailing_newline = body.endswith("\n")
    for line in body.splitlines():
        beat_match = beat_pattern.match(line)
        if not beat_match:
            lines.append(line)
            continue
        rest = beat_match.group("rest")
        shots = list(shot_pattern.finditer(rest))
        start = float(beat_match.group("start"))
        if not shots and start > 0 and implicit_cut_pattern.match(rest):
            previous_ids = [
                int(value)
                for prior in lines
                for value in re.findall(r"\[Shot (\d+)\]", prior)
            ]
            shot_id = max(previous_ids, default=1) + 1
            rest = f" [Shot {shot_id}]" + rest
            shots = list(shot_pattern.finditer(rest))
        if len(shots) != 1 or start <= 0:
            lines.append(line)
            continue

        shot = shots[0]
        before = rest[: shot.start()].strip(" \t,;:-")
        after = leading_time_pattern.sub("", rest[shot.end() :], count=1).strip()
        description = " ".join(part for part in (before, after) if part).strip()
        canonical = (
            f'{beat_match.group("indent")}{beat_match.group("beat")} '
            f'{shot.group(0)} At {_format_cut_timestamp(start)},'
        )
        if description:
            canonical += " " + description
        lines.append(canonical)
    normalized = "\n".join(lines)
    return normalized + ("\n" if trailing_newline else "")


def _canonicalize_invalid_ref_cut_times(body: str, duration: float) -> str:
    """Repair non-positive or unordered Ref2VA cuts deterministically.

    Gemma can preserve the correct shot order while assigning ``00:00.000``
    to a later shot, including after a repair pass. When any later timestamp is
    missing or invalid, distribute all later cuts evenly inside the requested
    duration. This changes only structural timing metadata; shot content and
    order remain untouched.
    """

    matches = list(re.finditer(r"\[Shot ([1-9]\d*)\]", body))
    ids = [int(match.group(1)) for match in matches]
    if len(matches) < 2 or ids != list(range(1, len(matches) + 1)):
        return body

    exact_timestamp = re.compile(r"^ At (\d{2}):(\d{2})\.(\d{3}),")
    timestamps: list[float] = []
    valid = True
    for index, match in enumerate(matches[1:], start=1):
        next_start = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        segment = body[match.end() : next_start]
        timestamp_match = exact_timestamp.match(segment)
        if timestamp_match is None:
            valid = False
            break
        minutes, seconds, millis = (int(value) for value in timestamp_match.groups())
        timestamp = minutes * 60 + seconds + millis / 1000.0
        if seconds >= 60 or timestamp <= 0 or timestamp >= duration - 1e-6:
            valid = False
            break
        timestamps.append(timestamp)
    if valid and timestamps == sorted(timestamps) and len(timestamps) == len(set(timestamps)):
        return body
    if duration <= 0:
        return body

    leading_time = re.compile(
        r"^\s*(?:(?i:at)\s+)?(?:\d{1,3}:\d{2}(?:\.\d+)?|"
        r"\d+(?:\.\d+)?\s*(?:s|seconds?))\s*,?\s*"
    )
    result = body
    shot_count = len(matches)
    for index in range(shot_count - 1, 0, -1):
        match = matches[index]
        segment_end = matches[index + 1].start() if index + 1 < shot_count else len(body)
        description = leading_time.sub("", body[match.end() : segment_end], count=1).strip()
        cut = duration * index / shot_count
        replacement = f"{match.group(0)} At {_format_cut_timestamp(cut)},"
        if description:
            replacement += " " + description
        result = result[: match.start()] + replacement + result[segment_end:]
    return result


def _isolate_last_base_schema(text: str, mode: str) -> str:
    """Keep the last complete base schema when Gemma drafts the main field twice."""

    candidate = sanitize_generated_text(text, mode)
    main_headers = _header_matches(candidate, BASE_FIELDS[0])
    sound_headers = _header_matches(candidate, BASE_FIELDS[1])
    music_headers = _header_matches(candidate, BASE_FIELDS[2])
    if len(main_headers) <= 1 or len(sound_headers) != 1 or len(music_headers) != 1:
        return candidate
    sound_start = sound_headers[0].start()
    eligible = [match for match in main_headers if match.start() < sound_start]
    if not eligible:
        return candidate
    return candidate[eligible[-1].start() :].strip()


def _canonicalize_speaker_groups(text: str) -> str:
    pattern = re.compile(
        r"\(\s*[sS]\s*([1-9]\d*)(?:\s*,\s*[sS]\s*([1-9]\d*))*\s*\)"
    )

    def replace(match: re.Match[str]) -> str:
        values = re.findall(r"[sS]\s*([1-9]\d*)", match.group(0))
        return "(" + ",".join(f"S{value}" for value in values) + ")"

    return pattern.sub(replace, text)


def _normalize_ref_headers(text: str) -> str:
    """Normalize harmless Ref2VA header variants before schema isolation."""

    value = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    value = re.sub(r"<think>.*?</think>", "", value, flags=re.DOTALL | re.IGNORECASE).strip()
    value = _strip_orphan_think_closures(value, "Ref2VA")
    if value.lower().startswith("<think>") and "</think>" not in value.lower():
        value = ""
    value = re.sub(r"^```(?:text|markdown)?\s*\n?", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\n?```\s*$", "", value).strip()

    aliases = {
        "subject_definitions:": r"subject[\s_-]*definitions",
        "summary:": r"summary",
        "retention_analysis:": r"retention[\s_-]*analysis",
        "detailed_description:": r"detailed[\s_-]*description",
        "overall_soundscape:": r"overall[\s_-]*soundscape",
        "non_diegetic_music:": r"non[\s_-]*diegetic[\s_-]*music",
    }
    for canonical, alias in aliases.items():
        value = re.sub(
            rf"(?im)^[ \t]*(?:\#{{1,6}}[ \t]+)?(?:[-*][ \t]+)?"
            rf"(?:\*\*|__)?{alias}(?:\*\*|__)?[ \t]*:",
            canonical,
            value,
        )

    def normalize_label(match: re.Match[str]) -> str:
        return f"<{match.group(1).title()} {int(match.group(2))}>"

    value = re.sub(
        r"<\s*(Subject|Picture|Video|Audio)\s*(\d+)\s*>",
        normalize_label,
        value,
        flags=re.IGNORECASE,
    )
    # Bare structured labels look like accidental top-level fields to the H3
    # schema. Convert them everywhere; the section-specific canonicalizer will
    # restore the retention colon when the line belongs to retention_analysis.
    value = re.sub(
        r"(?im)^[ \t]*(Subject|Picture|Video|Audio)\s*(\d+)\s*:\s*",
        lambda match: f"<{match.group(1).title()} {int(match.group(2))}> ",
        value,
    )
    value = re.sub(
        r"(?<!<)\b(Subject|Picture|Video|Audio)\s+([1-9]\d*)\b(?!>)",
        lambda match: f"<{match.group(1).title()} {int(match.group(2))}>",
        value,
        flags=re.IGNORECASE,
    )
    return value.strip()


def _isolate_last_ref_schema(text: str) -> str:
    """Keep the final Ref2VA draft and restore an omitted first header."""

    candidate = _normalize_ref_headers(text)
    summaries = _header_matches(candidate, REF_FIELDS[1])
    if not summaries:
        return candidate

    summary = summaries[-1]
    previous_music = [
        match
        for match in _header_matches(candidate, REF_FIELDS[-1])
        if match.start() < summary.start()
    ]
    attempt_start = previous_music[-1].end() if previous_music else 0
    subject_headers = [
        match
        for match in _header_matches(candidate, REF_FIELDS[0])
        if attempt_start <= match.start() < summary.start()
    ]
    start = subject_headers[-1].start() if subject_headers else attempt_start
    candidate = candidate[start:].strip()

    if not _header_matches(candidate, REF_FIELDS[0]):
        summary_headers = _header_matches(candidate, REF_FIELDS[1])
        remaining = [_header_matches(candidate, field) for field in REF_FIELDS[1:]]
        if (
            len(summary_headers) == 1
            and all(len(matches) == 1 for matches in remaining)
            and [matches[0].start() for matches in remaining]
            == sorted(matches[0].start() for matches in remaining)
        ):
            prefix = candidate[: summary_headers[0].start()].strip()
            prefix_lines = prefix.splitlines()
            if prefix_lines and re.match(
                r"(?i)^(?:here is|final|corrected|ref2va prompt\b)",
                prefix_lines[0].strip(" *_#:-"),
            ):
                prefix = "\n".join(prefix_lines[1:]).strip()
            suffix = candidate[summary_headers[0].start() :]
            candidate = "subject_definitions:\n" + prefix + "\n\n" + suffix
    return candidate.strip()


def _default_ref_task_types(summary: str, manifest: ReferenceManifest) -> list[str]:
    """Infer a conservative signature from existing prose and connected media."""

    lowered = summary.lower()
    task_types: list[str] = []
    if manifest.videos and re.search(r"edited version of\s+<video\s+\d+>", lowered):
        task_types.append("video editing")
    elif manifest.videos and re.search(r"\b(?:video )?continu(?:e|es|ed|ation)\b", lowered):
        task_types.append("video continuation")
    elif manifest.pictures and "keyframe completion" in lowered:
        task_types.append("keyframe completion")
    elif manifest.pictures or manifest.videos:
        task_types.append("reference generation")

    if manifest.audios:
        reuse = re.search(
            r"\b(?:audio reuse|reuse(?:s|d)? the (?:source )?(?:audio|signal)|"
            r"copy(?:ies|ied)? the (?:source )?(?:audio|signal)|as-is)\b",
            lowered,
        )
        task_types.append("audio reuse" if reuse else "audio reference")
    return task_types or ["reference generation"]


def _strip_leading_ref_task_echo(prose: str, allowed: set[str]) -> str:
    task_type_pattern = (
        r"(?:"
        + "|".join(
            re.escape(item) for item in sorted(allowed, key=len, reverse=True)
        )
        + r")"
    )
    repeated_types = re.compile(
        r"^(?:"
        + task_type_pattern
        + r")(?:\s*(?:\+|&|and|,|/)\s*"
        + task_type_pattern
        + r")*\s*[.:;,\-–—]\s*",
        flags=re.IGNORECASE,
    )
    value = prose.lstrip()
    while repeated_types.match(value):
        value = repeated_types.sub("", value, count=1).lstrip()
    return value


def _canonicalize_ref_summary(body: str, manifest: ReferenceManifest) -> str:
    value = body.strip()
    match = re.match(r"^\[([^\]\n]+)\](?=$|\s)", value)
    allowed = {
        "keyframe completion",
        "reference generation",
        "video editing",
        "video continuation",
        "audio reuse",
        "audio reference",
    }
    if match:
        values = [part.strip().lower() for part in match.group(1).split("+")]
        if values and set(values).issubset(allowed):
            unique = list(dict.fromkeys(values))
            prose = value[match.end() :].lstrip()
            # Models sometimes repeat the task-type token as a sentence before
            # the actual summary (for example, "reference generation."). The
            # bracketed signature already carries that metadata.
            # A repair can echo more than one signature item as a prose clause,
            # for example ``reference generation + audio reference: ...`` or
            # ``reference generation and audio reference. ...``.  Treat the
            # entire leading metadata clause as deterministic formatting rather
            # than spending (and possibly failing) another model pass.
            prose = _strip_leading_ref_task_echo(prose, allowed)
            signature = "[" + " + ".join(unique) + "]"
            return signature + (" " + prose if prose else "")
        return value
    signature = " + ".join(_default_ref_task_types(value, manifest))
    return f"[{signature}] {value}".rstrip()


def _subject_source_label(
    subject_id: int,
    candidate: str,
    manifest: ReferenceManifest,
) -> str | None:
    visible_labels = list(manifest.labels("image")) + list(manifest.labels("video"))
    subject_label = f"<Subject {subject_id}>"
    for match in re.finditer(re.escape(subject_label), candidate):
        context = candidate[max(0, match.start() - 180) : match.end() + 180]
        for label in visible_labels:
            if label in context:
                return label
    if visible_labels:
        return visible_labels[min(subject_id - 1, len(visible_labels) - 1)]
    return None


def _canonicalize_ref_subjects(
    body: str,
    candidate: str,
    manifest: ReferenceManifest,
) -> tuple[str, list[int]]:
    value = re.sub(
        r"(?im)^[ \t]*(?:<\s*)?Subject\s*(\d+)(?:\s*>)?\s*[:\-]\s*",
        lambda match: f"<Subject {int(match.group(1))}> ",
        body.strip(),
    )
    value = re.sub(
        r"(?m)^(<Subject [1-9]\d*>)\s*=\s*",
        r"\1 is ",
        value,
    )
    subject_ids = sorted(
        {int(raw) for raw in re.findall(r"<Subject\s+(\d+)>", candidate)}
    )
    lines = [line.rstrip() for line in value.splitlines() if line.strip()]
    defined = {
        int(match.group(1))
        for line in lines
        if (match := re.match(r"^<Subject (\d+)>(?=$|[ \t])", line))
    }
    for subject_id in subject_ids:
        if subject_id in defined:
            continue
        source = _subject_source_label(subject_id, candidate, manifest)
        if source:
            lines.append(
                f"<Subject {subject_id}> is the tracked visible subject sourced from "
                f"{source}, retaining the reference-critical appearance and role described below."
            )
        else:
            lines.append(
                f"<Subject {subject_id}> is the requested tracked visible subject, "
                "with its appearance and role defined by the target description."
            )

    if not lines:
        role_text = {
            "image": "connected reference image used as visible source evidence",
            "video": "connected reference video used as temporal and visible source evidence",
            "audio": "connected reference audio used as audible source evidence",
        }
        lines = [
            f"{asset.label} is the {role_text[asset.kind]}."
            for asset in manifest.presentation_order
        ]
    return "\n".join(lines), subject_ids


def _ref_marker_for(label: str, source_text: str) -> tuple[str, str]:
    lowered = source_text.lower()
    if label.startswith("<Audio"):
        if "partially_copy" in lowered or "partial copy" in lowered:
            return "partially_copy", "the identified source segment or layer is reused."
        if "fully_copy" in lowered or re.search(r"\b(?:copy|reuse)\b.*\bas-is\b", lowered):
            return "fully_copy", "the source signal is reused as-is."
        if "weak" in lowered:
            return "weak_reference", "only broad audible qualities guide the new result."
        return "reference", "its audible properties guide the target without copying the source signal."
    if "attribute_transfer" in lowered or "attribute transfer" in lowered or "style transfer" in lowered:
        return "attribute_transfer", "the specified visible attributes transfer while other content may change."
    if "weak_reference" in lowered or "weak reference" in lowered or "inspired" in lowered:
        return "weak_reference", "only the specified visible cues guide the target."
    if "partially_preserved" in lowered or "partial" in lowered or "edited version" in lowered:
        return "partially_preserved", "the specified visible traits remain while requested changes are applied."
    if "fully_preserved" in lowered or "fully preserved" in lowered:
        return "fully_preserved", "the reference-critical visible identity and attributes remain consistent."
    if label.startswith("<Video"):
        return "weak_reference", "its temporal and visible cues guide the target without frame-for-frame copying."
    return "fully_preserved", "the reference-critical visible identity and attributes remain consistent."


def _canonical_ref_relationship_line(label: str, original: str) -> str:
    annotation = ""
    match = re.match(rf"^{re.escape(label)}(\s+\([^\n)]*\))?:\s*(.*)$", original.strip())
    prose = original.strip()
    if match:
        annotation = match.group(1) or ""
        prose = match.group(2).strip()
    marker_match = re.match(
        r"^(fully_preserved|partially_preserved|attribute_transfer|weak_reference|"
        r"fully_copy|partially_copy|reference)\s*-\s*(.*)$",
        prose,
        flags=re.IGNORECASE,
    )
    allowed = (
        {"fully_copy", "partially_copy", "reference", "weak_reference"}
        if label.startswith("<Audio")
        else {"fully_preserved", "partially_preserved", "attribute_transfer", "weak_reference"}
    )
    if marker_match and marker_match.group(1).lower() in allowed:
        marker = marker_match.group(1).lower()
        prose = marker_match.group(2).strip()
        _, fallback = _ref_marker_for(label, marker)
    else:
        marker, fallback = _ref_marker_for(label, prose)
        prose = re.sub(
            r"^(?:fully[ _-]?preserved|partially[ _-]?preserved|attribute[ _-]?transfer|"
            r"weak[ _-]?reference|fully[ _-]?copy|partially[ _-]?copy|reference)"
            r"(?:\s*[-:]\s*|\s+)",
            "",
            prose,
            flags=re.IGNORECASE,
        ).strip()
    all_markers = {
        "fully_preserved",
        "partially_preserved",
        "attribute_transfer",
        "weak_reference",
        "fully_copy",
        "partially_copy",
        "reference",
    }
    if prose.lower().strip(" .:_-") in all_markers:
        prose = ""
    return f"{label}{annotation}: {marker} - {prose or fallback}"


def _canonicalize_ref_retention(
    body: str,
    subject_body: str,
    subject_ids: list[int],
    manifest: ReferenceManifest,
) -> str:
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    for index, line in enumerate(lines):
        malformed = re.match(
            r"^(<(?:Subject|Picture|Video|Audio) [1-9]\d*>)"
            r"(\s+\([^\n)]*\))?\s+(.*)$",
            line,
        )
        if malformed and not line.startswith(malformed.group(1) + ":"):
            lines[index] = (
                malformed.group(1)
                + (malformed.group(2) or "")
                + ": "
                + malformed.group(3)
            )
    removed: set[int] = set()

    def occurrences(label: str) -> list[int]:
        return [
            index
            for index, line in enumerate(lines)
            if re.match(rf"^{re.escape(label)}(?:\s+\([^\n)]*\))?:", line)
        ]

    def ensure(label: str, required: bool) -> None:
        indexes = occurrences(label)
        if indexes:
            first = indexes[0]
            lines[first] = _canonical_ref_relationship_line(label, lines[first])
            removed.update(indexes[1:])
        elif required:
            lines.append(_canonical_ref_relationship_line(label, ""))

    for subject_id in subject_ids:
        ensure(f"<Subject {subject_id}>", True)

    for asset in manifest.presentation_order:
        covered = asset.kind != "audio" and any(
            asset.label in line
            for line in subject_body.splitlines()
            if re.match(r"^<Subject [1-9]\d*>(?=$|[ \t])", line)
        )
        ensure(asset.label, not covered)
    return "\n".join(line for index, line in enumerate(lines) if index not in removed)


def _synchronize_ref_audio_signature(
    summary: str,
    retention: str,
    manifest: ReferenceManifest,
) -> str:
    """Make summary audio task types agree with canonical relationships."""

    match = re.match(r"^\[([^\]\n]+)\](?=$|\s)", summary.strip())
    if match is None or not manifest.audios:
        return summary
    signature = [part.strip().lower() for part in match.group(1).split("+")]
    signature = [
        item for item in signature if item not in {"audio reuse", "audio reference"}
    ]
    markers: list[str] = []
    for label in manifest.labels("audio"):
        relation = re.search(
            rf"(?m)^{re.escape(label)}(?:\s+\([^\n)]*\))?:\s*"
            r"(fully_copy|partially_copy|reference|weak_reference)\s*-",
            retention,
            flags=re.IGNORECASE,
        )
        if relation:
            markers.append(relation.group(1).lower())
    if any(marker in {"fully_copy", "partially_copy"} for marker in markers):
        signature.append("audio reuse")
    if any(marker in {"reference", "weak_reference"} for marker in markers):
        signature.append("audio reference")
    prose = summary.strip()[match.end() :].lstrip()
    canonical_signature = "[" + " + ".join(dict.fromkeys(signature)) + "]"
    return canonical_signature + (" " + prose if prose else "")


def _audio_reuse_is_explicit(raw_user_request: str) -> bool:
    value = str(raw_user_request or "").lower()
    return bool(
        re.search(
            r"\b(?:copy|copies|copied|reuse|reuses|reused|as-is|same signal|"
            r"retain (?:the )?(?:original )?(?:audio|soundtrack)|"
            r"keep (?:the )?(?:original )?(?:audio|soundtrack)|"
            r"copia(?:re)?|riusa(?:re)?|riutilizza(?:re)?|identic[oa]|"
            r"mantieni (?:l['’])?(?:audio|soundtrack) originale)\b",
            value,
        )
    )


def _default_ref_audio_to_reference(
    retention: str,
    manifest: ReferenceManifest,
    raw_user_request: str | None,
) -> str:
    """Do not turn generic use/guidance language into signal copying."""

    if raw_user_request is None or _audio_reuse_is_explicit(raw_user_request):
        return retention
    value = retention
    for label in manifest.labels("audio"):
        value = re.sub(
            rf"(?m)^{re.escape(label)}(\s+\([^\n)]*\))?:\s*"
            r"(?:fully_copy|partially_copy)\s*-\s*[^\n]*$",
            lambda match: (
                f"{label}{match.group(1) or ''}: reference - its audible properties "
                "guide the target without copying the source signal."
            ),
            value,
            flags=re.IGNORECASE,
        )
    return value


def _remove_unrequested_audio_copy_prose(
    body: str,
    manifest: ReferenceManifest,
    raw_user_request: str | None,
) -> str:
    if (
        raw_user_request is None
        or _audio_reuse_is_explicit(raw_user_request)
        or not manifest.audios
    ):
        return body
    value = body
    substitutions = (
        (r"\b(?:fully|partially)\s+copied\b", "used only as a reference"),
        (r"\b(?:copied|reused)\s+as-is\b", "used only as a reference"),
        (r"\b(?:copied|reused)\b", "referenced"),
        (r"\b(?:copy|reuse)\b", "reference"),
        (r"\b(?:an|the)\s+exact\s+(?:source\s+)?signal\b", "audible reference properties"),
    )
    for pattern, replacement in substitutions:
        value = re.sub(pattern, replacement, value, flags=re.IGNORECASE)
    return value


def _compact_seconds_value(value: float) -> str:
    return f"{float(value):.6f}".rstrip("0").rstrip(".")


def _canonicalize_requested_timeline_tail(
    body: str,
    requested_duration: float,
    effective_render_duration: float,
) -> str:
    """Fold grid-padding beats into the requested-duration endpoint."""

    pattern = re.compile(
        r"^(?P<indent>\s*)\[(?P<start>\d+(?:\.\d+)?)s-"
        r"(?P<end>\d+(?:\.\d+)?)s\](?P<description>.*)$"
    )
    # Gemma sometimes places the first range directly after ``Timeline:``.
    # Put that beat on its own line so the same deterministic duration folding
    # applies to inline and multiline timelines.
    body = re.sub(
        r"(?i)(\bTimeline:)\s*(?=\[\d+(?:\.\d+)?s-\d+(?:\.\d+)?s\])",
        r"\1\n",
        body,
    )
    lines = body.splitlines()
    last_beat_index: int | None = None
    changed = False
    for index, line in enumerate(lines):
        match = pattern.match(line)
        if not match:
            continue
        start = float(match.group("start"))
        end = float(match.group("end"))
        if start >= requested_duration - 1e-6:
            description = match.group("description").strip()
            if last_beat_index is not None and description:
                lines[last_beat_index] = lines[last_beat_index].rstrip() + " " + description
            lines[index] = ""
            changed = True
            continue
        if end > requested_duration + 1e-6:
            end_text = _compact_seconds_value(requested_duration)
            lines[index] = (
                f'{match.group("indent")}[{match.group("start")}s-{end_text}s]'
                + match.group("description")
            )
            changed = True
        last_beat_index = index

    result = "\n".join(line for line in lines if line != "").strip()
    if changed and effective_render_duration > requested_duration + 1e-6:
        hold = (
            "The final described state holds unchanged through the "
            f"{effective_render_duration:.6f}-second aligned render tail."
        )
        if hold not in result:
            result += "\n" + hold
    return result


def canonicalize_ref_structure(
    text: str,
    length: int,
    manifest: ReferenceManifest,
    requested_duration_seconds: float | None = None,
    raw_user_request: str | None = None,
) -> str:
    """Render deterministic Ref2VA metadata and formatting around model prose."""

    candidate = _isolate_last_ref_schema(text)
    matches = [_header_matches(candidate, field) for field in REF_FIELDS]
    if any(len(field_matches) != 1 for field_matches in matches):
        return candidate
    positions = [field_matches[0].start() for field_matches in matches]
    if positions != sorted(positions):
        return candidate

    bodies: list[str] = []
    for index, field_matches in enumerate(matches):
        start = field_matches[0].end()
        end = matches[index + 1][0].start() if index + 1 < len(matches) else len(candidate)
        bodies.append(candidate[start:end].strip())

    subject_body, subject_ids = _canonicalize_ref_subjects(bodies[0], candidate, manifest)
    bodies[0] = subject_body
    bodies[1] = _canonicalize_ref_summary(bodies[1], manifest)
    bodies[2] = _canonicalize_ref_retention(bodies[2], subject_body, subject_ids, manifest)
    bodies[2] = _default_ref_audio_to_reference(
        bodies[2], manifest, raw_user_request
    )
    for index in (0, 1, 3, 4, 5):
        bodies[index] = _remove_unrequested_audio_copy_prose(
            bodies[index], manifest, raw_user_request
        )
    bodies[1] = _synchronize_ref_audio_signature(bodies[1], bodies[2], manifest)
    bodies[3] = _canonicalize_timeline_cut_markers(bodies[3])
    if requested_duration_seconds is not None:
        bodies[3] = _canonicalize_requested_timeline_tail(
            bodies[3],
            float(requested_duration_seconds),
            effective_duration(length),
        )
    cut_duration = (
        float(requested_duration_seconds)
        if requested_duration_seconds is not None
        else effective_duration(length)
    )
    bodies[3] = _canonicalize_invalid_ref_cut_times(bodies[3], cut_duration)

    result = "\n\n".join(f"{field}\n{body}" for field, body in zip(REF_FIELDS, bodies))
    return _canonicalize_speaker_groups(result).strip()


def canonicalize_base_alignment(text: str, mode: str, length: int) -> str:
    """Replace model-authored base alignment prose with canonical metadata.

    The alignment prefix is fully determined by the connected keyframes, the
    target duration, and (for final-frame modes) the generated final shot.  It
    should therefore not consume a repair pass merely because the model used
    equivalent punctuation or wording.

    Only the prefix before the structured description is replaced.  If the
    required structured header or final shot cannot be identified, the
    sanitized candidate is returned unchanged so normal validation and repair
    still report the underlying structural problem.
    """

    candidate = sanitize_generated_text(text, mode)
    header = _header_matches(candidate, BASE_FIELDS[0])
    if len(header) != 1:
        return candidate

    body = candidate[header[0].start() :].strip()
    if mode == "T2VA":
        return body
    if mode == "I2VA":
        return (
            "For the target video, at 0.00 seconds into the target video, "
            "<Picture 1> (from [Shot 1]) is fully referenced.\n\n"
            + body
        )

    if mode not in ("FL2VA", "L2VA"):
        return candidate
    main_body = _field_body(body, BASE_FIELDS[0], BASE_FIELDS[1])
    final_shot = _last_shot_id(main_body)
    if final_shot is None:
        return candidate
    duration_text = duration_2dp(length)
    if mode == "FL2VA":
        return (
            "How the reference pictures align with the target video — "
            "Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; "
            f"Picture 2 (from Shot {final_shot}) aligns with the {duration_text}-second "
            "mark of the target video.\n\n"
            + body
        )
    return (
        "How the reference pictures align with the target video — "
        f"<Picture 1> (from [Shot {final_shot}]) aligns with the {duration_text}-second "
        "mark of the target video.\n\n"
        + body
    )


def canonicalize_base_structure(text: str, mode: str, length: int) -> str:
    """Canonicalize deterministic base metadata and section separators.

    Generated section bodies remain untouched.  Separators are normalized only
    when every required top-level field occurs exactly once in the right order;
    otherwise validation and the repair pass retain responsibility for the
    genuinely malformed structure.
    """

    candidate = _isolate_last_base_schema(text, mode)
    candidate = canonicalize_base_alignment(candidate, mode, length)
    candidate = _canonicalize_speaker_groups(candidate)
    matches = [_header_matches(candidate, field) for field in BASE_FIELDS]
    if any(len(field_matches) != 1 for field_matches in matches):
        return candidate
    if [field_matches[0].start() for field_matches in matches] != sorted(
        field_matches[0].start() for field_matches in matches
    ):
        return candidate

    main_start = matches[0][0].end()
    main_end = matches[1][0].start()
    candidate = (
        candidate[:main_start]
        + _canonicalize_timeline_cut_markers(candidate[main_start:main_end])
        + candidate[main_end:]
    )
    # A normalized implicit cut can introduce a new final Shot. Rebuild FL2VA/
    # L2VA alignment after that deterministic change so Picture 2 names it.
    candidate = canonicalize_base_alignment(candidate, mode, length)
    matches = [_header_matches(candidate, field) for field in BASE_FIELDS]

    # Work from right to left so earlier match positions stay valid.  rstrip
    # removes only formatting whitespace at the section boundary.
    for field_matches in reversed(matches[1:]):
        start = field_matches[0].start()
        candidate = candidate[:start].rstrip() + "\n\n" + candidate[start:]
    return candidate.strip()


def validate_base_prompt(text: str, mode: str, length: int) -> ValidationResult:
    candidate = sanitize_generated_text(text, mode)
    duration = effective_duration(length)
    duration_text = duration_2dp(length)
    issues = _validate_common(candidate, BASE_FIELDS)
    main_body = _field_body(candidate, BASE_FIELDS[0], BASE_FIELDS[1])
    issues.extend(
        _validate_shot_timeline(
            main_body,
            duration,
            shot_one_must_start_body=True,
        )
    )
    speaker_values = sorted(set(_speaker_ids(candidate)))
    if speaker_values and speaker_values != list(range(1, max(speaker_values) + 1)):
        issues.append("Speaker numbering must start at 1 and contain no gaps.")

    if mode == "T2VA":
        if not candidate.startswith(BASE_FIELDS[0]):
            issues.append("T2VA must begin directly with integrated_multimodal_description:.")
        allowed_pictures = 0
    elif mode == "I2VA":
        exact = (
            "For the target video, at 0.00 seconds into the target video, "
            "<Picture 1> (from [Shot 1]) is fully referenced.\n\n"
            + BASE_FIELDS[0]
        )
        if not candidate.startswith(exact):
            issues.append("I2VA alignment line or following blank line is not exact.")
        allowed_pictures = 1
    elif mode == "FL2VA":
        pattern = re.compile(
            r"^How the reference pictures align with the target video — "
            r"Picture 1 \(from Shot 1\) aligns with the 0\.00-second mark of the target video; "
            r"Picture 2 \(from Shot (\d+)\) aligns with the "
            + re.escape(duration_text)
            + r"-second mark of the target video\.\n\n"
            + re.escape(BASE_FIELDS[0])
        )
        match = pattern.match(candidate)
        if not match:
            issues.append("FL2VA alignment line, duration, or following blank line is not exact.")
        elif _last_shot_id(main_body) != int(match.group(1)):
            issues.append("FL2VA Picture 2 must name the actual final shot.")
        allowed_pictures = 2
    elif mode == "L2VA":
        pattern = re.compile(
            r"^How the reference pictures align with the target video — "
            r"<Picture 1> \(from \[Shot (\d+)\]\) aligns with the "
            + re.escape(duration_text)
            + r"-second mark of the target video\.\n\n"
            + re.escape(BASE_FIELDS[0])
        )
        match = pattern.match(candidate)
        if not match:
            issues.append("L2VA alignment line, duration, or following blank line is not exact.")
        elif _last_shot_id(main_body) != int(match.group(1)):
            issues.append("L2VA Picture 1 must name the actual final shot.")
        allowed_pictures = 1
    else:
        issues.append(f"Unknown base mode: {mode}.")
        allowed_pictures = 0

    raw_labels = re.findall(r"<Picture\s+(\d+)>", candidate)
    labels = [int(value) for value in raw_labels]
    if any(raw != str(int(raw)) for raw in raw_labels):
        issues.append("Picture labels must use canonical non-zero-padded integers.")
    if any(label < 1 for label in labels):
        issues.append("Picture numbering must start at 1.")
    if labels and max(labels) > allowed_pictures:
        issues.append("Prompt references a Picture label not provided by the keyframe mode.")
    if allowed_pictures == 0 and labels:
        issues.append("T2VA must not use Picture reference labels.")
    if re.search(r"<(?:Video|Audio|Subject)\s+\d+>", candidate):
        issues.append("Base modes must not introduce Ref2VA Subject/Video/Audio labels.")

    return ValidationResult(candidate, tuple(dict.fromkeys(issues)))


def _validate_sequential_ids(text: str, label: str, pattern: str) -> list[str]:
    raw_values = re.findall(pattern, text)
    values = sorted({int(value) for value in raw_values})
    issues: list[str] = []
    if any(raw != str(int(raw)) for raw in raw_values):
        issues.append(f"{label} labels must use canonical non-zero-padded integers.")
    if values and values != list(range(1, max(values) + 1)):
        issues.append(f"{label} numbering must start at 1 and contain no gaps.")
    return issues


def _speaker_ids(text: str) -> list[int]:
    values: list[int] = []
    for group in re.findall(r"\((S\d+(?:\s*,\s*S\d+)*)\)", text):
        values.extend(int(value) for value in re.findall(r"S(\d+)", group))
    return values


def _canonicalize_sanitized_ref_summary(
    candidate: str,
    manifest: ReferenceManifest,
) -> str:
    """Normalize summary metadata after the final model-token cleanup.

    Decoder residue can be removed by ``sanitize_generated_text`` only after
    the first structural pass.  Rebuild just this deterministic field on the
    sanitized string so a newly exposed task-type echo cannot consume or fail
    a repair pass.
    """

    starts = _header_matches(candidate, "summary:")
    ends = _header_matches(candidate, "retention_analysis:")
    if len(starts) != 1 or len(ends) != 1 or starts[0].start() >= ends[0].start():
        return candidate
    body_start = starts[0].end()
    body_end = ends[0].start()
    original = candidate[body_start:body_end]
    value = original.strip()
    match = re.match(r"^\[([^\]\n]+)\](?=$|\s)", value)
    allowed = {
        "keyframe completion",
        "reference generation",
        "video editing",
        "video continuation",
        "audio reuse",
        "audio reference",
    }
    if match is None:
        return candidate
    signature_items = [part.strip().lower() for part in match.group(1).split("+")]
    if not signature_items or not set(signature_items).issubset(allowed):
        return candidate
    prose = value[match.end() :].lstrip()
    normalized_prose = _strip_leading_ref_task_echo(prose, allowed)
    if normalized_prose == prose:
        return candidate
    normalized = value[: match.end()] + (
        " " + normalized_prose if normalized_prose else ""
    )
    return candidate[:body_start] + "\n" + normalized + "\n\n" + candidate[body_end:]


def validate_ref_prompt(
    text: str,
    length: int,
    manifest: ReferenceManifest,
) -> ValidationResult:
    candidate = sanitize_generated_text(text, "Ref2VA")
    candidate = _canonicalize_sanitized_ref_summary(candidate, manifest)
    duration = effective_duration(length)
    issues = _validate_common(candidate, REF_FIELDS)
    if not candidate.startswith("subject_definitions:"):
        issues.append("Ref2VA must begin with subject_definitions:.")
    detail_body = _field_body(
        candidate,
        "detailed_description:",
        "overall_soundscape:",
    )
    issues.extend(
        _validate_shot_timeline(
            detail_body,
            duration,
            shot_one_must_start_body=False,
        )
    )

    for label in manifest.labels():
        if label not in candidate:
            issues.append(f"Connected reference label {label} is never used.")

    allowed = {
        "Picture": {int(re.search(r"\d+", label).group()) for label in manifest.labels("image")},
        "Video": {int(re.search(r"\d+", label).group()) for label in manifest.labels("video")},
        "Audio": {int(re.search(r"\d+", label).group()) for label in manifest.labels("audio")},
    }
    for kind, raw_value in re.findall(r"<(Picture|Video|Audio)\s+(\d+)>", candidate):
        value = int(raw_value)
        if raw_value != str(value):
            issues.append(
                f"<{kind} {raw_value}> is malformed; reference labels cannot be zero-padded."
            )
        if value not in allowed[kind]:
            issues.append(f"Prompt uses nonexistent <{kind} {raw_value}>.")

    issues.extend(
        _validate_sequential_ids(candidate, "Subject", r"<Subject\s+(\d+)>")
    )
    speaker_values = sorted(set(_speaker_ids(candidate)))
    if speaker_values and speaker_values != list(range(1, max(speaker_values) + 1)):
        issues.append("Speaker numbering must start at 1 and contain no gaps.")

    subject_body = _field_body(candidate, "subject_definitions:", "summary:")
    subject_ids = sorted(
        {int(value) for value in re.findall(r"<Subject\s+(\d+)>", candidate)}
    )
    subject_definition_lines: dict[int, str] = {}
    for subject_id in subject_ids:
        definitions = re.findall(
            rf"(?m)^<Subject {subject_id}>(?=$|[ \t]).*$",
            subject_body,
        )
        if len(definitions) != 1:
            issues.append(
                f"<Subject {subject_id}> must have exactly one line in subject_definitions."
            )
        else:
            subject_definition_lines[subject_id] = definitions[0]

    for audio_label in manifest.labels("audio"):
        definitions = re.findall(
            rf"(?m)^{re.escape(audio_label)}(?=$|[ \t]).*$",
            subject_body,
        )
        if len(definitions) != 1:
            issues.append(
                f"{audio_label} must have exactly one source-role line in subject_definitions."
            )

    summary_body = _field_body(candidate, "summary:", "retention_analysis:")
    summary_match = re.match(r"^\[([^\]\n]+)\](?=$|\s)", summary_body)
    allowed_types = {
        "keyframe completion",
        "reference generation",
        "video editing",
        "video continuation",
        "audio reuse",
        "audio reference",
    }
    task_types: set[str] = set()
    if not summary_match:
        issues.append("summary must begin with a bracketed task-type signature.")
    else:
        signature = summary_match.group(1)
        task_type_list = [part.strip() for part in re.split(r"\s*\+\s*", signature)]
        task_types = set(task_type_list)
        if " + ".join(task_type_list) != signature:
            issues.append("summary task types must use the exact ` + ` separator.")
        if len(task_types) != len(task_type_list):
            issues.append("summary task types must not be duplicated.")
        if not task_types or not task_types.issubset(allowed_types):
            issues.append("summary contains an unsupported task type.")
        if task_types & {"video editing", "video continuation"} and not manifest.videos:
            issues.append("video editing/continuation requires a connected Video reference.")
        if "keyframe completion" in task_types and not manifest.pictures:
            issues.append("keyframe completion requires a connected Picture reference.")
        if task_types & {"audio reuse", "audio reference"} and not manifest.audios:
            issues.append("audio reuse/reference requires a connected Audio reference.")
        summary_prose = summary_body[summary_match.end() :].lstrip()
        if not re.search(r"[A-Za-z0-9<]", summary_prose):
            issues.append("summary needs substantive target-event prose after its signature.")
        elif re.match(
            r"(?i)^(?:keyframe completion|reference generation|video editing|"
            r"video continuation|audio reuse|audio reference)\s*[.:;,-]",
            summary_prose,
        ):
            issues.append("summary must not repeat a task type in its prose.")
        if "video editing" in task_types and manifest.videos:
            video_labels = "|".join(
                re.escape(label) for label in manifest.labels("video")
            )
            if not re.match(
                rf"The target video is an edited version of (?:{video_labels})\.",
                summary_prose,
            ):
                issues.append(
                    "A video-editing summary must begin with the prescribed edited-version sentence."
                )

    retention = _field_body(
        candidate,
        "retention_analysis:",
        "detailed_description:",
    )
    if _speaker_ids(retention):
        issues.append("retention_analysis must not assign Speaker identifiers.")
    visible_markers = {
        "fully_preserved",
        "partially_preserved",
        "attribute_transfer",
        "weak_reference",
    }
    audio_markers = {
        "fully_copy",
        "partially_copy",
        "reference",
        "weak_reference",
    }
    retention_lines = [line.strip() for line in retention.splitlines() if line.strip()]
    underscored_marker_pattern = re.compile(
        r"\b(?:fully_preserved|partially_preserved|attribute_transfer|"
        r"weak_reference|fully_copy|partially_copy)\b"
    )
    retention_line_pattern = re.compile(
        r"^<(?:Subject|Picture|Video|Audio) [1-9]\d*>"
        r"(?:\s+\([^\n)]*\))?:\s*"
        r"(fully_preserved|partially_preserved|attribute_transfer|"
        r"weak_reference|fully_copy|partially_copy|reference)\s*-\s*\S"
    )

    def marker_is_valid(line: str, allowed_markers: set[str]) -> bool:
        parsed = retention_line_pattern.match(line)
        if parsed is None:
            return False
        # Underscored relationship values must appear only once, in the marker
        # slot. The plain word "reference" may occur naturally in prose, so it
        # contributes one marker only when it is the parsed leading value.
        marker_count = len(underscored_marker_pattern.findall(line))
        if parsed.group(1) == "reference":
            marker_count += 1
        return (
            marker_count == 1
            and parsed.group(1) in allowed_markers
        )

    def relationship_lines_for(label: str) -> list[str]:
        return [
            line
            for line in retention_lines
            if re.match(rf"^{re.escape(label)}(?:\s+\([^\n)]*\))?:", line)
        ]

    subject_retention_lines: dict[int, str] = {}
    for subject_id in subject_ids:
        subject_label = f"<Subject {subject_id}>"
        relationship_lines = relationship_lines_for(subject_label)
        if len(relationship_lines) != 1:
            issues.append(
                f"retention_analysis must contain exactly one line beginning with {subject_label}."
            )
        elif not marker_is_valid(relationship_lines[0], visible_markers):
            issues.append(
                f"{subject_label} needs exactly one visible retention marker."
            )
        else:
            subject_retention_lines[subject_id] = relationship_lines[0]

    for asset in manifest.presentation_order:
        relationship_lines = relationship_lines_for(asset.label)
        allowed_markers = audio_markers if asset.kind == "audio" else visible_markers
        if len(relationship_lines) == 1:
            if not marker_is_valid(relationship_lines[0], allowed_markers):
                issues.append(
                    f"{asset.label} needs exactly one retention marker compatible with {asset.kind}."
                )
            continue
        if len(relationship_lines) > 1:
            issues.append(
                f"retention_analysis must contain exactly one line beginning with {asset.label}."
            )
            continue
        covered_by_subject = asset.kind != "audio" and any(
            asset.label in definition and subject_id in subject_retention_lines
            for subject_id, definition in subject_definition_lines.items()
        )
        if not covered_by_subject:
            issues.append(
                "retention_analysis needs a direct "
                f"{asset.label} relationship or a retained Subject explicitly bound to it."
            )

    if task_types:
        audio_relationships = []
        for label in manifest.labels("audio"):
            lines = relationship_lines_for(label)
            parsed = retention_line_pattern.match(lines[0]) if len(lines) == 1 else None
            if parsed is not None:
                audio_relationships.append(parsed.group(1))
        has_reuse = any(marker in {"fully_copy", "partially_copy"} for marker in audio_relationships)
        has_reference = any(marker in {"reference", "weak_reference"} for marker in audio_relationships)
        if has_reuse and "audio reuse" not in task_types:
            issues.append("Audio copy markers require `audio reuse` in the summary signature.")
        if has_reference and "audio reference" not in task_types:
            issues.append("Audio reference markers require `audio reference` in the summary signature.")
        if "audio reuse" in task_types and not has_reuse:
            issues.append("`audio reuse` requires a fully_copy or partially_copy Audio relationship.")
        if "audio reference" in task_types and not has_reference:
            issues.append("`audio reference` requires a reference or weak_reference Audio relationship.")
        if "audio reuse" not in task_types:
            music_headers = _header_matches(candidate, REF_FIELDS[5])
            non_retention = "\n".join(
                _field_body(candidate, REF_FIELDS[index], REF_FIELDS[index + 1])
                for index in (0, 1, 3, 4)
            )
            if music_headers:
                non_retention += "\n" + candidate[music_headers[0].end() :]
            if re.search(
                r"(?i)\b(?:fully|partially)\s+copied\b|"
                r"\b(?:copied|reused)\s+as-is\b|"
                r"\b(?:copy|reuse)\s+(?:the\s+)?(?:source\s+)?(?:audio|soundtrack|signal)\b",
                non_retention,
            ):
                issues.append(
                    "Audio-copy prose requires `audio reuse` and explicit copy intent."
                )

    return ValidationResult(candidate, tuple(dict.fromkeys(issues)))
