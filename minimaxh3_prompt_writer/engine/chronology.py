"""Small deterministic helpers for ordered frame-prompt chronology."""

from __future__ import annotations

import re


_SEPARATOR_LINE = re.compile(r"^[\s=\-_]{3,}$")
_SHOT_MARKER = re.compile(
    r"(?=\[Shot\s+[1-9]\d*\](?:\s+At\s+\d{2}:\d{2}\.\d{3},?)?)",
    flags=re.IGNORECASE,
)
_QUOTED_TEXT = re.compile(
    r'"([^"\n]+)"|“([^”\n]+)”|«([^»\n]+)»|„([^“\n]+)“'
)
_SPEECH_CUE = re.compile(
    r"\b(?:say|says|said|saying|speak|speaks|ask|asks|reply|replies|"
    r"whisper|whispers|shout|shouts|tell|tells|"
    r"dice|dicono|dicendo|parla|parlano|chiede|risponde|sussurra|grida|"
    r"habla|hablan|pregunta|responde|susurra|grita|diciendo|"
    r"dit|disent|parle|demande|répond|murmure|crie|"
    r"sagt|sagen|spricht|fragt|antwortet|flüstert|ruft|"
    r"diz|dizem|dizendo|fala|pergunta|responde|sussurra|grita)\b",
    flags=re.IGNORECASE,
)
_VISIBLE_TEXT_CUE = re.compile(
    r"(?:c['’]\s*è\s+scritto|con\s+scritto|mostra\s+(?:la\s+)?scritta|"
    r"(?:the\s+)?(?:sign|screen|label|card)\s+(?:reads|says|displays)|"
    r"written\s+on\s+(?:the\s+)?(?:sign|screen|label|card))\s*[:\-]?\s*",
    flags=re.IGNORECASE,
)
_VISIBLE_TEXT_ACTION_BOUNDARY = re.compile(
    r"\s*[,;]?\s+"
    r"(?:e(?:d)?|e\s+poi|poi|quindi|and|and\s+then|then)\s+"
    r"(?=(?:(?:il\s+personaggio|lui|lei|he|she)\s+)?"
    r"(?:dice|dicono|dicendo|parla|chiede|risponde|sussurra|grida|"
    r"says?|speaks?|asks?|repl(?:y|ies)|whispers?|shouts?)\b)",
    flags=re.IGNORECASE,
)
_VISIBLE_TEXT_QUOTES = {'"': '"', "“": "”", "«": "»", "„": "“"}


def _sentence_events(text: str) -> list[str]:
    """Split a one-line request without treating punctuation in quotes as a cut."""

    events: list[str] = []
    start = 0
    quote: str | None = None
    closing = {'"': '"', "“": "”", "«": "»", "„": "“"}
    for index, char in enumerate(text):
        if quote is None and char in closing:
            quote = closing[char]
            continue
        if quote is not None:
            if char == quote:
                quote = None
            continue
        if char not in ".!?":
            continue
        following = text[index + 1 : index + 2]
        if following and not following.isspace():
            continue
        event = text[start : index + 1].strip()
        if event:
            events.append(event)
        start = index + 1
    tail = text[start:].strip()
    if tail:
        events.append(tail)
    return events or ([text.strip()] if text.strip() else [])


def extract_protected_strings(text: str) -> list[str]:
    """Return exact user-authored strings while retaining their source event."""

    values: list[str] = []
    for match in _QUOTED_TEXT.finditer(text):
        value = next(group for group in match.groups() if group is not None).strip()
        if value:
            values.append(value)
    return values


def _visible_text_claims(text: str) -> list[tuple[str, bool]]:
    """Return visible strings and whether their right boundary was ambiguous."""

    claims: list[tuple[str, bool]] = []
    for cue in _VISIBLE_TEXT_CUE.finditer(text):
        tail = text[cue.end() :]
        if not tail:
            continue

        opening = tail[0]
        if opening in _VISIBLE_TEXT_QUOTES:
            closing = _VISIBLE_TEXT_QUOTES[opening]
            closing_index = tail.find(closing, 1)
            if closing_index >= 0:
                value = tail[1:closing_index].strip()
                if value:
                    claims.append((value, False))
                continue

        sentence_end = re.search(r"[\n.!?]", tail)
        fragment = tail[: sentence_end.start()] if sentence_end else tail
        action_boundary = _VISIBLE_TEXT_ACTION_BOUNDARY.search(fragment)
        ambiguous = action_boundary is not None
        if action_boundary is not None:
            fragment = fragment[: action_boundary.start()]
        value = fragment.strip().strip('"“”«»„').strip()
        if value:
            claims.append((value, ambiguous))
    return claims


def extract_visible_text_strings(text: str) -> list[str]:
    """Return literal display text following common English/Italian cues."""

    return [value for value, _ in _visible_text_claims(text)]


def ordered_event_ledger(raw_prompt: str) -> list[dict[str, object]]:
    """Turn a raw request into simple chronological beats for a small LLM.

    Line breaks are authoritative event boundaries because prompt writers often
    use one action per line.  A single-line prompt falls back to explicit Shot
    markers and then conservative sentence splitting.  No frame-to-event
    one-to-one mapping is inferred here.
    """

    normalized = str(raw_prompt or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [
        line.strip()
        for line in normalized.splitlines()
        if line.strip() and not _SEPARATOR_LINE.fullmatch(line.strip())
    ]
    if len(lines) > 1:
        events = lines
    else:
        compact = lines[0] if lines else ""
        shot_parts = [part.strip() for part in _SHOT_MARKER.split(compact) if part.strip()]
        events = shot_parts if len(shot_parts) > 1 else _sentence_events(compact)

    ledger: list[dict[str, object]] = []
    for index, event in enumerate(events, start=1):
        quoted = extract_protected_strings(event)
        visible_claims = _visible_text_claims(event)
        visible = [value for value, _ in visible_claims]
        ambiguous_visible = [
            value for value, ambiguous in visible_claims if ambiguous
        ]
        protected = list(dict.fromkeys(quoted + visible))
        ledger.append(
            {
                "event_index": index,
                "source_text": event,
                "protected_verbatim_strings": protected,
                "visible_verbatim_strings": visible,
                "ambiguous_visible_verbatim_strings": ambiguous_visible,
                "spoken_verbatim_strings": (
                    quoted if quoted and _SPEECH_CUE.search(event) else []
                ),
            }
        )
    return ledger


def _normalized_verbatim(text: str) -> str:
    value = re.sub(r"^\s*\[[^\]\n]+\]\s*", "", str(text or "").strip())
    value = value.replace("’", "'").replace("‘", "'")
    return re.sub(r"\s+", " ", value).strip().casefold()


def _dialogue_sentence_spans(text: str) -> list[tuple[int, int]]:
    masked = re.sub(
        r"<d>.*?</d>",
        lambda match: " " * len(match.group(0)),
        text,
        flags=re.DOTALL | re.IGNORECASE,
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


def validate_dialogue_event_ownership(text: str, raw_prompt: str) -> list[str]:
    """Catch dialogue from separate source events fused into one target action."""

    ledger = ordered_event_ledger(raw_prompt)
    source_event_by_string: dict[str, int] = {}
    required_spoken: list[tuple[int, str, str]] = []
    for event in ledger:
        event_index = int(event["event_index"])
        for value in event["protected_verbatim_strings"]:
            source_event_by_string.setdefault(_normalized_verbatim(str(value)), event_index)
        for value in event["spoken_verbatim_strings"]:
            required_spoken.append(
                (event_index, str(value), _normalized_verbatim(str(value)))
            )

    dialogue_matches = list(
        re.finditer(r"<d>(.*?)</d>", text, flags=re.DOTALL | re.IGNORECASE)
    )
    issues: list[str] = []
    opening_tags = len(re.findall(r"<d>", text, flags=re.IGNORECASE))
    closing_tags = len(re.findall(r"</d>", text, flags=re.IGNORECASE))
    if opening_tags != closing_tags or any(
        re.search(r"</?d>", match.group(1), flags=re.IGNORECASE)
        for match in dialogue_matches
    ):
        issues.append(
            "Dialogue blocks must be balanced and cannot contain nested <d> tags."
        )

    required_values = {normalized for _, _, normalized in required_spoken}
    unknown_dialogue = [
        match.group(1).strip()
        for match in dialogue_matches
        if _normalized_verbatim(match.group(1)) not in required_values
    ]
    if required_spoken and unknown_dialogue:
        issues.append(
            "Every <d> dialogue block must reproduce a spoken line from the "
            "source request exactly; remove invented or malformed dialogue."
        )
    owned: list[tuple[int, int, str]] = []
    spans = _dialogue_sentence_spans(text)
    for dialogue in dialogue_matches:
        normalized = _normalized_verbatim(dialogue.group(1))
        source_event = source_event_by_string.get(normalized)
        if source_event is None:
            continue
        sentence_index = next(
            (
                index
                for index, (start, end) in enumerate(spans)
                if start <= dialogue.start() < end
            ),
            -1,
        )
        owned.append((source_event, sentence_index, dialogue.group(1).strip()))

    output_dialogue_values = [
        _normalized_verbatim(match.group(1)) for match in dialogue_matches
    ]
    for source_event, exact_text, normalized in required_spoken:
        count = output_dialogue_values.count(normalized)
        if count != 1:
            issues.append(
                f'Source event {source_event} spoken line `{exact_text}` must appear '
                "exactly once inside its own <d>[Language] ...</d> block."
            )
    output_event_order = [source_event for source_event, _, _ in owned]
    if output_event_order != sorted(output_event_order):
        issues.append(
            "Spoken lines must retain the chronological order of their source events."
        )

    by_sentence: dict[int, list[tuple[int, str]]] = {}
    for source_event, sentence_index, dialogue in owned:
        by_sentence.setdefault(sentence_index, []).append((source_event, dialogue))
    for values in by_sentence.values():
        source_events = {event for event, _ in values}
        if len(source_events) > 1:
            descriptions = ", ".join(
                f'event {event} (`{dialogue}`)' for event, dialogue in values
            )
            issues.append(
                "Dialogue event ownership was merged into one action sentence: "
                f"{descriptions}. Place each line inside its own source event."
            )
    return issues
