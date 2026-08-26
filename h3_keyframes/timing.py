"""Parse RP H3 shot timestamps into native 24 fps keyframe positions."""

from __future__ import annotations

import re


FPS = 24

_SHOT_RE = re.compile(
    r"\[\s*Shot\s+(?P<shot>\d+)\s*\]"
    r"(?:\s+At\s+(?P<minutes>\d{2}):(?P<seconds>\d{2})\."
    r"(?P<milliseconds>\d{3}))?",
    re.IGNORECASE,
)
_DETAIL_START_RE = re.compile(r"(?im)^\s*detailed_description\s*:\s*")
_DETAIL_END_RE = re.compile(r"(?im)^\s*overall_soundscape\s*:")


def _timeline_section(prompt: str) -> str:
    """Prefer the narrative timeline and ignore earlier Shot cross-references."""
    start = _DETAIL_START_RE.search(prompt)
    if start is None:
        return prompt
    end = _DETAIL_END_RE.search(prompt, start.end())
    return prompt[start.end() : end.start() if end else None]


def _frame_from_timestamp(minutes: int, seconds: int, milliseconds: int) -> int:
    if seconds >= 60:
        raise ValueError(
            "RP H3-Keyframes: timestamp seconds must be between 00 and 59"
        )
    total_milliseconds = ((minutes * 60 + seconds) * 1000) + milliseconds
    # Exact integer round-half-up avoids Python's banker rounding at half frames.
    return (total_milliseconds * FPS + 500) // 1000


def shot_frame_positions(
    prompt: str,
    shot_numbers: list[int],
    frame_count: int,
) -> list[int]:
    """Return zero-based frame positions for the requested shot numbers.

    Shot 1 starts at frame zero. Every later shot must use the canonical
    ``[Shot N] At MM:SS.mmm`` form emitted by the RP H3 prompt writers.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("RP H3-Keyframes: prompt is empty")
    if frame_count < 1:
        raise ValueError("RP H3-Keyframes: target latent contains no video frames")

    matches: dict[int, tuple[int, int, int] | None] = {}
    for match in _SHOT_RE.finditer(_timeline_section(prompt)):
        shot = int(match.group("shot"))
        if shot in matches:
            raise ValueError(
                f"RP H3-Keyframes: [Shot {shot}] appears more than once in the timeline"
            )
        if match.group("minutes") is None:
            matches[shot] = None
        else:
            matches[shot] = (
                int(match.group("minutes")),
                int(match.group("seconds")),
                int(match.group("milliseconds")),
            )

    positions: list[int] = []
    for shot in shot_numbers:
        if shot not in matches:
            raise ValueError(
                f"RP H3-Keyframes: prompt has no [Shot {shot}] for keyframe_image_{shot}"
            )

        timestamp = matches[shot]
        if shot == 1:
            if timestamp is None:
                frame = 0
            else:
                frame = _frame_from_timestamp(*timestamp)
                if frame != 0:
                    raise ValueError(
                        "RP H3-Keyframes: [Shot 1] must start at 00:00.000"
                    )
        else:
            if timestamp is None:
                raise ValueError(
                    f"RP H3-Keyframes: [Shot {shot}] must be followed by "
                    "At MM:SS.mmm"
                )
            frame = _frame_from_timestamp(*timestamp)

        if frame >= frame_count:
            raise ValueError(
                f"RP H3-Keyframes: [Shot {shot}] resolves to frame {frame} "
                f"outside the target range 0..{frame_count - 1}"
            )
        if positions and frame <= positions[-1]:
            raise ValueError(
                f"RP H3-Keyframes: [Shot {shot}] must start after the previous "
                "connected keyframe shot"
            )
        positions.append(frame)

    return positions
