"""Mode inference and reference-label mapping matching ComfyUI's H3 nodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .constants import FPS


def align_frame_count(length: int) -> int:
    """Snap a frame count upward to MiniMax H3's native ``17k + 5`` grid."""

    value = max(5, int(length))
    return value + ((5 - value) % 17)


def seconds_to_aligned_frame_count(seconds: float, fps: int = FPS) -> int:
    """Convert requested seconds with the official workflow's grid formula."""

    requested_frames = max(5, round(float(seconds) * fps))
    return requested_frames + ((5 - (requested_frames % 17)) % 17)


def validate_whole_duration_seconds(value: float) -> float:
    """Accept only the whole-second range exposed by the H3 workflow widget."""

    seconds = float(value)
    if not seconds.is_integer():
        raise ValueError(
            "duration_seconds must be a whole number of seconds (1.0, 2.0, 3.0, ...)."
        )
    if not 1.0 <= seconds <= 149.0:
        raise ValueError("duration_seconds must be between 1.0 and 149.0 seconds.")
    return seconds


def effective_duration(length: int, fps: int = FPS) -> float:
    return align_frame_count(length) / float(fps)


def duration_2dp(length: int, fps: int = FPS) -> str:
    return f"{effective_duration(length, fps):.2f}"


def determine_base_mode(first_frame: Any = None, last_frame: Any = None) -> str:
    """Infer the native H3 base mode from connected keyframe sockets."""

    if first_frame is not None and last_frame is not None:
        return "FL2VA"
    if first_frame is not None:
        return "I2VA"
    if last_frame is not None:
        return "L2VA"
    return "T2VA"


@dataclass(frozen=True)
class AssetRef:
    socket: str
    label: str
    kind: str
    role: str
    paired_label: str | None = None

    def inventory_line(self) -> str:
        suffix = f"; paired with {self.paired_label}" if self.paired_label else ""
        return f"{self.label} <- {self.socket} ({self.role}{suffix})"


@dataclass(frozen=True)
class ReferenceManifest:
    """The exact labels the native ``MiniMaxH3ReferenceToVideo`` will assign.

    Labels are compacted by connected-item order; socket suffixes never become
    label numbers. Audio, video, and picture ordinals are independent.
    """

    pictures: tuple[AssetRef, ...]
    videos: tuple[AssetRef, ...]
    audios: tuple[AssetRef, ...]
    presentation_order: tuple[AssetRef, ...]

    @classmethod
    def from_inputs(
        cls,
        *,
        ref_image_0: Any = None,
        ref_image_1: Any = None,
        ref_image_2: Any = None,
        ref_video_0: Any = None,
        ref_video_1: Any = None,
        ref_video_audio_0: Any = None,
        ref_audio_0: Any = None,
    ) -> "ReferenceManifest":
        image_values = (
            ("ref_image_0", ref_image_0),
            ("ref_image_1", ref_image_1),
            ("ref_image_2", ref_image_2),
        )
        video_values = (
            ("ref_video_0", ref_video_0),
            ("ref_video_1", ref_video_1),
        )

        if ref_video_audio_0 is not None and ref_video_0 is None:
            raise ValueError(
                "ref_video_audio_0 requires ref_video_0. The native H3 node "
                "ignores an orphan video soundtrack, which would desynchronize labels."
            )

        pictures: list[AssetRef] = []
        for socket, value in image_values:
            if value is None:
                continue
            pictures.append(
                AssetRef(
                    socket=socket,
                    label=f"<Picture {len(pictures) + 1}>",
                    kind="image",
                    role="reference image",
                )
            )

        videos: list[AssetRef] = []
        video_by_socket: dict[str, AssetRef] = {}
        for socket, value in video_values:
            if value is None:
                continue
            asset = AssetRef(
                socket=socket,
                label=f"<Video {len(videos) + 1}>",
                kind="video",
                role="reference video at 24 fps",
            )
            videos.append(asset)
            video_by_socket[socket] = asset

        audios: list[AssetRef] = []
        presentation: list[AssetRef] = list(pictures)

        # This ordering mirrors nodes_minimax_h3.py: for each connected video,
        # its paired soundtrack is presented immediately before the video.
        for socket, value in video_values:
            if value is None:
                continue
            video_asset = video_by_socket[socket]
            if socket == "ref_video_0" and ref_video_audio_0 is not None:
                audio_asset = AssetRef(
                    socket="ref_video_audio_0",
                    label=f"<Audio {len(audios) + 1}>",
                    kind="audio",
                    role="enabled soundtrack of ref_video_0",
                    paired_label=video_asset.label,
                )
                audios.append(audio_asset)
                presentation.append(audio_asset)
            presentation.append(video_asset)

        if ref_audio_0 is not None:
            standalone = AssetRef(
                socket="ref_audio_0",
                label=f"<Audio {len(audios) + 1}>",
                kind="audio",
                role="standalone reference audio",
            )
            audios.append(standalone)
            presentation.append(standalone)

        manifest = cls(
            pictures=tuple(pictures),
            videos=tuple(videos),
            audios=tuple(audios),
            presentation_order=tuple(presentation),
        )
        if not manifest.presentation_order:
            raise ValueError(
                "RP H3-REF2V Prompt Writer needs at least one connected reference "
                "image, video, video soundtrack, or standalone audio."
            )
        return manifest

    def asset_for_socket(self, socket: str) -> AssetRef | None:
        for asset in self.presentation_order:
            if asset.socket == socket:
                return asset
        return None

    def labels(self, kind: str | None = None) -> tuple[str, ...]:
        assets: Iterable[AssetRef]
        if kind == "image":
            assets = self.pictures
        elif kind == "video":
            assets = self.videos
        elif kind == "audio":
            assets = self.audios
        else:
            assets = self.presentation_order
        return tuple(asset.label for asset in assets)

    def inventory_text(self) -> str:
        return "\n".join(f"- {asset.inventory_line()}" for asset in self.presentation_order)

    def to_dict(self) -> dict[str, list[dict[str, str | None]]]:
        def serialize(items: tuple[AssetRef, ...]) -> list[dict[str, str | None]]:
            return [
                {
                    "socket": item.socket,
                    "label": item.label,
                    "kind": item.kind,
                    "role": item.role,
                    "paired_label": item.paired_label,
                }
                for item in items
            ]

        return {
            "pictures": serialize(self.pictures),
            "videos": serialize(self.videos),
            "audios": serialize(self.audios),
            "presentation_order": serialize(self.presentation_order),
        }
