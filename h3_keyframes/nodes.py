# Copyright (C) 2026 NikoDemon80 and contributors
# Modified for ComfyUI-RPNodes in 2026. Licensed under GPL-3.0-or-later.

"""Prompt-timed still-image keyframes for MiniMax H3."""

from __future__ import annotations

import logging

import comfy.utils
import node_helpers
from comfy_api.latest import io

from .compat import ensure_h3_keyframe_support
from .timing import shot_frame_positions


_LOG = logging.getLogger("rp_h3_keyframes")
_FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
_MAX_KEYFRAMES = 32


def _pixel_frames(latent_steps: int) -> int:
    return sum(_FRAME_PER_TOKEN[index % 5] for index in range(latent_steps))


def _video_from_latent(latent):
    samples = latent["samples"]
    if hasattr(samples, "unbind"):
        streams = list(samples.unbind())
    elif isinstance(samples, (tuple, list)):
        streams = list(samples)
    else:
        raise ValueError(
            "RP H3-Keyframes: expected a MiniMax H3 AV latent with video and "
            f"audio streams, got {type(samples)!r}"
        )
    if not streams:
        raise ValueError("RP H3-Keyframes: the H3 AV latent has no streams")
    video = streams[0]
    if video.ndim == 4:
        video = video.unsqueeze(0)
    if video.ndim != 5:
        raise ValueError(
            "RP H3-Keyframes: expected video latent [B,C,T,H,W], got "
            f"{tuple(video.shape)}"
        )
    return video


def _resize(image, width: int, height: int, crop: str):
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, "lanczos", crop)
    return samples.movedim(1, -1)


def _connected_keyframes(keyframes) -> list[tuple[int, object]]:
    connected: list[tuple[int, object]] = []
    for name, image in (keyframes or {}).items():
        if image is None:
            continue
        try:
            slot = int(name.rsplit("_", 1)[1])
        except (IndexError, ValueError) as exc:
            raise ValueError(
                f"RP H3-Keyframes: invalid autogrow input name {name!r}"
            ) from exc
        connected.append((slot, image))
    connected.sort(key=lambda item: item[0])

    slots = [slot for slot, _ in connected]
    expected = list(range(1, len(connected) + 1))
    if slots != expected:
        raise ValueError(
            "RP H3-Keyframes: image inputs must be connected consecutively "
            "from keyframe_image_1"
        )
    return connected


class RPH3Keyframes(io.ComfyNode):
    """Attach each image at the timestamp of the same-numbered prompt shot."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="RPH3Keyframes",
            display_name="RP H3-Keyframes",
            category="RP/MiniMax H3",
            description=(
                "Pins each connected still to the start time of the matching "
                "[Shot N] in an RP H3 prompt. Shot 1 starts at frame zero; "
                "later shots use At MM:SS.mmm at H3's native 24 fps."
            ),
            search_aliases=["H3 keyframes", "MiniMax keyframes", "timed keyframes"],
            inputs=[
                io.Conditioning.Input(
                    "conditioning",
                    tooltip=(
                        "H3 conditioning. The node replaces its complete "
                        "minimax_keyframes list with the connected images."
                    ),
                ),
                io.Vae.Input(
                    "vae",
                    tooltip="MiniMax H3 video VAE used to encode each still.",
                ),
                io.Latent.Input(
                    "latent",
                    tooltip=(
                        "Target MiniMax H3 AV latent; defines output resolution "
                        "and exact frame count."
                    ),
                ),
                io.String.Input(
                    "prompt",
                    multiline=True,
                    dynamic_prompts=False,
                    tooltip=(
                        "Connect the same RP H3 prompt used by the native H3 "
                        "conditioning node. keyframe_image_N uses [Shot N]."
                    ),
                ),
                io.Combo.Input(
                    "crop",
                    options=["disabled", "center"],
                    default="disabled",
                ),
                io.Autogrow.Input(
                    "keyframes",
                    template=io.Autogrow.TemplateNames(
                        input=io.Image.Input(
                            "keyframe_image",
                            tooltip=(
                                "Still image for the same-numbered [Shot N]. "
                                "Connecting it reveals the next image input."
                            ),
                        ),
                        names=[
                            f"keyframe_image_{index}"
                            for index in range(1, _MAX_KEYFRAMES + 1)
                        ],
                        min=1,
                    ),
                    tooltip=f"One to {_MAX_KEYFRAMES} sequential shot keyframes.",
                ),
            ],
            outputs=[io.Conditioning.Output(display_name="conditioning")],
        )

    @classmethod
    def execute(
        cls,
        conditioning,
        vae,
        latent,
        prompt,
        crop="disabled",
        keyframes: io.Autogrow.Type = None,
    ) -> io.NodeOutput:
        ensure_h3_keyframe_support(conditioning)
        connected = _connected_keyframes(keyframes)
        if not connected:
            raise ValueError("RP H3-Keyframes: connect at least keyframe_image_1")

        video = _video_from_latent(latent)
        width = int(video.shape[4]) * 16
        height = int(video.shape[3]) * 16
        frame_count = _pixel_frames(int(video.shape[2]))
        slots = [slot for slot, _ in connected]
        positions = shot_frame_positions(prompt, slots, frame_count)

        encoded_keyframes = []
        for (slot, image), pixel_index in zip(connected, positions):
            if getattr(image, "ndim", 0) != 4:
                raise ValueError(
                    f"RP H3-Keyframes: keyframe_image_{slot} expected IMAGE "
                    "[B,H,W,C]"
                )
            if int(image.shape[0]) != 1:
                raise ValueError(
                    f"RP H3-Keyframes: keyframe_image_{slot} must receive exactly "
                    f"one image, not a batch of {int(image.shape[0])}"
                )

            encoded = vae.encode(_resize(image, width, height, crop))
            if getattr(encoded, "ndim", 0) != 5 or int(encoded.shape[2]) != 1:
                raise ValueError(
                    f"RP H3-Keyframes: keyframe_image_{slot} encoded to "
                    f"{tuple(getattr(encoded, 'shape', ()))}, expected one H3 "
                    "still latent [B,C,1,H,W]"
                )
            encoded_keyframes.append(
                {"resolved_frame_index": int(pixel_index), "latent": encoded}
            )

        output = node_helpers.conditioning_set_values(
            conditioning, {"minimax_keyframes": encoded_keyframes}
        )
        _LOG.info(
            "RP H3-Keyframes pinned %d shot images at zero-based frames %s "
            "in a %d-frame %dx%d target",
            len(encoded_keyframes),
            positions,
            frame_count,
            width,
            height,
        )
        return io.NodeOutput(output)
