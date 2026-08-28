"""ComfyUI V3 node definitions for RP H3 prompt writing."""

from __future__ import annotations

from typing import Any

import comfy.model_management
from comfy_api.latest import io

from .engine.analyzers import (
    analyze_reference_media,
)
from .engine.composer import (
    compose_ref_prompt,
    compose_t2v_prompt,
)
from .engine.constants import SKILL_CHOICES
from .engine.gemma import GemmaRunner, SamplingConfig
from .engine.manifests import (
    ReferenceManifest,
    seconds_to_aligned_frame_count,
    validate_whole_duration_seconds,
)


MAX_SEED = 0xFFFFFFFFFFFFFFFF


def _check_interrupted() -> None:
    comfy.model_management.throw_exception_if_processing_interrupted()


def _release_clip_vram(clip: Any) -> None:
    """Offload the prompt-writer CLIP before downstream H3 generation."""

    patcher = getattr(clip, "patcher", None)
    if patcher is None:
        return
    comfy.model_management.unload_model_and_clones(
        patcher,
        unload_additional_models=True,
        all_devices=True,
    )
    comfy.model_management.soft_empty_cache()


def _sampling_config(
    *,
    sampling: bool,
    temperature: float,
    top_k: int,
    top_p: float,
    min_p: float,
    repetition_penalty: float,
    seed: int,
) -> SamplingConfig:
    return SamplingConfig(
        do_sample=bool(sampling),
        temperature=float(temperature),
        top_k=int(top_k),
        top_p=float(top_p),
        min_p=float(min_p),
        repetition_penalty=float(repetition_penalty),
        seed=int(seed),
    )


def _connected_frames(frames: io.Autogrow.Type | None) -> list[tuple[int, Any]]:
    connected: list[tuple[int, Any]] = []
    for name, image in (frames or {}).items():
        if image is None:
            continue
        try:
            slot = int(name.rsplit("_", 1)[1])
        except (IndexError, ValueError) as exc:
            raise ValueError(
                f"RP H3-I2V Prompt Writer: invalid input name {name!r}"
            ) from exc
        connected.append((slot, image))
    connected.sort(key=lambda item: item[0])
    slots = [slot for slot, _ in connected]
    if slots != list(range(1, len(connected) + 1)):
        raise ValueError(
            "RP H3-I2V Prompt Writer: frame inputs must be connected "
            "consecutively from frame_1"
        )
    if len(connected) > 9:
        raise ValueError(
            "RP H3-I2V Prompt Writer: at most 9 frames are supported"
        )
    return connected


def _ordered_prompt_inputs() -> list[Any]:
    return [
        io.String.Input(
            f"prompt_{index}",
            multiline=True,
            dynamic_prompts=False,
            default="",
            tooltip=(
                f"Draft for frame_{index} only. Describe the intended action or "
                "state around this exact frame; Gemma keeps it bound to "
                f"<Picture {index}> and merges all frame drafts into one "
                "continuous H3 video prompt."
            ),
        )
        for index in range(1, 10)
    ]


def _common_inputs(*, include_frame_prompts: bool = False) -> list[Any]:
    inputs = [
        io.Clip.Input(
            "clip",
            tooltip=(
                "Generative Gemma 4 CLIP only. Load "
                "gemma4_e4b_it_fp8_scaled.safetensors with Load CLIP type "
                "stable_diffusion. Do not use the H3 Qwen3-VL CLIP here."
            ),
        ),
        io.Combo.Input(
            "skill",
            options=list(SKILL_CHOICES),
            default=SKILL_CHOICES[0],
            tooltip=(
                "Core H3 formatting is always active. This selector adds one "
                "creative production profile, or lets Gemma choose it."
            ),
        ),
        io.Float.Input(
            "duration_seconds",
            default=3.0,
            min=1.0,
            max=149.0,
            step=1.0,
            tooltip=(
                "Whole requested seconds. The aligned_length output applies "
                "max(5, round(seconds * 24)) and snaps upward to H3's 17k+5 grid; "
                "connect that output to the native H3 length input."
            ),
        ),
        io.String.Input(
            "prompt",
            multiline=True,
            dynamic_prompts=False,
            default="",
            tooltip=(
                "Raw request in any language. Structural output is English; "
                "dialogue, lyrics, and visible text remain verbatim."
            ),
        ),
        io.Int.Input(
            "max_token_length",
            default=2048,
            min=512,
            max=8192,
            step=64,
            advanced=True,
            tooltip="Maximum new tokens for final synthesis and a possible repair pass.",
        ),
        io.Int.Input(
            "media_analysis_tokens",
            default=256,
            min=256,
            max=1536,
            step=64,
            advanced=True,
            tooltip=(
                "Token budget for each connected media asset. Every image, "
                "video, and audio source is analyzed independently so no input "
                "can be hidden by another; an incomplete observation is retried "
                "once automatically."
            ),
        ),
        io.Boolean.Input(
            "sampling",
            default=False,
            advanced=True,
            tooltip="Off is deterministic and recommended for strict H3 structure.",
        ),
        io.Float.Input(
            "temperature",
            default=0.7,
            min=0.0,
            max=2.0,
            step=0.01,
            advanced=True,
        ),
        io.Int.Input(
            "top_k",
            default=64,
            min=0,
            max=1000,
            advanced=True,
        ),
        io.Float.Input(
            "top_p",
            default=0.95,
            min=0.0,
            max=1.0,
            step=0.01,
            advanced=True,
        ),
        io.Float.Input(
            "min_p",
            default=0.05,
            min=0.0,
            max=1.0,
            step=0.01,
            advanced=True,
        ),
        io.Float.Input(
            "repetition_penalty",
            default=1.05,
            min=0.0,
            max=5.0,
            step=0.01,
            advanced=True,
        ),
        io.Int.Input(
            "seed",
            default=42,
            min=0,
            max=MAX_SEED,
            control_after_generate=io.ControlAfterGenerate.fixed,
            advanced=True,
        ),
        io.Boolean.Input(
            "strict_validation",
            default=True,
            advanced=True,
            tooltip=(
                "Validate schema, labels, timestamps, fields, and 7000-character "
                "limit; run one repair pass and fail loudly if still invalid."
            ),
        ),
    ]
    if include_frame_prompts:
        # I2V replaces the one global prompt with prompt_1...prompt_9 in the
        # same UI position. The frontend restores historical positional
        # workflows by widget name before they are displayed.
        inputs[3:4] = _ordered_prompt_inputs()
    return inputs


def _reference_media_inputs() -> list[Any]:
    return [
        io.Autogrow.Input(
            "ref_images",
            optional=True,
            template=io.Autogrow.TemplatePrefix(
                input=io.Image.Input("ref_image", tooltip="Reference image."),
                prefix="ref_image_",
                min=0,
                max=9,
            ),
        ),
        io.Autogrow.Input(
            "ref_videos",
            optional=True,
            template=io.Autogrow.TemplatePrefix(
                input=io.Image.Input(
                    "ref_video",
                    tooltip="Reference video as an IMAGE frame batch at 24 fps.",
                ),
                prefix="ref_video_",
                min=0,
                max=3,
            ),
        ),
        io.Autogrow.Input(
            "ref_video_audios",
            optional=True,
            template=io.Autogrow.TemplatePrefix(
                input=io.Audio.Input(
                    "ref_video_audio",
                    tooltip="Enabled soundtrack paired with the same-numbered video.",
                ),
                prefix="ref_video_audio_",
                min=0,
                max=3,
            ),
        ),
        io.Autogrow.Input(
            "ref_audios",
            optional=True,
            template=io.Autogrow.TemplatePrefix(
                input=io.Audio.Input(
                    "ref_audio",
                    tooltip="Standalone reference audio.",
                ),
                prefix="ref_audio_",
                min=0,
                max=3,
            ),
        ),
    ]


class RPH3I2VPromptWriter(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="RPH3I2VPromptWriter",
            display_name="RP H3-I2V Prompt Writer",
            category="RP/MiniMax H3",
            description=(
                "Analyzes up to nine ordered H3 frame images, binds each local "
                "required prompt_N draft to its matching frame, and writes one "
                "continuous structured multimodal prompt that uses them all."
            ),
            search_aliases=[
                "H3 frames prompt",
                "Gemma prompt writer",
                "I2V frames prompt",
            ],
            inputs=_common_inputs(include_frame_prompts=True)
            + [
                io.Autogrow.Input(
                    "frames",
                    template=io.Autogrow.TemplateNames(
                        input=io.Image.Input(
                            "frame",
                            tooltip=(
                                "Ordered H3 frame image. Connecting it reveals "
                                "the next frame input."
                            ),
                        ),
                        names=[f"frame_{index}" for index in range(1, 10)],
                        min=1,
                    ),
                    tooltip="One to nine ordered H3 frame images.",
                )
            ],
            outputs=[
                io.String.Output(display_name="prompt"),
                io.Int.Output(display_name="aligned_length"),
                io.String.Output(display_name="analysis_report"),
            ],
        )

    @classmethod
    def execute(
        cls,
        clip,
        skill,
        duration_seconds,
        prompt_1,
        prompt_2,
        prompt_3,
        prompt_4,
        prompt_5,
        prompt_6,
        prompt_7,
        prompt_8,
        prompt_9,
        max_token_length,
        media_analysis_tokens,
        sampling,
        temperature,
        top_k,
        top_p,
        min_p,
        repetition_penalty,
        seed,
        strict_validation,
        frames: io.Autogrow.Type = None,
    ) -> io.NodeOutput:
        connected = _connected_frames(frames)
        if not connected:
            raise ValueError(
                "RP H3-I2V Prompt Writer: connect at least frame_1"
            )
        duration_seconds = validate_whole_duration_seconds(duration_seconds)
        aligned_length = seconds_to_aligned_frame_count(duration_seconds)
        # Keep the REF2V implementation intact internally. The public frame_n
        # sockets are mapped to its native image sockets before all processing.
        ref_images = {
            f"ref_image_{slot - 1}": image for slot, image in connected
        }
        supplied_prompts = (
            prompt_1,
            prompt_2,
            prompt_3,
            prompt_4,
            prompt_5,
            prompt_6,
            prompt_7,
            prompt_8,
            prompt_9,
        )
        frame_prompts = {
            slot: str(supplied_prompts[slot - 1] or "").strip()
            for slot, _ in connected
        }
        missing_prompts = [
            f"prompt_{slot}" for slot, value in frame_prompts.items() if not value
        ]
        if missing_prompts:
            raise ValueError(
                "RP H3-I2V Prompt Writer: every connected frame requires its "
                "matching draft; fill " + ", ".join(missing_prompts)
            )
        manifest = ReferenceManifest.from_inputs(ref_images=ref_images)
        runner = GemmaRunner(clip)
        observations = analyze_reference_media(
            runner,
            manifest=manifest,
            ref_images=ref_images,
            target_frame_count=aligned_length,
            max_new_tokens=media_analysis_tokens,
            seed=seed,
            after_call=_check_interrupted,
        )
        result = compose_ref_prompt(
            runner,
            raw_prompt="",
            length=aligned_length,
            selected_skill_label=skill,
            observations=observations,
            manifest=manifest,
            max_new_tokens=max_token_length,
            sampling=_sampling_config(
                sampling=sampling,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                seed=seed,
            ),
            requested_duration_seconds=duration_seconds,
            strict_validation=strict_validation,
            after_call=_check_interrupted,
            i2v_detailed_description=True,
            frame_prompts=frame_prompts,
        )
        report = result.analysis_report(
            mode="Frames2VA",
            length=aligned_length,
            requested_duration_seconds=duration_seconds,
            observations=observations,
            manifest=manifest,
        )
        node_output = io.NodeOutput(result.prompt, aligned_length, report)
        _release_clip_vram(clip)
        return node_output


class RPH3T2VPromptWriter(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="RPH3T2VPromptWriter",
            display_name="RP H3-T2V Prompt Writer",
            category="RP/MiniMax H3",
            description=(
                "Uses optional images, video frame batches, and audio only as "
                "internal planning evidence, then writes a self-contained H3 "
                "text-to-video prompt without reference tags."
            ),
            search_aliases=["H3 text to video", "Gemma prompt writer", "T2V prompt"],
            inputs=_common_inputs() + _reference_media_inputs(),
            outputs=[
                io.String.Output(display_name="prompt"),
                io.Int.Output(display_name="aligned_length"),
                io.String.Output(display_name="analysis_report"),
            ],
        )

    @classmethod
    def execute(
        cls,
        clip,
        skill,
        duration_seconds,
        prompt,
        max_token_length,
        media_analysis_tokens,
        sampling,
        temperature,
        top_k,
        top_p,
        min_p,
        repetition_penalty,
        seed,
        strict_validation,
        ref_images=None,
        ref_videos=None,
        ref_video_audios=None,
        ref_audios=None,
    ) -> io.NodeOutput:
        duration_seconds = validate_whole_duration_seconds(duration_seconds)
        aligned_length = seconds_to_aligned_frame_count(duration_seconds)
        manifest = ReferenceManifest.from_inputs(
            ref_images=ref_images,
            ref_videos=ref_videos,
            ref_video_audios=ref_video_audios,
            ref_audios=ref_audios,
            require_reference=False,
        )
        runner = GemmaRunner(clip)
        observations = analyze_reference_media(
            runner,
            manifest=manifest,
            ref_images=ref_images,
            ref_videos=ref_videos,
            ref_video_audios=ref_video_audios,
            ref_audios=ref_audios,
            target_frame_count=aligned_length,
            max_new_tokens=media_analysis_tokens,
            seed=seed,
            after_call=_check_interrupted,
        )
        result = compose_t2v_prompt(
            runner,
            raw_prompt=prompt,
            length=aligned_length,
            selected_skill_label=skill,
            observations=observations,
            manifest=manifest,
            max_new_tokens=max_token_length,
            sampling=_sampling_config(
                sampling=sampling,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                seed=seed,
            ),
            requested_duration_seconds=duration_seconds,
            strict_validation=strict_validation,
            after_call=_check_interrupted,
        )
        report = result.analysis_report(
            mode="T2V",
            length=aligned_length,
            requested_duration_seconds=duration_seconds,
            observations=observations,
            manifest=manifest,
        )
        node_output = io.NodeOutput(result.prompt, aligned_length, report)
        _release_clip_vram(clip)
        return node_output


class RPH3REF2VPromptWriter(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="RPH3REF2VPromptWriter",
            display_name="RP H3-REF2V Prompt Writer",
            category="RP/MiniMax H3",
            description=(
                "Analyzes every connected image, video, paired soundtrack, and "
                "standalone audio source independently, reproduces native "
                "Picture/Video/Audio numbering, and "
                "writes the strict six-section Ref2VA prompt."
            ),
            search_aliases=["H3 reference prompt", "Gemma prompt writer", "Ref2V prompt"],
            inputs=_common_inputs() + _reference_media_inputs(),
            outputs=[
                io.String.Output(display_name="prompt"),
                io.Int.Output(display_name="aligned_length"),
                io.String.Output(display_name="analysis_report"),
            ],
        )

    @classmethod
    def execute(
        cls,
        clip,
        skill,
        duration_seconds,
        prompt,
        max_token_length,
        media_analysis_tokens,
        sampling,
        temperature,
        top_k,
        top_p,
        min_p,
        repetition_penalty,
        seed,
        strict_validation,
        ref_images=None,
        ref_videos=None,
        ref_video_audios=None,
        ref_audios=None,
    ) -> io.NodeOutput:
        duration_seconds = validate_whole_duration_seconds(duration_seconds)
        aligned_length = seconds_to_aligned_frame_count(duration_seconds)
        manifest = ReferenceManifest.from_inputs(
            ref_images=ref_images,
            ref_videos=ref_videos,
            ref_video_audios=ref_video_audios,
            ref_audios=ref_audios,
        )
        runner = GemmaRunner(clip)
        observations = analyze_reference_media(
            runner,
            manifest=manifest,
            ref_images=ref_images,
            ref_videos=ref_videos,
            ref_video_audios=ref_video_audios,
            ref_audios=ref_audios,
            target_frame_count=aligned_length,
            max_new_tokens=media_analysis_tokens,
            seed=seed,
            after_call=_check_interrupted,
        )
        result = compose_ref_prompt(
            runner,
            raw_prompt=prompt,
            length=aligned_length,
            selected_skill_label=skill,
            observations=observations,
            manifest=manifest,
            max_new_tokens=max_token_length,
            sampling=_sampling_config(
                sampling=sampling,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                seed=seed,
            ),
            requested_duration_seconds=duration_seconds,
            strict_validation=strict_validation,
            after_call=_check_interrupted,
        )
        report = result.analysis_report(
            mode="Ref2VA",
            length=aligned_length,
            requested_duration_seconds=duration_seconds,
            observations=observations,
            manifest=manifest,
        )
        node_output = io.NodeOutput(result.prompt, aligned_length, report)
        _release_clip_vram(clip)
        return node_output
