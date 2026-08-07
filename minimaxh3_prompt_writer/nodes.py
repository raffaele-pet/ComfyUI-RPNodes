"""ComfyUI V3 node definitions for RP H3 prompt writing."""

from __future__ import annotations

from typing import Any

import comfy.model_management
from comfy_api.latest import io

from .engine.analyzers import analyze_base_media, analyze_reference_media
from .engine.composer import (
    compose_base_prompt,
    compose_ref_prompt,
    compose_t2v_prompt,
)
from .engine.constants import SKILL_CHOICES
from .engine.gemma import GemmaRunner, SamplingConfig
from .engine.manifests import (
    ReferenceManifest,
    determine_base_mode,
    seconds_to_aligned_frame_count,
    validate_whole_duration_seconds,
)


MAX_SEED = 0xFFFFFFFFFFFFFFFF


def _check_interrupted() -> None:
    comfy.model_management.throw_exception_if_processing_interrupted()


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


def _common_inputs(*, max_new_tokens: int) -> list[Any]:
    return [
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
            "max_new_tokens",
            default=max_new_tokens,
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
            default=0,
            min=0,
            max=MAX_SEED,
            control_after_generate=True,
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


def _reference_media_inputs() -> list[Any]:
    return [
        io.Image.Input("ref_image_0", optional=True),
        io.Image.Input("ref_image_1", optional=True),
        io.Image.Input("ref_image_2", optional=True),
        io.Image.Input(
            "ref_video_0",
            optional=True,
            tooltip="Reference video as an IMAGE frame batch at 24 fps.",
        ),
        io.Image.Input(
            "ref_video_1",
            optional=True,
            tooltip="Reference video as an IMAGE frame batch at 24 fps.",
        ),
        io.Audio.Input(
            "ref_video_audio_0",
            optional=True,
            tooltip="Enabled soundtrack paired with ref_video_0.",
        ),
        io.Audio.Input("ref_audio_0", optional=True),
    ]


class RPH3I2VPromptWriter(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="RPH3I2VPromptWriter",
            display_name="RP H3-I2V Prompt Writer",
            category="RP/MiniMax H3",
            description=(
                "Uses a generative multimodal Gemma 4 CLIP to analyze optional "
                "H3 keyframes and rewrite a raw request into strict T2VA, I2VA, "
                "FL2VA, or L2VA prompt structure."
            ),
            search_aliases=["H3 prompt", "Gemma prompt writer", "I2V prompt"],
            inputs=_common_inputs(max_new_tokens=2048)
            + [
                io.Image.Input(
                    "first_frame",
                    optional=True,
                    tooltip="Literal first video frame; only batch element 0 is analyzed.",
                ),
                io.Image.Input(
                    "last_frame",
                    optional=True,
                    tooltip="Literal final video frame; only batch element 0 is analyzed.",
                ),
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
        prompt,
        max_new_tokens,
        media_analysis_tokens,
        sampling,
        temperature,
        top_k,
        top_p,
        min_p,
        repetition_penalty,
        seed,
        strict_validation,
        first_frame=None,
        last_frame=None,
    ) -> io.NodeOutput:
        duration_seconds = validate_whole_duration_seconds(duration_seconds)
        aligned_length = seconds_to_aligned_frame_count(duration_seconds)
        mode = determine_base_mode(first_frame, last_frame)
        runner = GemmaRunner(clip)
        observations = analyze_base_media(
            runner,
            mode=mode,
            first_frame=first_frame,
            last_frame=last_frame,
            max_new_tokens=media_analysis_tokens,
            seed=seed,
            after_call=_check_interrupted,
        )
        result = compose_base_prompt(
            runner,
            raw_prompt=prompt,
            mode=mode,
            length=aligned_length,
            selected_skill_label=skill,
            observations=observations,
            max_new_tokens=max_new_tokens,
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
            mode=mode,
            length=aligned_length,
            requested_duration_seconds=duration_seconds,
            observations=observations,
        )
        return io.NodeOutput(result.prompt, aligned_length, report)


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
            inputs=_common_inputs(max_new_tokens=2560) + _reference_media_inputs(),
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
        max_new_tokens,
        media_analysis_tokens,
        sampling,
        temperature,
        top_k,
        top_p,
        min_p,
        repetition_penalty,
        seed,
        strict_validation,
        ref_image_0=None,
        ref_image_1=None,
        ref_image_2=None,
        ref_video_0=None,
        ref_video_1=None,
        ref_video_audio_0=None,
        ref_audio_0=None,
    ) -> io.NodeOutput:
        duration_seconds = validate_whole_duration_seconds(duration_seconds)
        aligned_length = seconds_to_aligned_frame_count(duration_seconds)
        manifest = ReferenceManifest.from_inputs(
            ref_image_0=ref_image_0,
            ref_image_1=ref_image_1,
            ref_image_2=ref_image_2,
            ref_video_0=ref_video_0,
            ref_video_1=ref_video_1,
            ref_video_audio_0=ref_video_audio_0,
            ref_audio_0=ref_audio_0,
            require_reference=False,
        )
        runner = GemmaRunner(clip)
        observations = analyze_reference_media(
            runner,
            manifest=manifest,
            ref_image_0=ref_image_0,
            ref_image_1=ref_image_1,
            ref_image_2=ref_image_2,
            ref_video_0=ref_video_0,
            ref_video_1=ref_video_1,
            ref_video_audio_0=ref_video_audio_0,
            ref_audio_0=ref_audio_0,
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
            max_new_tokens=max_new_tokens,
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
        return io.NodeOutput(result.prompt, aligned_length, report)


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
            inputs=_common_inputs(max_new_tokens=2560) + _reference_media_inputs(),
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
        max_new_tokens,
        media_analysis_tokens,
        sampling,
        temperature,
        top_k,
        top_p,
        min_p,
        repetition_penalty,
        seed,
        strict_validation,
        ref_image_0=None,
        ref_image_1=None,
        ref_image_2=None,
        ref_video_0=None,
        ref_video_1=None,
        ref_video_audio_0=None,
        ref_audio_0=None,
    ) -> io.NodeOutput:
        duration_seconds = validate_whole_duration_seconds(duration_seconds)
        aligned_length = seconds_to_aligned_frame_count(duration_seconds)
        manifest = ReferenceManifest.from_inputs(
            ref_image_0=ref_image_0,
            ref_image_1=ref_image_1,
            ref_image_2=ref_image_2,
            ref_video_0=ref_video_0,
            ref_video_1=ref_video_1,
            ref_video_audio_0=ref_video_audio_0,
            ref_audio_0=ref_audio_0,
        )
        runner = GemmaRunner(clip)
        observations = analyze_reference_media(
            runner,
            manifest=manifest,
            ref_image_0=ref_image_0,
            ref_image_1=ref_image_1,
            ref_image_2=ref_image_2,
            ref_video_0=ref_video_0,
            ref_video_1=ref_video_1,
            ref_video_audio_0=ref_video_audio_0,
            ref_audio_0=ref_audio_0,
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
            max_new_tokens=max_new_tokens,
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
        return io.NodeOutput(result.prompt, aligned_length, report)
