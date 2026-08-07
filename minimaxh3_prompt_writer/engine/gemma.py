"""Thin, public-API-only wrapper around ComfyUI's generative CLIP object."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .prompts import gemma4_chat


@dataclass(frozen=True)
class SamplingConfig:
    do_sample: bool = False
    temperature: float = 0.7
    top_k: int = 64
    top_p: float = 0.95
    min_p: float = 0.05
    repetition_penalty: float = 1.05
    presence_penalty: float = 0.0
    seed: int = 0


class GemmaRunner:
    """Generate text using only ``clip.tokenize/generate/decode``.

    This deliberately avoids Gemma tokenizer internals so the node remains
    aligned with ComfyUI's supported TextGenerate surface.
    """

    def __init__(self, clip: Any):
        self.clip = clip
        self._validate_clip()

    def _validate_clip(self) -> None:
        missing = [name for name in ("tokenize", "generate", "decode") if not callable(getattr(self.clip, name, None))]
        if missing:
            raise ValueError(
                "The connected CLIP cannot generate text; missing methods: "
                + ", ".join(missing)
                + ". Load gemma4_e4b_it_fp8_scaled.safetensors with Load CLIP."
            )
        tokenizer = getattr(self.clip, "tokenizer", None)
        clip_name = getattr(tokenizer, "clip_name", None)
        if clip_name is not None and clip_name != "gemma4":
            raise ValueError(
                f"Expected the Gemma 4 generative CLIP, but tokenizer is {clip_name!r}. "
                "Use gemma4_e4b_it_fp8_scaled.safetensors (Load CLIP type: stable_diffusion). "
                "The Qwen3-VL MiniMax CLIP belongs on the downstream H3 node instead."
            )

    def _decode(self, generated_ids: Any) -> str:
        decoded = self.clip.decode(generated_ids)
        return str(decoded or "").strip()

    def _generate_ids(
        self,
        tokens: Any,
        *,
        max_new_tokens: int,
        sampling: SamplingConfig,
    ) -> Any:
        return self.clip.generate(
            tokens,
            do_sample=bool(sampling.do_sample),
            max_length=int(max_new_tokens),
            temperature=float(sampling.temperature),
            top_k=int(sampling.top_k),
            top_p=float(sampling.top_p),
            min_p=float(sampling.min_p),
            repetition_penalty=float(sampling.repetition_penalty),
            presence_penalty=float(sampling.presence_penalty),
            seed=int(sampling.seed),
        )

    def generate_media_analysis(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        image: Any = None,
        video: Any = None,
        audio: Any = None,
        seed: int = 0,
    ) -> str:
        if image is not None and video is not None:
            raise ValueError(
                "Gemma 4 media analysis cannot receive image and video together; "
                "ComfyUI would silently ignore the image input."
            )
        tokens = self.clip.tokenize(
            prompt,
            image=image,
            video=video,
            audio=audio,
            skip_template=False,
            min_length=1,
            thinking=False,
        )
        generated_ids = self._generate_ids(
            tokens,
            max_new_tokens=max_new_tokens,
            sampling=SamplingConfig(do_sample=False, seed=seed),
        )
        return self._decode(generated_ids)

    def generate_chat(
        self,
        system_prompt: str,
        user_payload: str,
        *,
        max_new_tokens: int,
        sampling: SamplingConfig,
        assistant_prefix: str = "",
    ) -> str:
        formatted = gemma4_chat(system_prompt, user_payload, assistant_prefix)
        tokens = self.clip.tokenize(
            formatted,
            skip_template=True,
            min_length=1,
            thinking=False,
        )
        generated_ids = self._generate_ids(
            tokens,
            max_new_tokens=max_new_tokens,
            sampling=sampling,
        )
        return assistant_prefix + self._decode(generated_ids)
