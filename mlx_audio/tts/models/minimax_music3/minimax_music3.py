"""Native MLX MiniMax Music 3 pipeline.

The model implementation is adapted and modified from
mikolaj92/minimax-music3-mlx (Apache-2.0) and integrated with mlx-audio's model,
quantization, and result APIs. See LICENSE and NOTICE.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_audio.lm.models.qwen3 import Model as Qwen3Model
from mlx_audio.lm.models.qwen3 import ModelArgs as Qwen3Args
from mlx_audio.tts.models.base import GenerationResult

from .ar import generate_frame_hiddens
from .config import (
    CHUNK_FRAMES,
    CHUNK_HOP,
    CROP_LEFT_LATENT,
    CROP_RIGHT_LATENT,
    DIT_CFG_SCALE,
    LATENT_HOP_LENGTH,
    MAX_PROMPT_TOKENS,
    OVERLAP_LATENT_LENGTH,
    ModelConfig,
)
from .depth import RVQDepthDecoder
from .dit import FlowMatchingTransformer
from .euler import denoise_chunk
from .fusion import ConditionEncoder
from .prompt import assemble_prompt
from .vocoder import Vocoder


def _make_qwen3(config: ModelConfig) -> Qwen3Model:
    return Qwen3Model(
        Qwen3Args(
            model_type="qwen3",
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            intermediate_size=config.intermediate_size,
            num_attention_heads=config.num_attention_heads,
            rms_norm_eps=config.rms_norm_eps,
            vocab_size=config.vocab_size,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            head_dim=config.head_dim,
            tie_word_embeddings=config.tie_word_embeddings,
        )
    )


def _encode_tiny_text_pair(
    text: str, config: ModelConfig, max_length: int = 32
) -> mx.array:
    ids = [
        ((ord(character) * 17 + index * 3) % 180) + 1
        for index, character in enumerate(text[: max_length - 3])
    ]
    ids = [1] + (ids or [1]) + [2]
    conditional = mx.array(ids, dtype=mx.int32)[None, :]
    unconditional = conditional
    if unconditional.shape[1] > 3:
        middle = mx.full(
            (1, unconditional.shape[1] - 3),
            config.audio_cfg_token_id,
            dtype=mx.int32,
        )
        unconditional = mx.concatenate(
            [unconditional[:, :1], middle, unconditional[:, -2:]], axis=1
        )
    return mx.concatenate([conditional, unconditional], axis=0)


def _encode_official_text_pair(
    text: str, config: ModelConfig, tokenizer_dir: Path
) -> mx.array:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    input_ids = tokenizer(text, return_tensors="np")["input_ids"]
    if input_ids.shape[1] > MAX_PROMPT_TOKENS:
        raise ValueError(
            f"The assembled prompt has {input_ids.shape[1]} tokens; "
            f"the maximum is {MAX_PROMPT_TOKENS}"
        )
    conditional = mx.array(input_ids.astype("int32"))
    unconditional = conditional
    if unconditional.shape[1] > 3:
        middle = mx.full(
            (1, unconditional.shape[1] - 3),
            config.audio_cfg_token_id,
            dtype=mx.int32,
        )
        unconditional = mx.concatenate(
            [unconditional[:, :1], middle, unconditional[:, -2:]], axis=1
        )
    return mx.concatenate([conditional, unconditional], axis=0)


def _chunk_starts(num_frames: int) -> list[int]:
    if num_frames <= CHUNK_FRAMES:
        return [0]
    return list(range(0, num_frames - CHUNK_HOP, CHUNK_HOP))


def _crop_waveform(wave: mx.array, chunk_index: int, num_chunks: int) -> mx.array:
    left = 0 if chunk_index == 0 else CROP_LEFT_LATENT * LATENT_HOP_LENGTH
    right = (
        0 if chunk_index == num_chunks - 1 else CROP_RIGHT_LATENT * LATENT_HOP_LENGTH
    )
    end = wave.shape[-1] - right
    return wave if end <= left else wave[..., left:end]


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.sample_rate = config.sample_rate
        self.model_type = config.model_type
        self.language_model = _make_qwen3(config)
        self.rvq_depth_decoder = RVQDepthDecoder(config)
        self.condition_encoder = ConditionEncoder(config)
        self.transformer = FlowMatchingTransformer(config)
        self.vocoder = Vocoder(config)

    @staticmethod
    def model_quant_predicate(path: str, module: nn.Module) -> bool:
        if not isinstance(module, nn.Linear):
            return False
        if path.endswith("lm_head") or ".audio_heads." in path:
            return False
        return path.startswith(
            (
                "language_model.model.layers.",
                "rvq_depth_decoder.layers.",
                "transformer.proj_in",
                "transformer.proj_out",
                "transformer.transformer_blocks.",
            )
        )

    def _text_ids(self, prompt: str, lyrics: str) -> mx.array:
        assembled = assemble_prompt(prompt, lyrics)
        tokenizer_dir = Path(self.config.model_path) / "tokenizer"
        if self.config.model_path and tokenizer_dir.is_dir():
            return _encode_official_text_pair(assembled, self.config, tokenizer_dir)
        return _encode_tiny_text_pair(assembled, self.config)

    def _run_flow(
        self,
        frame_hiddens: mx.array,
        num_inference_steps: int,
        seed: int,
    ) -> mx.array:
        starts = _chunk_starts(frame_hiddens.shape[1])
        waves = []
        previous_latent = None
        previous_condition = None
        mx.random.seed(seed + 7)
        for start in starts:
            end = min(start + CHUNK_FRAMES, frame_hiddens.shape[1])
            condition = self.condition_encoder(frame_hiddens[:, start:end])
            noise = mx.random.normal(
                (1, self.config.dit_in_channels, condition.shape[1])
            ).astype(condition.dtype)
            latents, condition = denoise_chunk(
                self.transformer,
                noise,
                condition,
                num_inference_steps=num_inference_steps,
                guidance_scale=DIT_CFG_SCALE,
                previous_latent=previous_latent,
                previous_condition=previous_condition,
            )
            carry_start = max(0, latents.shape[-1] - 2 * OVERLAP_LATENT_LENGTH)
            carry_end = max(carry_start, latents.shape[-1] - OVERLAP_LATENT_LENGTH)
            previous_latent = latents[..., carry_start:carry_end]
            previous_condition = condition[:, carry_start:carry_end]
            waves.append(self.vocoder(latents))
        return mx.concatenate(
            [
                _crop_waveform(wave, index, len(waves))
                for index, wave in enumerate(waves)
            ],
            axis=-1,
        )

    def generate(
        self,
        text: str,
        *,
        lyrics: str,
        duration: float | None = None,
        gen_duration: float | None = None,
        steps: int = 30,
        seed: int = 0,
        **_: object,
    ) -> Iterator[GenerationResult]:
        requested_duration = (
            duration
            if duration is not None
            else gen_duration if gen_duration is not None else 60.0
        )
        if not text.strip():
            raise ValueError("A music description is required")
        if not lyrics.strip():
            raise ValueError("Lyrics are required; use [instrumental] explicitly")
        if requested_duration <= 0 or requested_duration > 360:
            raise ValueError("duration must be between 0 and 360 seconds")
        if steps < 1 or steps > 30:
            raise ValueError("steps must be between 1 and 30")

        started = time.perf_counter()
        max_frames = max(1, int(requested_duration * self.config.frame_rate))
        frame_hiddens = generate_frame_hiddens(
            self.language_model,
            self.rvq_depth_decoder,
            self.config,
            self._text_ids(text, lyrics),
            max_frames=max_frames,
            seed=seed,
        )
        mx.eval(frame_hiddens)
        audio = self._run_flow(frame_hiddens, steps, seed)
        mx.eval(audio)
        waveform = mx.clip(audio[0].transpose(1, 0).astype(mx.float32), -1.0, 1.0)
        if not np.isfinite(np.asarray(waveform)).all():
            raise RuntimeError("MiniMax Music 3 produced non-finite audio")

        elapsed = time.perf_counter() - started
        samples = int(waveform.shape[0])
        duration_seconds = samples / self.sample_rate
        duration_hours = int(duration_seconds // 3600)
        duration_minutes = int((duration_seconds % 3600) // 60)
        duration_whole_seconds = int(duration_seconds % 60)
        duration_milliseconds = int((duration_seconds % 1) * 1000)
        duration_string = (
            f"{duration_hours:02d}:{duration_minutes:02d}:"
            f"{duration_whole_seconds:02d}.{duration_milliseconds:03d}"
        )
        token_count = int(frame_hiddens.shape[1])
        yield GenerationResult(
            audio=waveform,
            samples=samples,
            sample_rate=self.sample_rate,
            segment_idx=0,
            token_count=token_count,
            audio_duration=duration_string,
            real_time_factor=elapsed / max(duration_seconds, 1e-9),
            prompt={
                "tokens": token_count,
                "tokens-per-sec": token_count / max(elapsed, 1e-9),
                "text": text,
                "lyrics": lyrics,
                "seed": seed,
                "steps": steps,
            },
            audio_samples={
                "samples": samples,
                "samples-per-sec": samples / max(elapsed, 1e-9),
                "requested_seconds": requested_duration,
            },
            processing_time_seconds=elapsed,
            peak_memory_usage=mx.get_peak_memory() / 1e9,
        )


__all__ = ["Model", "ModelConfig"]
