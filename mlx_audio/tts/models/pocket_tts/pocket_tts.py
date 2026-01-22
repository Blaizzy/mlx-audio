from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Iterable, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.tts.generate import load_audio
from mlx_audio.tts.models.base import GenerationResult

from .conditioners import TokenizedText
from .config import ModelConfig
from .flow_lm import FlowLMModel
from .mimi import MimiAdapter
from .utils import PREDEFINED_VOICES, download_if_necessary, load_predefined_voice

DEFAULT_TEMPERATURE = 0.7
DEFAULT_LSD_DECODE_STEPS = 1
DEFAULT_NOISE_CLAMP = None
DEFAULT_EOS_THRESHOLD = -4.0
DEFAULT_AUDIO_PROMPT = "alba"


class Model(nn.Module):
    def __init__(self, config: ModelConfig | dict[str, Any]):
        super().__init__()
        if isinstance(config, dict):
            config = ModelConfig.from_dict(config)
        self.config = config
        if config.flow_lm is None or config.mimi is None:
            raise ValueError("PocketTTS requires flow_lm and mimi config sections.")

        self.flow_lm = FlowLMModel.from_config(
            config.flow_lm, latent_dim=config.mimi.quantizer.dimension
        )
        self.mimi = MimiAdapter.from_config(config.mimi)

        self.temp = DEFAULT_TEMPERATURE
        self.lsd_decode_steps = DEFAULT_LSD_DECODE_STEPS
        self.noise_clamp = DEFAULT_NOISE_CLAMP
        self.eos_threshold = DEFAULT_EOS_THRESHOLD

        self.speaker_proj_weight = mx.zeros(
            (
                config.flow_lm.transformer.d_model,
                config.mimi.quantizer.output_dimension,
            ),
            dtype=mx.float32,
        )

    @property
    def sample_rate(self) -> int | None:
        if self.config.mimi is None:
            return None
        return self.config.mimi.sample_rate

    @property
    def model_type(self) -> str:
        return self.config.model_type

    def load_weights(self, weights, strict: bool = True):
        return super().load_weights(weights, strict=strict)

    def load_from_config(self, strict: bool = True):
        if self.config.weights_path:
            weights = mx.load(download_if_necessary(self.config.weights_path))
            return self.load_weights(weights, strict=strict)

        loaded = False
        if self.config.flow_lm and self.config.flow_lm.weights_path:
            weights = mx.load(download_if_necessary(self.config.flow_lm.weights_path))
            self.load_weights(weights, strict=strict)
            loaded = True
        if self.config.mimi and self.config.mimi.weights_path:
            weights = mx.load(download_if_necessary(self.config.mimi.weights_path))
            self.load_weights(weights, strict=strict)
            loaded = True
        if not loaded:
            raise ValueError("No weights_path configured for PocketTTS.")

    def init_state(self) -> dict[str, Any]:
        return {"flow_cache": self.flow_lm.make_cache()}

    def _run_flow_lm(
        self,
        model_state: dict[str, Any],
        text_tokens: mx.array,
        backbone_input_latents: mx.array,
        audio_conditioning: mx.array,
    ) -> tuple[mx.array, mx.array]:
        text_embeddings = self.flow_lm.conditioner(TokenizedText(text_tokens))
        text_embeddings = mx.concatenate([text_embeddings, audio_conditioning], axis=1)
        output_embeddings, is_eos = self.flow_lm._sample_next_latent(
            sequence=backbone_input_latents,
            text_embeddings=text_embeddings,
            cache=model_state["flow_cache"],
            lsd_decode_steps=self.lsd_decode_steps,
            temp=self.temp,
            noise_clamp=self.noise_clamp,
            eos_threshold=self.eos_threshold,
        )
        return output_embeddings[:, None, :], is_eos

    def _run_flow_lm_and_increment_step(
        self,
        model_state: dict[str, Any],
        text_tokens: mx.array | None = None,
        backbone_input_latents: mx.array | None = None,
        audio_conditioning: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        if text_tokens is None:
            text_tokens = mx.zeros((1, 0), dtype=mx.int32)
        if backbone_input_latents is None:
            backbone_input_latents = mx.zeros(
                (1, 0, self.flow_lm.ldim), dtype=mx.float32
            )
        if audio_conditioning is None:
            audio_conditioning = mx.zeros((1, 0, self.flow_lm.dim), dtype=mx.float32)
        return self._run_flow_lm(
            model_state,
            text_tokens=text_tokens,
            backbone_input_latents=backbone_input_latents,
            audio_conditioning=audio_conditioning,
        )

    def _encode_audio(self, audio: mx.array) -> mx.array:
        encoded = self.mimi.encode_to_latent(audio)
        latents = encoded.transpose(0, 2, 1).astype(mx.float32)
        conditioning = latents @ self.speaker_proj_weight.T
        return conditioning

    def get_state_for_audio_prompt(self, audio_conditioning: mx.array | Path | str):
        if (
            isinstance(audio_conditioning, str)
            and audio_conditioning in PREDEFINED_VOICES
        ):
            prompt = load_predefined_voice(audio_conditioning)
        else:
            audio = self._load_audio(audio_conditioning)
            prompt = self._encode_audio(audio)
        model_state = self.init_state()
        self._run_flow_lm_and_increment_step(
            model_state=model_state, audio_conditioning=prompt
        )
        self._slice_flow_cache(model_state, prompt.shape[1])
        return model_state

    def generate_audio(
        self,
        model_state: dict[str, Any] | None,
        text_to_generate: str,
        frames_after_eos: int | None = None,
    ) -> mx.array:
        if model_state is None:
            model_state = self.get_state_for_audio_prompt(DEFAULT_AUDIO_PROMPT)
        chunks = []
        for chunk in self.generate_audio_stream(
            model_state=model_state,
            text_to_generate=text_to_generate,
            frames_after_eos=frames_after_eos,
        ):
            chunks.append(chunk)
        if not chunks:
            return mx.zeros((0,), dtype=mx.float32)
        return mx.concatenate(chunks, axis=0)

    def generate_audio_stream(
        self,
        model_state: dict[str, Any] | None,
        text_to_generate: str,
        frames_after_eos: int | None = None,
    ) -> Iterable[mx.array]:
        if model_state is None:
            model_state = self.get_state_for_audio_prompt(DEFAULT_AUDIO_PROMPT)
        chunks = split_into_best_sentences(
            self.flow_lm.conditioner.tokenizer, text_to_generate
        )
        for chunk in chunks:
            _, frames_after_eos_guess = prepare_text_prompt(chunk)
            if frames_after_eos is None:
                frames_after_eos = frames_after_eos_guess + 2
            yield from self._generate_audio_stream_short_text(
                model_state=model_state,
                text_to_generate=chunk,
                frames_after_eos=frames_after_eos,
            )

    def _generate_audio_stream_short_text(
        self, model_state: dict[str, Any], text_to_generate: str, frames_after_eos: int
    ) -> Iterable[mx.array]:
        self.mimi.reset_state()
        self._expand_flow_cache(model_state, sequence_length=1000)
        gen_len_sec = len(text_to_generate.split()) * 1 + 2.0
        max_gen_len = int(gen_len_sec * self.mimi.frame_rate)

        prepared = self.flow_lm.conditioner.prepare(text_to_generate)
        self._run_flow_lm_and_increment_step(
            model_state=model_state, text_tokens=prepared.tokens
        )

        backbone_input = mx.full(
            (1, 1, self.flow_lm.ldim), float("NaN"), dtype=mx.float32
        )
        eos_step = None
        for step in range(max_gen_len):
            next_latent, is_eos = self._run_flow_lm_and_increment_step(
                model_state=model_state, backbone_input_latents=backbone_input
            )
            if bool(is_eos.item()) and eos_step is None:
                eos_step = step
            if eos_step is not None and step >= eos_step + frames_after_eos:
                break

            decoding_input = next_latent * self.flow_lm.emb_std + self.flow_lm.emb_mean
            quantized = self.mimi.quantizer(decoding_input.transpose(0, 2, 1))
            audio_chunk = self.mimi.decode_step(quantized)
            yield audio_chunk[0, 0]
            backbone_input = next_latent

    def generate(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        lang_code: str = "en",
        ref_audio: Optional[mx.array] = None,
        ref_text: Optional[str] = None,
        cfg_scale: Optional[float] = None,
        ddpm_steps: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 1200,
        verbose: bool = False,
        stream: bool = False,
        streaming_interval: float = 2.0,
        frames_after_eos: Optional[int] = None,
        **kwargs,
    ) -> Iterable[GenerationResult]:
        del lang_code, ref_text, cfg_scale, ddpm_steps, max_tokens, speed

        prompt = self._resolve_audio_prompt(voice, ref_audio, verbose)
        model_state = self.get_state_for_audio_prompt(prompt)
        token_count = len(text.split())

        prev_temp = self.temp
        prev_lsd = self.lsd_decode_steps
        prev_noise = self.noise_clamp
        prev_eos = self.eos_threshold
        if temperature is not None:
            self.temp = temperature
        lsd_decode_steps = kwargs.get("lsd_decode_steps")
        if lsd_decode_steps is not None:
            self.lsd_decode_steps = lsd_decode_steps
        noise_clamp = kwargs.get("noise_clamp")
        if noise_clamp is not None:
            self.noise_clamp = noise_clamp
        eos_threshold = kwargs.get("eos_threshold")
        if eos_threshold is not None:
            self.eos_threshold = eos_threshold

        try:
            if not stream:
                start_time = time.perf_counter()
                audio = self.generate_audio(
                    model_state=model_state,
                    text_to_generate=text,
                    frames_after_eos=frames_after_eos,
                )
                yield _build_generation_result(
                    audio=audio,
                    sample_rate=self.sample_rate,
                    start_time=start_time,
                    segment_idx=0,
                    token_count=token_count,
                )
                return

            interval_samples = int(streaming_interval * self.sample_rate)
            if interval_samples <= 0:
                interval_samples = 1
            buffer = []
            buffered_samples = 0
            segment_idx = 0
            start_time = time.perf_counter()
            for chunk in self.generate_audio_stream(
                model_state=model_state,
                text_to_generate=text,
                frames_after_eos=frames_after_eos,
            ):
                if not isinstance(chunk, mx.array):
                    chunk = mx.array(chunk)
                buffer.append(chunk)
                buffered_samples += int(chunk.shape[0])
                if buffered_samples < interval_samples:
                    continue
                audio = mx.concatenate(buffer, axis=0) if len(buffer) > 1 else buffer[0]
                yield _build_generation_result(
                    audio=audio,
                    sample_rate=self.sample_rate,
                    start_time=start_time,
                    segment_idx=segment_idx,
                    token_count=token_count,
                )
                segment_idx += 1
                buffer = []
                buffered_samples = 0
                start_time = time.perf_counter()

            if buffer:
                audio = mx.concatenate(buffer, axis=0) if len(buffer) > 1 else buffer[0]
                yield _build_generation_result(
                    audio=audio,
                    sample_rate=self.sample_rate,
                    start_time=start_time,
                    segment_idx=segment_idx,
                    token_count=token_count,
                )
        finally:
            self.temp = prev_temp
            self.lsd_decode_steps = prev_lsd
            self.noise_clamp = prev_noise
            self.eos_threshold = prev_eos

    def _load_audio(self, audio_conditioning: mx.array | Path | str) -> mx.array:
        if isinstance(audio_conditioning, (str, Path)):
            audio_path = download_if_necessary(str(audio_conditioning))
            audio = load_audio(str(audio_path), sample_rate=self.sample_rate)
        else:
            audio = audio_conditioning
        if not isinstance(audio, mx.array):
            audio = mx.array(audio)
        if audio.ndim == 1:
            audio = audio[None, None, :]
        elif audio.ndim == 2:
            if audio.shape[0] > 1:
                audio = mx.mean(audio, axis=0, keepdims=True)
            audio = audio[None, :, :]
        return audio.astype(mx.float32)

    def _slice_flow_cache(self, model_state: dict[str, Any], num_frames: int) -> None:
        caches = model_state.get("flow_cache", [])
        for cache in caches:
            if cache.keys is None:
                continue
            cache.keys = cache.keys[..., :num_frames, :]
            cache.values = cache.values[..., :num_frames, :]
            cache.offset = min(cache.offset, num_frames)

    def _expand_flow_cache(
        self, model_state: dict[str, Any], sequence_length: int
    ) -> None:
        caches = model_state.get("flow_cache", [])
        for cache in caches:
            if cache.keys is None:
                continue
            current_length = cache.keys.shape[2]
            if current_length >= sequence_length:
                continue
            pad = sequence_length - current_length
            zeros_k = mx.zeros(
                (
                    cache.keys.shape[0],
                    cache.keys.shape[1],
                    pad,
                    cache.keys.shape[3],
                ),
                cache.keys.dtype,
            )
            zeros_v = mx.zeros(
                (
                    cache.values.shape[0],
                    cache.values.shape[1],
                    pad,
                    cache.values.shape[3],
                ),
                cache.values.dtype,
            )
            cache.keys = mx.concatenate([cache.keys, zeros_k], axis=2)
            cache.values = mx.concatenate([cache.values, zeros_v], axis=2)

    def _resolve_audio_prompt(
        self,
        voice: Optional[str],
        ref_audio: Optional[mx.array],
        verbose: bool,
    ):
        if ref_audio is not None:
            return ref_audio

        prompt = voice or DEFAULT_AUDIO_PROMPT
        if isinstance(prompt, str):
            if prompt in PREDEFINED_VOICES:
                return prompt
            normalized = prompt.lower()
            if normalized in PREDEFINED_VOICES:
                return normalized
            if prompt.startswith(("http://", "https://", "hf://")):
                return prompt
            if os.path.exists(prompt):
                return prompt
            if verbose:
                print(
                    f"Voice '{prompt}' not found for pocket_tts; using '{DEFAULT_AUDIO_PROMPT}' instead."
                )
            return DEFAULT_AUDIO_PROMPT

        return prompt


def _format_duration(duration_seconds: float) -> str:
    duration_mins = int(duration_seconds // 60)
    duration_secs = int(duration_seconds % 60)
    duration_ms = int((duration_seconds % 1) * 1000)
    duration_hours = int(duration_seconds // 3600)
    return f"{duration_hours:02d}:{duration_mins:02d}:{duration_secs:02d}.{duration_ms:03d}"


def _build_generation_result(
    audio: mx.array,
    sample_rate: int,
    start_time: float,
    segment_idx: int,
    token_count: int,
) -> GenerationResult:
    samples = int(audio.shape[0])
    audio_duration_seconds = samples / sample_rate if sample_rate else 0.0
    elapsed_time = time.perf_counter() - start_time
    real_time_factor = (
        audio_duration_seconds / elapsed_time if elapsed_time > 0 else 0.0
    )

    return GenerationResult(
        audio=audio,
        samples=samples,
        sample_rate=sample_rate,
        segment_idx=segment_idx,
        token_count=token_count,
        audio_duration=_format_duration(audio_duration_seconds),
        real_time_factor=real_time_factor,
        prompt={
            "tokens": token_count,
            "tokens-per-sec": (
                round(token_count / elapsed_time, 2) if elapsed_time > 0 else 0.0
            ),
        },
        audio_samples={
            "samples": samples,
            "samples-per-sec": (
                round(samples / elapsed_time, 2) if elapsed_time > 0 else 0.0
            ),
        },
        processing_time_seconds=elapsed_time,
        peak_memory_usage=mx.get_peak_memory() / 1e9,
    )


def prepare_text_prompt(text: str) -> tuple[str, int]:
    text = text.strip()
    if text == "":
        raise ValueError("Text prompt cannot be empty")
    text = text.replace("\n", " ").replace("\r", " ").replace("  ", " ")
    number_of_words = len(text.split())
    if number_of_words <= 4:
        frames_after_eos_guess = 3
    else:
        frames_after_eos_guess = 1

    if not text[0].isupper():
        text = text[0].upper() + text[1:]

    if text[-1].isalnum():
        text = text + "."

    if len(text.split()) < 5:
        text = " " * 8 + text

    return text, frames_after_eos_guess


def split_into_best_sentences(tokenizer, text_to_generate: str) -> list[str]:
    text_to_generate, _ = prepare_text_prompt(text_to_generate)
    text_to_generate = text_to_generate.strip()
    tokens = tokenizer(text_to_generate)
    list_of_tokens = tokens.tokens[0].tolist()

    _, *end_of_sentence_tokens = tokenizer(".!...?").tokens[0].tolist()

    end_of_sentences_indices = [0]
    previous_was_end_of_sentence_token = False

    for token_idx, token in enumerate(list_of_tokens):
        if token in end_of_sentence_tokens:
            previous_was_end_of_sentence_token = True
        else:
            if previous_was_end_of_sentence_token:
                end_of_sentences_indices.append(token_idx)
            previous_was_end_of_sentence_token = False
    end_of_sentences_indices.append(len(list_of_tokens))

    nb_tokens_and_sentences = []
    for i in range(len(end_of_sentences_indices) - 1):
        start = end_of_sentences_indices[i]
        end = end_of_sentences_indices[i + 1]
        text = tokenizer.sp.decode(list_of_tokens[start:end])
        nb_tokens_and_sentences.append((end - start, text))

    max_nb_tokens_in_a_chunk = 50
    chunks = []
    current_chunk = ""
    current_nb_of_tokens_in_chunk = 0
    for nb_tokens, sentence in nb_tokens_and_sentences:
        if current_chunk == "":
            current_chunk = sentence
            current_nb_of_tokens_in_chunk = nb_tokens
            continue

        if current_nb_of_tokens_in_chunk + nb_tokens > max_nb_tokens_in_a_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_nb_of_tokens_in_chunk = nb_tokens
        else:
            current_chunk += " " + sentence
            current_nb_of_tokens_in_chunk += nb_tokens

    if current_chunk != "":
        chunks.append(current_chunk.strip())

    return chunks
