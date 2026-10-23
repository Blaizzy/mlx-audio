from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Any, Iterable, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.codec.models.mimi.modules.conv import ConvTranspose1d
from mlx_audio.tts.models.base import GenerationResult
from mlx_audio.utils import load_audio, load_weights

from .conditioners import ContinuousConditioner, SpeakerCountConditioner, TokenizedText
from .config import ModelConfig
from .flow_lm import FlowLMModel
from .mimi import MimiAdapter
from .utils import PREDEFINED_VOICES, download_if_necessary, load_predefined_voice
from .weights import map_weights

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
            config.flow_lm,
            latent_dim=config.flow_lm.ldim or config.mimi.quantizer.dimension,
        )
        self.mimi = MimiAdapter.from_config(config.mimi)
        for conditioner in config.conditioners:
            if conditioner["name"] == "num_speakers" and conditioner["type"] == "lut":
                lut = conditioner["lut"]
                self.flow_lm.num_speakers = SpeakerCountConditioner(
                    lut["n_bins"], lut["dim"], config.flow_lm.transformer.d_model
                )
            elif (
                conditioner["name"] == "duration_delta"
                and conditioner["type"] == "continuous"
            ):
                self.flow_lm.duration_delta = ContinuousConditioner(
                    output_dim=config.flow_lm.transformer.d_model,
                    **conditioner["continuous"],
                )
            else:
                raise ValueError(f"Unsupported conditioner: {conditioner['name']}")

        self.temp = config.temp
        self.lsd_decode_steps = config.lsd_decode_steps
        self.noise_clamp = config.noise_clamp
        self.eos_threshold = config.eos_threshold
        self._empty_conditioning = mx.zeros((1, 0, self.flow_lm.dim))
        # Private buffers are excluded from parameters(), which the loader evals.
        # Materialize this on its owning stream before an audio worker uses it.
        mx.eval(self._empty_conditioning)
        self._validate_sampling()

        if (
            config.fuser.get("cross")
            or config.fuser.get("streaming_sum")
            or set(config.fuser.get("sum", [])) - {"num_speakers", "duration_delta"}
        ):
            raise ValueError("Unsupported Pocket TTS conditioning fuser.")

        self.speaker_proj_weight = mx.zeros(
            (
                config.flow_lm.transformer.d_model,
                config.mimi.seanet.dimension,
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

    def sanitize(self, weights):
        # Conditioner padding is unused during inference.
        for key in list(weights):
            if key.endswith(".learnt_padding"):
                weights.pop(key)
        # Voice sidecars are loaded separately from the model parameters.
        for key in ("speaker_wavs", "emb", "audio_prompt"):
            weights.pop(key, None)
        weights = map_weights(self, weights)
        # Native inference packages can omit the audio encoder. Quantization is
        # handled by the standard loader using the exported config.
        if not any(key.startswith("mimi.encoder.") for key in weights):
            self.config.mimi.encoder_enabled = False
            self.mimi.encoder = self.mimi.encoder_transformer = self.mimi.downsample = (
                None
            )
            self.mimi.encoder_cache = []
        return weights

    def _validate_sampling(self):
        if not isinstance(self.lsd_decode_steps, int) or self.lsd_decode_steps <= 0:
            raise ValueError("lsd_decode_steps must be a positive integer.")
        if not math.isfinite(self.temp) or self.temp < 0:
            raise ValueError("temperature must be finite and >= 0.")
        if self.noise_clamp is not None and (
            not math.isfinite(self.noise_clamp) or self.noise_clamp < 0
        ):
            raise ValueError("noise_clamp must be finite and >= 0.")

    def load_weights(self, weights, strict: bool = True):
        m = super().load_weights(weights, strict=strict)

        def _filter_fn(module, name, _):
            if isinstance(module, ConvTranspose1d) and name == "weight":
                module.update_in_place()
            return True

        m.filter_and_map(_filter_fn)
        self.flow_lm.cache_conditioning()
        return m

    def load_from_config(self, strict: bool = True):
        from dataclasses import asdict

        from mlx_audio.utils import apply_quantization

        if self.config.model_path:
            weights = load_weights(Path(self.config.model_path))
        elif self.config.weights_path:
            weights = mx.load(download_if_necessary(self.config.weights_path))
        else:
            weights = {}
            for cfg in (self.config.flow_lm, self.config.mimi):
                if cfg and cfg.weights_path:
                    weights.update(mx.load(download_if_necessary(cfg.weights_path)))
            if not weights:
                raise ValueError("No weights_path configured for PocketTTS.")
        weights = self.sanitize(weights)
        apply_quantization(self, asdict(self.config), weights)
        return self.load_weights(list(weights.items()), strict=strict)

    def init_state(self) -> dict[str, Any]:
        return {"flow_cache": self.flow_lm.make_cache()}

    def _run_flow_lm(
        self,
        model_state: dict[str, Any],
        text_tokens: mx.array | None = None,
        backbone_input_latents: mx.array | None = None,
        audio_conditioning: mx.array | None = None,
    ) -> tuple[mx.array, mx.array] | None:
        text_embeddings = (
            self.flow_lm.conditioner(TokenizedText(text_tokens))
            if text_tokens is not None and text_tokens.shape[1]
            else self._empty_conditioning
        )
        if audio_conditioning is not None and audio_conditioning.shape[1]:
            text_embeddings = mx.concatenate(
                [text_embeddings, audio_conditioning], axis=1
            )
        if backbone_input_latents is None or backbone_input_latents.shape[1] == 0:
            # Prefill the transformer cache with float32 conditioning.
            text_embeddings = text_embeddings.astype(mx.float32)
            self.flow_lm.transformer(text_embeddings, model_state["flow_cache"])
            # Advance the sampling RNG once per prefill without evaluating a latent.
            mx.random.normal((1, self.flow_lm.ldim))
            return None
        output_embeddings, is_eos = self.flow_lm(
            sequence=backbone_input_latents,
            text_embeddings=text_embeddings,
            cache=model_state["flow_cache"],
            lsd_decode_steps=self.lsd_decode_steps,
            temp=self.temp,
            noise_clamp=self.noise_clamp,
            eos_threshold=self.eos_threshold,
        )
        return output_embeddings[:, None, :], is_eos

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
        elif isinstance(audio_conditioning, (str, Path)) and str(
            audio_conditioning
        ).split("@")[0].endswith(".safetensors"):
            weights = mx.load(download_if_necessary(str(audio_conditioning)))
            if "speaker_wavs" in weights:
                latents = weights["speaker_wavs"]
                if latents.ndim == 2:
                    latents = latents[None]
                prompt = (
                    latents.transpose(0, 2, 1).astype(mx.float32)
                    @ self.speaker_proj_weight.T
                )
            else:
                key = "emb" if "emb" in weights else "audio_prompt"
                prompt = weights[key]
        else:
            audio = self._load_audio(audio_conditioning)
            min_samples = int(self.config.audio_prompt_min_duration * self.sample_rate)
            if audio.shape[-1] < min_samples:
                raise ValueError(
                    f"Audio prompt must be at least {self.config.audio_prompt_min_duration} seconds."
                )
            if self.config.audio_prompt_max_duration is not None:
                audio = audio[
                    ...,
                    : int(self.config.audio_prompt_max_duration * self.sample_rate),
                ]
            if audio.shape[-1] == 0:
                prompt = mx.zeros((1, 0, self.flow_lm.dim))
            else:
                prompt = self._encode_audio(audio)
        if prompt.ndim == 2:
            prompt = prompt[None]
        if (
            prompt.ndim != 3
            or prompt.shape[0] != 1
            or prompt.shape[-1] != self.flow_lm.dim
        ):
            raise ValueError(
                f"Voice embedding must have shape [1, T, {self.flow_lm.dim}], got {prompt.shape}."
            )
        model_state = self.init_state()
        if prompt.shape[1]:
            self._run_flow_lm(model_state=model_state, audio_conditioning=prompt)
        self._slice_flow_cache(model_state, prompt.shape[1])
        return model_state

    def generate_audio(
        self,
        model_state: dict[str, Any] | None,
        text_to_generate: str,
        frames_after_eos: int | None = None,
    ) -> mx.array:
        if model_state is None:
            model_state = self.get_state_for_audio_prompt(
                self._resolve_audio_prompt(None, None, False)
            )
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
            model_state = self.get_state_for_audio_prompt(
                self._resolve_audio_prompt(None, None, False)
            )
        prompt_num_frames = self._get_flow_cache_num_frames(model_state)
        chunks = split_into_best_sentences(
            self.flow_lm.conditioner.tokenizer, text_to_generate
        )
        for chunk in chunks:
            self._slice_flow_cache(model_state, prompt_num_frames)
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
        self._run_flow_lm(model_state=model_state, text_tokens=prepared.tokens)

        backbone_input = mx.full(
            (1, 1, self.flow_lm.ldim), float("NaN"), dtype=mx.float32
        )
        eos_step = None
        for step in range(max_gen_len):
            next_latent, is_eos = self._run_flow_lm(
                model_state=model_state, backbone_input_latents=backbone_input
            )
            if eos_step is None and bool(is_eos.item()):
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
        ref_audio: Optional[Union[str, mx.array]] = None,
        temperature: Optional[float] = None,
        verbose: bool = False,
        stream: bool = False,
        streaming_interval: float = 2.0,
        frames_after_eos: Optional[int] = None,
        **kwargs,
    ) -> Iterable[GenerationResult]:
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
            self._validate_sampling()
            if ref_audio is not None:
                ref_audio = load_audio(ref_audio, sample_rate=self.sample_rate)
            prompt = self._resolve_audio_prompt(voice, ref_audio, verbose)
            model_state = self.get_state_for_audio_prompt(prompt)
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
                    is_streaming_chunk=True,
                )
                segment_idx += 1
                buffer = []
                buffered_samples = 0
                start_time = time.perf_counter()

            if buffer:
                audio = mx.concatenate(buffer, axis=0) if len(buffer) > 1 else buffer[0]
            else:
                # The last audio already went out as a partial chunk.
                audio = mx.zeros((0,), dtype=mx.float32)
            yield _build_generation_result(
                audio=audio,
                sample_rate=self.sample_rate,
                start_time=start_time,
                segment_idx=segment_idx,
                token_count=token_count,
                is_streaming_chunk=True,
                is_final_chunk=True,
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

    def _get_flow_cache_num_frames(self, model_state: dict[str, Any]) -> int:
        caches = model_state.get("flow_cache", [])
        for cache in caches:
            if cache.keys is None:
                continue
            return min(cache.offset, cache.keys.shape[2])
        return 0

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

        if self.config.model_path:
            root = Path(self.config.model_path)
            voices = {
                p.stem.lower(): p
                for directory in (root / "voices", root / "embeddings")
                for p in sorted(directory.glob("*.safetensors"))
            }
            default_voice = root / "default-voice.safetensors"
            if default_voice.is_file():
                voices.setdefault("default", default_voice)
            if voice is None and voices:
                return voices.get(
                    "default",
                    voices.get(DEFAULT_AUDIO_PROMPT, next(iter(voices.values()))),
                )
            if voice and voice.lower() in voices:
                return voices[voice.lower()]
            if (
                voices
                and self.mimi.encoder is None
                and voice
                and not (
                    voice.startswith(("http://", "https://", "hf://"))
                    or os.path.exists(voice)
                )
            ):
                raise ValueError(
                    f"Unknown voice '{voice}'. Available voices: {', '.join(voices)}"
                )

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
    is_streaming_chunk: bool = False,
    is_final_chunk: bool = False,
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
        is_streaming_chunk=is_streaming_chunk,
        is_final_chunk=is_final_chunk,
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
        text = tokenizer.decode(list_of_tokens[start:end])
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
