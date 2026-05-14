from __future__ import annotations

import time
from pathlib import Path
from typing import Generator

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_audio.audio_io import read as read_audio
from mlx_audio.tts.models.base import GenerationResult
from mlx_audio.utils import resample_audio

from .audio_vae import AudioVAE
from .config import ModelConfig
from .convert import sanitize_weights
from .duration import estimate_speech_duration
from .gemma import encode_prompts_hidden_states, load_text_encoder
from .guidance import MultiModalGuiderParams, auto_rescale_for_cfg
from .latent import (
    AudioLatentTools,
    AudioPatchifier,
    add_gaussian_noise,
    append_reference_latent,
)
from .sampling import (
    guided_euler_loop,
    patch_long_clip_silence_prior,
    resolve_generation_duration,
    target_shape_for_duration,
)
from .text_conditioning import DramaboxTextConditioner
from .transformer import AudioOnlyLTXModel, X0Model
from .vocoder import build_dramabox_vocoder


def _log_mel_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    hop_length: int,
    n_fft: int = 1024,
    n_mels: int = 64,
) -> mx.array:
    from mlx_audio.dsp import hanning, mel_filters, stft

    waveform = mx.array(audio.astype(np.float32, copy=False))
    spectrogram = stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=hanning(n_fft, periodic=True),
        center=True,
        pad_mode="reflect",
    )
    magnitude = mx.abs(spectrogram)
    filters = mel_filters(
        sample_rate,
        n_fft,
        n_mels,
        f_min=0.0,
        f_max=sample_rate / 2.0,
        norm="slaney",
        mel_scale="slaney",
    )
    mel = magnitude @ filters.T
    return mx.log(mx.maximum(mel, 1e-5))


class Model(nn.Module):
    def __init__(self, config: ModelConfig | dict):
        super().__init__()
        self.config = (
            ModelConfig.from_dict(config) if isinstance(config, dict) else config
        )
        self.transformer = AudioOnlyLTXModel(self.config.transformer)
        self.text_conditioner = DramaboxTextConditioner(
            embedding_dim=self.config.text_encoder_hidden_size,
            audio_inner_dim=self.config.transformer.audio_cross_attention_dim,
            num_gemma_layers=49,
            connector_layers=self.config.transformer.connector_num_layers,
            connector_heads=self.config.transformer.audio_connector_num_attention_heads,
            connector_head_dim=self.config.transformer.audio_connector_attention_head_dim,
            connector_num_learnable_registers=(
                self.config.transformer.connector_num_learnable_registers
            ),
        )
        self.audio_vae = AudioVAE()
        self.vocoder = build_dramabox_vocoder()
        self._text_encoder = None
        self._tokenizer = None
        self._text_encoder_id = None

    @property
    def sample_rate(self) -> int:
        return self.config.audio.sample_rate

    @property
    def model_type(self) -> str:
        return self.config.model_type

    def estimate_duration(self, text: str, speed: float = 1.0) -> float:
        return estimate_speech_duration(text, speed=speed)

    def _ensure_text_encoder(self, model_id: str | None = None):
        model_id = model_id or self.config.mlx_text_encoder
        if (
            self._text_encoder is None
            or self._tokenizer is None
            or self._text_encoder_id != model_id
        ):
            self._text_encoder, self._tokenizer = load_text_encoder(model_id)
            self._text_encoder_id = model_id
        return self._text_encoder, self._tokenizer

    def _encode_prompt_contexts(
        self,
        prompts: list[str],
        text_encoder_id: str | None = None,
    ) -> list[tuple[mx.array, mx.array]]:
        text_encoder, tokenizer = self._ensure_text_encoder(text_encoder_id)
        encoded = encode_prompts_hidden_states(text_encoder, tokenizer, prompts)
        contexts = []
        for item in encoded:
            context, mask = self.text_conditioner(
                item.hidden_states,
                item.attention_mask,
            )
            contexts.append((context, mask))
        return contexts

    def _encode_voice_reference(self, voice_ref: str | Path) -> mx.array:
        audio, sample_rate = read_audio(voice_ref, always_2d=True, dtype="float32")
        audio = audio.astype(np.float32)
        if audio.shape[1] == 1:
            audio = np.repeat(audio, 2, axis=1)
        elif audio.shape[1] > 2:
            audio = audio[:, :2]
        if sample_rate != self.config.audio.latent_sample_rate:
            audio = resample_audio(
                audio,
                sample_rate,
                self.config.audio.latent_sample_rate,
                axis=0,
            )
        max_samples = int(
            self.config.inference_defaults.ref_duration
            * self.config.audio.latent_sample_rate
        )
        if audio.shape[0] < max_samples:
            repeats = (max_samples // max(audio.shape[0], 1)) + 1
            audio = np.tile(audio, (repeats, 1))
        audio = audio[:max_samples]
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio * ((10 ** (-4.0 / 20.0)) / peak)
        mel_channels = []
        for idx in range(2):
            mel_channels.append(
                _log_mel_spectrogram(
                    audio[:, idx],
                    sample_rate=self.config.audio.latent_sample_rate,
                    hop_length=self.config.audio.hop_length,
                )
            )
        spectrogram = mx.stack(mel_channels, axis=0)[None]
        return self.audio_vae.encode(spectrogram.astype(mx.bfloat16))

    def generate(self, text: str, **kwargs) -> Generator[GenerationResult, None, None]:
        start = time.time()
        defaults = self.config.inference_defaults
        cfg_scale = float(kwargs.get("cfg_scale", defaults.cfg_scale))
        stg_scale = float(kwargs.get("stg_scale", defaults.stg_scale))
        steps = int(kwargs.get("steps", defaults.steps))
        speed = float(kwargs.get("speed", 1.0))
        gen_duration = float(kwargs.get("gen_duration", kwargs.get("duration", 0.0)))
        duration = resolve_generation_duration(
            text,
            speed=speed,
            duration_multiplier=float(
                kwargs.get("duration_multiplier", defaults.duration_multiplier)
            ),
            gen_duration=gen_duration,
        )
        pad_start = float(kwargs.get("pad_start", 0.0))
        shape = target_shape_for_duration(duration + pad_start, self.config.audio)
        tools = AudioLatentTools(AudioPatchifier(), shape)
        state = tools.create_initial_state(dtype=mx.bfloat16)
        voice_ref = kwargs.get("voice_ref", kwargs.get("ref_audio", None))
        if voice_ref is not None:
            reference_latent = self._encode_voice_reference(voice_ref)
            state = append_reference_latent(state, tools, reference_latent)
        state = add_gaussian_noise(
            state,
            seed=int(kwargs.get("seed", defaults.seed)),
            noise_scale=1.0,
        )

        prompts = [text]
        if cfg_scale > 1.0:
            prompts.append(str(kwargs.get("negative_prompt", defaults.negative_prompt)))
        contexts = self._encode_prompt_contexts(
            prompts,
            text_encoder_id=kwargs.get(
                "mlx_text_encoder",
                kwargs.get("text_encoder_model", self.config.mlx_text_encoder),
            ),
        )
        context, context_mask = contexts[0]
        negative_context = contexts[1][0] if cfg_scale > 1.0 else None
        negative_context_mask = contexts[1][1] if cfg_scale > 1.0 else None
        # The reference Dramabox denoiser uses prompt masks only inside the
        # PromptEncoder/EmbeddingsProcessor. The DiT cross-attention receives
        # the compacted context without an additional mask; passing one here
        # audibly degrades generation even when it is all valid tokens.
        context_mask = None
        negative_context_mask = None

        rescale_scale = kwargs.get("rescale_scale", defaults.rescale_scale)
        rescale = (
            auto_rescale_for_cfg(cfg_scale)
            if rescale_scale == "auto"
            else float(rescale_scale)
        )
        guider = MultiModalGuiderParams(
            cfg_scale=cfg_scale,
            stg_scale=stg_scale,
            stg_blocks=(int(kwargs.get("stg_block", defaults.stg_block)),),
            rescale_scale=rescale,
            modality_scale=float(kwargs.get("modality_scale", defaults.modality_scale)),
        )
        denoised_state = guided_euler_loop(
            state=state,
            x0_model=X0Model(self.transformer),
            context=context,
            negative_context=negative_context,
            context_mask=context_mask,
            negative_context_mask=negative_context_mask,
            steps=steps,
            guider_params=guider,
        )
        denoised_state = tools.clear_conditioning(denoised_state)
        latents = tools.unpatchify(denoised_state).latent
        latents = patch_long_clip_silence_prior(latents)
        mel = self.audio_vae.decode(latents)
        waveform = self.vocoder(mel)
        audio = waveform[0].transpose(1, 0).astype(mx.float32)
        if pad_start > 0:
            trim_samples = int(pad_start * self.sample_rate)
            audio = audio[trim_samples:]
        samples = audio.shape[0]
        elapsed = time.time() - start
        duration_seconds = samples / self.sample_rate
        duration_str = (
            f"{int(duration_seconds // 3600):02d}:"
            f"{int(duration_seconds % 3600 // 60):02d}:"
            f"{int(duration_seconds % 60):02d}."
            f"{int((duration_seconds % 1) * 1000):03d}"
        )
        yield GenerationResult(
            audio=audio,
            samples=samples,
            sample_rate=self.sample_rate,
            segment_idx=0,
            token_count=shape.token_count(),
            audio_duration=duration_str,
            real_time_factor=duration_seconds / elapsed if elapsed > 0 else 0.0,
            prompt={
                "tokens": shape.token_count(),
                "tokens-per-sec": (
                    round(shape.token_count() / elapsed, 2) if elapsed > 0 else 0.0
                ),
            },
            audio_samples={
                "samples": samples,
                "samples-per-sec": round(samples / elapsed, 2) if elapsed > 0 else 0.0,
            },
            processing_time_seconds=elapsed,
            peak_memory_usage=mx.get_peak_memory() / 1e9,
        )

    def sanitize(self, weights):
        return sanitize_weights(weights)
