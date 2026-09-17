"""Offline, two-speaker DialogueSidon separation on Apple Silicon."""

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx import nn

from mlx_audio.codec.models.descript.dac import Decoder

from .config import ModelConfig
from .diffusion import DiffusionHead, DPMSolver
from .encoder import Encoder
from .frontend import extract_features, normalize_chunk, resample


@dataclass
class SeparationResult:
    speakers: mx.array  # [2, samples], anonymous speaker slots
    sample_rate: int


def align_speakers(previous, current, overlap):
    """Match waveform channels by centered correlation in their overlap."""
    if overlap <= 0:
        return current
    a, b = previous[:, -overlap:], current[:, :overlap]
    a = a - a.mean(axis=-1, keepdims=True)
    b = b - b.mean(axis=-1, keepdims=True)
    norms = (
        mx.sqrt(mx.sum(a * a, axis=-1))[:, None]
        * mx.sqrt(mx.sum(b * b, axis=-1))[None, :]
    )
    correlations = mx.where(norms > 1e-8, (a @ b.T) / mx.maximum(norms, 1e-8), 0)
    swap = (
        correlations[0, 1] + correlations[1, 0]
        > correlations[0, 0] + correlations[1, 1]
    )
    return mx.where(swap, current[::-1], current)


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config.encoder)
        self.linear1 = nn.Linear(config.encoder.hidden_size, config.latent_dim)
        self.linear2 = nn.Linear(config.encoder.hidden_size, config.latent_dim)
        self.diffusion_head = DiffusionHead(
            config.diffusion,
            config.latent_dim * 2,
            config.encoder.hidden_size + config.latent_dim * 2,
        )
        self.decoder = Decoder(
            input_channel=config.latent_dim,
            channels=config.decoder_channels,
            rates=config.decoder_rates,
            d_out=1,
        )
        # Config-owned constants, not learned model parameters.
        self._latent_mean = mx.array(
            config.latent_norm_mean or [0.0] * (config.latent_dim * 2)
        )
        self._latent_std = mx.array(
            config.latent_norm_std or [1.0] * (config.latent_dim * 2)
        )

    @property
    def sample_rate(self):
        return self.config.sample_rate

    @property
    def dtype(self):
        return self.linear1.weight.dtype

    def normalize_latents(self, latents):
        if not self.config.latent_norm_initialized:
            return latents
        return (
            (latents.astype(mx.float32) - self._latent_mean) / self._latent_std
        ).astype(latents.dtype)

    def denormalize_latents(self, latents):
        if not self.config.latent_norm_initialized:
            return latents
        return (
            latents.astype(mx.float32) * self._latent_std + self._latent_mean
        ).astype(latents.dtype)

    def encode(self, input_features, attention_mask=None):
        features = self.encoder(input_features.astype(self.dtype), attention_mask)
        return features, self.linear1(features), self.linear2(features)

    def sample_latents(self, conditioning, num_steps=30, initial_noise=None, key=None):
        shape = (*conditioning.shape[:2], self.config.latent_dim * 2)
        if initial_noise is not None and initial_noise.shape != shape:
            raise ValueError(
                f"Expected initial_noise shape {shape}, got {initial_noise.shape}"
            )
        latents = (
            mx.random.normal(shape, dtype=self.dtype, key=key)
            if initial_noise is None
            else initial_noise.astype(self.dtype)
        )
        scheduler = DPMSolver(self.config.diffusion, num_steps)
        for t in scheduler.timesteps:
            timesteps = mx.full((shape[0],), int(t), dtype=mx.float32)
            prediction = self.diffusion_head(latents, timesteps, conditioning)
            latents = scheduler.step(prediction, latents)
            mx.eval(latents, scheduler.previous)
        return latents

    def predict_latents(self, waveform, num_steps=30, initial_noise=None, key=None):
        """One mono 16 kHz chunk -> normalized [1, frames, 2 * latent_dim]."""
        features, mask = extract_features(normalize_chunk(waveform))
        features, first, second = self.encode(features, mask)
        predicted = mx.concatenate((first, second), axis=-1)
        conditioning = mx.concatenate(
            (self.normalize_latents(predicted), features), axis=-1
        )
        mx.eval(conditioning)
        return self.sample_latents(conditioning, num_steps, initial_noise, key)

    def decode_latents(self, latents):
        """Normalized [batch, frames, 2 * latent_dim] -> [batch, 2, samples]."""
        latents = self.denormalize_latents(latents).astype(self.dtype)
        first, second = mx.split(latents, 2, axis=-1)
        # Sequential decoding limits decoder activation memory for long chunks.
        first = self.decoder(first)[:, :, 0]
        mx.eval(first)
        second = self.decoder(second)[:, :, 0]
        output = mx.stack((first, second), axis=1)
        mx.eval(output)
        return output

    def separate(
        self,
        audio,
        sample_rate=None,
        *,
        num_steps=30,
        seed=None,
        chunk_seconds=20.0,
        overlap_seconds=5.0,
    ) -> SeparationResult:
        """Separate a file or array into two mono waveforms at 24 kHz.

        Arrays must be [samples] or [samples, channels], with sample_rate set.
        chunk_seconds=None enables whole-file inference. Long files use the
        published demo's channel alignment and waveform crossfade. A seed is
        local to this call and does not reset MLX's global random stream.
        """
        DPMSolver(self.config.diffusion, num_steps)  # Validate before model work.
        if chunk_seconds is not None and (
            not np.isfinite(chunk_seconds)
            or chunk_seconds <= 0
            or not np.isfinite(overlap_seconds)
            or not 0 <= overlap_seconds < chunk_seconds
        ):
            raise ValueError("Require chunk_seconds > overlap_seconds >= 0")
        if isinstance(audio, (str, Path)):
            from mlx_audio import audio_io

            audio, sample_rate = audio_io.read(str(audio))
        if sample_rate is None or int(sample_rate) != sample_rate or sample_rate <= 0:
            raise ValueError("A positive integer sample_rate is required for arrays")
        waveform = mx.array(audio).astype(mx.float32)
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=-1)
        if waveform.ndim != 1 or waveform.shape[0] == 0:
            raise ValueError("Expected nonempty [samples] or [samples, channels] audio")
        if not mx.all(mx.isfinite(waveform)).item():
            raise ValueError("Audio must contain only finite values")
        output_length = max(
            1, round(waveform.shape[0] * self.sample_rate / sample_rate)
        )
        waveform = resample(waveform, int(sample_rate))
        total = waveform.shape[0]
        chunk = (
            max(1, int(chunk_seconds * 16000)) if chunk_seconds is not None else total
        )
        overlap = int(overlap_seconds * 16000) if chunk_seconds is not None else 0
        hop = chunk - overlap
        key = mx.random.key(seed) if seed is not None else None
        stitched = None
        previous_end = 0
        for start in range(0, total, hop):
            end = min(start + chunk, total)
            if key is not None:
                key, chunk_key = mx.random.split(key)
            else:
                chunk_key = None
            latent = self.predict_latents(waveform[start:end], num_steps, key=chunk_key)
            predicted = self.decode_latents(latent)[0].astype(mx.float32)
            target = max(1, round((end - start) * self.sample_rate / 16000))
            predicted = self._match_length(predicted, target)
            if stitched is None:
                stitched = predicted
            else:
                count = min(
                    max(0, round((previous_end - start) * self.sample_rate / 16000)),
                    stitched.shape[-1],
                    predicted.shape[-1],
                )
                if count:
                    predicted = align_speakers(stitched, predicted, count)
                    fade = mx.linspace(0, 1, count)[None, :]
                    blended = (
                        stitched[:, -count:] * (1 - fade) + predicted[:, :count] * fade
                    )
                    stitched = mx.concatenate(
                        (stitched[:, :-count], blended, predicted[:, count:]), axis=-1
                    )
                else:
                    stitched = mx.concatenate((stitched, predicted), axis=-1)
            previous_end = end
            mx.eval(stitched)
        return SeparationResult(
            self._match_length(stitched, output_length), self.sample_rate
        )

    @staticmethod
    def _match_length(audio, length):
        if audio.shape[-1] < length:
            return mx.pad(audio, ((0, 0), (0, length - audio.shape[-1])))
        return audio[:, :length]

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        from mlx_audio.sts.utils import load_model

        return load_model(path, **kwargs)
