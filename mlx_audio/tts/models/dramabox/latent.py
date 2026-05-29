from __future__ import annotations

from dataclasses import dataclass, replace
from typing import NamedTuple

import mlx.core as mx


class AudioLatentShape(NamedTuple):
    batch: int
    channels: int
    frames: int
    mel_bins: int

    def to_mlx_shape(self) -> tuple[int, int, int, int]:
        return (self.batch, self.channels, self.frames, self.mel_bins)

    def token_count(self) -> int:
        return self.frames

    def mask_shape(self) -> "AudioLatentShape":
        return self._replace(channels=1, mel_bins=1)

    @staticmethod
    def from_duration(
        batch: int,
        duration: float,
        channels: int = 8,
        mel_bins: int = 16,
        sample_rate: int = 16000,
        hop_length: int = 160,
        audio_latent_downsample_factor: int = 4,
    ) -> "AudioLatentShape":
        latents_per_second = (
            float(sample_rate)
            / float(hop_length)
            / float(audio_latent_downsample_factor)
        )
        return AudioLatentShape(
            batch=batch,
            channels=channels,
            frames=round(duration * latents_per_second),
            mel_bins=mel_bins,
        )


@dataclass(frozen=True)
class LatentState:
    latent: mx.array
    denoise_mask: mx.array
    positions: mx.array
    clean_latent: mx.array
    attention_mask: mx.array | None = None


@dataclass(frozen=True)
class AudioPatchifier:
    patch_size: int = 1
    sample_rate: int = 16000
    hop_length: int = 160
    audio_latent_downsample_factor: int = 4
    is_causal: bool = True
    shift: int = 0

    def get_token_count(self, shape: AudioLatentShape) -> int:
        return shape.frames

    def patchify(self, audio_latents: mx.array) -> mx.array:
        batch, channels, frames, mel_bins = audio_latents.shape
        return audio_latents.transpose(0, 2, 1, 3).reshape(
            batch, frames, channels * mel_bins
        )

    def unpatchify(
        self, audio_latents: mx.array, output_shape: AudioLatentShape
    ) -> mx.array:
        batch, frames, channels_mels = audio_latents.shape
        if batch != output_shape.batch or frames != output_shape.frames:
            raise ValueError(
                "Patched latent shape does not match requested output shape: "
                f"{audio_latents.shape} vs {output_shape}"
            )
        expected = output_shape.channels * output_shape.mel_bins
        if channels_mels != expected:
            raise ValueError(f"Expected last dim {expected}, got {channels_mels}")
        return audio_latents.reshape(
            batch, frames, output_shape.channels, output_shape.mel_bins
        ).transpose(0, 2, 1, 3)

    def _get_audio_latent_time_in_sec(
        self,
        start_latent: int,
        end_latent: int,
        dtype=mx.float32,
    ) -> mx.array:
        audio_latent_frame = mx.arange(start_latent, end_latent, dtype=dtype)
        audio_mel_frame = audio_latent_frame * self.audio_latent_downsample_factor
        if self.is_causal:
            audio_mel_frame = mx.maximum(
                audio_mel_frame + 1 - self.audio_latent_downsample_factor, 0
            )
        return audio_mel_frame * self.hop_length / self.sample_rate

    def get_patch_grid_bounds(self, output_shape: AudioLatentShape) -> mx.array:
        start = self._get_audio_latent_time_in_sec(
            self.shift, output_shape.frames + self.shift
        )
        end = self._get_audio_latent_time_in_sec(
            self.shift + 1, output_shape.frames + self.shift + 1
        )
        start = mx.broadcast_to(
            start[None, None, :], (output_shape.batch, 1, output_shape.frames)
        )
        end = mx.broadcast_to(
            end[None, None, :], (output_shape.batch, 1, output_shape.frames)
        )
        return mx.stack([start, end], axis=-1)


@dataclass(frozen=True)
class AudioLatentTools:
    patchifier: AudioPatchifier
    target_shape: AudioLatentShape

    def create_initial_state(
        self,
        dtype=mx.float32,
        initial_latent: mx.array | None = None,
    ) -> LatentState:
        if initial_latent is None:
            initial_latent = mx.zeros(self.target_shape.to_mlx_shape(), dtype=dtype)
        elif initial_latent.shape != self.target_shape.to_mlx_shape():
            raise ValueError(
                f"Latent shape {initial_latent.shape} does not match {self.target_shape}"
            )

        denoise_mask = mx.ones(
            self.target_shape.mask_shape().to_mlx_shape(), dtype=mx.float32
        )
        positions = self.patchifier.get_patch_grid_bounds(self.target_shape).astype(
            dtype
        )
        return self.patchify(
            LatentState(
                latent=initial_latent,
                denoise_mask=denoise_mask,
                positions=positions,
                clean_latent=mx.array(initial_latent),
            )
        )

    def patchify(self, latent_state: LatentState) -> LatentState:
        return replace(
            latent_state,
            latent=self.patchifier.patchify(latent_state.latent),
            denoise_mask=self.patchifier.patchify(latent_state.denoise_mask),
            clean_latent=self.patchifier.patchify(latent_state.clean_latent),
        )

    def unpatchify(self, latent_state: LatentState) -> LatentState:
        return replace(
            latent_state,
            latent=self.patchifier.unpatchify(latent_state.latent, self.target_shape),
            denoise_mask=self.patchifier.unpatchify(
                latent_state.denoise_mask, self.target_shape.mask_shape()
            ),
            clean_latent=self.patchifier.unpatchify(
                latent_state.clean_latent, self.target_shape
            ),
        )

    def clear_conditioning(self, latent_state: LatentState) -> LatentState:
        num_tokens = self.patchifier.get_token_count(self.target_shape)
        return LatentState(
            latent=latent_state.latent[:, :num_tokens],
            denoise_mask=mx.ones_like(latent_state.denoise_mask[:, :num_tokens]),
            positions=latent_state.positions[:, :, :num_tokens],
            clean_latent=latent_state.clean_latent[:, :num_tokens],
            attention_mask=None,
        )


def add_gaussian_noise(state: LatentState, seed: int = 42, noise_scale: float = 1.0):
    mx.random.seed(seed)
    noise = mx.random.normal(state.latent.shape).astype(state.latent.dtype)
    scaled_mask = state.denoise_mask * noise_scale
    noised = noise * scaled_mask + state.latent * (1.0 - scaled_mask)
    return replace(state, latent=noised)


def append_reference_latent(
    latent_state: LatentState,
    latent_tools: AudioLatentTools,
    reference_latent: mx.array,
    strength: float = 1.0,
    position_offset: float = 0.5,
) -> LatentState:
    tokens = latent_tools.patchifier.patchify(reference_latent)
    ref_shape = AudioLatentShape(
        batch=reference_latent.shape[0],
        channels=reference_latent.shape[1],
        frames=reference_latent.shape[2],
        mel_bins=reference_latent.shape[3],
    )
    positions = latent_tools.patchifier.get_patch_grid_bounds(ref_shape)
    positions = (positions + position_offset).astype(latent_state.positions.dtype)
    denoise_mask = mx.full((*tokens.shape[:2], 1), 1.0 - strength, dtype=mx.float32)

    batch_size = tokens.shape[0]
    num_target = latent_state.latent.shape[1]
    num_ref = tokens.shape[1]
    total = num_target + num_ref
    mask = mx.zeros((batch_size, total, total), dtype=mx.float32)

    if latent_state.attention_mask is not None:
        mask[:, :num_target, :num_target] = latent_state.attention_mask
    else:
        mask[:, :num_target, :num_target] = 1.0
    mask[:, :num_target, num_target:] = 1.0
    mask[:, num_target:, num_target:] = 1.0

    return LatentState(
        latent=mx.concatenate([latent_state.latent, tokens], axis=1),
        denoise_mask=mx.concatenate([latent_state.denoise_mask, denoise_mask], axis=1),
        positions=mx.concatenate([latent_state.positions, positions], axis=2),
        clean_latent=mx.concatenate([latent_state.clean_latent, tokens], axis=1),
        attention_mask=mask,
    )
