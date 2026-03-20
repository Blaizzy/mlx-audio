from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import mlx.core as mx

from mlx_audio.codec.models.fish_s1_dac import DAC


@dataclass
class PCAState:
    pca_components: mx.array
    pca_mean: mx.array
    latent_scale: float


def load_pca_state(path: str) -> PCAState:
    t = mx.load(path)
    latent_scale = float(mx.array(t["latent_scale"]).item())
    return PCAState(
        pca_components=mx.array(t["pca_components"]),
        pca_mean=mx.array(t["pca_mean"]),
        latent_scale=latent_scale,
    )


def ae_encode(fish_ae: DAC, pca_state: PCAState, audio: mx.array) -> mx.array:
    # audio: [batch, 1, samples]
    z_q = fish_ae.encode_zq(audio).astype(mx.float32)  # [B, 1024, T]
    z_q = z_q.transpose(0, 2, 1)  # [B, T, 1024]
    z_q = (z_q - pca_state.pca_mean) @ pca_state.pca_components.T
    z_q = z_q * pca_state.latent_scale
    return z_q


def ae_decode(fish_ae: DAC, pca_state: PCAState, z_q: mx.array) -> mx.array:
    # z_q: [B, T, 80]
    z_q = (z_q / pca_state.latent_scale) @ pca_state.pca_components + pca_state.pca_mean
    z_q = z_q.transpose(0, 2, 1)  # [B, 1024, T]
    return fish_ae.decode_zq(z_q.astype(mx.float32)).astype(mx.float32)


def find_flattening_point(
    data: mx.array,
    target_value: float = 0.0,
    window_size: int = 20,
    std_threshold: float = 0.05,
) -> int:
    # data: [T, 80]
    padded = mx.concatenate(
        [data, mx.zeros((window_size, data.shape[-1]), dtype=data.dtype)], axis=0
    )
    for i in range(int(padded.shape[0] - window_size)):
        window = padded[i : i + window_size]
        if (
            float(window.std()) < std_threshold
            and abs(float(window.mean()) - target_value) < 0.1
        ):
            return i
    return int(data.shape[0])


def crop_audio_to_flattening_point(audio: mx.array, latent: mx.array) -> mx.array:
    # audio: [B, 1, samples], latent: [T, 80]
    flattening_point = find_flattening_point(latent)
    return audio[..., : flattening_point * 2048]


def get_speaker_latent_and_mask(
    fish_ae: DAC,
    pca_state: PCAState,
    audio: mx.array,  # [1, samples]
    max_speaker_latent_length: int = 6400,
    audio_chunk_size: int = 640 * 2048,
    pad_to_max: bool = False,
    divis_by_patch_size: int | None = 4,
) -> Tuple[mx.array, mx.array]:
    ae_downsample_factor = 2048
    max_audio_len = max_speaker_latent_length * ae_downsample_factor

    audio = audio[:, :max_audio_len]
    latent_arr = []

    for i in range(0, int(audio.shape[1]), audio_chunk_size):
        audio_chunk = audio[:, i : i + audio_chunk_size]
        if audio_chunk.shape[1] < audio_chunk_size:
            pad = audio_chunk_size - int(audio_chunk.shape[1])
            audio_chunk = mx.pad(audio_chunk, [(0, 0), (0, pad)])

        latent_chunk = ae_encode(fish_ae, pca_state, audio_chunk[:, None, :])
        latent_arr.append(latent_chunk)

    speaker_latent = (
        mx.concatenate(latent_arr, axis=1) if latent_arr else mx.zeros((1, 0, 80))
    )

    actual_latent_length = int(audio.shape[1]) // ae_downsample_factor
    speaker_mask = (
        mx.arange(speaker_latent.shape[1], dtype=mx.int32)[None, :]
        < actual_latent_length
    )

    if pad_to_max and speaker_latent.shape[1] < max_speaker_latent_length:
        pad_t = max_speaker_latent_length - int(speaker_latent.shape[1])
        speaker_latent = mx.pad(speaker_latent, [(0, 0), (0, pad_t), (0, 0)])
        speaker_mask = mx.pad(speaker_mask, [(0, 0), (0, pad_t)])
    elif not pad_to_max:
        speaker_latent = speaker_latent[:, :actual_latent_length]
        speaker_mask = speaker_mask[:, :actual_latent_length]

    if divis_by_patch_size is not None and speaker_latent.shape[1] > 0:
        limit = (
            int(speaker_latent.shape[1]) // divis_by_patch_size
        ) * divis_by_patch_size
        speaker_latent = speaker_latent[:, :limit]
        speaker_mask = speaker_mask[:, :limit]

    return speaker_latent, speaker_mask
