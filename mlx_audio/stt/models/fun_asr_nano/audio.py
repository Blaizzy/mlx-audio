from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple, Union

import mlx.core as mx
import numpy as np

from mlx_audio.stt.utils import load_audio

from .config import FrontendConfig


def compute_fbank(waveform: mx.array, config: FrontendConfig) -> mx.array:
    from mlx_audio.dsp import compute_fbank_kaldi

    win_len = int(config.fs * config.frame_length / 1000)
    win_inc = int(config.fs * config.frame_shift / 1000)
    return compute_fbank_kaldi(
        waveform * (1 << 15),
        sample_rate=config.fs,
        win_len=win_len,
        win_inc=win_inc,
        num_mels=config.n_mels,
        win_type=config.window,
        preemphasis=0.97,
        dither=0.0,
        snip_edges=True,
        low_freq=20.0,
        high_freq=0.0,
    )


def apply_lfr(feats: mx.array, lfr_m: int = 7, lfr_n: int = 6) -> mx.array:
    if lfr_m == 1 and lfr_n == 1:
        return feats

    T, D = feats.shape
    T_lfr = math.ceil(T / lfr_n)
    left_padding = (lfr_m - 1) // 2
    if left_padding > 0:
        feats = mx.concatenate([mx.tile(feats[:1], (left_padding, 1)), feats], axis=0)

    padded_T = feats.shape[0]
    frames = []
    for i in range(T_lfr):
        start = i * lfr_n
        end = start + lfr_m
        if end <= padded_T:
            frame = feats[start:end]
        else:
            available = feats[start:padded_T]
            pad = mx.tile(feats[-1:], (end - padded_T, 1))
            frame = mx.concatenate([available, pad], axis=0)
        frames.append(frame.reshape(lfr_m * D))
    return mx.stack(frames, axis=0)


def fake_token_length(speech_length: int) -> int:
    olens = 1 + (int(speech_length) - 3 + 2 * 1) // 2
    olens = 1 + (olens - 3 + 2 * 1) // 2
    return max(1, (olens - 1) // 2 + 1)


def prepare_audio(
    audio: Union[str, Path, mx.array, np.ndarray],
    config: FrontendConfig,
) -> Tuple[mx.array, mx.array, int]:
    if isinstance(audio, (str, Path)):
        audio_data = load_audio(str(audio), sr=config.fs)
    elif isinstance(audio, np.ndarray):
        audio_data = mx.array(audio.astype(np.float32, copy=False))
    elif isinstance(audio, mx.array):
        audio_data = audio.astype(mx.float32)
    else:
        raise TypeError(f"Unsupported audio type: {type(audio)}")

    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=-1)

    feats = compute_fbank(audio_data, config)
    feats = apply_lfr(feats, lfr_m=config.lfr_m, lfr_n=config.lfr_n)
    speech_len = int(feats.shape[0])

    return (
        feats[None, :, :],
        mx.array([speech_len], dtype=mx.int32),
        fake_token_length(speech_len),
    )
