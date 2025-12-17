# Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

import logging
from functools import lru_cache
from typing import Union

import mlx.core as mx
import numpy as np
from librosa.filters import mel as librosa_mel_fn

logger = logging.getLogger(__name__)


@lru_cache(maxsize=10)
def get_mel_basis(
    n_fft: int,
    num_mels: int,
    sampling_rate: int,
    fmin: int,
    fmax: int,
) -> np.ndarray:
    """Get cached mel filter bank."""
    mel = librosa_mel_fn(
        sr=sampling_rate,
        n_fft=n_fft,
        n_mels=num_mels,
        fmin=fmin,
        fmax=fmax,
    )
    return mel.astype(np.float32)


@lru_cache(maxsize=10)
def get_hann_window(win_size: int) -> np.ndarray:
    """Get cached Hann window."""
    return np.hanning(win_size).astype(np.float32)


def dynamic_range_compression(
    x: np.ndarray, C: float = 1, clip_val: float = 1e-5
) -> np.ndarray:
    """Apply dynamic range compression."""
    return np.log(np.maximum(x, clip_val) * C)


def spectral_normalize(magnitudes: np.ndarray) -> np.ndarray:
    """Normalize spectral magnitudes."""
    return dynamic_range_compression(magnitudes)


def mel_spectrogram(
    y: Union[np.ndarray, mx.array],
    n_fft: int = 1920,
    num_mels: int = 80,
    sampling_rate: int = 24000,
    hop_size: int = 480,
    win_size: int = 1920,
    fmin: int = 0,
    fmax: int = 8000,
    center: bool = False,
) -> mx.array:
    """
    Compute mel-spectrogram from audio waveform.

    Args:
        y: Audio waveform, shape (samples,) or (batch, samples)
        n_fft: FFT size
        num_mels: Number of mel bands
        sampling_rate: Sample rate
        hop_size: Hop size
        win_size: Window size
        fmin: Minimum frequency
        fmax: Maximum frequency
        center: Whether to pad signal

    Returns:
        Mel-spectrogram of shape (batch, num_mels, time)
    """
    if isinstance(y, mx.array):
        y = np.array(y)

    if y.ndim == 1:
        y = y[None, :]

    # Check for audio clipping
    min_val = np.min(y)
    max_val = np.max(y)
    if min_val < -1.0 or max_val > 1.0:
        logger.warning(
            f"Audio values outside normalized range: min={min_val:.4f}, max={max_val:.4f}"
        )

    # Get mel basis and window
    mel_basis = get_mel_basis(n_fft, num_mels, sampling_rate, fmin, fmax)
    window = get_hann_window(win_size)

    # Pad signal
    pad_amount = (n_fft - hop_size) // 2
    y_padded = np.pad(y, ((0, 0), (pad_amount, pad_amount)), mode="reflect")

    # Compute STFT using numpy
    batch_size = y_padded.shape[0]
    n_frames = 1 + (y_padded.shape[1] - n_fft) // hop_size
    specs = []

    for b in range(batch_size):
        # Frame the signal
        frames = np.lib.stride_tricks.sliding_window_view(y_padded[b], n_fft)[
            ::hop_size
        ][:n_frames]

        # Apply window and FFT
        windowed = frames * window
        spec_complex = np.fft.rfft(windowed, n=n_fft, axis=-1)

        # Get magnitude
        spec_mag = np.abs(spec_complex).T  # (n_freq, n_frames)
        specs.append(spec_mag)

    specs = np.stack(specs, axis=0)  # (batch, n_freq, n_frames)

    # Apply mel filterbank
    mel_spec = np.einsum("mf,bft->bmt", mel_basis, specs)

    # Normalize
    mel_spec = spectral_normalize(mel_spec)

    return mx.array(mel_spec)
