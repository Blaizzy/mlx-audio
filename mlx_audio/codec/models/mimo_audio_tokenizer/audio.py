"""MiMo's magnitude-mel frontend and same-padding waveform synthesis."""

from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_audio import audio_io
from mlx_audio.dsp import hanning, istft, mel_filters, stft
from mlx_audio.resample import resample_audio_array

from .config import ModelConfig


def load_waveform(audio, sample_rate: int | None, target_rate: int):
    """Accept a path or a mono/time-major multichannel waveform."""
    if isinstance(audio, (str, Path)):
        audio, sample_rate = audio_io.read(str(audio))
    elif sample_rate is None:
        raise ValueError("sample_rate is required for an in-memory waveform")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=-1)
    if audio.ndim != 1 or audio.size == 0:
        raise ValueError("Expected a nonempty mono or time-major waveform")
    if not np.isfinite(audio).all():
        raise ValueError("Audio contains non-finite samples")
    audio = resample_audio_array(audio, sample_rate, target_rate)
    return mx.array(audio)


def log_mel_spectrogram(audio: mx.array, config: ModelConfig) -> mx.array:
    # Reflect padding needs at least n_fft / 2 + 1 samples. Extend very short
    # clips explicitly, retaining the original length in callers when needed.
    if audio.ndim != 1 or audio.size == 0:
        raise ValueError("Expected a nonempty mono waveform")
    if audio.size <= config.nfft // 2:
        audio = mx.pad(audio, (0, config.nfft // 2 + 1 - audio.size))
    spectrum = stft(
        audio.astype(mx.float32),
        n_fft=config.nfft,
        hop_length=config.hop_length,
        win_length=config.window_size,
        window=hanning(config.window_size, periodic=True),
        center=True,
    )
    filters = mel_filters(
        config.sampling_rate,
        config.nfft,
        config.n_mels,
        f_min=config.fmin,
        f_max=config.fmax,
        norm=None,
        mel_scale="htk",
    )
    return mx.log(mx.maximum(mx.abs(spectrum) @ filters.T, 1e-7))


def same_istft(
    spectrum: mx.array, n_fft: int, hop_length: int, window=None
) -> mx.array:
    """(frames, frequencies) -> waveform, normalized by squared Hann window."""
    window = (
        hanning(n_fft, periodic=True) if window is None else window.astype(mx.float32)
    )
    wave = istft(
        spectrum.T,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        center=False,
        normalized=True,
    )
    trim = (n_fft - hop_length) // 2
    return wave[trim:-trim] if trim else wave
