"""Experimental, backend-independent RE-USE spectral analysis."""

import numpy as np


def stable_stft(audio, n_fft, hop_size, win_size, compress_factor):
    """Return compressed magnitude, phase, and a float32 synthesis window.

    Analysis uses NumPy's CPU float64 FFT. Only analytically real bins are
    assigned zero imaginary part; there is no near-zero phase threshold.
    This defines a new input convention, not exact NVIDIA CUDA parity.
    """
    audio = np.asarray(audio, dtype=np.float64)
    if n_fft % 2 or win_size % 2 or win_size > n_fft:
        raise ValueError(
            "Stable analysis requires even FFT/window sizes and win <= FFT"
        )
    # Mirror one half so the periodic window is exactly even about its center.
    half = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(win_size // 2 + 1) / win_size)
    window = np.concatenate((half, half[-2:0:-1]))
    padding = (n_fft - win_size) // 2
    analysis_window = np.pad(window, (padding, padding))
    padded = np.pad(audio, (n_fft // 2, n_fft // 2), mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(padded, n_fft)[::hop_size]
    spectrum = np.fft.rfft(frames * analysis_window, axis=-1).T
    # DC/Nyquist and a frame centered on either reflected boundary are real.
    spectrum.imag[[0, -1], :] = 0.0
    spectrum.imag[:, 0] = 0.0
    if (len(audio) - 1) % hop_size == 0:
        spectrum.imag[:, -1] = 0.0
    imaginary = np.where(spectrum.imag == 0, 0.0, spectrum.imag)
    real = np.where(spectrum.real == 0, 0.0, spectrum.real)
    phase = np.arctan2(imaginary, real).astype(np.float32)
    magnitude = np.abs(spectrum)
    if compress_factor in ("log1p", "relu_log1p", "signed_log1p"):
        magnitude = np.log1p(magnitude)
    else:
        magnitude = magnitude**compress_factor
    return magnitude.astype(np.float32), phase, window.astype(np.float32)
