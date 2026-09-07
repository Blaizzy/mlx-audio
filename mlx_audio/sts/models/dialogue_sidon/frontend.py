import math
from functools import lru_cache

import mlx.core as mx
import numpy as np


@lru_cache(maxsize=1)
def _filters():
    # Torchaudio's Kaldi mel bank: float64 scalar bounds, float32 tensor math.
    low = 1127.0 * math.log(1 + 20.0 / 700)
    high = 1127.0 * math.log(1 + 8000.0 / 700)
    delta = (high - low) / 81
    bins = np.arange(80, dtype=np.float32)[:, None]
    left = low + bins * delta
    center = low + (bins + 1) * delta
    right = low + (bins + 2) * delta
    mel = 1127 * np.log(1 + np.arange(256, dtype=np.float32) * 31.25 / 700)
    bank = np.maximum(
        0, np.minimum((mel - left) / (center - left), (right - mel) / (right - center))
    )
    bank = np.pad(bank, ((0, 0), (0, 1)))
    n = np.arange(400, dtype=np.float32)
    window = (0.5 - 0.5 * np.cos(n * np.float32(2 * math.pi / 399))) ** 0.85
    return mx.array(bank.T), mx.array(window)


def extract_features(waveform: mx.array) -> tuple[mx.array, mx.array]:
    """Padded/normalized 16 kHz mono waveform -> [1, frames, 160] and mask.

    This deliberately does not apply PCM scaling. Per-bin sample variance,
    epsilon 1e-5, and padding (rather than dropping) the odd frame match Sidon.
    """
    waveform = waveform.astype(mx.float32)
    if waveform.ndim != 1 or waveform.shape[0] < 560:
        raise ValueError("Feature extraction requires at least 560 mono samples")
    count = 1 + (waveform.shape[0] - 400) // 160
    indices = mx.arange(count)[:, None] * 160 + mx.arange(400)[None, :]
    frames = waveform[indices]
    frames = frames - mx.mean(frames, axis=1, keepdims=True)
    frames = mx.concatenate(
        (frames[:, :1] * 0.03, frames[:, 1:] - 0.97 * frames[:, :-1]), axis=1
    )
    filters, window = _filters()
    spectrum = mx.abs(mx.fft.rfft(frames * window, n=512, axis=-1)) ** 2
    feats = mx.log(mx.maximum(spectrum @ filters, 1.1920928955078125e-7))
    centered = feats - mx.mean(feats, axis=0, keepdims=True)
    variance = mx.sum(centered**2, axis=0, keepdims=True) / (count - 1)
    feats = centered * mx.rsqrt(variance + 1e-5)
    mask = mx.ones((count,), dtype=mx.bool_)
    if count % 2:
        feats = mx.pad(feats, ((0, 1), (0, 0)))
        mask = mx.pad(mask, ((0, 1),))
    return feats.reshape(1, -1, 160), mask[1::2][None, :]


def normalize_chunk(waveform: mx.array) -> mx.array:
    peak = mx.maximum(mx.max(mx.abs(waveform)), 1e-6)
    waveform = mx.pad(0.9 * waveform / peak, ((160, 160),))
    if waveform.shape[0] < 560:
        waveform = mx.pad(waveform, ((0, 560 - waveform.shape[0]),))
    return waveform


@lru_cache(maxsize=16)
def _resample_kernel(orig_freq: int, new_freq: int):
    base = min(orig_freq, new_freq) * 0.99
    width = math.ceil(6 * orig_freq / base)
    idx = mx.arange(-width, width + orig_freq, dtype=mx.float32) / orig_freq
    t = mx.arange(0, -new_freq, -1, dtype=mx.float32)[:, None] / new_freq + idx
    t = mx.clip(t * base, -6, 6)
    window = mx.cos(t * math.pi / 6 / 2) ** 2
    t = t * math.pi
    sinc = mx.sin(t) / mx.where(t == 0, 1, t)
    kernel = mx.where(t == 0, 1, sinc) * window * (base / orig_freq)
    return kernel[:, :, None], width


def resample(waveform: mx.array, sample_rate: int, target_rate: int = 16000):
    if sample_rate <= 0 or target_rate <= 0:
        raise ValueError("Sample rates must be positive")
    if sample_rate == target_rate:
        return waveform
    gcd = math.gcd(sample_rate, target_rate)
    old, new = sample_rate // gcd, target_rate // gcd
    kernel, width = _resample_kernel(old, new)
    padded = mx.pad(waveform.astype(mx.float32), ((width, width + old),))
    output = mx.conv1d(padded[None, :, None], kernel, stride=old).reshape(-1)
    return output[: math.ceil(waveform.shape[0] * new / old)]
