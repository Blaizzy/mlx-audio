from __future__ import annotations

import math
from functools import lru_cache
from typing import Optional

import mlx.core as mx


@lru_cache(maxsize=None)
def mel_filters(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    f_min: float = 0,
    f_max: Optional[float] = None,
    norm: Optional[str] = None,
    mel_scale: str = "htk",
) -> mx.array:
    def hz_to_mel(freq, mel_scale="htk"):
        if mel_scale == "htk":
            return 2595.0 * math.log10(1.0 + freq / 700.0)

        # slaney scale
        f_min, f_sp = 0.0, 200.0 / 3
        mels = (freq - f_min) / f_sp
        min_log_hz = 1000.0
        min_log_mel = (min_log_hz - f_min) / f_sp
        logstep = math.log(6.4) / 27.0
        if freq >= min_log_hz:
            mels = min_log_mel + math.log(freq / min_log_hz) / logstep
        return mels

    def mel_to_hz(mels, mel_scale="htk"):
        if mel_scale == "htk":
            return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

        # slaney scale
        f_min, f_sp = 0.0, 200.0 / 3
        freqs = f_min + f_sp * mels
        min_log_hz = 1000.0
        min_log_mel = (min_log_hz - f_min) / f_sp
        logstep = math.log(6.4) / 27.0
        freqs = mx.where(
            mels >= min_log_mel,
            min_log_hz * mx.exp(logstep * (mels - min_log_mel)),
            freqs,
        )
        return freqs

    f_max = f_max or sample_rate / 2

    # generate frequency points

    n_freqs = n_fft // 2 + 1
    all_freqs = mx.linspace(0, sample_rate // 2, n_freqs)

    # convert frequencies to mel and back to hz

    m_min = hz_to_mel(f_min, mel_scale)
    m_max = hz_to_mel(f_max, mel_scale)
    m_pts = mx.linspace(m_min, m_max, n_mels + 2)
    f_pts = mel_to_hz(m_pts, mel_scale)

    # compute slopes for filterbank

    f_diff = f_pts[1:] - f_pts[:-1]
    slopes = mx.expand_dims(f_pts, 0) - mx.expand_dims(all_freqs, 1)

    # calculate overlapping triangular filters

    down_slopes = (-slopes[:, :-2]) / f_diff[:-1]
    up_slopes = slopes[:, 2:] / f_diff[1:]
    filterbank = mx.maximum(
        mx.zeros_like(down_slopes), mx.minimum(down_slopes, up_slopes)
    )

    if norm == "slaney":
        enorm = 2.0 / (f_pts[2 : n_mels + 2] - f_pts[:n_mels])
        filterbank *= mx.expand_dims(enorm, 0)

    filterbank = filterbank.moveaxis(0, 1)
    return filterbank


@lru_cache(maxsize=None)
def hanning(size):
    return mx.array(
        [0.5 * (1 - math.cos(2 * math.pi * n / (size - 1))) for n in range(size)]
    )


def stft(x, window, nperseg=256, noverlap=None, nfft=None, pad_mode="constant"):
    if nfft is None:
        nfft = nperseg
    if noverlap is None:
        noverlap = nfft // 4

    def _pad(x, padding, pad_mode="constant"):
        if pad_mode == "constant":
            return mx.pad(x, [(padding, padding)])
        elif pad_mode == "reflect":
            prefix = x[1 : padding + 1][::-1]
            suffix = x[-(padding + 1) : -1][::-1]
            return mx.concatenate([prefix, x, suffix])
        else:
            raise ValueError(f"Invalid pad_mode {pad_mode}")

    if window.shape[0] < nfft:
        window = mx.pad(window, (0, nfft - window.shape[0]))

    padding = nperseg // 2
    x = _pad(x, padding, pad_mode)

    strides = [noverlap, 1]
    t = (x.size - nperseg + noverlap) // noverlap
    shape = [t, nfft]
    x = mx.as_strided(x, shape=shape, strides=strides)
    return mx.fft.rfft(x * window)


def istft(x, window, nperseg=256, noverlap=None, nfft=None):
    if nfft is None:
        nfft = nperseg
    if noverlap is None:
        noverlap = nfft // 4

    t = (x.shape[0] - 1) * noverlap + nperseg
    reconstructed = mx.zeros(t)
    window_sum = mx.zeros(t)

    for i in range(x.shape[0]):
        # inverse FFT of each frame
        frame_time = mx.fft.irfft(x[i])

        # get the position in the time-domain signal to add the frame
        start = i * noverlap
        end = start + nperseg

        # overlap-add the inverse transformed frame, scaled by the window
        reconstructed[start:end] += frame_time * window
        window_sum[start:end] += window

    # normalize by the sum of the window values
    reconstructed = mx.where(window_sum != 0, reconstructed / window_sum, reconstructed)

    return reconstructed


def log_mel_spectrogram(
    audio: mx.array,
    sample_rate: int = 24_000,
    n_mels: int = 100,
    n_fft: int = 1024,
    hop_length: int = 256,
    padding: int = 0,
):
    if not isinstance(audio, mx.array):
        audio = mx.array(audio)

    if padding > 0:
        audio = mx.pad(audio, (0, padding))

    freqs = stft(audio, hanning(n_fft), nperseg=n_fft, noverlap=hop_length)
    magnitudes = freqs[:-1, :].abs()
    filters = mel_filters(
        sample_rate=sample_rate,
        n_fft=n_fft,
        n_mels=n_mels,
        norm=None,
        mel_scale="htk",
    )
    mel_spec = magnitudes @ filters.T
    log_spec = mx.maximum(mel_spec, 1e-5).log()
    return mx.expand_dims(log_spec, axis=0)
