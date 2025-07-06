import math
from functools import lru_cache
from typing import Optional

import mlx.core as mx

# Common window functions


@lru_cache(maxsize=None)
def hanning(size):
    return mx.array(
        [0.5 * (1 - math.cos(2 * math.pi * n / (size - 1))) for n in range(size)]
    )


@lru_cache(maxsize=None)
def hamming(size):
    return mx.array(
        [0.54 - 0.46 * math.cos(2 * math.pi * n / (size - 1)) for n in range(size)]
    )


@lru_cache(maxsize=None)
def blackman(size):
    return mx.array(
        [
            0.42
            - 0.5 * math.cos(2 * math.pi * n / (size - 1))
            + 0.08 * math.cos(4 * math.pi * n / (size - 1))
            for n in range(size)
        ]
    )


@lru_cache(maxsize=None)
def bartlett(size):
    return mx.array([1 - 2 * abs(n - (size - 1) / 2) / (size - 1) for n in range(size)])


STR_TO_WINDOW_FN = {
    "hann": hanning,
    "hanning": hanning,
    "hamming": hamming,
    "blackman": blackman,
    "bartlett": bartlett,
}

# STFT and ISTFT


def stft(
    x,
    n_fft=800,
    hop_length=None,
    win_length=None,
    window: mx.array | str = "hann",
    center=True,
    pad_mode="reflect",
):
    if hop_length is None:
        hop_length = n_fft // 4
    if win_length is None:
        win_length = n_fft

    if isinstance(window, str):
        window_fn = STR_TO_WINDOW_FN.get(window.lower())
        if window_fn is None:
            raise ValueError(f"Unknown window function: {window}")
        w = window_fn(win_length)
    else:
        w = window

    if w.shape[0] < n_fft:
        pad_size = n_fft - w.shape[0]
        w = mx.concatenate([w, mx.zeros((pad_size,))], axis=0)

    def _pad(x, padding, pad_mode="reflect"):
        if pad_mode == "constant":
            return mx.pad(x, [(padding, padding)])
        elif pad_mode == "reflect":
            prefix = x[1 : padding + 1][::-1]
            suffix = x[-(padding + 1) : -1][::-1]
            return mx.concatenate([prefix, x, suffix])
        else:
            raise ValueError(f"Invalid pad_mode {pad_mode}")

    if center:
        x = _pad(x, n_fft // 2, pad_mode)

    num_frames = 1 + (x.shape[0] - n_fft) // hop_length
    if num_frames <= 0:
        raise ValueError(
            f"Input is too short (length={x.shape[0]}) for n_fft={n_fft} with "
            f"hop_length={hop_length} and center={center}."
        )

    shape = (num_frames, n_fft)
    strides = (hop_length, 1)
    frames = mx.as_strided(x, shape=shape, strides=strides)
    return mx.fft.rfft(frames * w)


def istft(
    x,
    hop_length=None,
    win_length=None,
    window="hann",
    center=True,
    length=None,
):
    if win_length is None:
        win_length = (x.shape[1] - 1) * 2
    if hop_length is None:
        hop_length = win_length // 4

    if isinstance(window, str):
        window_fn = STR_TO_WINDOW_FN.get(window.lower())
        if window_fn is None:
            raise ValueError(f"Unknown window function: {window}")
        w = window_fn(win_length + 1)[:-1]
    else:
        w = window

    if w.shape[0] < win_length:
        w = mx.concatenate([w, mx.zeros((win_length - w.shape[0],))], axis=0)

    num_frames = x.shape[1]
    t = (num_frames - 1) * hop_length + win_length

    reconstructed = mx.zeros(t)
    window_sum = mx.zeros(t)

    # inverse FFT of each frame
    frames_time = mx.fft.irfft(x, axis=0).transpose(1, 0)

    # get the position in the time-domain signal to add the frame
    frame_offsets = mx.arange(num_frames) * hop_length
    indices = frame_offsets[:, None] + mx.arange(win_length)
    indices_flat = indices.flatten()

    updates_reconstructed = (frames_time * w).flatten()
    updates_window = mx.tile(w, (num_frames,)).flatten()

    # overlap-add the inverse transformed frame, scaled by the window
    reconstructed = reconstructed.at[indices_flat].add(updates_reconstructed)
    window_sum = window_sum.at[indices_flat].add(updates_window)

    # normalize by the sum of the window values
    reconstructed = mx.where(window_sum != 0, reconstructed / window_sum, reconstructed)

    if center and length is None:
        reconstructed = reconstructed[win_length // 2 : -win_length // 2]

    if length is not None:
        reconstructed = reconstructed[:length]

    return reconstructed


# Mel filterbank


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
