"""Pure audio processing utilities - no TTS/STT imports.

This module contains only audio processing functions (window functions, STFT, mel filterbanks)
that can be imported without pulling in TTS or STT dependencies.
"""

import math

__all__ = [
    "hanning",
    "hamming",
    "blackman",
    "bartlett",
    "STR_TO_WINDOW_FN",
    "stft",
    "istft",
    "ISTFTCache",
    "mel_filters",
]
from functools import lru_cache
from typing import Optional

import mlx.core as mx


# Common window functions
@lru_cache(maxsize=None)
def hanning(size, periodic=False):
    """Hanning (Hann) window.

    Args:
        size: Window length
        periodic: If True, use periodic window (for spectral analysis)
    """
    denom = size if periodic else size - 1
    return mx.array(
        [0.5 * (1 - math.cos(2 * math.pi * n / denom)) for n in range(size)]
    )


@lru_cache(maxsize=None)
def hamming(size, periodic=False):
    """Hamming window.

    Args:
        size: Window length
        periodic: If True, use periodic window (for spectral analysis)
    """
    denom = size if periodic else size - 1
    return mx.array(
        [0.54 - 0.46 * math.cos(2 * math.pi * n / denom) for n in range(size)]
    )


@lru_cache(maxsize=None)
def blackman(size, periodic=False):
    """Blackman window."""
    denom = size if periodic else size - 1
    return mx.array(
        [
            0.42
            - 0.5 * math.cos(2 * math.pi * n / denom)
            + 0.08 * math.cos(4 * math.pi * n / denom)
            for n in range(size)
        ]
    )


@lru_cache(maxsize=None)
def bartlett(size, periodic=False):
    """Bartlett (triangular) window."""
    denom = size if periodic else size - 1
    return mx.array([1 - 2 * abs(n - denom / 2) / denom for n in range(size)])


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


class ISTFTCache:
    """
    Advanced caching for iSTFT operations. Fully vectorized Overlap-Add for MLX.
    Handles multiple configurations efficiently.
    Automatically caches normalization buffers and position indices for maximum performance.
    """

    def __init__(self):
        self.norm_buffer_cache = {}
        self.position_cache = {}

    def get_positions(self, num_frames: int, frame_length: int, hop_length: int):
        """Get cached position indices or create new ones"""
        key = (num_frames, frame_length, hop_length)

        if key not in self.position_cache:
            positions = (
                mx.arange(num_frames)[:, None] * hop_length
                + mx.arange(frame_length)[None, :]
            )
            self.position_cache[key] = positions.reshape(-1)

        return self.position_cache[key]

    def get_norm_buffer(
        self,
        n_fft: int,
        hop_length: int,
        win_length: int,
        window: mx.array,
        num_frames: int,
    ):
        """Get cached normalization buffer or create new one"""
        window_hash = hash(tuple(window.tolist()))
        key = (n_fft, hop_length, win_length, window_hash, num_frames)

        if key not in self.norm_buffer_cache:
            frame_length = window.shape[0]
            ola_len = (num_frames - 1) * hop_length + frame_length
            positions_flat = self.get_positions(num_frames, frame_length, hop_length)

            window_squared = window**2
            norm_buffer = mx.zeros(ola_len, dtype=mx.float32)
            window_sq_tiled = mx.tile(window_squared, num_frames)
            norm_buffer = norm_buffer.at[positions_flat].add(window_sq_tiled)
            norm_buffer = mx.maximum(norm_buffer, 1e-10)

            self.norm_buffer_cache[key] = norm_buffer

        return self.norm_buffer_cache[key]

    def istft(
        self,
        real_part: mx.array,
        imag_part: mx.array,
        n_fft: int,
        hop_length: int,
        win_length: int,
        window: mx.array,
        center: bool = True,
        audio_length: int = None,
    ) -> mx.array:
        """
        iSTFT with automatic caching and vectorized overlap-add.

        Args:
            real_part: Real part of STFT output (batch, freq, time)
            imag_part: Imaginary part of STFT output (batch, freq, time)
            n_fft: FFT size
            hop_length: Hop length
            win_length: Window length
            window: Window function
            center: If True, remove center padding
            audio_length: Target audio length

        Returns:
            Reconstructed audio (batch, samples)
        """
        # Window padding safety check
        if window.shape[0] < n_fft:
            pad = n_fft - window.shape[0]
            window = mx.concatenate([window, mx.zeros((pad,), dtype=window.dtype)])

        # Inverse FFT
        stft_complex = real_part + 1j * imag_part
        time_frames = mx.fft.irfft(stft_complex.transpose(0, 2, 1), n=n_fft, axis=-1)

        # Apply synthesis window
        windowed_frames = time_frames * window

        batch_size, num_frames, frame_length = windowed_frames.shape
        ola_len = (num_frames - 1) * hop_length + frame_length

        # Get cached items
        norm_buffer = self.get_norm_buffer(
            n_fft, hop_length, win_length, window, num_frames
        )
        positions_flat = self.get_positions(num_frames, frame_length, hop_length)

        # Vectorized overlap-add
        batch_offsets = mx.arange(batch_size) * ola_len
        global_indices = positions_flat[None, :] + batch_offsets[:, None]

        output = mx.zeros((batch_size * ola_len), dtype=mx.float32)
        output = output.at[global_indices.reshape(-1)].add(windowed_frames.reshape(-1))
        output = output.reshape(batch_size, ola_len)

        # Apply normalization
        output = output / norm_buffer[None, :]

        # Final trimming
        if center:
            start_cut = n_fft // 2
            output = output[:, start_cut:]

        if audio_length is not None:
            output = output[:, :audio_length]

        return output

    def clear_cache(self):
        """Clear all cached data to free memory"""
        self.norm_buffer_cache.clear()
        self.position_cache.clear()

    def cache_info(self):
        """Get information about cached items"""
        return {
            "norm_buffers": len(self.norm_buffer_cache),
            "position_indices": len(self.position_cache),
            "total_cached_items": len(self.norm_buffer_cache)
            + len(self.position_cache),
        }
