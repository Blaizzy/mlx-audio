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
    # Kaldi-compatible features
    "compute_deltas_kaldi",
    "mel_scale_kaldi",
    "inverse_mel_scale_kaldi",
    "get_mel_banks_kaldi",
    "compute_fbank_kaldi",
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


# =============================================================================
# Kaldi-compatible audio feature extraction
# =============================================================================


def compute_deltas_kaldi(
    specgram: mx.array, win_length: int = 5, mode: str = "edge"
) -> mx.array:
    """
    Compute delta coefficients of a spectrogram (Kaldi-compatible).

    The formula is:
    d_t = sum_{n=1}^{N} n * (c_{t+n} - c_{t-n}) / (2 * sum_{n=1}^{N} n^2)

    Args:
        specgram: MLX array of dimension (..., freq, time)
        win_length: The window length used for computing delta (default: 5)
        mode: Padding mode - "edge" or "constant" (default: "edge")

    Returns:
        MLX array of deltas of dimension (..., freq, time)
    """
    if win_length < 3:
        raise ValueError(f"win_length should be >= 3, got {win_length}")

    original_shape = specgram.shape
    specgram = specgram.reshape(-1, original_shape[-1])
    num_features = specgram.shape[0]

    n = (win_length - 1) // 2
    denom = float(n * (n + 1) * (2 * n + 1)) / 3.0

    # Pad the specgram
    if mode == "edge":
        pad_left = mx.repeat(specgram[:, 0:1], n, axis=1)
        pad_right = mx.repeat(specgram[:, -1:], n, axis=1)
        padded = mx.concatenate([pad_left, specgram, pad_right], axis=1)
    else:
        padded = mx.pad(specgram, [(0, 0), (n, n)])

    kernel_weights = mx.arange(-n, n + 1, dtype=padded.dtype)
    time_steps = padded.shape[1] - 2 * n
    output = mx.zeros((num_features, time_steps), dtype=padded.dtype)

    for i in range(time_steps):
        window = padded[:, i : i + win_length]
        weighted = window * kernel_weights
        output[:, i] = mx.sum(weighted, axis=1) / denom

    return output.reshape(original_shape)


def mel_scale_kaldi(freq: mx.array) -> mx.array:
    """Convert frequency to mel scale (Kaldi formula)."""
    return 1127.0 * mx.log(1.0 + freq / 700.0)


def inverse_mel_scale_kaldi(mel_freq: mx.array) -> mx.array:
    """Convert mel scale to frequency (Kaldi formula)."""
    return 700.0 * (mx.exp(mel_freq / 1127.0) - 1.0)


def _next_power_of_2(x: int) -> int:
    """Returns the smallest power of 2 that is greater than x."""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def _get_strided_kaldi(
    waveform: mx.array, window_size: int, window_shift: int, snip_edges: bool
) -> mx.array:
    """Extract frames from waveform using strided windowing (Kaldi-style)."""
    num_samples = waveform.shape[0]

    if snip_edges:
        if num_samples < window_size:
            return mx.zeros((0, 0))
        m = 1 + (num_samples - window_size) // window_shift
    else:
        m = (num_samples + (window_shift // 2)) // window_shift
        pad = window_size // 2 - window_shift // 2

        if pad > 0:
            pad_left = waveform[1 : pad + 1][::-1]
            pad_right = waveform[-1 : -pad - 1 : -1] if pad > 1 else waveform[-1:0:-1]
            waveform = mx.concatenate([pad_left, waveform, pad_right])
        else:
            pad_right = waveform[::-1]
            waveform = mx.concatenate([waveform[-pad:], pad_right])

    return mx.as_strided(waveform, shape=(m, window_size), strides=(window_shift, 1))


def get_mel_banks_kaldi(
    num_bins: int,
    window_length_padded: int,
    sample_freq: float,
    low_freq: float,
    high_freq: float,
) -> tuple[mx.array, mx.array]:
    """
    Create Kaldi-compatible mel filterbank matrix.

    Args:
        num_bins: Number of mel bins
        window_length_padded: Padded window length (FFT size)
        sample_freq: Sample frequency in Hz
        low_freq: Low frequency cutoff
        high_freq: High frequency cutoff (0 or negative = relative to Nyquist)

    Returns:
        (bins, center_freqs): Mel filterbank matrix and center frequencies
    """
    assert num_bins > 3, "Must have at least 3 mel bins"
    assert window_length_padded % 2 == 0

    num_fft_bins = window_length_padded // 2
    nyquist = 0.5 * sample_freq

    if high_freq <= 0.0:
        high_freq += nyquist

    assert (0.0 <= low_freq < nyquist) and (0.0 < high_freq <= nyquist)

    fft_bin_width = sample_freq / window_length_padded
    mel_low_freq = float(mel_scale_kaldi(mx.array(low_freq)))
    mel_high_freq = float(mel_scale_kaldi(mx.array(high_freq)))
    mel_freq_delta = (mel_high_freq - mel_low_freq) / (num_bins + 1)

    bin_idx = mx.arange(num_bins).reshape(-1, 1)
    left_mel = mel_low_freq + bin_idx * mel_freq_delta
    center_mel = mel_low_freq + (bin_idx + 1.0) * mel_freq_delta
    right_mel = mel_low_freq + (bin_idx + 2.0) * mel_freq_delta

    center_freqs = inverse_mel_scale_kaldi(center_mel)
    mel = mel_scale_kaldi(fft_bin_width * mx.arange(num_fft_bins)).reshape(1, -1)

    up_slope = (mel - left_mel) / (center_mel - left_mel)
    down_slope = (right_mel - mel) / (right_mel - center_mel)
    bins = mx.maximum(mx.zeros(1), mx.minimum(up_slope, down_slope))

    return bins, center_freqs.squeeze()


def compute_fbank_kaldi(
    waveform: mx.array,
    sample_rate: int = 48000,
    win_len: int = 1920,
    win_inc: int = 384,
    num_mels: int = 60,
    win_type: str = "hamming",
    preemphasis: float = 0.97,
    dither: float = 1.0,
    snip_edges: bool = True,
    low_freq: float = 20.0,
    high_freq: float = 0.0,
) -> mx.array:
    """
    Compute Kaldi-compatible log mel-filterbank features.

    Args:
        waveform: Input audio (1D array)
        sample_rate: Sample rate in Hz
        win_len: Window length in samples
        win_inc: Window shift in samples
        num_mels: Number of mel bins
        win_type: Window type ("hamming", "hanning", "povey", "rectangular")
        preemphasis: Preemphasis coefficient
        dither: Dither amount (0 to disable)
        snip_edges: If True, discard incomplete frames at edges
        low_freq: Low frequency cutoff
        high_freq: High frequency cutoff (0 = Nyquist)

    Returns:
        Log mel-filterbank features (time, num_mels)
    """
    if waveform.ndim == 2:
        waveform = waveform[0]

    frame_length_ms = win_len / sample_rate * 1000
    frame_shift_ms = win_inc / sample_rate * 1000
    window_shift_samples = int(sample_rate * frame_shift_ms * 0.001)
    window_size = int(sample_rate * frame_length_ms * 0.001)
    padded_window_size = _next_power_of_2(window_size)

    # Get frames
    strided_input = _get_strided_kaldi(
        waveform, window_size, window_shift_samples, snip_edges
    )

    if strided_input.shape[0] == 0:
        return mx.zeros((0, num_mels))

    # Apply dither
    if dither != 0.0:
        rand_gauss = mx.random.normal(strided_input.shape) * dither
        strided_input = strided_input + rand_gauss

    # Remove DC offset
    row_means = mx.mean(strided_input, axis=1, keepdims=True)
    strided_input = strided_input - row_means

    # Apply preemphasis
    if preemphasis != 0.0:
        first_col = strided_input[:, 0:1]
        other_cols = strided_input[:, 1:] - preemphasis * strided_input[:, :-1]
        strided_input = mx.concatenate([first_col, other_cols], axis=1)

    # Apply window function
    if win_type == "hamming":
        n = mx.arange(window_size)
        window = 0.54 - 0.46 * mx.cos(2 * mx.pi * n / (window_size - 1))
    elif win_type == "hanning":
        n = mx.arange(window_size)
        window = 0.5 - 0.5 * mx.cos(2 * mx.pi * n / (window_size - 1))
    elif win_type == "povey":
        n = mx.arange(window_size)
        hann = 0.5 - 0.5 * mx.cos(2 * mx.pi * n / (window_size - 1))
        window = mx.power(hann, 0.85)
    else:
        window = mx.ones(window_size)

    strided_input = strided_input * window

    # Pad to power of 2
    if padded_window_size != window_size:
        padding = padded_window_size - window_size
        strided_input = mx.pad(strided_input, [(0, 0), (0, padding)])

    # FFT and power spectrum
    fft_result = mx.fft.rfft(strided_input, n=padded_window_size, axis=1)
    spectrum = mx.abs(fft_result) ** 2.0

    # Get mel filterbank
    mel_energies, _ = get_mel_banks_kaldi(
        num_mels, padded_window_size, float(sample_rate), low_freq, high_freq
    )
    mel_energies = mx.pad(mel_energies, [(0, 0), (0, 1)])

    # Apply mel filterbank and log
    mel_features = mx.matmul(spectrum, mel_energies.T)
    mel_features = mx.log(mx.maximum(mel_features, 1e-8))

    return mel_features
