# Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

from functools import lru_cache

import numpy as np
from scipy import signal

from .config import VoiceEncConfig


def _hz_to_mel(freq: np.ndarray, htk: bool = False) -> np.ndarray:
    """Convert Hz to Mel scale."""
    if htk:
        return 2595.0 * np.log10(1.0 + freq / 700.0)
    # Slaney formula
    f_min = 0.0
    f_sp = 200.0 / 3
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0

    mels = np.where(
        freq >= min_log_hz,
        min_log_mel + np.log(freq / min_log_hz) / logstep,
        (freq - f_min) / f_sp,
    )
    return mels


def _mel_to_hz(mel: np.ndarray, htk: bool = False) -> np.ndarray:
    """Convert Mel scale to Hz."""
    if htk:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
    # Slaney formula
    f_min = 0.0
    f_sp = 200.0 / 3
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0

    freqs = np.where(
        mel >= min_log_mel,
        min_log_hz * np.exp(logstep * (mel - min_log_mel)),
        f_min + f_sp * mel,
    )
    return freqs


def _mel_filterbank(
    sr: int,
    n_fft: int,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: float = None,
    htk: bool = False,
    norm: str = "slaney",
) -> np.ndarray:
    """Create a mel filterbank matrix."""
    if fmax is None:
        fmax = sr / 2.0

    n_fft_bins = 1 + n_fft // 2
    fft_freqs = np.linspace(0, sr / 2, n_fft_bins)

    mel_min = _hz_to_mel(np.array([fmin]), htk=htk)[0]
    mel_max = _hz_to_mel(np.array([fmax]), htk=htk)[0]
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = _mel_to_hz(mel_points, htk=htk)

    filterbank = np.zeros((n_mels, n_fft_bins))

    for i in range(n_mels):
        left = hz_points[i]
        center = hz_points[i + 1]
        right = hz_points[i + 2]

        up_slope = (fft_freqs - left) / (center - left + 1e-10)
        down_slope = (right - fft_freqs) / (right - center + 1e-10)
        filterbank[i] = np.maximum(0, np.minimum(up_slope, down_slope))

    if norm == "slaney":
        enorm = 2.0 / (hz_points[2 : n_mels + 2] - hz_points[:n_mels])
        filterbank *= enorm[:, np.newaxis]

    return filterbank


@lru_cache()
def mel_basis(hp: VoiceEncConfig):
    assert hp.fmax <= hp.sample_rate // 2
    return _mel_filterbank(
        sr=hp.sample_rate,
        n_fft=hp.n_fft,
        n_mels=hp.num_mels,
        fmin=hp.fmin,
        fmax=hp.fmax,
    )  # -> (nmel, nfreq)


def preemphasis(wav, hp: VoiceEncConfig):
    assert hp.preemphasis != 0
    wav = signal.lfilter([1, -hp.preemphasis], [1], wav)
    wav = np.clip(wav, -1, 1)
    return wav


def melspectrogram(wav, hp: VoiceEncConfig, pad=True):
    """Compute mel-spectrogram from waveform."""
    # Run through pre-emphasis
    if hp.preemphasis > 0:
        wav = preemphasis(wav, hp)
        assert np.abs(wav).max() - 1 < 1e-07

    # Do the stft
    spec_complex = _stft(wav, hp, pad=pad)

    # Get the magnitudes
    spec_magnitudes = np.abs(spec_complex)

    if hp.mel_power != 1.0:
        spec_magnitudes **= hp.mel_power

    # Get the mel and convert magnitudes->db
    mel = np.dot(mel_basis(hp), spec_magnitudes)
    if hp.mel_type == "db":
        mel = _amp_to_db(mel, hp)

    # Normalise the mel from db to 0,1
    if hp.normalized_mels:
        mel = _normalize(mel, hp).astype(np.float32)

    assert not pad or mel.shape[1] == 1 + len(wav) // hp.hop_size  # Sanity check
    return mel  # (M, T)


def _stft(y, hp: VoiceEncConfig, pad=True):
    """Compute STFT using scipy/numpy."""
    n_fft = hp.n_fft
    hop_length = hp.hop_size
    win_length = hp.win_size

    # Get window
    window = signal.get_window("hann", win_length, fftbins=True)

    # Pad window to n_fft if needed
    if win_length < n_fft:
        pad_left = (n_fft - win_length) // 2
        pad_right = n_fft - win_length - pad_left
        window = np.pad(window, (pad_left, pad_right))

    # Center pad the signal
    if pad:
        pad_amount = n_fft // 2
        y = np.pad(y, (pad_amount, pad_amount), mode="reflect")

    # Compute number of frames
    n_frames = 1 + (len(y) - n_fft) // hop_length

    # Frame the signal using stride tricks
    frames = np.lib.stride_tricks.sliding_window_view(y, n_fft)[::hop_length][:n_frames]

    # Apply window and compute FFT
    windowed = frames * window
    stft_result = np.fft.rfft(windowed, n=n_fft, axis=-1).T  # (n_freq, n_frames)

    return stft_result


def _amp_to_db(x, hp: VoiceEncConfig):
    return 20 * np.log10(np.maximum(hp.stft_magnitude_min, x))


def _db_to_amp(x):
    return np.power(10.0, x * 0.05)


def _normalize(s, hp: VoiceEncConfig, headroom_db=15):
    min_level_db = 20 * np.log10(hp.stft_magnitude_min)
    s = (s - min_level_db) / (-min_level_db + headroom_db)
    return s
