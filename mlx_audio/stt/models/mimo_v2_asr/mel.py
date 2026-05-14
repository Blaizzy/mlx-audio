"""
Mel spectrogram for MiMo Audio Tokenizer.

Parameters match MiMo-Audio-Tokenizer config:
  - sample_rate=24000
  - n_fft=960
  - hop_length=240
  - win_length=960
  - n_mels=128
  - f_min=0, f_max=None
  - power=1.0 (magnitude spectrum)
  - center=True

Uses natural logarithm (not log10 as in Whisper).
"""

from typing import Optional, Union

import mlx.core as mx
import numpy as np

from mlx_audio.utils import hanning, mel_filters, stft


# ── MiMo audio constants ───────────────────────────────────────────
SAMPLE_RATE = 24000
N_FFT = 960
HOP_LENGTH = 240
WIN_LENGTH = 960
N_MELS = 128
F_MIN = 0.0
F_MAX: Optional[float] = None


def _maybe_load_audio(audio: Union[str, np.ndarray, mx.array]) -> mx.array:
    """Ensure audio is an MLX array, loading from file if needed."""
    if isinstance(audio, str):
        from mlx_audio.stt.utils import load_audio

        audio = load_audio(audio, sr=SAMPLE_RATE)
    if not isinstance(audio, mx.array):
        audio = mx.array(audio)
    return audio


def log_mel_spectrogram(
    audio: Union[str, np.ndarray, mx.array],
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    win_length: int = WIN_LENGTH,
    f_min: float = F_MIN,
    f_max: Optional[float] = F_MAX,
    center: bool = True,
    pad_mode: str = "reflect",
) -> mx.array:
    """
    Compute the log-Mel spectrogram with MiMo parameters.

    Equivalent to torchaudio.transforms.MelSpectrogram(power=1.0) followed by
    ``torch.log(torch.clip(spec, min=1e-7))``.

    Returns
    -------
    mx.array, shape = (n_mels, n_frames)
    """
    audio = _maybe_load_audio(audio)

    window = hanning(win_length)
    # STFT → complex spectrogram
    freqs = stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
    )

    # Magnitude spectrum (power=1.0 → take absolute value)
    # stft returns (n_fft//2+1, n_frames) for real input;
    # keep Nyquist bin since mel_filters also include it.
    magnitudes = freqs.abs()

    # Mel filterbank
    filters = mel_filters(
        sample_rate=SAMPLE_RATE,
        n_fft=n_fft,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
        norm=None,
        mel_scale="htk",
    )
    mel_spec = magnitudes @ filters.T  # (n_frames, n_mels)

    # Natural log
    log_spec = mx.log(mx.maximum(mel_spec, 1e-7))

    return log_spec.T  # (n_mels, n_frames)


def wav_to_mel(
    audio: Union[str, np.ndarray, mx.array],
    **kwargs,
) -> mx.array:
    """Convenience: audio → log-mel spectrogram (n_mels, n_frames)."""
    return log_mel_spectrogram(audio, **kwargs)
