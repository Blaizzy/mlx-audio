# Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

import mlx.core as mx

from mlx_audio.utils import mel_filters, stft


def _log_mel_spectrogram_cpu(
    audio: mx.array,
    n_mels: int = 128,
    padding: int = 0,
) -> mx.array:
    """CPU fallback for log mel spectrogram using scipy (for non-Metal devices)."""
    import numpy as np
    from scipy import signal as scipy_signal

    was_1d = len(audio.shape) == 1
    if was_1d:
        audio = mx.expand_dims(audio, 0)

    if padding > 0:
        audio = mx.pad(audio, [(0, 0), (0, padding)])

    # Force evaluation before converting to numpy
    audio_float = audio.astype(mx.float32)
    mx.eval(audio_float)
    audio_np = np.array(audio_float, dtype=np.float32)

    # STFT with S3Tokenizer parameters
    specs = []
    for i in range(audio_np.shape[0]):
        _, _, Zxx = scipy_signal.stft(
            audio_np[i],
            fs=16000,
            window="hann",
            nperseg=400,
            noverlap=400 - 160,
            nfft=400,
            boundary=None,
            padded=False,
        )
        # Zxx shape: (n_freqs, n_frames), transpose to (n_frames, n_freqs)
        specs.append(Zxx.T)

    # Stack: each spec is (T', F) -> stack to (B, T', F)
    spec = np.stack(specs, axis=0)

    # Magnitude squared - drop last frame to match PyTorch torch.stft behavior
    magnitudes = np.abs(spec[:, :-1, :]) ** 2

    # Use slaney-style mel filterbank
    filters = mel_filters(
        sample_rate=16000,
        n_fft=400,
        n_mels=n_mels,
        norm="slaney",
        mel_scale="slaney",
    )
    filters_np = np.array(filters)

    # Apply mel filterbank: (B, T, F) @ (F, M) -> (B, T, M)
    mel_spec = magnitudes @ filters_np.T
    mel_spec = np.transpose(mel_spec, [0, 2, 1])  # (B, M, T)

    # Log compression with S3Tokenizer-style normalization
    log_spec = np.log10(np.maximum(mel_spec, 1e-10))
    log_spec = np.maximum(log_spec, np.max(log_spec) - 8.0)
    log_spec = (log_spec + 4.0) / 4.0

    result = mx.array(log_spec, dtype=audio.dtype)
    return result.squeeze(0) if was_1d else result


def log_mel_spectrogram(
    audio: mx.array,
    n_mels: int = 128,
    padding: int = 0,
) -> mx.array:
    """
    Compute the log-Mel spectrogram.

    This implementation matches the PyTorch S3Tokenizer which uses:
    - torch.stft with return_complex=True
    - Drops the last frame (magnitudes[..., :-1])
    - librosa-style mel filterbank (slaney norm, slaney mel scale)

    Args:
        audio: Audio waveform (T,) or (B, T) in 16 kHz
        n_mels: Number of Mel-frequency filters (80 or 128)
        padding: Number of zero samples to pad to the right

    Returns:
        Log-Mel spectrogram (n_mels, T') or (B, n_mels, T')
    """
    # Use CPU fallback for non-Metal devices (FFT not available on CUDA)
    # TODO: Create CUDA kernel for FFT
    if not mx.metal.is_available():
        return _log_mel_spectrogram_cpu(audio, n_mels, padding)

    was_1d = len(audio.shape) == 1
    if was_1d:
        audio = mx.expand_dims(audio, 0)

    if padding > 0:
        audio = mx.pad(audio, [(0, 0), (0, padding)])

    # STFT with S3Tokenizer parameters
    specs = []
    for i in range(audio.shape[0]):
        spec = stft(
            audio[i],
            window="hann",
            n_fft=400,
            hop_length=160,
            win_length=400,
        )
        specs.append(spec)

    # Stack: each spec is (T', F) -> stack to (B, T', F)
    spec = mx.stack(specs, axis=0)

    # Magnitude squared - drop last frame to match PyTorch torch.stft behavior
    magnitudes = mx.abs(spec[:, :-1, :]) ** 2

    # Use slaney-style mel filterbank
    filters = mel_filters(
        sample_rate=16000,
        n_fft=400,
        n_mels=n_mels,
        norm="slaney",
        mel_scale="slaney",
    )

    # Apply mel filterbank: (B, T, F) @ (F, M) -> (B, T, M)
    mel_spec = magnitudes @ filters.T
    mel_spec = mx.transpose(mel_spec, [0, 2, 1])  # (B, M, T)

    # Log compression with S3Tokenizer-style normalization
    log_spec = mx.log10(mx.maximum(mel_spec, 1e-10))
    log_spec = mx.maximum(log_spec, mx.max(log_spec) - 8.0)
    log_spec = (log_spec + 4.0) / 4.0

    return log_spec.squeeze(0) if was_1d else log_spec
