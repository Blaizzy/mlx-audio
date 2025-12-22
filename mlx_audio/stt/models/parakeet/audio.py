from dataclasses import dataclass

import mlx.core as mx

from mlx_audio.utils import (
    STR_TO_WINDOW_FN,
    bartlett,
    blackman,
    hamming,
    hanning,
    mel_filters,
    stft,
)


@dataclass
class PreprocessArgs:
    sample_rate: int
    normalize: str
    window_size: float
    window_stride: float
    window: str
    features: int
    n_fft: int
    dither: float
    pad_to: int = 0
    pad_value: float = 0
    preemph: float = 0.97  # Preemphasis coefficient (set to 0.0 to disable)

    @property
    def win_length(self) -> int:
        return int(self.window_size * self.sample_rate)

    @property
    def hop_length(self) -> int:
        return int(self.window_stride * self.sample_rate)


def _log_mel_spectrogram_cpu(x: mx.array, args: PreprocessArgs) -> mx.array:
    """CPU fallback for log mel spectrogram using scipy (for non-Metal devices)."""
    import numpy as np
    from scipy import signal as scipy_signal

    original_dtype = x.dtype
    # Force evaluation before converting to numpy
    x_float = x.astype(mx.float32)
    mx.eval(x_float)
    x_np = np.array(x_float, dtype=np.float32)

    if args.pad_to > 0:
        if x_np.shape[-1] < args.pad_to:
            pad_length = args.pad_to - x_np.shape[-1]
            x_np = np.pad(x_np, (0, pad_length), constant_values=args.pad_value)

    # Apply preemphasis high-pass filter
    preemph = getattr(args, "preemph", 0.97)
    if preemph > 0:
        x_np = np.concatenate([x_np[:1], x_np[1:] - preemph * x_np[:-1]], axis=0)

    # Map window name to scipy window
    window_map = {
        "hann": "hann",
        "hanning": "hann",
        "hamming": "hamming",
        "blackman": "blackman",
        "bartlett": "bartlett",
    }
    window_name = window_map.get(args.window, "hann")

    # Compute STFT using scipy
    _, _, Zxx = scipy_signal.stft(
        x_np,
        fs=args.sample_rate,
        window=window_name,
        nperseg=args.win_length,
        noverlap=args.win_length - args.hop_length,
        nfft=args.n_fft,
        boundary=None,
        padded=False,
    )

    # Compute power spectrogram: Zxx shape is (n_freqs, n_frames)
    power_spec = np.abs(Zxx) ** 2

    # Get mel filters
    filters = mel_filters(
        args.sample_rate, args.n_fft, args.features, norm=args.normalize, mel_scale=None
    )
    filters_np = np.array(filters)

    # Apply mel filters: filters @ power_spec
    # filters: (n_mels, n_freqs), power_spec: (n_freqs, n_frames)
    # Match dimensions if needed
    if power_spec.shape[0] != filters_np.shape[1]:
        min_freqs = min(power_spec.shape[0], filters_np.shape[1])
        power_spec = power_spec[:min_freqs, :]
        filters_np = filters_np[:, :min_freqs]

    mel_spec = filters_np @ power_spec

    # Log mel spectrogram
    log_mel = np.log(mel_spec + 1e-5)

    # Normalize
    if args.normalize == "per_feature":
        mean = np.mean(log_mel, axis=1, keepdims=True)
        std = np.std(log_mel, axis=1, keepdims=True)
        normalized_mel = (log_mel - mean) / (std + 1e-5)
    else:
        mean = np.mean(log_mel)
        std = np.std(log_mel)
        normalized_mel = (log_mel - mean) / (std + 1e-5)

    # Transpose and add batch dimension: (n_mels, n_frames) -> (1, n_frames, n_mels)
    normalized_mel = normalized_mel.T
    normalized_mel = np.expand_dims(normalized_mel, axis=0)

    return mx.array(normalized_mel, dtype=original_dtype)


def log_mel_spectrogram(x: mx.array, args: PreprocessArgs) -> mx.array:
    original_dtype = x.dtype

    # Use CPU fallback for non-Metal devices (FFT not available on CUDA)
    # TODO: Create CUDA kernel for FFT
    if not mx.metal.is_available():
        return _log_mel_spectrogram_cpu(x, args)

    if args.pad_to > 0:
        if x.shape[-1] < args.pad_to:
            pad_length = args.pad_to - x.shape[-1]
            x = mx.pad(x, ((0, pad_length),), constant_values=args.pad_value)

    window_fn = STR_TO_WINDOW_FN.get(args.window, None)
    window = window_fn(args.win_length) if window_fn else hanning(args.win_length)

    # Apply preemphasis high-pass filter (matches NeMo training preprocessing)
    # Formula: y[n] = x[n] - α*x[n-1] where α=0.97
    # Boosts high frequencies for better consonant recognition
    preemph = getattr(args, "preemph", 0.97)  # Backward compatible with old configs
    if preemph > 0:
        x = mx.concat([x[:1], x[1:] - preemph * x[:-1]], axis=0)

    x = stft(x, args.n_fft, args.hop_length, args.win_length, window)
    x = mx.square(mx.abs(x)).astype(original_dtype)
    filters = mel_filters(
        args.sample_rate, args.n_fft, args.features, norm=args.normalize, mel_scale=None
    )
    x = filters.astype(x.dtype) @ x.T

    x = mx.log(x + 1e-5)

    if args.normalize == "per_feature":
        mean = mx.mean(x, axis=1, keepdims=True)
        std = mx.std(x, axis=1, keepdims=True)
        normalized_mel = (x - mean) / (std + 1e-5)
    else:
        mean = mx.mean(x)
        std = mx.std(x)
        normalized_mel = (x - mean) / (std + 1e-5)

    normalized_mel = normalized_mel.T
    normalized_mel = mx.expand_dims(normalized_mel, axis=0)

    return normalized_mel.astype(original_dtype)
