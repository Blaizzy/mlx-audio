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

    @property
    def win_length(self) -> int:
        return int(self.window_size * self.sample_rate)

    @property
    def hop_length(self) -> int:
        return int(self.window_stride * self.sample_rate)


def log_mel_spectrogram(x: mx.array, args: PreprocessArgs) -> mx.array:
    original_dtype = x.dtype

    if args.pad_to > 0:
        if x.shape[-1] < args.pad_to:
            pad_length = args.pad_to - x.shape[-1]
            x = mx.pad(x, ((0, pad_length),), constant_values=args.pad_value)

    window_fn = STR_TO_WINDOW_FN.get(args.window, None)
    window = window_fn(args.win_length) if window_fn else hanning(args.win_length)

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
