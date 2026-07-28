"""Log-mel spectrogram preprocessing matching NeMo's AudioToMelSpectrogramPreprocessor.

Differences from the Parakeet preprocessor:
  * 128 mel bins (vs 80).
  * ``normalize: NA`` => the features are NOT normalized (NeMo's normalize_batch
    falls through and returns the input unchanged for an unrecognized type).

Everything else matches NeMo at inference time: preemphasis 0.97, symmetric Hann
window, power spectrogram (mag_power=2.0), Slaney mel filters, log with a small
additive guard, and no dither (dither is training-only in NeMo).
"""

from collections.abc import Iterator

import mlx.core as mx

from mlx_audio.utils import STR_TO_WINDOW_FN, hanning, mel_filters, stft

from .config import PreprocessArgs


def _padded_window(args: PreprocessArgs) -> mx.array:
    window_fn = STR_TO_WINDOW_FN.get(args.window, None)
    window = window_fn(args.win_length) if window_fn else hanning(args.win_length)

    # Center-pad the window to n_fft (matches torch.stft / NeMo).
    if window.shape[0] < args.n_fft:
        left = (args.n_fft - window.shape[0]) // 2
        right = args.n_fft - window.shape[0] - left
        window = mx.concatenate(
            [
                mx.zeros((left,), dtype=window.dtype),
                window,
                mx.zeros((right,), dtype=window.dtype),
            ]
        )

    return window


def _preemphasize(x: mx.array, args: PreprocessArgs) -> mx.array:
    if args.preemph and args.preemph > 0:
        return mx.concat([x[:1], x[1:] - args.preemph * x[:-1]], axis=0)
    return x


def _power_to_log_mel(
    power: mx.array, args: PreprocessArgs, dtype: mx.Dtype
) -> mx.array:
    filters = mel_filters(
        args.sample_rate, args.n_fft, args.features, norm="slaney", mel_scale="slaney"
    )
    x = filters.astype(power.dtype) @ power.T

    log_guard = mx.array(args.log_zero_guard_value, dtype=x.dtype)
    x = mx.log(x + log_guard)

    # normalize: NeMo only normalizes for "per_feature" / "all_features". "NA" (and
    # anything else) is a no-op, which is what this model was trained with.
    if args.normalize == "per_feature":
        mean = mx.mean(x, axis=1, keepdims=True)
        n = max(x.shape[1] - 1, 1)
        variance = mx.sum((x - mean) ** 2, axis=1, keepdims=True) / n
        std = mx.sqrt(variance)
        x = (x - mean) / (std + 1e-5)
    elif args.normalize == "all_features":
        x = (x - mx.mean(x)) / (mx.std(x) + 1e-5)

    x = mx.expand_dims(x.T, axis=0)
    return x.astype(dtype)


def log_mel_spectrogram(x: mx.array, args: PreprocessArgs) -> mx.array:
    """Compute (1, T, features) log-mel features for a mono waveform ``x`` (samples,)."""
    original_dtype = x.dtype

    if args.pad_to > 0 and x.shape[-1] < args.pad_to:
        pad_length = args.pad_to - x.shape[-1]
        x = mx.pad(x, ((0, pad_length),), constant_values=args.pad_value)

    window = _padded_window(args)
    x = _preemphasize(x, args)

    x = stft(x, args.n_fft, args.hop_length, args.n_fft, window, pad_mode="constant")
    # Power spectrum (mag_power = 2.0).
    x = mx.square(mx.abs(x)).astype(original_dtype)

    return _power_to_log_mel(x, args, original_dtype)


def log_mel_spectrogram_frames(
    x: mx.array, args: PreprocessArgs, frame_start: int, frame_end: int
) -> mx.array:
    """Compute a contiguous full-audio-equivalent log-mel frame range.

    This avoids materializing the full STFT for long files while preserving the
    same frame centers, center padding, and preemphasis state as
    :func:`log_mel_spectrogram`.
    """
    original_dtype = x.dtype

    if frame_end <= frame_start:
        return mx.zeros((1, 0, args.features), dtype=original_dtype)

    if args.pad_to > 0 and x.shape[-1] < args.pad_to:
        pad_length = args.pad_to - x.shape[-1]
        x = mx.pad(x, ((0, pad_length),), constant_values=args.pad_value)

    if args.normalize in ("per_feature", "all_features"):
        raise NotImplementedError(
            "chunked Nemotron mel extraction only supports normalize='NA'"
        )

    hop = args.hop_length
    n_fft = args.n_fft
    num_frames = frame_end - frame_start
    sample_start = frame_start * hop - n_fft // 2
    sample_end = (frame_end - 1) * hop - n_fft // 2 + n_fft
    total_samples = x.shape[-1]

    raw_start = max(sample_start, 0)
    raw_end = min(sample_end, total_samples)
    raw = x[raw_start:raw_end]

    # Preemphasis high-pass filter with the same previous-sample state as a
    # full-utterance pass: y[n] = x[n] - alpha * x[n-1].
    if args.preemph and args.preemph > 0 and raw.shape[0] > 0:
        if raw_start > 0:
            first = raw[:1] - args.preemph * x[raw_start - 1 : raw_start]
            raw = mx.concat([first, raw[1:] - args.preemph * raw[:-1]], axis=0)
        else:
            raw = _preemphasize(raw, args)

    left_pad = max(-sample_start, 0)
    right_pad = max(sample_end - total_samples, 0)
    pieces = []
    if left_pad:
        pieces.append(mx.zeros((left_pad,), dtype=original_dtype))
    pieces.append(raw.astype(original_dtype))
    if right_pad:
        pieces.append(mx.zeros((right_pad,), dtype=original_dtype))
    segment = mx.concatenate(pieces, axis=0) if len(pieces) > 1 else pieces[0]

    expected_len = (num_frames - 1) * hop + n_fft
    if segment.shape[0] < expected_len:
        segment = mx.pad(
            segment,
            ((0, expected_len - segment.shape[0]),),
            constant_values=0.0,
        )

    window = _padded_window(args)
    frames = mx.as_strided(segment, shape=(num_frames, n_fft), strides=(hop, 1))
    power = mx.square(mx.abs(mx.fft.rfft(frames * window))).astype(original_dtype)
    return _power_to_log_mel(power, args, original_dtype)


def iter_log_mel_spectrogram(
    x: mx.array, args: PreprocessArgs, chunk_frames: int
) -> Iterator[mx.array]:
    """Yield full-audio-equivalent log-mel chunks with bounded STFT memory."""
    if chunk_frames <= 0:
        raise ValueError("chunk_frames must be positive")

    total_samples = x.shape[-1]
    if args.pad_to > 0 and total_samples < args.pad_to:
        total_samples = args.pad_to

    total_frames = total_samples // args.hop_length + 1
    for frame_start in range(0, total_frames, chunk_frames):
        frame_end = min(frame_start + chunk_frames, total_frames)
        yield log_mel_spectrogram_frames(x, args, frame_start, frame_end)
