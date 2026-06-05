"""Log-mel preprocessing for Nemotron 3.5 ASR.

Matches NeMo's AudioToMelSpectrogramPreprocessor for this model:
  - 16 kHz, n_fft=512, win 400 (Hann), hop 160, 128 mel bins
  - preemphasis 0.97, power spectrum (|stft|^2), log(x + 2^-24)
  - normalize="NA"  -> NO normalization (unlike Parakeet's per_feature/global)

The mel filterbank (`fb`) and `window` are taken VERBATIM from the .nemo
(`preprocessor.featurizer.{fb,window}`) so the filters are bit-identical to the
reference; only the STFT/log math is re-implemented here.
"""

from dataclasses import dataclass

import mlx.core as mx

from mlx_audio.dsp import stft


@dataclass
class PreprocessArgs:
    sample_rate: int = 16000
    n_fft: int = 512
    win_length: int = 400
    hop_length: int = 160
    features: int = 128
    preemph: float = 0.97
    log_zero_guard_value: float = 2.0**-24


def log_mel_spectrogram(
    audio: mx.array,
    fb: mx.array,
    window: mx.array,
    args: PreprocessArgs,
) -> mx.array:
    """audio: (n_samples,) -> mel: (1, n_frames, features)."""
    orig_dtype = audio.dtype
    x = audio.astype(mx.float32)

    # preemphasis high-pass: y[n] = x[n] - 0.97*x[n-1]
    if args.preemph and args.preemph > 0:
        x = mx.concatenate([x[:1], x[1:] - args.preemph * x[:-1]], axis=0)

    # center-pad the win_length window to n_fft (matches torch.stft centering)
    w = window.astype(mx.float32)
    if w.shape[0] < args.n_fft:
        left = (args.n_fft - w.shape[0]) // 2
        right = args.n_fft - w.shape[0] - left
        w = mx.concatenate([mx.zeros((left,)), w, mx.zeros((right,))], axis=0)

    spec = stft(
        x,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.n_fft,
        window=w,
        center=True,
        pad_mode="constant",  # NeMo FilterbankFeatures uses zero (constant) padding
    )  # (n_frames, n_fft//2+1)
    power = mx.square(mx.abs(spec))  # power spectrum

    # mel: fb (128, 257) @ power.T (257, n_frames) -> (128, n_frames)
    fb2 = fb.astype(mx.float32)
    if fb2.ndim == 3:
        fb2 = fb2[0]
    mel = fb2 @ power.T
    mel = mx.log(mel + args.log_zero_guard_value)

    # normalize="NA": no normalization
    mel = mx.expand_dims(mel.T, axis=0)  # (1, n_frames, features)
    return mel.astype(orig_dtype)
