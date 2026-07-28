# Copyright © 2023-2024 Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)
"""SeamlessM4T-style fbank (160-d) for the w2v-bert frontend, in MLX.

Reimplements transformers' SeamlessM4TFeatureExtractor: povey window, remove-dc
+ preemphasis 0.97 per frame, 80 kaldi-mel, log(mel_floor), per-mel-bin CMVN
(ddof=1), stride-2 frame stacking. The mel matrix + window are precomputed at
convert time (fbank_filters.npz). Validated to ~1e-6 vs transformers; the MLX
path matches that reference to float32 precision.
"""

import mlx.core as mx

FRAME, HOP, NFFT, MEL_FLOOR = 400, 160, 512, 1.192092955078125e-07


def fbank_160(audio: mx.array, mel_filters: mx.array, window: mx.array) -> mx.array:
    """audio (T,) 16 kHz -> (1, num_frames // 2, 160), all MLX."""
    wav = audio.astype(mx.float32) * (2**15)
    nfr = 1 + (wav.shape[0] - FRAME) // HOP

    # frame the signal into (nfr, FRAME) via a strided gather
    idx = mx.arange(nfr)[:, None] * HOP + mx.arange(FRAME)[None, :]
    frames = mx.take(wav, idx, axis=0)
    frames = frames - frames.mean(axis=1, keepdims=True)  # remove_dc_offset

    # preemphasis: buf[0] = b[0] * 0.03 ; buf[1:] = b[1:] - 0.97 * b[:-1]
    emph = mx.concatenate(
        [frames[:, :1] * 0.03, frames[:, 1:] - 0.97 * frames[:, :-1]], axis=1
    )
    emph = emph * window  # povey window (FRAME,)
    buf = mx.concatenate([emph, mx.zeros((nfr, NFFT - FRAME))], axis=1)

    spec = mx.abs(mx.fft.rfft(buf, n=NFFT, axis=1)) ** 2  # (nfr, NFFT // 2 + 1)
    out = mx.log(mx.maximum(MEL_FLOOR, spec @ mel_filters))  # (nfr, 80)

    # per-mel-bin CMVN (unbiased variance, ddof=1)
    mean = out.mean(axis=0, keepdims=True)
    var = ((out - mean) ** 2).sum(axis=0, keepdims=True) / (nfr - 1)
    out = (out - mean) / mx.sqrt(var + 1e-7)

    n = nfr - (nfr % 2)
    return out[:n].reshape(1, n // 2, 160)
