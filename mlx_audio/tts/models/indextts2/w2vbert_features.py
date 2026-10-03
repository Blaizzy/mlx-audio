from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import mlx.core as mx

from mlx_audio.dsp import compute_fbank_kaldi


@dataclass
class W2VBertFeatureExtractorConfig:
    # From facebook/w2v-bert-2.0 preprocessor_config.json
    sampling_rate: int = 16000
    num_mel_bins: int = 80
    stride: int = 2
    padding_value: float = 1.0

    # Reasonable defaults (SeamlessM4TFeatureExtractor-like)
    win_len: int = 400  # 25ms @ 16k
    win_inc: int = 160  # 10ms @ 16k
    preemphasis: float = 0.97
    dither: float = 0.0
    low_freq: float = 20.0
    high_freq: float = 0.0
    snip_edges: bool = False


def _pad_2d(x: mx.array, target_len: int, value: float) -> mx.array:
    if x.shape[0] >= target_len:
        return x
    pad = target_len - x.shape[0]
    return mx.pad(x, [(0, pad), (0, 0)], mode="constant", constant_values=value)


class W2VBertFeatureExtractor:
    """MLX feature extractor compatible with Wav2Vec2BertModel inputs.

    Produces log-mel filterbank features (80 bins), then applies `stride=2`
    stacking to yield a 160-dim feature vector per frame.
    """

    def __init__(self, cfg: Optional[W2VBertFeatureExtractorConfig] = None):
        self.cfg = cfg or W2VBertFeatureExtractorConfig()

    def __call__(
        self, audio: mx.array, *, lengths: Optional[mx.array] = None
    ) -> Tuple[mx.array, mx.array]:
        """Extract features.

        Args:
            audio: (T,) or (B, T) float waveform in [-1, 1]
            lengths: Optional (B,) lengths for each batch item.

        Returns:
            input_features: (B, T', 160)
            attention_mask: (B, T') with 1 for real frames, 0 for padded.
        """

        if audio.ndim == 1:
            audio = audio[None, :]
        if audio.ndim != 2:
            raise ValueError(f"audio must be shape (T,) or (B,T), got {audio.shape}")

        B, T = audio.shape
        if lengths is None:
            lengths = mx.array([T] * B)

        feats = []
        frame_lens = []

        for i in range(B):
            wav = audio[i, : int(lengths[i].item())]
            fb = compute_fbank_kaldi(
                wav,
                sample_rate=self.cfg.sampling_rate,
                win_len=self.cfg.win_len,
                win_inc=self.cfg.win_inc,
                num_mels=self.cfg.num_mel_bins,
                win_type="hamming",
                preemphasis=self.cfg.preemphasis,
                dither=self.cfg.dither,
                snip_edges=self.cfg.snip_edges,
                low_freq=self.cfg.low_freq,
                high_freq=self.cfg.high_freq,
            )  # (frames, 80)

            # Stride stacking: (frames, 80) -> (frames//2, 160)
            stride = self.cfg.stride
            n = (fb.shape[0] // stride) * stride
            fb = fb[:n]
            fb = fb.reshape(n // stride, stride * fb.shape[1])
            feats.append(fb)
            frame_lens.append(fb.shape[0])

        max_frames = int(max(frame_lens) if frame_lens else 0)

        padded = [_pad_2d(f, max_frames, self.cfg.padding_value) for f in feats]
        input_features = mx.stack(padded, axis=0).astype(mx.float32)

        # Attention mask over frames
        mask_rows = []
        for fl in frame_lens:
            fl = int(fl)
            if fl < 0 or fl > max_frames:
                raise ValueError("Invalid frame length")
            ones = mx.ones((fl,), dtype=mx.int32)
            zeros = mx.zeros((max_frames - fl,), dtype=mx.int32)
            mask_rows.append(mx.concatenate([ones, zeros], axis=0))
        mask = mx.stack(mask_rows, axis=0) if mask_rows else mx.zeros((B, 0), dtype=mx.int32)

        return input_features, mask
