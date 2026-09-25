from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class W2VBertStats(nn.Module):
    """Mean/std normalization for W2V-BERT hidden states.

    The official IndexTTS2 pipeline normalizes hidden_states[17] as:
      (feat - mean) / std
    where mean/std are loaded from wav2vec2bert_stats.pt.
    """

    def __init__(self, dim: int = 1024):
        super().__init__()
        self.mean = mx.zeros((dim,), dtype=mx.float32)
        self.std = mx.ones((dim,), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, T, C)
        return (x - self.mean[None, None, :]) / (self.std[None, None, :] + 1e-12)
