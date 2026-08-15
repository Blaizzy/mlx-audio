"""Top-k sampling used by MiniMax Music 3.

Adapted and modified from mikolaj92/minimax-music3-mlx under Apache-2.0.
See LICENSE and NOTICE.
"""

from __future__ import annotations

import mlx.core as mx

from .config import AR_SAMPLING_TOP_K


def sample_top_k(
    logits: mx.array,
    key: mx.array,
    top_k: int = AR_SAMPLING_TOP_K,
) -> tuple[mx.array, mx.array]:
    values = mx.where(
        mx.isnan(logits), mx.array(-1e9, dtype=logits.dtype), logits
    ).astype(mx.float32)
    k = min(top_k, values.shape[-1])
    threshold = mx.min(mx.topk(values, k, axis=-1), axis=-1, keepdims=True)
    masked = mx.where(values < threshold, mx.array(-1e9, dtype=mx.float32), values)
    next_key = mx.random.split(key)[1]
    return mx.random.categorical(masked, axis=-1, key=key), next_key
