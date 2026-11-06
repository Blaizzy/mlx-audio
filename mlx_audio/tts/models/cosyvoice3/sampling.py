"""Sampling utilities for CosyVoice3's speech-token LM.

``nucleus_sampling``/``random_sampling`` wrap ``mlx_lm``'s ``make_sampler``.
``ras_sampling`` adds repetition-aware fallback on top: if the nucleus pick
repeats too often within the last ``win_size`` decoded tokens
(>= win_size*tau_r), it re-samples with that token masked out.
"""

from typing import List

import mlx.core as mx
from mlx_lm.sample_utils import make_sampler


def _log_softmax(logits: mx.array) -> mx.array:
    logits = logits.astype(mx.float32)
    return logits - mx.logsumexp(logits, axis=-1, keepdims=True)


def nucleus_sampling(logits: mx.array, top_p: float = 0.8, top_k: int = 25) -> int:
    """logits: (V,) unnormalized. Returns a sampled token id (int)."""
    top_k = min(top_k, logits.shape[-1] - 1)
    sampler = make_sampler(temp=1.0, top_p=top_p, top_k=top_k)
    return int(sampler(_log_softmax(logits)).item())


def random_sampling(logits: mx.array) -> int:
    sampler = make_sampler(temp=1.0)
    return int(sampler(_log_softmax(logits)).item())


def ras_sampling(
    logits: mx.array,
    decoded_tokens: List[int],
    sampling: int = 25,
    top_p: float = 0.8,
    top_k: int = 25,
    win_size: int = 10,
    tau_r: float = 0.1,
) -> int:
    """Repetition-aware sampling: nucleus_sampling with a masked fallback."""
    top_id = nucleus_sampling(logits, top_p=top_p, top_k=top_k)
    window = decoded_tokens[-win_size:]
    rep_num = sum(1 for t in window if t == top_id)
    if rep_num >= win_size * tau_r:
        masked = logits.at[mx.array([top_id])].add(mx.array(float("-inf")))
        top_id = random_sampling(masked)
    return top_id
