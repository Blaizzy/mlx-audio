from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.sample_utils import apply_top_k as _apply_top_k_logprobs
from mlx_lm.sample_utils import apply_top_p as _apply_top_p_logprobs


def _mask_logits_from_logprobs(logits: mx.array, logprobs: mx.array) -> mx.array:
    return mx.where(logprobs == -mx.inf, -mx.inf, logits)


def apply_top_k(logits: mx.array, top_k: int | None) -> mx.array:
    if top_k is None or int(top_k) <= 0 or int(top_k) >= logits.shape[-1]:
        return logits
    logprobs = nn.log_softmax(logits, axis=-1)
    return _mask_logits_from_logprobs(
        logits, _apply_top_k_logprobs(logprobs, int(top_k))
    )


def apply_top_p(logits: mx.array, top_p: float | None) -> mx.array:
    if top_p is None or float(top_p) <= 0.0 or float(top_p) >= 1.0:
        return logits
    logprobs = nn.log_softmax(logits, axis=-1)
    return _mask_logits_from_logprobs(
        logits, _apply_top_p_logprobs(logprobs, float(top_p))
    )


def apply_repetition_penalty_delay_pattern(
    logits: mx.array,
    prev_tokens: mx.array | None,
    penalty: float,
) -> mx.array:
    if prev_tokens is None or float(penalty) == 1.0 or prev_tokens.size == 0:
        return logits

    vocab_size = int(logits.shape[-1])
    previous = prev_tokens.astype(mx.int32)
    previous = mx.where(previous < 0, 0, previous)
    previous = mx.where(previous >= vocab_size, 0, previous)

    if logits.ndim == 2:
        one_hot = mx.any(
            mx.eye(vocab_size, dtype=mx.bool_)[previous.reshape(-1)], axis=0
        )
        penalized = mx.where(
            logits > 0, logits / float(penalty), logits * float(penalty)
        )
        return mx.where(one_hot[None, :], penalized, logits)

    if logits.ndim != 3:
        raise ValueError(f"Expected logits rank 2 or 3, got {logits.shape}")

    heads = int(logits.shape[1])
    one_hot_heads = []
    for head_index in range(heads):
        prev_h = previous[..., head_index].reshape(-1)
        one_hot_heads.append(mx.any(mx.eye(vocab_size, dtype=mx.bool_)[prev_h], axis=0))
    one_hot = mx.stack(one_hot_heads, axis=0)[None, :, :]
    penalized = mx.where(logits > 0, logits / float(penalty), logits * float(penalty))
    return mx.where(one_hot, penalized, logits)


def sample_token(
    logits: mx.array,
    *,
    prev_tokens: mx.array | None = None,
    repetition_penalty: float = 1.0,
    top_p: float | None = None,
    top_k: int | None = None,
    do_sample: bool = True,
) -> mx.array:
    logits = apply_repetition_penalty_delay_pattern(
        logits, prev_tokens=prev_tokens, penalty=repetition_penalty
    )
    if not do_sample:
        return mx.argmax(logits, axis=-1).astype(mx.int32)

    original_shape = logits.shape
    vocab_size = int(original_shape[-1])
    logits = logits.reshape(-1, vocab_size)
    logits = apply_top_k(logits, top_k)
    logits = apply_top_p(logits, top_p)
    return (
        mx.random.categorical(logits, axis=-1)
        .reshape(original_shape[:-1])
        .astype(mx.int32)
    )
