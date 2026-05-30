from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.sample_utils import apply_top_k as _apply_top_k_logprobs
from mlx_lm.sample_utils import apply_top_p as _apply_top_p_logprobs


def _mask_logits_from_logprobs(logits: mx.array, logprobs: mx.array) -> mx.array:
    return mx.where(logprobs == -mx.inf, -mx.inf, logits)


def apply_repetition_penalty(
    logits: mx.array,
    previous_token_ids: mx.array | None,
    repetition_penalty: float = 1.0,
) -> mx.array:
    if previous_token_ids is None or repetition_penalty == 1.0:
        return logits
    if previous_token_ids.size == 0:
        return logits

    scores = logits
    vocab_size = logits.shape[-1]
    previous = previous_token_ids.astype(mx.int32)
    previous = mx.where(previous < 0, 0, previous)
    previous = mx.where(previous >= vocab_size, 0, previous)

    one_hot = mx.any(mx.eye(vocab_size, dtype=mx.bool_)[previous], axis=-2)
    penalized = mx.where(
        scores < 0,
        scores * float(repetition_penalty),
        scores / float(repetition_penalty),
    )
    return mx.where(one_hot, penalized, scores)


def apply_top_k(logits: mx.array, top_k: int | None) -> mx.array:
    if top_k is None or int(top_k) <= 0 or int(top_k) >= logits.shape[-1]:
        return logits
    logprobs = nn.log_softmax(logits, axis=-1)
    filtered_logprobs = _apply_top_k_logprobs(logprobs, int(top_k))
    return _mask_logits_from_logprobs(logits, filtered_logprobs)


def apply_top_p(logits: mx.array, top_p: float | None) -> mx.array:
    if top_p is None or float(top_p) <= 0.0 or float(top_p) >= 1.0:
        return logits
    logprobs = nn.log_softmax(logits, axis=-1)
    filtered_logprobs = _apply_top_p_logprobs(logprobs, float(top_p))
    return _mask_logits_from_logprobs(logits, filtered_logprobs)


def sample_next_token(
    logits: mx.array,
    *,
    do_sample: bool,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    previous_token_ids: mx.array | None = None,
    repetition_penalty: float = 1.0,
) -> mx.array:
    scores = apply_repetition_penalty(
        logits,
        previous_token_ids=previous_token_ids,
        repetition_penalty=repetition_penalty,
    )
    if not do_sample:
        return mx.argmax(scores, axis=-1).astype(mx.int32)
    if temperature <= 0:
        raise ValueError("temperature must be positive when do_sample=True")
    scores = scores / float(temperature)
    scores = apply_top_k(scores, top_k)
    scores = apply_top_p(scores, top_p)
    return mx.random.categorical(scores, axis=-1).astype(mx.int32)


def sample_assistant_text_token(
    text_logits: mx.array,
    *,
    audio_assistant_slot_token_id: int,
    audio_end_token_id: int,
    do_sample: bool,
    temperature: float,
    top_k: int,
    top_p: float,
) -> mx.array:
    candidate_ids = mx.array(
        [audio_assistant_slot_token_id, audio_end_token_id],
        dtype=mx.int32,
    )
    candidate_scores = text_logits[..., candidate_ids]
    sampled_candidate = sample_next_token(
        candidate_scores,
        do_sample=do_sample,
        temperature=temperature,
        top_k=min(int(top_k), 2),
        top_p=top_p,
    )
    return candidate_ids[sampled_candidate]
