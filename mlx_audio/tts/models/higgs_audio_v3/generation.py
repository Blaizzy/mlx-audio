from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.sample_utils import apply_top_k as _apply_top_k_logprobs
from mlx_lm.sample_utils import apply_top_p as _apply_top_p_logprobs

STOP_CODE = -1


@dataclass
class HiggsSamplerState:
    num_codebooks: int
    delay_count: int = 0
    eoc_countdown: Optional[int] = None
    generation_done: bool = False
    last_codes: Optional[mx.array] = None


def apply_delay_pattern(
    codes_tn: mx.array,
    *,
    boc_id: int,
    eoc_id: int,
) -> mx.array:
    """Convert raw codec codes [T, N] to delayed rows [T + N - 1, N]."""
    if codes_tn.ndim != 2:
        raise ValueError(f"codes_tn must be 2-D [T, N], got {codes_tn.shape}")
    t, n = codes_tn.shape
    out = mx.full((t + n - 1, n), eoc_id, dtype=codes_tn.dtype)
    for codebook in range(n):
        if codebook:
            out[:codebook, codebook] = boc_id
        out[codebook : codebook + t, codebook] = codes_tn[:, codebook]
    return out


def reverse_delay_pattern(delayed_ln: mx.array) -> mx.array:
    """Convert delayed rows [L, N] back to raw codec codes [L - N + 1, N]."""
    if delayed_ln.ndim != 2:
        raise ValueError(f"delayed_ln must be 2-D [L, N], got {delayed_ln.shape}")
    length, num_codebooks = delayed_ln.shape
    t = length - num_codebooks + 1
    if t <= 0:
        raise ValueError(
            f"delayed rows have L={length}, N={num_codebooks}; need L >= N"
        )
    cols = [
        delayed_ln[codebook : codebook + t, codebook : codebook + 1]
        for codebook in range(num_codebooks)
    ]
    return mx.concatenate(cols, axis=1)


def _mask_logits_from_logprobs(logits: mx.array, logprobs: mx.array) -> mx.array:
    return mx.where(logprobs == -mx.inf, -mx.inf, logits)


def _apply_top_k(logits: mx.array, top_k: Optional[int]) -> mx.array:
    if top_k is None or int(top_k) <= 0 or int(top_k) >= logits.shape[-1]:
        return logits
    logprobs = nn.log_softmax(logits, axis=-1)
    return _mask_logits_from_logprobs(
        logits, _apply_top_k_logprobs(logprobs, int(top_k))
    )


def _apply_top_p(logits: mx.array, top_p: Optional[float]) -> mx.array:
    if top_p is None or float(top_p) <= 0.0 or float(top_p) >= 1.0:
        return logits
    logprobs = nn.log_softmax(logits, axis=-1)
    return _mask_logits_from_logprobs(
        logits, _apply_top_p_logprobs(logprobs, float(top_p))
    )


def sample_independent(
    logits_nv: mx.array,
    *,
    temperature: float,
    top_p: Optional[float],
    top_k: Optional[int],
) -> mx.array:
    if logits_nv.ndim != 2:
        raise ValueError(f"logits_nv must be [N, V], got {logits_nv.shape}")
    if temperature <= 1e-5 or (top_k is not None and int(top_k) == 1):
        return mx.argmax(logits_nv, axis=-1).astype(mx.int32)
    logits = logits_nv / float(temperature)
    logits = _apply_top_k(logits, top_k)
    logits = _apply_top_p(logits, top_p)
    return mx.random.categorical(logits, axis=-1).astype(mx.int32)


def sample_batch(
    logits_bnv: mx.array,
    *,
    temperature: float,
    top_p: Optional[float],
    top_k: Optional[int],
) -> mx.array:
    if logits_bnv.ndim != 3:
        raise ValueError(f"logits_bnv must be [B, N, V], got {logits_bnv.shape}")
    if temperature <= 1e-5 or (top_k is not None and int(top_k) == 1):
        return mx.argmax(logits_bnv, axis=-1).astype(mx.int32)
    logits = logits_bnv / float(temperature)
    logits = _apply_top_k(logits, top_k)
    logits = _apply_top_p(logits, top_p)
    return mx.random.categorical(logits, axis=-1).astype(mx.int32)


def step(
    logits_nv: mx.array,
    state: HiggsSamplerState,
    *,
    temperature: float,
    top_p: Optional[float],
    top_k: Optional[int],
    boc_id: int,
    eoc_id: int,
) -> mx.array:
    """Run one SGLang-compatible delayed multi-codebook sampler step."""
    n = state.num_codebooks
    if logits_nv.ndim != 2 or logits_nv.shape[0] != n:
        raise ValueError(
            f"logits shape {logits_nv.shape} incompatible with num_codebooks={n}"
        )
    if state.generation_done:
        return mx.full((n,), STOP_CODE, dtype=mx.int32)

    codes_n = sample_independent(
        logits_nv,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )

    if state.delay_count < n:
        next_codebook = state.delay_count + 1
        if next_codebook < n:
            tail_mask = mx.arange(n, dtype=mx.int32) >= next_codebook
            codes_n = mx.where(tail_mask, mx.array(boc_id, dtype=mx.int32), codes_n)
        state.delay_count += 1
    elif state.eoc_countdown is not None:
        state.eoc_countdown -= 1
        if state.eoc_countdown <= 0:
            state.generation_done = True
    elif int(codes_n[0].item()) == eoc_id:
        if n <= 2:
            state.generation_done = True
        else:
            state.eoc_countdown = n - 2

    if not state.generation_done:
        state.last_codes = codes_n
    return codes_n
