"""CTC greedy collapse and insertion-slot interleaving for Granite Speech NAR.

These are pure utilities used by the transcribe pipeline. No model dependencies.
"""

from __future__ import annotations

import mlx.core as mx


def ctc_collapse_decode(tokens: mx.array, blank_id: int) -> mx.array:
    """Standard CTC greedy collapse: dedup adjacent repeats, then drop blanks.

    Args:
        tokens: 1-D integer mx.array of argmax token IDs.
        blank_id: the blank token ID (100257 for Granite Speech).

    Returns:
        1-D int mx.array of collapsed token IDs.
    """
    n = tokens.shape[0]
    if n == 0:
        return tokens
    is_first = mx.concatenate([mx.array([True]), tokens[1:] != tokens[:-1]])
    keep = is_first & (tokens != blank_id)
    n_keep = int(mx.sum(keep).item())
    if n_keep == 0:
        return tokens[:0]
    # Scatter compact: dropped tokens go to a trash slot at index n_keep; kept
    # tokens each write to their unique position. Then slice off the trash.
    positions = mx.cumsum(keep.astype(mx.int32)) - 1
    scatter_idx = mx.where(keep, positions, n_keep)
    out = mx.zeros((n_keep + 1,), dtype=tokens.dtype)
    out[scatter_idx] = tokens
    return out[:n_keep]


def add_insertion_slots(
    token_ids: mx.array, blank_id: int, min_len: int = 8
) -> mx.array:
    """Insert blank tokens between each CTC token as editing slots for the LLM.

    For N CTC tokens, the output has length `max(2N+1, min_len)`. CTC tokens occupy
    ODD indices (1, 3, 5, ..., 2N-1); EVEN indices and any trailing positions are blanks.

    Args:
        token_ids: 1-D integer mx.array of CTC-collapsed token IDs.
        blank_id: the blank token ID (100257 for Granite Speech).
        min_len: minimum output length (8 for Granite Speech NAR).

    Returns:
        1-D int mx.array of length max(2N+1, min_len).
    """
    n = token_ids.shape[0]
    total_len = max(2 * n + 1, min_len)
    dtype = token_ids.dtype
    if n == 0:
        return mx.full((total_len,), blank_id, dtype=dtype)
    blanks = mx.full((n,), blank_id, dtype=dtype)
    interleaved = mx.stack([blanks, token_ids], axis=1).reshape(2 * n)
    padding = mx.full((total_len - 2 * n,), blank_id, dtype=dtype)
    return mx.concatenate([interleaved, padding])
