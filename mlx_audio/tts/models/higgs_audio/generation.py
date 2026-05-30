"""Higgs Audio v2 generation primitives — delay pattern, audio embedding
lookup, and a minimal generate_frames() loop.

Not yet integrated with ChatML / reference audio / streaming — those
layers build on top of what's here. This module handles the per-frame
mechanics of autoregressive multi-codebook audio generation.
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .config import HiggsAudioConfig


def revert_delay_pattern(data: mx.array) -> mx.array:
    """Undo the delay pattern applied during generation.

    Input shape: (K, seq_len + K - 1) — codebook i is offset by i frames.
    Output shape: (K, seq_len) — codebooks aligned at each timestep.

    Ported from boson_multimodal/model/higgs_audio/utils.py.
    """
    assert data.ndim == 2, f"expected 2D, got {data.shape}"
    K = data.shape[0]
    L = data.shape[1]
    rows = [data[i : i + 1, i : L - K + 1 + i] for i in range(K)]
    return mx.concatenate(rows, axis=0)


def apply_delay_pattern(codebook_ids: mx.array, bos_id: int) -> mx.array:
    """Apply the delay pattern to a sequence of aligned codebook frames.

    Inverse of revert_delay_pattern. Given aligned tokens [K, L], produce
    delayed [K, L + K - 1] where codebook i starts emitting at frame i.

    Positions before codebook i's start are filled with bos_id.
    Positions after codebook i's real content tail are untouched (we
    typically don't need them, but they'd be EOS in a full generation).
    """
    K = codebook_ids.shape[0]
    L = codebook_ids.shape[1]
    out = mx.full((K, L + K - 1), bos_id, dtype=codebook_ids.dtype)
    for i in range(K):
        out[i, i : i + L] = codebook_ids[i]
    return out


def build_delay_pattern_mask(
    input_ids: mx.array, bos_token_id: int, pad_token_id: int
) -> mx.array:
    """Apply the delay pattern to a conditioning/prompt sequence of codebook tokens.

    Takes aligned [K, L] codebook ids and produces delayed [K, L + K - 1]:
      - Lower triangle (i > j): bos_token_id
      - Upper triangle (j >= L + i): pad_token_id
      - Middle band: original input_ids aligned so codebook i is offset by i frames

    Ported from boson_multimodal/model/higgs_audio/utils.py build_delay_pattern_mask
    (non-generation variant — no -1 placeholders since all positions are known).
    """
    K, L = input_ids.shape
    new_L = L + K - 1
    i_idx = mx.arange(K)[:, None]  # [K, 1]
    j_idx = mx.arange(new_L)[None, :]  # [1, new_L]
    bos_mask = j_idx < i_idx  # below diag → BOS
    eos_mask = j_idx >= (L + i_idx)  # past content end → EOS
    # Middle: input_ids[i, j - i]
    src_j = mx.clip(j_idx - i_idx, 0, L - 1)
    src_j_broad = mx.broadcast_to(src_j, (K, new_L))
    gathered = mx.take_along_axis(input_ids, src_j_broad, axis=1)
    out = mx.where(bos_mask, mx.array(bos_token_id, dtype=input_ids.dtype), gathered)
    out = mx.where(eos_mask, mx.array(pad_token_id, dtype=input_ids.dtype), out)
    return out


def lookup_audio_embedding(
    audio_codebook_embeddings: nn.Embedding,
    codebook_ids: mx.array,
    codebook_size_plus2: int,
) -> mx.array:
    """Convert [K, T] codebook token ids to [T, hidden] summed embeddings.

    Each codebook's tokens are shifted by `k * (codebook_size + 2)` to index
    into the shared embedding table. Per-codebook embeddings are summed
    (per Higgs v2 config.audio_embed_avg = False).

    Args:
        codebook_ids: shape [K, T] int32, values in [0, codebook_size + 1]
            (includes stream BOS at codebook_size and EOS at codebook_size+1).
        codebook_size_plus2: codebook_size + 2 (per-codebook stride in the
            shared embedding table).

    Returns:
        [T, hidden_size] float array.
    """
    K, T = codebook_ids.shape
    shift = mx.arange(K, dtype=mx.int32) * codebook_size_plus2  # [K]
    shifted = codebook_ids + shift[:, None]  # [K, T]
    per_codebook = audio_codebook_embeddings(shifted)  # [K, T, hidden]
    return mx.sum(per_codebook, axis=0)  # [T, hidden]


def greedy_sample_audio(audio_logits: mx.array) -> mx.array:
    """Argmax per codebook on audio_logits.

    Args:
        audio_logits: shape [B, T, K, C+2] (raw audio head output).

    Returns:
        [B, T, K] int32 token ids.
    """
    return mx.argmax(audio_logits, axis=-1).astype(mx.int32)


def sample_audio(
    audio_logits: mx.array,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: Optional[int] = None,
) -> mx.array:
    """Temperature + (optional) top-p / top-k sampling per codebook.

    Args:
        audio_logits: [B, T, K, C+2].
        temperature: 0 → greedy.
        top_p: nucleus cutoff (applied per codebook independently).
        top_k: top-k cutoff (applied per codebook independently, None → off).

    Returns:
        [B, T, K] int32.
    """
    if temperature <= 0.0:
        return greedy_sample_audio(audio_logits)

    logits = audio_logits / temperature

    if top_k is not None and top_k > 0:
        # Mask logits below top-k per codebook.
        kth = mx.sort(logits, axis=-1)[..., -top_k : -top_k + 1]
        logits = mx.where(logits < kth, -mx.inf, logits)

    if top_p is not None and 0.0 < top_p < 1.0:
        # Nucleus sampling per codebook on the last axis.
        sorted_idx = mx.argsort(-logits, axis=-1)  # descending order
        sorted_logits = mx.take_along_axis(logits, sorted_idx, axis=-1)
        sorted_probs = mx.softmax(sorted_logits, axis=-1)
        cumulative = mx.cumsum(sorted_probs, axis=-1)
        # Keep positions where previous-cumsum < top_p (always keep idx 0).
        shifted = mx.concatenate(
            [mx.zeros_like(cumulative[..., :1]), cumulative[..., :-1]], axis=-1
        )
        keep_sorted = shifted < top_p
        masked_sorted = mx.where(keep_sorted, sorted_logits, -mx.inf)
        # Scatter back to original order.
        inv_idx = mx.argsort(sorted_idx, axis=-1)
        logits = mx.take_along_axis(masked_sorted, inv_idx, axis=-1)

    # Gumbel-max trick for categorical sampling.
    u = mx.random.uniform(shape=logits.shape)
    g = -mx.log(-mx.log(u + 1e-20) + 1e-20)
    return mx.argmax(logits + g, axis=-1).astype(mx.int32)
