"""Drama 3 codebook tower.

Port of fish_speech/models/text2semantic/fish_omni/tower.py. Unlike the S2 Pro
fast decoder, every tower layer attends causally over *time* and keeps its own
KV cache across frames: row p of the tower sees backbone hidden state p and is
labelled with frame p + 1.

All RMSNorms here are zero-centered (scale = 1 + weight) and kept that way:
storing 1 + weight in bf16 rounds away most of the weight and visibly
degrades the deeper codebook heads.
"""

from __future__ import annotations

import math
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.lm.models.base import create_attention_mask
from mlx_audio.lm.models.cache import KVCache

from .config import AudioDecoderConfig

# Repetition-aware sampling of the main token, fixed by the training recipe:
# when the first draw repeats one of the last RAS_WINDOW_FRAMES main tokens, a
# second draw with these parameters is used instead.
RAS_WINDOW_FRAMES = 16
RAS_TEMPERATURE = 1.0
RAS_TOP_K = 30
RAS_TOP_P = 0.95
MIN_TEMPERATURE = 1e-5


class ZeroCenteredRMSNorm(nn.Module):
    """RMSNorm scaling by 1 + weight, computed in fp32."""

    def __init__(self, dims: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = mx.zeros((dims,))

    def __call__(self, x: mx.array) -> mx.array:
        out = mx.fast.rms_norm(x.astype(mx.float32), None, self.eps)
        return (out * (1.0 + self.weight.astype(mx.float32))).astype(x.dtype)


def sample_top_k_top_p(
    logits: mx.array, temperature: float, top_k: int, top_p: float
) -> mx.array:
    """One id per row of `logits` [N, V].

    Matches the reference: probabilities come from the full softmax, the token
    that crosses top_p is dropped too, and the top-ranked token is always kept.
    Only the top_k candidates are sorted since every lower rank is dropped.
    """
    logits = logits.astype(mx.float32) / max(temperature, MIN_TEMPERATURE)
    k = min(top_k, logits.shape[-1])
    top_ids = mx.argpartition(-logits, kth=k - 1, axis=-1)[..., :k]
    top_values = mx.take_along_axis(logits, top_ids, axis=-1)
    order = mx.argsort(-top_values, axis=-1)
    top_ids = mx.take_along_axis(top_ids, order, axis=-1)
    top_values = mx.take_along_axis(top_values, order, axis=-1)

    log_norm = mx.logsumexp(logits, axis=-1, keepdims=True)
    cumulative = mx.cumsum(mx.exp(top_values - log_norm), axis=-1)
    dropped = cumulative > top_p
    dropped[..., 0] = False
    masked = mx.where(dropped, -mx.inf, top_values)
    choice = mx.random.categorical(masked, axis=-1)
    return mx.take_along_axis(top_ids, choice[..., None], axis=-1).squeeze(-1)


class TowerAttention(nn.Module):
    """Gated attention over time; checkpoint wqkv rows are [Q | Gate | K | V]."""

    def __init__(self, config: AudioDecoderConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_local_heads = config.n_local_heads
        self.head_dim = config.head_dim
        self.q_width = config.n_head * config.head_dim
        self.kv_width = config.n_local_heads * config.head_dim
        self.scale = config.head_dim**-0.5
        self.wqkv = nn.Linear(
            config.dim, 2 * self.q_width + 2 * self.kv_width, bias=False
        )
        self.wo = nn.Linear(self.q_width, config.dim, bias=False)
        # Always on, whatever attention_qk_norm says in config.json (the exporter
        # never writes it, so released configs carry the default false).
        self.q_norm = ZeroCenteredRMSNorm(config.head_dim, config.norm_eps)
        self.k_norm = ZeroCenteredRMSNorm(config.head_dim, config.norm_eps)
        # Interleaved (non-neox) partial RoPE, unlike the backbone.
        self.rope = nn.RoPE(config.rotary_dim, traditional=True, base=config.rope_base)

    def __call__(self, x: mx.array, mask, cache: KVCache) -> mx.array:
        B, L, _ = x.shape
        q, gate, k, v = mx.split(
            self.wqkv(x),
            [self.q_width, 2 * self.q_width, 2 * self.q_width + self.kv_width],
            axis=-1,
        )
        q = self.q_norm(q.reshape(B, L, self.n_head, self.head_dim))
        k = self.k_norm(k.reshape(B, L, self.n_local_heads, self.head_dim))
        q = self.rope(q.transpose(0, 2, 1, 3), offset=cache.offset)
        k = self.rope(k.transpose(0, 2, 1, 3), offset=cache.offset)
        v = v.reshape(B, L, self.n_local_heads, self.head_dim).transpose(0, 2, 1, 3)
        k, v = cache.update_and_fetch(k, v)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, self.q_width)
        return self.wo(out * mx.sigmoid(gate))


class TowerMLP(nn.Module):
    def __init__(self, config: AudioDecoderConfig):
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class TowerTransformerLayer(nn.Module):
    def __init__(self, config: AudioDecoderConfig):
        super().__init__()
        self.attention = TowerAttention(config)
        self.attention_norm = ZeroCenteredRMSNorm(config.dim, config.norm_eps)
        self.ffn_norm = ZeroCenteredRMSNorm(config.dim, config.norm_eps)
        self.feed_forward = TowerMLP(config)

    def __call__(self, x: mx.array, mask, cache: KVCache) -> mx.array:
        h = x + self.attention(self.attention_norm(x), mask, cache)
        return h + self.feed_forward(self.ffn_norm(h))


class TowerLayer(nn.Module):
    """Mix a code embedding into the hidden state, then one pre-norm block."""

    def __init__(self, config: AudioDecoderConfig, shared: bool):
        super().__init__()
        self.shared = shared
        # Checkpoint name of the embedding differs between shared / residual layers.
        if shared:
            self.semantic_embedding = nn.Embedding(config.vocab_size, config.dim)
        else:
            self.codebook_embedding = nn.Embedding(config.vocab_size, config.dim)
        self.vocab_size = config.vocab_size
        self.enorm = ZeroCenteredRMSNorm(config.dim, config.norm_eps)
        self.hnorm = ZeroCenteredRMSNorm(config.dim, config.norm_eps)
        self.eh_proj = nn.Linear(2 * config.dim, config.dim, bias=False)
        self.transformer_layer = TowerTransformerLayer(config)
        self.final_layernorm = ZeroCenteredRMSNorm(config.dim, config.norm_eps)

    def __call__(
        self,
        hidden: mx.array,
        codes: mx.array,
        semantic: mx.array,
        mask,
        cache: KVCache,
        shared_input: Optional[mx.array] = None,
    ) -> mx.array:
        embedding = self.semantic_embedding if self.shared else self.codebook_embedding
        e = embedding(mx.clip(codes, 0, self.vocab_size - 1))
        e = mx.where(semantic[..., None], e, mx.zeros_like(e))
        if shared_input is not None:
            e = e + shared_input
        mixed = self.eh_proj(
            mx.concatenate([self.enorm(e), self.hnorm(hidden)], axis=-1)
        )
        return self.final_layernorm(self.transformer_layer(mixed, mask, cache))


class AudioDecoder(nn.Module):
    """Predicts codes 1..N-1 of each frame, attending over time.

    A frame row is [main token id, code 0, ..., code N-1]; code 0 is the main
    token's offset in the semantic band. Arrays here are [B, T, ...] with B = 1.
    """

    def __init__(
        self,
        config: AudioDecoderConfig,
        semantic_begin: int,
        semantic_end: int,
    ):
        super().__init__()
        self.config = config
        self.semantic_begin = semantic_begin
        self.semantic_end = semantic_end
        self.codebook_embeddings = nn.Embedding(
            config.vocab_size * config.num_codebooks, config.dim
        )
        self.shared_layers = [
            TowerLayer(config, shared=True)
            for _ in range(config.shared_enrichment_layers)
        ]
        self.residual_layers = [
            TowerLayer(config, shared=False) for _ in range(config.num_codebooks - 1)
        ]
        self.codebook_heads = [
            nn.Linear(config.dim, config.vocab_size, bias=False)
            for _ in range(config.num_codebooks - 1)
        ]

    @property
    def layers(self) -> List[TowerLayer]:
        return [*self.shared_layers, *self.residual_layers]

    def make_cache(self) -> List[KVCache]:
        return [KVCache() for _ in self.layers]

    def is_semantic(self, tokens: mx.array) -> mx.array:
        return (tokens >= self.semantic_begin) & (tokens <= self.semantic_end)

    def embedding_sum(self, frames: mx.array, first: int, end: int) -> mx.array:
        """Summed embeddings of codes first..end-1 per frame row."""
        offsets = mx.arange(first, end) * self.config.vocab_size
        rows = frames[..., 1 + first : 1 + end] + offsets
        return self.codebook_embeddings(rows).sum(axis=-2)

    def embed_backbone(self, frames: mx.array, text_embeddings: mx.array) -> mx.array:
        """Backbone input: semantic frames add codes 1..K to the token embedding."""
        count = self.config.vq_backbone_codebooks
        extra = self.embedding_sum(frames, 1, 1 + count).astype(text_embeddings.dtype)
        mixed = (text_embeddings + extra) / math.sqrt(1 + count)
        semantic = self.is_semantic(frames[..., 0])
        return mx.where(semantic[..., None], mixed, text_embeddings)

    def run_shared(
        self,
        hidden: mx.array,
        c0: mx.array,
        semantic: mx.array,
        frames: mx.array,
        mask,
        caches: List[KVCache],
    ) -> mx.array:
        # The first shared layer also sees the codes the backbone did not.
        unseen = self.embedding_sum(
            frames, 1 + self.config.vq_backbone_codebooks, self.config.num_codebooks
        )
        unseen = mx.where(
            self.is_semantic(frames[..., 0])[..., None], unseen, mx.zeros_like(unseen)
        )
        for index, layer in enumerate(self.shared_layers):
            hidden = layer(
                hidden,
                c0,
                semantic,
                mask,
                caches[index],
                unseen if index == 0 else None,
            )
        return hidden

    def prefill(
        self,
        hidden: mx.array,
        frames: mx.array,
        next_frames: mx.array,
        caches: List[KVCache],
    ) -> None:
        """Teacher-force the tower caches; row p is labelled with frame p + 1."""
        mask = create_attention_mask(hidden, caches[0])
        semantic = self.is_semantic(next_frames[..., 0])
        n_shared = len(self.shared_layers)
        hidden = self.run_shared(
            hidden, next_frames[..., 1], semantic, frames, mask, caches[:n_shared]
        )
        for index, layer in enumerate(self.residual_layers):
            hidden = layer(
                hidden,
                next_frames[..., index + 1],
                semantic,
                mask,
                caches[n_shared + index],
            )
        return hidden

    def decode_step(
        self,
        main: mx.array,
        hidden: mx.array,
        frames: mx.array,
        caches: List[KVCache],
        *,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> mx.array:
        """Codes 0..N-1 of the frame after `frames` [1, 1, W], given its main token."""
        c0 = mx.clip(main - self.semantic_begin, 0, self.config.vocab_size - 1)
        semantic = self.is_semantic(main)
        n_shared = len(self.shared_layers)
        hidden = self.run_shared(hidden, c0, semantic, frames, None, caches[:n_shared])
        codes = [c0]
        for index, (layer, head) in enumerate(
            zip(self.residual_layers, self.codebook_heads)
        ):
            hidden = layer(hidden, codes[-1], semantic, None, caches[n_shared + index])
            codes.append(
                sample_top_k_top_p(head(hidden)[:, -1], temperature, top_k, top_p)[
                    :, None
                ]
            )
        return mx.stack(codes, axis=-1)
