"""Drama 3's audio tower: samples codebooks 1..9 of a frame, attending over time.

Unlike the S2 Pro fast decoder, which restarts a 10-step cache for every frame,
each of the 13 tower blocks keeps a KV cache across frames. All blocks run at the
same temporal position; the codebook index selects the block, not the RoPE
position. A frame row is [main token, c0, ..., c9], where c0 is the main token's
offset in the semantic band.
"""

from typing import Callable, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.lm.models.cache import KVCache

from .backbone import Fp32RMSNorm
from .config import TowerConfig


class TowerAttention(nn.Module):
    """Gated GQA. wqkv rows are [all Q | all gates | K | V]."""

    def __init__(self, cfg: TowerConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.n_kv = cfg.n_local_heads
        self.head_dim = cfg.head_dim
        self.q_width = cfg.n_head * cfg.head_dim
        self.kv_width = cfg.n_local_heads * cfg.head_dim
        self.rotary_dims = int(cfg.head_dim * cfg.rotary_percent)
        self.rope_base = cfg.rope_base
        self.wqkv = nn.Linear(cfg.dim, 2 * self.q_width + 2 * self.kv_width, bias=False)
        self.wo = nn.Linear(self.q_width, cfg.dim, bias=False)
        self.q_norm = Fp32RMSNorm(cfg.head_dim, cfg.norm_eps)
        self.k_norm = Fp32RMSNorm(cfg.head_dim, cfg.norm_eps)

    def _rope(self, x: mx.array, offset: int) -> mx.array:
        # Rotated in float32, as in the reference.
        y = mx.fast.rope(
            x.astype(mx.float32),
            self.rotary_dims,
            traditional=True,
            base=self.rope_base,
            scale=1.0,
            offset=offset,
        )
        return y.astype(x.dtype)

    def __call__(self, x: mx.array, cache: KVCache) -> mx.array:
        B, T, _ = x.shape
        splits = [self.q_width, 2 * self.q_width, 2 * self.q_width + self.kv_width]
        q, gate, k, v = mx.split(self.wqkv(x), splits, axis=-1)
        q = self.q_norm(q.reshape(B, T, self.n_head, self.head_dim))
        k = self.k_norm(k.reshape(B, T, self.n_kv, self.head_dim))
        v = v.reshape(B, T, self.n_kv, self.head_dim).transpose(0, 2, 1, 3)
        q = self._rope(q.transpose(0, 2, 1, 3), cache.offset)
        k = self._rope(k.transpose(0, 2, 1, 3), cache.offset)
        k, v = cache.update_and_fetch(k, v)
        out = mx.fast.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=self.head_dim**-0.5,
            mask="causal" if T > 1 else None,
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, T, self.q_width)
        return self.wo(out * mx.sigmoid(gate))


class TowerMLP(nn.Module):
    def __init__(self, cfg: TowerConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.dim, cfg.intermediate_size, bias=False)
        self.w3 = nn.Linear(cfg.dim, cfg.intermediate_size, bias=False)
        self.w2 = nn.Linear(cfg.intermediate_size, cfg.dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class TowerBlock(nn.Module):
    def __init__(self, cfg: TowerConfig):
        super().__init__()
        self.attention = TowerAttention(cfg)
        self.attention_norm = Fp32RMSNorm(cfg.dim, cfg.norm_eps)
        self.ffn_norm = Fp32RMSNorm(cfg.dim, cfg.norm_eps)
        self.feed_forward = TowerMLP(cfg)

    def __call__(self, x: mx.array, cache: KVCache) -> mx.array:
        x = x + self.attention(self.attention_norm(x), cache)
        return x + self.feed_forward(self.ffn_norm(x))


class TowerLayer(nn.Module):
    """Mixes a code embedding (zero on text rows) into the hidden state, then
    runs one block."""

    def __init__(self, cfg: TowerConfig, enrichment: bool):
        super().__init__()
        self.enrichment = enrichment
        embedding = nn.Embedding(cfg.vocab_size, cfg.dim)
        # Enrichment and residual layers name their embedding differently.
        if enrichment:
            self.semantic_embedding = embedding
        else:
            self.codebook_embedding = embedding
        self.enorm = Fp32RMSNorm(cfg.dim, cfg.norm_eps)
        self.hnorm = Fp32RMSNorm(cfg.dim, cfg.norm_eps)
        self.eh_proj = nn.Linear(2 * cfg.dim, cfg.dim, bias=False)
        self.transformer_layer = TowerBlock(cfg)
        self.final_layernorm = Fp32RMSNorm(cfg.dim, cfg.norm_eps)
        self.vocab_size = cfg.vocab_size

    def __call__(
        self,
        hidden: mx.array,
        codes: mx.array,
        semantic: mx.array,
        cache: KVCache,
        extra: Optional[mx.array] = None,
    ) -> mx.array:
        embed = self.semantic_embedding if self.enrichment else self.codebook_embedding
        e = embed(mx.clip(codes, 0, self.vocab_size - 1))
        e = mx.where(semantic[..., None], e, mx.zeros_like(e))
        if extra is not None:
            e = e + extra
        mixed = self.eh_proj(
            mx.concatenate([self.enorm(e), self.hnorm(hidden)], axis=-1)
        )
        out = self.final_layernorm(self.transformer_layer(mixed, cache))
        return out.astype(mixed.dtype)


class Tower(nn.Module):
    def __init__(self, cfg: TowerConfig, semantic_begin: int, semantic_end: int):
        super().__init__()
        self.vocab_size = cfg.vocab_size
        self.num_codebooks = cfg.num_codebooks
        self.backbone_codebooks = cfg.vq_backbone_codebooks
        self.semantic_begin = semantic_begin
        self.semantic_end = semantic_end
        # One table for all codebooks; codebook i uses rows [i * vocab, (i+1) * vocab).
        self.codebook_embeddings = nn.Embedding(
            cfg.vocab_size * cfg.num_codebooks, cfg.dim
        )
        self.shared_layers = [
            TowerLayer(cfg, enrichment=True)
            for _ in range(cfg.shared_enrichment_layers)
        ]
        self.residual_layers = [
            TowerLayer(cfg, enrichment=False) for _ in range(cfg.num_codebooks - 1)
        ]
        self.codebook_heads = [
            nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
            for _ in range(cfg.num_codebooks - 1)
        ]

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.shared_layers + self.residual_layers]

    def is_semantic(self, tokens: mx.array) -> mx.array:
        return (tokens >= self.semantic_begin) & (tokens <= self.semantic_end)

    def embedding_sum(self, frames: mx.array, first: int, end: int) -> mx.array:
        """Sum of the embeddings of codebooks first..end-1 for frames [B, T, 11]."""
        rows = (
            frames[..., 1 + first : 1 + end] + mx.arange(first, end) * self.vocab_size
        )
        return self.codebook_embeddings(rows).sum(axis=-2)

    def embed_backbone(self, frames: mx.array, text: mx.array) -> mx.array:
        """Backbone input. Semantic rows add the first residual codebooks to the
        token embedding and rescale; text rows keep the token embedding."""
        count = self.backbone_codebooks
        mixed = (text + self.embedding_sum(frames, 1, 1 + count)) * (1 + count) ** -0.5
        semantic = self.is_semantic(frames[..., 0])[..., None]
        return mx.where(semantic, mixed, text)

    def _enrich(self, hidden, c0, semantic, frames, caches) -> mx.array:
        # The first enrichment layer also sees the frame's codebooks that the
        # backbone input left out.
        unseen = self.embedding_sum(
            frames, 1 + self.backbone_codebooks, self.num_codebooks
        )
        unseen = mx.where(
            self.is_semantic(frames[..., 0])[..., None], unseen, mx.zeros_like(unseen)
        )
        for index, layer in enumerate(self.shared_layers):
            extra = unseen if index == 0 else None
            hidden = layer(hidden, c0, semantic, caches[index], extra)
        return hidden

    def prefill(self, hidden, frames, next_frames, caches) -> None:
        """Teacher-force the caches over a prompt; row t is labelled by row t+1."""
        semantic = self.is_semantic(next_frames[..., 0])
        hidden = self._enrich(hidden, next_frames[..., 1], semantic, frames, caches)
        offset = len(self.shared_layers)
        for index, layer in enumerate(self.residual_layers):
            codes = next_frames[..., index + 1]
            hidden = layer(hidden, codes, semantic, caches[offset + index])

    def residual_codes(
        self,
        hidden: mx.array,
        c0: mx.array,
        semantic: mx.array,
        frame: mx.array,
        caches: list[KVCache],
        sample: Callable[[mx.array], mx.array],
    ) -> list[mx.array]:
        """Codes c0..c9 of one frame, sampling c1..c9 with sample(logits)."""
        hidden = self._enrich(hidden, c0, semantic, frame, caches)
        codes = [c0]
        offset = len(self.shared_layers)
        layers = zip(self.residual_layers, self.codebook_heads)
        for index, (layer, head) in enumerate(layers):
            hidden = layer(hidden, codes[-1], semantic, caches[offset + index])
            codes.append(sample(head(hidden)))
        return codes
