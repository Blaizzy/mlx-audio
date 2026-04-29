"""LLM decoder for Voxtral Realtime.

26-layer decoder-only transformer with:
- GQA (32 query heads, 8 KV heads, head_dim=128)
- Sliding window attention (8192)
- Interleaved RoPE (theta=1M)
- Adaptive RMSNorm with time conditioning
- Tied embeddings (tok_embeddings used as both input and LM head)
- No biases anywhere in the decoder

Optimizations:
- GQA handled natively by mx.fast.scaled_dot_product_attention (no repeat)
- KV cache is a ring buffer (mlx_lm RotatingKVCache) so steady-state decode
  does O(1) in-place writes instead of rebuilding the cache every token
- Single-token generation skips mask computation
- RoPE inv_freq cached in attention module
"""

import math

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import RotatingKVCache

from .config import DecoderConfig


def compute_time_embedding(
    t_value: float, dim: int, theta: float = 10000.0
) -> mx.array:
    """Sinusoidal time embedding for adaptive RMSNorm conditioning.

    Args:
        t_value: Number of delay tokens (e.g. 6.0 for 480ms)
        dim: Embedding dimension (decoder dim, e.g. 3072)
        theta: Frequency base

    Returns:
        mx.array: [dim] time conditioning vector
    """
    half_dim = dim // 2
    inv_freq = mx.exp(
        -math.log(theta) * mx.arange(half_dim, dtype=mx.float32) / half_dim
    )
    emb = t_value * inv_freq
    return mx.concatenate([mx.cos(emb), mx.sin(emb)])  # [dim]


class AdaRMSNorm(nn.Module):
    """Adaptive RMSNorm with time conditioning.

    Per-layer MLP: Linear(dim -> bottleneck) -> GELU -> Linear(bottleneck -> dim)
    Applied as: h_norm * (1 + ada_scale)
    """

    def __init__(self, dim: int, bottleneck_dim: int):
        super().__init__()
        self.ada_down = nn.Linear(dim, bottleneck_dim, bias=False)
        self.ada_up = nn.Linear(bottleneck_dim, dim, bias=False)

    def compute_scale(self, t_cond: mx.array) -> mx.array:
        """Precompute ada_scale from time conditioning. Returns [dim]."""
        hidden = nn.gelu(self.ada_down(t_cond))
        return self.ada_up(hidden)

    def __call__(self, x: mx.array, ada_scale: mx.array) -> mx.array:
        return x * (1.0 + ada_scale)


class DecoderAttention(nn.Module):
    """GQA attention for decoder (no biases).

    GQA is handled natively by mx.fast.scaled_dot_product_attention
    which broadcasts KV heads to match Q heads internally.
    """

    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.sliding_window = config.sliding_window
        self.rope_theta = config.rope_theta
        self.scale = 1.0 / math.sqrt(config.head_dim)

        q_dim = config.n_heads * config.head_dim
        kv_dim = config.n_kv_heads * config.head_dim

        self.wq = nn.Linear(config.dim, q_dim, bias=False)
        self.wk = nn.Linear(config.dim, kv_dim, bias=False)
        self.wv = nn.Linear(config.dim, kv_dim, bias=False)
        self.wo = nn.Linear(q_dim, config.dim, bias=False)

    def __call__(self, x, start_pos, cache=None):
        """
        Args:
            x: [seq, dim]
            start_pos: absolute position of the first token in ``x``
            cache: Optional RotatingKVCache for this layer (state is mutated in place).

        Returns:
            output tensor [seq, dim].
        """
        seq_len = x.shape[0]
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        # Reshape to [B=1, n_heads, seq, head_dim] for fused RoPE + SDPA kernels.
        q = q.reshape(1, seq_len, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(1, seq_len, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(1, seq_len, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Fused RoPE kernel (GPT-J style = traditional=True interleaved pairs).
        q = mx.fast.rope(
            q,
            self.head_dim,
            traditional=True,
            base=self.rope_theta,
            scale=1.0,
            offset=start_pos,
        )
        k = mx.fast.rope(
            k,
            self.head_dim,
            traditional=True,
            base=self.rope_theta,
            scale=1.0,
            offset=start_pos,
        )

        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
            # update_and_fetch returns the view of the ring buffer containing
            # all tokens still in the sliding window (bounded by max_size).
            if seq_len == 1:
                mask = None
            else:
                mask = cache.make_mask(seq_len, window_size=self.sliding_window)
        else:
            # No cache: either prefill without caching, or encoder-style one-shot.
            mask = "causal" if seq_len > 1 else None

        attn_out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=mask
        )

        # Reshape back: [1, n_heads, seq, head_dim] -> [seq, n_heads * head_dim]
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(
            seq_len, self.n_heads * self.head_dim
        )

        return self.wo(attn_out)


class DecoderLayer(nn.Module):
    """Single decoder transformer layer with adaptive RMSNorm."""

    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.attention_norm = nn.RMSNorm(config.dim, eps=config.norm_eps)
        self.attention = DecoderAttention(config)
        self.ffn_norm = nn.RMSNorm(config.dim, eps=config.norm_eps)

        if config.ada_rms_norm_t_cond:
            self.ada_rms_norm_t_cond = AdaRMSNorm(
                config.dim, config.ada_rms_norm_t_cond_dim
            )
        else:
            self.ada_rms_norm_t_cond = None

        # SwiGLU FFN (no biases in decoder)
        self.feed_forward_w1 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.feed_forward_w3 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.feed_forward_w2 = nn.Linear(config.hidden_dim, config.dim, bias=False)

    def __call__(self, x, start_pos, ada_scale=None, cache=None):
        # Attention (cache is mutated in place by RotatingKVCache.update_and_fetch)
        h = self.attention_norm(x)
        h = self.attention(h, start_pos, cache=cache)
        x = x + h

        # FFN with adaptive norm
        h = self.ffn_norm(x)
        if self.ada_rms_norm_t_cond is not None and ada_scale is not None:
            h = self.ada_rms_norm_t_cond(h, ada_scale)

        gate = nn.silu(self.feed_forward_w1(h))
        up = self.feed_forward_w3(h)
        x = x + self.feed_forward_w2(gate * up)

        return x


class Decoder(nn.Module):
    """Full LLM decoder with tied embeddings."""

    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = [DecoderLayer(config) for _ in range(config.n_layers)]
        self.norm = nn.RMSNorm(config.dim, eps=config.norm_eps)

        # ada_scale per layer (precomputed from t_cond)
        self._ada_scales = None

    def precompute_ada_scales(self, t_cond: mx.array):
        """Precompute adaptive norm scales for all layers. Call once after loading."""
        scales = []
        for layer in self.layers:
            if layer.ada_rms_norm_t_cond is not None:
                scales.append(layer.ada_rms_norm_t_cond.compute_scale(t_cond))
            else:
                scales.append(None)
        self._ada_scales = scales

    def embed_token(self, token_id: int) -> mx.array:
        """Look up the dequantized embedding for a single token id."""
        return self.tok_embeddings(mx.array([token_id]))[0]

    def embed_tokens(self, token_ids: mx.array) -> mx.array:
        return self.tok_embeddings(token_ids)

    def make_cache(self):
        """Allocate a fresh per-layer ring-buffer KV cache sized to the sliding window."""
        return [
            RotatingKVCache(max_size=self.config.sliding_window, keep=0)
            for _ in self.layers
        ]

    def forward(self, embeds, start_pos=0, cache=None):
        """Run decoder forward.

        Args:
            embeds: [seq, dim] input embeddings (audio_embed + tok_embed)
            start_pos: Starting position for RoPE
            cache: list of RotatingKVCache per layer. If None, a fresh cache is
                allocated and returned; otherwise the provided caches are mutated
                in place to avoid per-step reallocation.

        Returns:
            (hidden_states, cache) — cache is returned so callers can keep a
            handle to the same list across steps.
        """
        if cache is None:
            cache = self.make_cache()

        h = embeds
        for i, layer in enumerate(self.layers):
            ada_scale = self._ada_scales[i] if self._ada_scales is not None else None
            h = layer(h, start_pos, ada_scale=ada_scale, cache=cache[i])

        h = self.norm(h)
        return h, cache

    def logits(self, h):
        """Compute logits via tied embeddings.

        Uses ``Embedding.as_linear`` so quantized and full-precision embeddings
        both route through the right kernel (mx.quantized_matmul vs
        ``h @ weight.T``). This is what makes the 4-bit tok-embedding path
        actually fast.
        """
        return self.tok_embeddings.as_linear(h)
