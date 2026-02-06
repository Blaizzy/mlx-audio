"""LLM decoder for Voxtral Realtime.

26-layer decoder-only transformer with:
- GQA (32 query heads, 8 KV heads, head_dim=128)
- Sliding window attention (8192)
- Interleaved RoPE (theta=1M)
- Adaptive RMSNorm with time conditioning
- Tied embeddings (tok_embeddings used as both input and LM head)
- No biases anywhere in the decoder
"""

import math

import mlx.core as mx
import mlx.nn as nn

from .config import DecoderConfig
from .encoder import _compute_rope_freqs, _interleaved_rope


def compute_time_embedding(t_value: float, dim: int, theta: float = 10000.0) -> mx.array:
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
        # Weight indices .0. and .2. (GELU at .1. has no params)
        self.ada_down = nn.Linear(dim, bottleneck_dim, bias=False)
        self.ada_up = nn.Linear(bottleneck_dim, dim, bias=False)

    def compute_scale(self, t_cond: mx.array) -> mx.array:
        """Precompute ada_scale from time conditioning. Returns [dim]."""
        hidden = nn.gelu(self.ada_down(t_cond))
        return self.ada_up(hidden)

    def __call__(self, x: mx.array, ada_scale: mx.array) -> mx.array:
        """Apply adaptive modulation.

        Args:
            x: [seq, dim] normalized hidden states
            ada_scale: [dim] precomputed scale from compute_scale()
        """
        return x * (1.0 + ada_scale)


class DecoderAttention(nn.Module):
    """GQA attention for decoder (no biases)."""

    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.sliding_window = config.sliding_window
        self.rope_theta = config.rope_theta

        q_dim = config.n_heads * config.head_dim
        kv_dim = config.n_kv_heads * config.head_dim

        self.wq = nn.Linear(config.dim, q_dim, bias=False)
        self.wk = nn.Linear(config.dim, kv_dim, bias=False)
        self.wv = nn.Linear(config.dim, kv_dim, bias=False)
        self.wo = nn.Linear(q_dim, config.dim, bias=False)

    def __call__(self, x, positions, cache=None):
        """
        Args:
            x: [seq, dim]
            positions: [seq] position indices
            cache: Optional (k_cache, v_cache) tuple

        Returns:
            (output, new_cache)
        """
        seq_len = x.shape[0]
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        # RoPE
        cos, sin = _compute_rope_freqs(positions, self.head_dim, self.rope_theta)
        q = _interleaved_rope(q, cos, sin, self.n_heads, self.head_dim)
        k = _interleaved_rope(k, cos, sin, self.n_kv_heads, self.head_dim)

        # Update KV cache (store BEFORE GQA expansion)
        if cache is not None:
            k_cache, v_cache = cache
            k = mx.concatenate([k_cache, k], axis=0)
            v = mx.concatenate([v_cache, v], axis=0)

        kv_len = k.shape[0]

        # Save cache before GQA expansion
        k_for_cache = k
        v_for_cache = v

        # Reshape for attention: [1, n_heads, seq, head_dim] (MLX expects B,H,T,D)
        q = q.reshape(1, seq_len, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(1, kv_len, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(1, kv_len, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        # GQA: expand KV heads
        gqa_ratio = self.n_heads // self.n_kv_heads
        if gqa_ratio > 1:
            k = mx.repeat(k, gqa_ratio, axis=1)  # axis=1 is heads dim in B,H,T,D
            v = mx.repeat(v, gqa_ratio, axis=1)

        # Causal sliding window mask [seq_q, kv_len]
        q_pos = positions[:, None]  # [seq_q, 1]
        k_global = mx.arange(kv_len)[None, :]  # [1, kv_len]

        causal = k_global <= q_pos
        window = k_global >= (q_pos - self.sliding_window + 1)
        mask = mx.where(causal & window, mx.array(0.0), mx.array(-1e9))

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=scale, mask=mask
        )

        # Reshape back: [1, n_heads, seq, head_dim] -> [seq, n_heads * head_dim]
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(seq_len, self.n_heads * self.head_dim)

        return self.wo(attn_out), (k_for_cache, v_for_cache)


class DecoderLayer(nn.Module):
    """Single decoder transformer layer with adaptive RMSNorm."""

    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.attention_norm = nn.RMSNorm(config.dim, eps=config.norm_eps)
        self.attention = DecoderAttention(config)
        self.ffn_norm = nn.RMSNorm(config.dim, eps=config.norm_eps)

        if config.ada_rms_norm_t_cond:
            self.ada_rms_norm_t_cond = AdaRMSNorm(config.dim, config.ada_rms_norm_t_cond_dim)
        else:
            self.ada_rms_norm_t_cond = None

        # SwiGLU FFN (no biases in decoder)
        self.feed_forward_w1 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.feed_forward_w3 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.feed_forward_w2 = nn.Linear(config.hidden_dim, config.dim, bias=False)

    def __call__(self, x, positions, ada_scale=None, cache=None):
        # Attention
        h = self.attention_norm(x)
        h, new_cache = self.attention(h, positions, cache=cache)
        x = x + h

        # FFN with adaptive norm
        h = self.ffn_norm(x)
        if self.ada_rms_norm_t_cond is not None and ada_scale is not None:
            h = self.ada_rms_norm_t_cond(h, ada_scale)

        gate = nn.silu(self.feed_forward_w1(h))
        up = self.feed_forward_w3(h)
        x = x + self.feed_forward_w2(gate * up)

        return x, new_cache


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
        return self.tok_embeddings.weight[token_id]

    def embed_tokens(self, token_ids: mx.array) -> mx.array:
        return self.tok_embeddings(token_ids)

    def forward(self, embeds, start_pos=0, cache=None):
        """Run decoder forward.

        Args:
            embeds: [seq, dim] input embeddings (audio_embed + tok_embed)
            start_pos: Starting position for RoPE
            cache: List of (k, v) per layer, or None

        Returns:
            (hidden_states, new_cache)
        """
        h = embeds
        seq_len = h.shape[0]
        positions = mx.arange(start_pos, start_pos + seq_len)

        new_cache = []
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            ada_scale = self._ada_scales[i] if self._ada_scales is not None else None
            h, kv = layer(h, positions, ada_scale=ada_scale, cache=layer_cache)
            new_cache.append(kv)

        h = self.norm(h)
        return h, new_cache

    def logits(self, h):
        """Compute logits via tied embeddings: h @ tok_embeddings^T."""
        return h @ self.tok_embeddings.weight.T
