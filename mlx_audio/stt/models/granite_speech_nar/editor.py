"""MLX implementation of the bidirectional Granite editor LM (Plan 4).

40-layer pre-norm decoder with:
- RMSNorm (no bias, fp32 internally)
- GQA self-attention (16 Q heads, 4 KV heads, head_dim 128)
- SwiGLU MLP (no bias)
- RoPE on Q and K (theta=10000)
- Four Granite-specific multipliers applied at exact locations
- Bidirectional attention (no causal mask)
- Tied embeddings (lm_head shares embed_tokens.weight)

The full forward signature is (inputs_embeds, position_ids) -> logits, where
`inputs_embeds` is the caller-built [audio | hypothesis_tokens] concatenation
already pre-divided by `embedding_multiplier=12`.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class GraniteRMSNorm(nn.Module):
    """RMSNorm: x * weight / sqrt(mean(x**2) + eps). fp32 internally."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones(dim)

    def __call__(self, x: mx.array) -> mx.array:
        dtype = x.dtype
        h = x.astype(mx.float32)
        variance = (h * h).mean(axis=-1, keepdims=True)
        h = h * mx.rsqrt(variance + self.eps)
        return (self.weight.astype(mx.float32) * h).astype(dtype)


class GraniteRotaryEmbedding(nn.Module):
    """RoPE position encoding.

    inv_freq is precomputed in __init__ and stored under an underscored name so
    MLX's parameter tree skips it — the analog of PyTorch's
    `register_buffer(..., persistent=False)` used upstream.

    Forward returns (cos, sin) tensors of shape [seq_len, head_dim] in the caller's
    requested dtype. cos/sin are computed in fp32 for numerical stability.
    """

    def __init__(self, head_dim: int, max_position: int, theta: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_position = max_position
        self.theta = theta
        # inv_freq[i] = 1 / (theta ** (2i / head_dim)) for i in 0..head_dim/2-1
        idx = mx.arange(0, head_dim, 2, dtype=mx.float32)
        self._inv_freq = 1.0 / (theta ** (idx / head_dim))  # [head_dim/2]

    def __call__(self, position_ids: mx.array, dtype) -> tuple[mx.array, mx.array]:
        # position_ids: [seq_len], int
        positions = position_ids.astype(mx.float32)
        # Outer product via broadcast: [seq_len, head_dim/2]
        freqs = positions[:, None] * self._inv_freq[None, :]
        # Duplicate along last dim to match head_dim: [seq_len, head_dim]
        emb = mx.concatenate([freqs, freqs], axis=-1)
        cos = mx.cos(emb)
        sin = mx.sin(emb)
        return cos.astype(dtype), sin.astype(dtype)


def rotate_half(x: mx.array) -> mx.array:
    """Splits last dim in half, returns concat([-second_half, first_half])."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(
    q: mx.array, k: mx.array, cos: mx.array, sin: mx.array
) -> tuple[mx.array, mx.array]:
    """Apply RoPE rotation to query and key.

    q, k: [batch, num_heads, seq, head_dim]  (or any shape ending in [seq, head_dim])
    cos, sin: [seq, head_dim]
    """
    # Broadcast cos/sin: prepend (q.ndim - cos.ndim) leading size-1 axes so that
    # shapes align regardless of whether q is 3-D or 4-D.
    n_extra = q.ndim - cos.ndim
    for _ in range(n_extra):
        cos = cos[None]
        sin = sin[None]
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot


class GraniteAttention(nn.Module):
    """Bidirectional GQA self-attention with Granite's custom `attention_multiplier` scale.

    Critical Granite quirk: the QK^T scale is `attention_multiplier=1/128`, NOT the
    standard `1/sqrt(head_dim)=1/sqrt(128)≈0.0884`.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        attention_multiplier: float,
    ):
        super().__init__()
        assert num_heads % num_kv_heads == 0
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.kv_groups = num_heads // num_kv_heads
        self.attention_multiplier = attention_multiplier  # = 1/128, not 1/sqrt(128)

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        B, T, H = x.shape
        dtype = x.dtype

        # Project + reshape to [B, num_heads, T, head_dim] (Q) / [B, num_kv_heads, T, head_dim] (K, V)
        q = (
            self.q_proj(x)
            .reshape(B, T, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        k = (
            self.k_proj(x)
            .reshape(B, T, self.num_kv_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        v = (
            self.v_proj(x)
            .reshape(B, T, self.num_kv_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )

        # Apply RoPE to Q and K (V is not rotated)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # GQA: repeat K and V along the head axis to match num_heads.
        # mx.repeat gives repeat_interleave semantics: [a, b] → [a, a, b, b] (verified).
        k = mx.repeat(k, self.kv_groups, axis=1)  # [B, num_heads, T, head_dim]
        v = mx.repeat(v, self.kv_groups, axis=1)

        # Bidirectional attention: no mask. Granite scale = attention_multiplier (1/128).
        attn = mx.fast.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=self.attention_multiplier,
        )
        # attn: [B, num_heads, T, head_dim] → [B, T, num_heads*head_dim]
        attn = attn.transpose(0, 2, 1, 3).reshape(B, T, self.num_heads * self.head_dim)
        return self.o_proj(attn).astype(dtype)


class GraniteMLP(nn.Module):
    """SwiGLU MLP: down_proj(silu(gate_proj(x)) * up_proj(x)). No bias."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        dtype = x.dtype
        gate = nn.silu(self.gate_proj(x))
        up = self.up_proj(x)
        h = self.down_proj(gate * up)
        return h.astype(dtype)


class GraniteDecoderLayer(nn.Module):
    """Pre-norm decoder layer with scaled residuals.

    Forward:
      residual = x
      x = input_layernorm(x); x = self_attn(x, cos, sin)
      x = residual + x * residual_multiplier      # 0.22
      residual = x
      x = post_attention_layernorm(x); x = mlp(x)
      x = residual + x * residual_multiplier
      return x
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        attention_multiplier: float,
        residual_multiplier: float,
        eps: float,
    ):
        super().__init__()
        self.residual_multiplier = residual_multiplier
        self.input_layernorm = GraniteRMSNorm(hidden_size, eps=eps)
        self.self_attn = GraniteAttention(
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            attention_multiplier,
        )
        self.post_attention_layernorm = GraniteRMSNorm(hidden_size, eps=eps)
        self.mlp = GraniteMLP(hidden_size, intermediate_size)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        dtype = x.dtype
        # Attention block
        residual = x
        h = self.input_layernorm(x)
        h = self.self_attn(h, cos, sin)
        x = residual + h * self.residual_multiplier
        # MLP block
        residual = x
        h = self.post_attention_layernorm(x)
        h = self.mlp(h)
        x = residual + h * self.residual_multiplier
        return x.astype(dtype)


class GraniteEditor(nn.Module):
    """Full Granite editor LM. Bidirectional. Tied embeddings.

    Forward signature: (inputs_embeds, position_ids) → logits.
      - inputs_embeds: [B, T, hidden_size] — caller-built, already pre-divided by
        embedding_multiplier (for audio_embeds) where applicable.
      - position_ids: [T] or [B, T] — contiguous range over the input sequence.
        For multi-segment packed inputs, the controller is responsible for
        constructing valid per-segment masking (out of scope for Plan 4).
      - logits: [B, T, vocab_size] — already divided by logits_scaling=8.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding_multiplier = cfg.embedding_multiplier
        self.logits_scaling = cfg.logits_scaling

        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.rotary_emb = GraniteRotaryEmbedding(
            head_dim=head_dim,
            max_position=cfg.max_position_embeddings,
            theta=cfg.rope_theta,
        )
        self.layers = [
            GraniteDecoderLayer(
                hidden_size=cfg.hidden_size,
                num_heads=cfg.num_attention_heads,
                num_kv_heads=cfg.num_key_value_heads,
                head_dim=head_dim,
                intermediate_size=cfg.intermediate_size,
                attention_multiplier=cfg.attention_multiplier,
                residual_multiplier=cfg.residual_multiplier,
                eps=cfg.rms_norm_eps,
            )
            for _ in range(cfg.num_hidden_layers)
        ]
        self.norm = GraniteRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        # No separate lm_head: weight is tied to embed_tokens. We compute logits
        # via `self.embed_tokens.as_linear(h)` in the forward — this method works
        # for both nn.Embedding (bf16 weight) and nn.QuantizedEmbedding (packed
        # weight + scales/biases), so the tied lm_head survives quantization.

    def __call__(self, inputs_embeds: mx.array, position_ids: mx.array) -> mx.array:
        # inputs_embeds: [B, T, hidden]; position_ids: [T] or [B, T]
        B, T, _ = inputs_embeds.shape
        dtype = inputs_embeds.dtype

        # Apply embedding multiplier to ALL inputs uniformly
        h = inputs_embeds * self.embedding_multiplier

        # Compute RoPE cos/sin once for the full sequence
        if position_ids.ndim == 2:
            # Take the first row — we don't support packed sequences in Plan 4
            position_ids = position_ids[0]
        cos, sin = self.rotary_emb(position_ids, dtype=h.dtype)

        # 40 decoder layers
        for layer in self.layers:
            h = layer(h, cos, sin)

        h = self.norm(h)

        # Tied lm_head via as_linear (quantization-safe; equivalent to h @ W.T
        # when the embedding is unquantized).
        logits = self.embed_tokens.as_linear(h)
        logits = logits / self.logits_scaling
        return logits.astype(dtype)
