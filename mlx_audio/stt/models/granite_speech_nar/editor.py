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
        return mx.fast.rms_norm(x, self.weight, self.eps).astype(x.dtype)


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
        rope_theta: float,
    ):
        super().__init__()
        assert num_heads % num_kv_heads == 0
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.attention_multiplier = attention_multiplier  # = 1/128, not 1/sqrt(128)
        self.rope_theta = rope_theta

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
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

        # Apply RoPE to Q and K (V is not rotated) via the fused mx.fast.rope kernel.
        # traditional=False = half rotation (NeoX/Llama, matching rotate_half from
        # previous code); offset=0 reproduces the contiguous arange positions.
        q = mx.fast.rope(
            q,
            self.head_dim,
            traditional=False,
            base=self.rope_theta,
            scale=1.0,
            offset=0,
        )
        k = mx.fast.rope(
            k,
            self.head_dim,
            traditional=False,
            base=self.rope_theta,
            scale=1.0,
            offset=0,
        )

        # Bidirectional GQA attention w/o mask. MLX's scaled_dot_product_attention
        # handles grouped-query heads natively (gqa_factor = num_heads/num_kv_heads),
        # so we pass the un-repeated K,V (num_kv_heads=4) directly to avoid explicitly
        # materializing the 4x K,V repeat.
        # Granite scale = attention_multiplier (1/128), not 1/sqrt(head_dim).
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
        rope_theta: float,
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
            rope_theta,
        )
        self.post_attention_layernorm = GraniteRMSNorm(hidden_size, eps=eps)
        self.mlp = GraniteMLP(hidden_size, intermediate_size)

    def __call__(self, x: mx.array) -> mx.array:
        dtype = x.dtype
        # Attention block
        residual = x
        h = self.input_layernorm(x)
        h = self.self_attn(h)
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
                rope_theta=cfg.rope_theta,
            )
            for _ in range(cfg.num_hidden_layers)
        ]
        self.norm = GraniteRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        # No separate lm_head: weight is tied to embed_tokens. We compute logits
        # via `self.embed_tokens.as_linear(h)` in the forward — this method works
        # for both nn.Embedding (bf16 weight) and nn.QuantizedEmbedding (packed
        # weight + scales/biases), so the tied lm_head survives quantization.

    def __call__(
        self,
        inputs_embeds: mx.array,
        position_ids: mx.array,
        logits_start: int | None = None,
    ) -> mx.array:
        # inputs_embeds: [B, T, hidden]; position_ids: [T] or [B, T]
        # logits_start: if given, only compute vocab logits for positions >=
        # logits_start (usually just the text tail), skipping the audio prefix.
        B, T, _ = inputs_embeds.shape
        dtype = inputs_embeds.dtype

        # Apply embedding multiplier to ALL inputs uniformly
        h = inputs_embeds * self.embedding_multiplier

        # RoPE is applied per-layer inside each GraniteAttention via mx.fast.rope
        # (offset=0 reproduces the contiguous arange positions the caller passes).
        # 40 decoder layers (run over the FULL sequence -- attention needs it)
        for layer in self.layers:
            h = layer(h)

        h = self.norm(h)

        # Tied lm_head via as_linear (quantization-safe; equivalent to h @ W.T
        # when the embedding is unquantized). Slice to the text tail first when
        # logits_start is given so we don't project (and discard) audio positions.
        if logits_start is not None:
            h = h[:, logits_start:, :]
        logits = self.embed_tokens.as_linear(h)
        logits = logits / self.logits_scaling
        return logits.astype(dtype)
