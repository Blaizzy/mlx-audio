# Ported from https://github.com/shivammehta25/Matcha-TTS
# Original: matcha/models/components/transformer.py

import math

import mlx.core as mx
import mlx.nn as nn


class DiffusersAttention(nn.Module):
    """
    Attention module matching diffusers.models.attention_processor.Attention.

    PyTorch diffusers uses:
        inner_dim = heads * dim_head  (e.g., 8 * 64 = 512)
        to_q, to_k, to_v: (query_dim, inner_dim)  (256 -> 512)
        to_out.0: (inner_dim, query_dim)  (512 -> 256)

    This differs from standard MHA where all dims equal query_dim.

    Weight names match sanitized format: query_proj, key_proj, value_proj, out_proj
    """

    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        bias: bool = False,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = heads * dim_head
        self.scale = dim_head**-0.5

        # Match sanitized weight naming: query_proj, key_proj, value_proj, out_proj
        self.query_proj = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.key_proj = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.value_proj = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.out_proj = nn.Linear(self.inner_dim, query_dim, bias=bias)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array = None,
    ) -> mx.array:
        B, T, _ = hidden_states.shape

        # Project to q, k, v
        q = self.query_proj(hidden_states)  # (B, T, inner_dim)
        k = self.key_proj(hidden_states)  # (B, T, inner_dim)
        v = self.value_proj(hidden_states)  # (B, T, inner_dim)

        # Reshape to (B, heads, T, dim_head)
        q = q.reshape(B, T, self.heads, self.dim_head).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.heads, self.dim_head).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.heads, self.dim_head).transpose(0, 2, 1, 3)

        # Prepare mask for mx.fast.scaled_dot_product_attention
        # MLX expects boolean mask broadcastable to (B, heads, T_q, T_kv)
        # where True = attend, False = mask out
        mask = None
        if attention_mask is not None:
            # attention_mask: (B, T) -> (B, 1, 1, T)
            if attention_mask.ndim == 2:
                mask = attention_mask[:, None, None, :]
            else:
                mask = attention_mask

        # Use MLX fast attention (fused kernel)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)

        # Reshape back: (B, heads, T, dim_head) -> (B, T, inner_dim)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, self.inner_dim)

        # Output projection
        out = self.out_proj(out)

        return out


class BasicTransformerBlock(nn.Module):
    """
    Basic transformer block for decoder.

    This is a simplified version used by Chatterbox. The full Matcha-TTS
    implementation includes cross-attention and other features not needed here.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout: float = 0.0,
        activation_fn: str = "gelu",
    ):
        super().__init__()
        # Separate norms for attention and feed-forward (matches original)
        self.norm1 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        # Use DiffusersAttention to match PyTorch weight shapes
        # PyTorch: inner_dim = heads * dim_head = 8 * 64 = 512
        # Projections: (256, 512) for q/k/v, (512, 256) for out
        # Named 'attn' to match sanitized weight keys (e.g., .attn.query_proj)
        self.attn = DiffusersAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            bias=False,
        )
        # Feed-forward with GEGLU
        # Sanitize converts: ff.net.0.proj -> ff.layers.0, ff.net.2 -> ff.layers.1
        self.ff = FeedForward(dim, dim * 4)
        self.activation_fn = activation_fn

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array = None,
        timestep: mx.array = None,
    ) -> mx.array:
        # Self-attention
        normed = self.norm1(hidden_states)
        attn_out = self.attn(normed, attention_mask=attention_mask)
        hidden_states = hidden_states + attn_out

        # Feed-forward
        normed = self.norm3(hidden_states)
        ff_out = self.ff(normed)
        hidden_states = hidden_states + ff_out

        return hidden_states


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        inner_dim = mult  # mult is passed as dim * 4, so inner_dim = dim * 4
        # Weights: ff.net.0.proj (256->1024), ff.net.2 (1024->256)
        # Sanitize renames: ff.net.0.proj -> ff.layers.0, ff.net.2 -> ff.layers.1
        self.layers = [
            nn.Linear(dim, inner_dim),  # 256 -> 1024
            nn.Linear(inner_dim, dim),  # 1024 -> 256
        ]

    def __call__(self, x: mx.array) -> mx.array:
        x = self.layers[0](x)
        x = nn.gelu(x)
        x = self.layers[1](x)
        return x
