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
        qkv_bias: bool = False,
        out_bias: bool = True,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = heads * dim_head
        self.scale = dim_head**-0.5

        # Match sanitized weight naming: query_proj, key_proj, value_proj, out_proj
        # Note: CosyVoice2 has no bias on q/k/v but has bias on out_proj
        self.query_proj = nn.Linear(query_dim, self.inner_dim, bias=qkv_bias)
        self.key_proj = nn.Linear(query_dim, self.inner_dim, bias=qkv_bias)
        self.value_proj = nn.Linear(query_dim, self.inner_dim, bias=qkv_bias)
        self.out_proj = nn.Linear(self.inner_dim, query_dim, bias=out_bias)

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

        # Handle attention mask
        # attention_mask can be:
        # - None: no masking
        # - Boolean mask (B, T) or (B, T, T): True = attend
        # - Additive bias (B, T, T): 0 = attend, large negative = mask
        if attention_mask is not None:
            # Check if it's additive bias (contains large negative values)
            # or boolean mask
            if attention_mask.dtype == mx.bool_:
                # Boolean mask - convert to proper shape for SDPA
                if attention_mask.ndim == 2:
                    mask = attention_mask[:, None, None, :]
                elif attention_mask.ndim == 3:
                    # (B, T_q, T_kv) -> (B, 1, T_q, T_kv)
                    mask = attention_mask[:, None, :, :]
                else:
                    mask = attention_mask
                out = mx.fast.scaled_dot_product_attention(
                    q, k, v, scale=self.scale, mask=mask
                )
            else:
                # Additive bias - compute attention manually
                # attention_mask shape: (B, T_q, T_kv) with values 0 or -1e10
                scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale  # (B, heads, T, T)
                # Add bias (broadcast across heads)
                if attention_mask.ndim == 3:
                    scores = scores + attention_mask[:, None, :, :]
                else:
                    scores = scores + attention_mask
                weights = mx.softmax(scores, axis=-1)
                out = weights @ v
        else:
            # No mask - use fast path
            out = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=self.scale, mask=None
            )

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
        # Note: CosyVoice2 has no bias on q/k/v but has bias on out_proj
        self.attn = DiffusersAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            qkv_bias=False,
            out_bias=True,
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
        # Keys: ff.layers.0.weight/bias, ff.layers.1.weight/bias
        # Use a container class that exposes indexed attributes
        self.layers = LayerList(
            [
                nn.Linear(dim, inner_dim),  # 256 -> 1024
                nn.Linear(inner_dim, dim),  # 1024 -> 256
            ]
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = self.layers[0](x)
        x = nn.gelu(x)
        x = self.layers[1](x)
        return x


class LayerList(nn.Module):
    """
    A list-like container for layers that exposes indexed attributes.

    This allows weight loading with keys like 'layers.0.weight'.
    """

    def __init__(self, layers):
        super().__init__()
        for i, layer in enumerate(layers):
            setattr(self, str(i), layer)
        self._n_layers = len(layers)

    def __getitem__(self, idx):
        return getattr(self, str(idx))

    def __len__(self):
        return self._n_layers
