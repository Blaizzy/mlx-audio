# Copyright (c) 2025 Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)
# LFM2.5-Audio: Transformer backbone implementation

import math
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .config import LFM2Config


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.scale = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        rms = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return x / rms * self.scale


def precompute_freqs_cis(
    dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
) -> mx.array:
    """Precompute rotary embedding frequencies."""
    freqs = 1.0 / (theta ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
    t = mx.arange(max_seq_len, dtype=mx.float32)
    freqs = mx.outer(t, freqs)
    return freqs


def apply_rotary_emb(
    xq: mx.array,
    xk: mx.array,
    freqs: mx.array,
    offset: int = 0,
) -> Tuple[mx.array, mx.array]:
    """Apply rotary embeddings to queries and keys."""
    seq_len = xq.shape[1]
    freqs = freqs[offset : offset + seq_len]

    # Reshape to match xq shape
    freqs = mx.expand_dims(freqs, axis=(0, 2))

    # Split into real and imaginary parts
    xq_r, xq_i = mx.split(xq.reshape(*xq.shape[:-1], -1, 2), 2, axis=-1)
    xk_r, xk_i = mx.split(xk.reshape(*xk.shape[:-1], -1, 2), 2, axis=-1)

    xq_r = xq_r.squeeze(-1)
    xq_i = xq_i.squeeze(-1)
    xk_r = xk_r.squeeze(-1)
    xk_i = xk_i.squeeze(-1)

    # Apply rotation
    cos = mx.cos(freqs)
    sin = mx.sin(freqs)

    xq_out_r = xq_r * cos - xq_i * sin
    xq_out_i = xq_r * sin + xq_i * cos
    xk_out_r = xk_r * cos - xk_i * sin
    xk_out_i = xk_r * sin + xk_i * cos

    # Interleave back
    xq_out = mx.stack([xq_out_r, xq_out_i], axis=-1).reshape(xq.shape)
    xk_out = mx.stack([xk_out_r, xk_out_i], axis=-1).reshape(xk.shape)

    return xq_out, xk_out


class SwiGLU(nn.Module):
    """SwiGLU activation with gated linear unit."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int = 256,
    ):
        super().__init__()
        # Round hidden_dim to multiple_of
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class Attention(nn.Module):
    """Multi-head attention with GQA support, RoPE, and optional bounded attention norms."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        max_seq_len: int = 128000,
        rope_theta: float = 1000000.0,
        use_qk_norm: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.use_qk_norm = use_qk_norm

        self.q_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, dim, bias=False)

        # Optional Q/K layer norms (bounded attention)
        if use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        # RoPE frequencies
        self.freqs = precompute_freqs_cis(self.head_dim, max_seq_len, rope_theta)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array]]]:
        B, L, _ = x.shape

        # Projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to (B, L, num_heads, head_dim)
        q = q.reshape(B, L, self.num_heads, self.head_dim)
        k = k.reshape(B, L, self.num_kv_heads, self.head_dim)
        v = v.reshape(B, L, self.num_kv_heads, self.head_dim)

        # Apply Q/K norms if using bounded attention
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Apply RoPE
        offset = 0 if cache is None else cache[0].shape[1]
        q, k = apply_rotary_emb(q, k, self.freqs, offset)

        # Handle KV cache
        if cache is not None:
            k_cache, v_cache = cache
            k = mx.concatenate([k_cache, k], axis=1)
            v = mx.concatenate([v_cache, v], axis=1)

        new_cache = (k, v)

        # Transpose to (B, num_heads, L, head_dim)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Expand KV for GQA
        if self.num_kv_heads < self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = mx.repeat(k, n_rep, axis=1)
            v = mx.repeat(v, n_rep, axis=1)

        # Scaled dot-product attention
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale

        if mask is not None:
            scores = scores + mask

        attn = mx.softmax(scores, axis=-1)
        out = attn @ v

        # Reshape back
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out), new_cache


class ConvBlock(nn.Module):
    """Convolutional block for hybrid LFM architecture with gating."""

    def __init__(
        self,
        dim: int,
        kernel_size: int = 4,
        bias: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size

        # Input projection with 3x expansion for gating (GLU-like)
        self.in_proj = nn.Linear(dim, 3 * dim, bias=bias)

        # Depthwise causal conv1d - MLX expects (batch, seq, channels)
        # Using groups=dim for depthwise convolution (each channel processed independently)
        self.conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            bias=bias,
            groups=dim,  # Depthwise: each channel has its own kernel
        )

        # Output projection
        self.out_proj = nn.Linear(dim, dim, bias=bias)

    def __call__(
        self,
        x: mx.array,
        cache: Optional[mx.array] = None,
    ) -> Tuple[mx.array, Optional[mx.array]]:
        B, L, D = x.shape

        # Input projection with gating
        projected = self.in_proj(x)  # (B, L, 3*D)
        gate, a, b = mx.split(projected, 3, axis=-1)  # Each (B, L, D)

        # Apply gating: silu(gate) * a + b-like structure
        # This is similar to GLU/SwiGLU gating
        x_gated = nn.silu(gate) * a

        # Handle caching for causal convolution
        if cache is not None:
            x_conv = mx.concatenate([cache, x_gated], axis=1)
            new_cache = x_conv[:, -(self.kernel_size - 1):, :]
        else:
            # Pad for causal conv
            pad = mx.zeros((B, self.kernel_size - 1, D))
            x_conv = mx.concatenate([pad, x_gated], axis=1)
            new_cache = x_gated[:, -(self.kernel_size - 1):, :] if L >= self.kernel_size - 1 else x_gated

        # Apply convolution
        out = self.conv(x_conv)

        # Add the residual path (b) and project
        out = self.out_proj(out + b)

        return out, new_cache


class TransformerBlock(nn.Module):
    """Transformer block with attention and feed-forward."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        ff_dim: int,
        max_seq_len: int = 128000,
        rope_theta: float = 1000000.0,
        norm_eps: float = 1e-5,
        multiple_of: int = 256,
        use_qk_norm: bool = True,
    ):
        super().__init__()
        self.attn_norm = RMSNorm(dim, eps=norm_eps)
        self.attn = Attention(dim, num_heads, num_kv_heads, max_seq_len, rope_theta, use_qk_norm)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn = SwiGLU(dim, ff_dim, multiple_of)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array]]]:
        # Attention with residual
        h, new_cache = self.attn(self.attn_norm(x), mask, cache)
        x = x + h

        # FFN with residual
        x = x + self.ffn(self.ffn_norm(x))

        return x, new_cache


class ConvTransformerBlock(nn.Module):
    """Convolutional block with feed-forward."""

    def __init__(
        self,
        dim: int,
        ff_dim: int,
        kernel_size: int = 4,
        norm_eps: float = 1e-5,
        multiple_of: int = 256,
        bias: bool = False,
    ):
        super().__init__()
        self.conv_norm = RMSNorm(dim, eps=norm_eps)
        self.conv = ConvBlock(dim, kernel_size, bias)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn = SwiGLU(dim, ff_dim, multiple_of)

    def __call__(
        self,
        x: mx.array,
        cache: Optional[mx.array] = None,
    ) -> Tuple[mx.array, Optional[mx.array]]:
        # Conv with residual
        h, new_cache = self.conv(self.conv_norm(x), cache)
        x = x + h

        # FFN with residual
        x = x + self.ffn(self.ffn_norm(x))

        return x, new_cache


class LFM2Backbone(nn.Module):
    """LFM2 language model backbone with hybrid conv+attention architecture."""

    def __init__(self, config: LFM2Config):
        super().__init__()
        self.config = config

        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Build layers based on layer_types
        self.layers = []
        for layer_type in config.layer_types:
            if layer_type == "full_attention":
                layer = TransformerBlock(
                    dim=config.hidden_size,
                    num_heads=config.num_attention_heads,
                    num_kv_heads=config.num_key_value_heads,
                    ff_dim=config.intermediate_size,
                    max_seq_len=config.max_position_embeddings,
                    rope_theta=config.rope_theta,
                    norm_eps=config.norm_eps,
                    multiple_of=config.block_multiple_of,
                )
            else:  # conv
                layer = ConvTransformerBlock(
                    dim=config.conv_dim,
                    ff_dim=config.block_ff_dim,
                    kernel_size=config.conv_L_cache,  # conv_L_cache is the kernel size
                    norm_eps=config.block_norm_eps,
                    multiple_of=config.block_multiple_of,
                    bias=config.conv_bias,
                )
            self.layers.append(layer)

        # Final norm
        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        cache: Optional[List[Any]] = None,
        use_cache: bool = False,
    ) -> Tuple[mx.array, Optional[List[Any]]]:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        x = inputs_embeds
        B, L, _ = x.shape

        # Create causal mask if needed
        if attention_mask is None and L > 1:
            attention_mask = mx.triu(
                mx.full((L, L), float("-inf")), k=1
            )
            attention_mask = mx.expand_dims(attention_mask, axis=(0, 1))

        new_cache = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None

            if isinstance(layer, TransformerBlock):
                x, layer_new_cache = layer(x, attention_mask, layer_cache)
            else:  # ConvTransformerBlock
                x, layer_new_cache = layer(x, layer_cache)

            if use_cache:
                new_cache.append(layer_new_cache)

        x = self.norm(x)

        return x, new_cache


class Depthformer(nn.Module):
    """Depthformer for audio frame generation using self-attention."""

    def __init__(
        self,
        layers: int,
        dim: int,
        num_heads: int = 32,
        num_kv_heads: int = 8,
        ff_dim: Optional[int] = None,
        tie: bool = True,
    ):
        super().__init__()
        self.layers_count = layers
        self.dim = dim
        self.tie = tie

        ff_dim = ff_dim or dim * 4

        # Always create separate blocks for weight loading compatibility
        # The tie parameter affects runtime behavior (weight sharing) but
        # for loading pretrained weights, we need separate blocks
        self.blocks = [
            TransformerBlock(
                dim=dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                ff_dim=ff_dim,
                max_seq_len=4096,
                rope_theta=10000.0,
                use_qk_norm=True,  # Depthformer uses bounded attention
            )
            for _ in range(layers)
        ]

    def __call__(
        self,
        x: mx.array,
        cache: Optional[List[Any]] = None,
        use_cache: bool = False,
    ) -> Tuple[mx.array, Optional[List[Any]]]:
        new_cache = [] if use_cache else None

        for i in range(self.layers_count):
            layer_cache = cache[i] if cache is not None else None
            x, layer_new_cache = self.blocks[i](x, cache=layer_cache)

            if use_cache:
                new_cache.append(layer_new_cache)

        return x, new_cache


class SharedEmbedding(nn.Module):
    """Shared embedding with optional weight tying to output projection."""

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        tie_weights: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.tie_weights = tie_weights

        self.weight = mx.zeros((vocab_size, dim))

    def embed(self, x: mx.array) -> mx.array:
        """Embed tokens."""
        return self.weight[x]

    def project(self, x: mx.array) -> mx.array:
        """Project to vocabulary."""
        return x @ self.weight.T
