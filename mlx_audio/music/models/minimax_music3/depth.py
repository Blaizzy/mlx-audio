"""Local residual-codebook decoder for MiniMax Music 3."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from .config import ModelConfig


class DepthAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        batch, length, _ = x.shape
        q = self.to_q(x).reshape(batch, length, self.heads, self.head_dim)
        k = self.to_k(x).reshape(batch, length, self.heads, self.head_dim)
        v = self.to_v(x).reshape(batch, length, self.heads, self.head_dim)
        q, k, v = (array.transpose(0, 2, 1, 3) for array in (q, k, v))
        y = mx.fast.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=1.0 / math.sqrt(self.head_dim),
            mask="causal",
        )
        return self.to_out(y.transpose(0, 2, 1, 3).reshape(batch, length, -1))


class DepthBlock(nn.Module):
    def __init__(self, dim: int, heads: int, intermediate: int, eps: float):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(dim, eps=eps)
        self.attn = DepthAttention(dim, heads)
        self.post_attention_layernorm = nn.RMSNorm(dim, eps=eps)
        self.gate_proj = nn.Linear(dim, intermediate, bias=False)
        self.up_proj = nn.Linear(dim, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.input_layernorm(x))
        normalized = self.post_attention_layernorm(x)
        return x + self.down_proj(
            nn.silu(self.gate_proj(normalized)) * self.up_proj(normalized)
        )


class RVQDepthDecoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        residual = config.residual_codebooks
        self.audio_embeddings = nn.Embedding(
            config.audio_vocab_size * residual, config.hidden_size
        )
        self.projection = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.pos_embedding = nn.Embedding(
            config.depth_max_position_embeddings, config.hidden_size
        )
        self.layers = [
            DepthBlock(
                config.hidden_size,
                config.depth_num_heads,
                config.depth_intermediate_size,
                config.rms_norm_eps,
            )
            for _ in range(config.depth_num_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.audio_heads = [
            nn.Linear(config.hidden_size, config.audio_vocab_size, bias=False)
            for _ in range(residual)
        ]

    def __call__(self, input_embeddings: mx.array) -> mx.array:
        hidden = input_embeddings + self.pos_embedding(
            mx.arange(input_embeddings.shape[1])
        )
        for layer in self.layers:
            hidden = layer(hidden)
        return self.norm(hidden)
