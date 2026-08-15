"""One-dimensional flow-matching transformer for MiniMax Music 3.

Adapted and modified from mikolaj92/minimax-music3-mlx under Apache-2.0.
See LICENSE and NOTICE.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from .config import ModelConfig


def _apply_partial_rotary(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    rotary_dim = cos.shape[-1]
    cos = cos[:, None, :].astype(x.dtype)
    sin = sin[:, None, :].astype(x.dtype)
    rotated = x[..., :rotary_dim]
    half = rotary_dim // 2
    rotate_half = mx.concatenate([-rotated[..., half:], rotated[..., :half]], axis=-1)
    rotated = rotated * cos + rotate_half * sin
    return mx.concatenate([rotated, x[..., rotary_dim:]], axis=-1)


class FourierEmbedding(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.weight = mx.random.normal((embedding_dim // 2, 1))

    def __call__(self, timestep: mx.array) -> mx.array:
        angles = 2.0 * math.pi * timestep.reshape(-1, 1) @ self.weight.T
        return mx.concatenate([mx.cos(angles), mx.sin(angles)], axis=-1)


class TimestepEmbedding(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(input_dim, output_dim)
        self.linear_2 = nn.Linear(output_dim, output_dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear_2(nn.silu(self.linear_1(x)))


class DiTAttention(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        inner = heads * head_dim
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_k = nn.Linear(dim, inner, bias=False)
        self.to_v = nn.Linear(dim, inner, bias=False)
        self.to_out = [nn.Linear(inner, dim, bias=False), nn.Dropout(0.0)]

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        batch, length, _ = x.shape
        q = self.to_q(x).reshape(batch, length, self.heads, self.head_dim)
        k = self.to_k(x).reshape(batch, length, self.heads, self.head_dim)
        v = self.to_v(x).reshape(batch, length, self.heads, self.head_dim)
        q = _apply_partial_rotary(q, cos, sin).transpose(0, 2, 1, 3)
        k = _apply_partial_rotary(k, cos, sin).transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        y = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=1.0 / math.sqrt(self.head_dim)
        )
        y = y.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self.to_out[1](self.to_out[0](y))


class DiTBlock(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int, ff_inner: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = DiTAttention(dim, heads, head_dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff_in = nn.Linear(dim, ff_inner * 2)
        self.ff_out = nn.Linear(ff_inner, dim)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        x = x + self.attn(self.norm1(x), cos, sin)
        states, gate = mx.split(self.ff_in(self.norm2(x)), 2, axis=-1)
        return x + self.ff_out(states * nn.silu(gate))


class FlowMatchingTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        inner = config.dit_num_heads * config.dit_head_dim
        concat = 2 * config.dit_in_channels + config.condition_out_dim
        self.time_proj = FourierEmbedding(config.dit_fourier_dim)
        self.time_embed = TimestepEmbedding(config.dit_fourier_dim, inner)
        self.preprocess_conv = nn.Conv1d(concat, concat, kernel_size=1, bias=False)
        self.proj_in = nn.Linear(concat, inner, bias=False)
        self.rotary_dim = config.dit_rotary_dim
        self.transformer_blocks = [
            DiTBlock(
                inner,
                config.dit_num_heads,
                config.dit_head_dim,
                config.dit_ff_inner_dim,
            )
            for _ in range(config.dit_num_layers)
        ]
        self.proj_out = nn.Linear(inner, config.dit_in_channels, bias=False)
        self.postprocess_conv = nn.Conv1d(
            config.dit_in_channels,
            config.dit_in_channels,
            kernel_size=1,
            bias=False,
        )

    def _rotary(self, length: int) -> tuple[mx.array, mx.array]:
        inv_freq = 1.0 / (
            10_000.0
            ** (mx.arange(0, self.rotary_dim, 2).astype(mx.float32) / self.rotary_dim)
        )
        frequencies = mx.outer(mx.arange(length).astype(mx.float32), inv_freq)
        frequencies = mx.concatenate([frequencies, frequencies], axis=-1)
        return mx.cos(frequencies), mx.sin(frequencies)

    def __call__(
        self,
        hidden_states: mx.array,
        timestep: mx.array,
        encoder_hidden_states: mx.array,
    ) -> mx.array:
        condition = encoder_hidden_states.transpose(0, 2, 1)
        x = mx.concatenate(
            [hidden_states, mx.zeros_like(hidden_states), condition], axis=1
        ).transpose(0, 2, 1)
        x = self.preprocess_conv(x) + x
        time_embedding = self.time_embed(self.time_proj(timestep))
        x = self.proj_in(x)
        x = mx.concatenate([time_embedding[:, None, :], x], axis=1)
        cos, sin = self._rotary(x.shape[1])
        for block in self.transformer_blocks:
            x = block(x, cos, sin)
        x = self.proj_out(x[:, 1:])
        x = self.postprocess_conv(x) + x
        return x.transpose(0, 2, 1)
