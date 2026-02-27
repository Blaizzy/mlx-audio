from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


def _find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


def _apply_rotary_emb(x: mx.array, freqs: mx.array) -> mx.array:
    # x: (B, L, H, D)
    # freqs: (L, D/2, 2) with real/imag
    xshaped = x.astype(mx.float32).reshape(*x.shape[:-1], -1, 2)
    freqs = freqs.reshape(1, xshaped.shape[1], 1, xshaped.shape[3], 2)
    re = xshaped[..., 0] * freqs[..., 0] - xshaped[..., 1] * freqs[..., 1]
    im = xshaped[..., 1] * freqs[..., 0] + xshaped[..., 0] * freqs[..., 1]
    out = mx.stack([re, im], axis=-1).reshape(x.shape)
    return out.astype(x.dtype)


def _precompute_freqs_cis(seq_len: int, n_elem: int, base: int = 10000) -> mx.array:
    freqs = 1.0 / (base ** (mx.arange(0, n_elem, 2, dtype=mx.float32) / n_elem))
    t = mx.arange(seq_len, dtype=mx.float32)
    freqs = mx.outer(t, freqs)
    return mx.stack([mx.cos(freqs), mx.sin(freqs)], axis=-1).astype(mx.float32)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        x_f = x.astype(mx.float32)
        denom = mx.rsqrt(mx.mean(x_f * x_f, axis=-1, keepdims=True) + self.eps)
        y = x_f * denom
        return (y * self.weight).astype(x.dtype)


class AdaptiveLayerNorm(nn.Module):
    def __init__(self, d_model: int, norm: nn.Module):
        super().__init__()
        self.project_layer = nn.Linear(d_model, 2 * d_model)
        self.norm = norm
        self.d_model = d_model

    def __call__(self, x: mx.array, embedding: Optional[mx.array] = None) -> mx.array:
        if embedding is None:
            return self.norm(x)
        weight, bias = mx.split(self.project_layer(embedding), 2, axis=-1)
        return weight[:, None, :] * self.norm(x) + bias[:, None, :]


@dataclass
class GPTFastArgs:
    block_size: int = 16384
    n_layer: int = 13
    n_head: int = 8
    dim: int = 512
    head_dim: int = 64
    n_local_heads: int = 8
    intermediate_size: int = 1536
    rope_base: float = 10000
    norm_eps: float = 1e-5
    uvit_skip_connection: bool = True
    time_as_token: bool = False


class GPTFastAttention(nn.Module):
    def __init__(self, config: GPTFastArgs):
        super().__init__()
        assert config.dim % config.n_head == 0

        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        self.wqkv = nn.Linear(config.dim, total_head_dim, bias=False)
        self.wo = nn.Linear(config.head_dim * config.n_head, config.dim, bias=False)
        self.n_head = config.n_head
        self.n_local_heads = config.n_local_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5

    def __call__(
        self,
        x: mx.array,
        freqs_cis: mx.array,
        mask: mx.array,
    ) -> mx.array:
        B, L, _ = x.shape
        kv_size = self.n_local_heads * self.head_dim
        q, k, v = mx.split(self.wqkv(x), [kv_size, 2 * kv_size], axis=-1)

        q = q.reshape(B, L, self.n_head, self.head_dim)
        k = k.reshape(B, L, self.n_local_heads, self.head_dim)
        v = v.reshape(B, L, self.n_local_heads, self.head_dim)

        q = _apply_rotary_emb(q, freqs_cis)
        k = _apply_rotary_emb(k, freqs_cis)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        if self.n_local_heads < self.n_head:
            n_rep = self.n_head // self.n_local_heads
            k = mx.repeat(k, n_rep, axis=1)
            v = mx.repeat(v, n_rep, axis=1)

        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        y = y.transpose(0, 2, 1, 3).reshape(B, L, self.n_head * self.head_dim)
        return self.wo(y)


class GPTFastFeedForward(nn.Module):
    def __init__(self, config: GPTFastArgs):
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class GPTFastBlock(nn.Module):
    def __init__(self, config: GPTFastArgs):
        super().__init__()
        self.attention = GPTFastAttention(config)
        self.feed_forward = GPTFastFeedForward(config)
        self.ffn_norm = AdaptiveLayerNorm(config.dim, RMSNorm(config.dim, eps=config.norm_eps))
        self.attention_norm = AdaptiveLayerNorm(
            config.dim, RMSNorm(config.dim, eps=config.norm_eps)
        )

        if config.uvit_skip_connection:
            self.skip_in_linear = nn.Linear(config.dim * 2, config.dim)
            self.uvit_skip_connection = True
        else:
            self.uvit_skip_connection = False

        self.time_as_token = config.time_as_token

    def __call__(
        self,
        x: mx.array,
        c: mx.array,
        input_pos: mx.array,
        freqs_cis: mx.array,
        mask: mx.array,
        skip_in_x: Optional[mx.array] = None,
    ) -> mx.array:
        c_in = None if self.time_as_token else c
        if self.uvit_skip_connection and skip_in_x is not None:
            x = self.skip_in_linear(mx.concatenate([x, skip_in_x], axis=-1))

        h = x + self.attention(self.attention_norm(x, c_in), freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h, c_in))
        return out


class GPTFastTransformer(nn.Module):
    def __init__(self, config: GPTFastArgs):
        super().__init__()
        self.config = config
        self.layers = [GPTFastBlock(config) for _ in range(config.n_layer)]
        self.norm = AdaptiveLayerNorm(config.dim, RMSNorm(config.dim, eps=config.norm_eps))

        self.freqs_cis = _precompute_freqs_cis(config.block_size, config.head_dim, int(config.rope_base))

        self.uvit_skip_connection = config.uvit_skip_connection
        if self.uvit_skip_connection:
            self.layers_emit_skip = [i for i in range(config.n_layer) if i < config.n_layer // 2]
            self.layers_receive_skip = [i for i in range(config.n_layer) if i > config.n_layer // 2]
        else:
            self.layers_emit_skip = []
            self.layers_receive_skip = []

    def __call__(
        self,
        x: mx.array,
        c: mx.array,
        input_pos: mx.array,
        mask: mx.array,
    ) -> mx.array:
        freqs = self.freqs_cis[input_pos]
        skip_stack = []
        for i, layer in enumerate(self.layers):
            if self.uvit_skip_connection and i in self.layers_receive_skip:
                skip_in_x = skip_stack.pop()
            else:
                skip_in_x = None
            x = layer(x, c, input_pos, freqs, mask, skip_in_x=skip_in_x)
            if self.uvit_skip_connection and i in self.layers_emit_skip:
                skip_stack.append(x)
        return self.norm(x, c)
