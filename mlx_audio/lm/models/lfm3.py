# Copyright 2026 The HuggingFace Team. All rights reserved.
# Adapted from LiquidAI's Apache-2.0 LFM3 reference implementation.
# SPDX-License-Identifier: Apache-2.0
# See mlx_audio/sts/models/lfm3_audio/README.md for license information.
"""LFM3 hybrid convolution, global attention, and sliding attention backbone."""

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask
from .cache import ArraysCache, KVCache, RotatingKVCache
from .lfm2 import MLP
from .lfm2 import Attention as Lfm2Attention
from .lfm2 import ShortConv


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "lfm3"
    vocab_size: int = 128000
    hidden_size: int = 2048
    intermediate_size: int = 7680
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int | None = None
    norm_eps: float = 1e-5
    conv_bias: bool = False
    conv_L_cache: int = 7
    layer_types: list[str] | None = None
    sliding_window: int | None = 256
    use_sandwich_norm: bool = True
    rope_parameters: dict | None = None
    rope_theta: float = 1000000.0
    block_multiple_of: int = 256
    block_ffn_dim_multiplier: float = 1.0
    block_auto_adjust_ff_dim: bool = True
    tie_word_embeddings: bool = True
    max_position_embeddings: int = 128000

    def __post_init__(self):
        if self.layer_types is None:
            self.layer_types = [
                "conv",
                "conv",
                "full_attention",
                "conv",
                "sliding_attention",
                "conv",
                "full_attention",
                "conv",
                "sliding_attention",
                "conv",
                "full_attention",
                "conv",
                "sliding_attention",
                "conv",
                "full_attention",
                "conv",
                "sliding_attention",
                "conv",
                "full_attention",
                "conv",
                "sliding_attention",
                "conv",
                "full_attention",
                "conv",
            ][: self.num_hidden_layers]
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types must contain one entry per layer")
        if set(self.layer_types) - {"conv", "full_attention", "sliding_attention"}:
            raise ValueError("Unsupported LFM layer type")
        if "sliding_attention" in self.layer_types and (
            self.sliding_window is None or self.sliding_window < 1
        ):
            raise ValueError("Sliding attention requires a positive sliding_window")
        self.head_dim = self.head_dim or self.hidden_size // self.num_attention_heads
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("Attention heads must be divisible by key/value heads")
        if self.rope_parameters is None:
            self.rope_parameters = {
                "full_attention": {
                    "rope_theta": self.rope_theta,
                    "partial_rotary_factor": 0.5,
                },
                "sliding_attention": {"rope_theta": 10000.0},
            }
        if "rope_theta" in self.rope_parameters:
            self.rope_theta = self.rope_parameters["rope_theta"]


class RMSNorm(nn.Module):
    """Backbone norms round the normalized input before multiplying the weight."""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x):
        normalized = mx.fast.rms_norm(x.astype(mx.float32), None, self.eps)
        return normalized.astype(x.dtype) * self.weight


class Attention(nn.Module):
    def __init__(self, args, layer_type):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(
            args.hidden_size, self.n_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            args.hidden_size, self.n_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            args.hidden_size, self.n_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim, args.hidden_size, bias=False
        )
        self.q_norm = RMSNorm(self.head_dim, args.norm_eps)
        self.k_norm = RMSNorm(self.head_dim, args.norm_eps)
        self.gate_proj = nn.Linear(args.hidden_size, self.n_heads, bias=False)
        params = args.rope_parameters.get(layer_type, args.rope_parameters)
        if params.get("rope_type", "default") != "default":
            raise ValueError("LFM3 currently supports default RoPE only")
        self._theta = params.get("rope_theta", args.rope_theta)
        self._rotated = (
            int(params.get("partial_rotary_factor", 1.0) * self.head_dim) // 2
        )
        if not 0 < self._rotated <= self.head_dim // 2:
            raise ValueError("Invalid partial_rotary_factor")

    def _rope(self, x, offset):
        # Partial rotation uses the full head's frequency spacing and a zero tail,
        # rather than rotating a contiguous prefix with a smaller RoPE dimension.
        index = mx.arange(self.head_dim // 2)
        inv = mx.where(
            index < self._rotated, self._theta ** (-2 * index / self.head_dim), 0
        )
        angles = (mx.arange(x.shape[-2]) + offset)[:, None].astype(mx.float32) * inv
        angles = mx.concatenate([angles, angles], axis=-1)
        half = self.head_dim // 2
        rotated = mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)
        return (
            x.astype(mx.float32) * mx.cos(angles)
            + rotated.astype(mx.float32) * mx.sin(angles)
        ).astype(x.dtype)

    def __call__(self, x, mask=None, cache=None):
        batch, length, _ = x.shape
        q = self.q_norm(
            self.q_proj(x).reshape(batch, length, self.n_heads, self.head_dim)
        ).transpose(0, 2, 1, 3)
        k = self.k_norm(
            self.k_proj(x).reshape(batch, length, self.n_kv_heads, self.head_dim)
        ).transpose(0, 2, 1, 3)
        v = (
            self.v_proj(x)
            .reshape(batch, length, self.n_kv_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        offset = cache.offset if cache is not None else 0
        q, k = self._rope(q, offset), self._rope(k, offset)
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        y = y.transpose(0, 2, 1, 3) * mx.sigmoid(self.gate_proj(x))[..., None]
        return self.o_proj(y.reshape(batch, length, -1))


class DecoderLayer(nn.Module):
    def __init__(self, args, index):
        super().__init__()
        self.layer_type = args.layer_types[index]
        self.is_attention_layer = self.layer_type != "conv"
        self._sandwich = args.use_sandwich_norm and args.model_type != "lfm2"
        if self.is_attention_layer:
            self.self_attn = (
                Lfm2Attention(args)
                if args.model_type == "lfm2"
                else Attention(args, self.layer_type)
            )
        else:
            self.conv = ShortConv(args, index)
        self.feed_forward = MLP(
            args.hidden_size,
            args.intermediate_size,
            args.block_multiple_of,
            args.model_type == "lfm2" and args.block_auto_adjust_ff_dim,
            args.block_ffn_dim_multiplier,
        )
        self.operator_norm = RMSNorm(args.hidden_size, args.norm_eps)
        self.ffn_norm = RMSNorm(args.hidden_size, args.norm_eps)
        if self._sandwich:
            self.post_operator_norm = RMSNorm(args.hidden_size, args.norm_eps)
            self.post_ffn_norm = RMSNorm(args.hidden_size, args.norm_eps)

    def __call__(self, x, mask=None, cache=None):
        op = self.self_attn if self.is_attention_layer else self.conv
        y = op(self.operator_norm(x), mask=mask, cache=cache)
        if self._sandwich:
            y = self.post_operator_norm(y)
        x = x + y
        y = self.feed_forward(self.ffn_norm(x))
        if self._sandwich:
            y = self.post_ffn_norm(y)
        return x + y


class Lfm3Model(nn.Module):
    def __init__(self, args, *, embed_tokens=True):
        super().__init__()
        self.args = args
        if embed_tokens:
            self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.embedding_norm = RMSNorm(args.hidden_size, args.norm_eps)

    def make_cache(self):
        return [
            (
                ArraysCache(size=1)
                if kind == "conv"
                else (
                    RotatingKVCache(max_size=self.args.sliding_window, keep=0)
                    if kind == "sliding_attention"
                    else KVCache()
                )
            )
            for kind in self.args.layer_types
        ]

    def __call__(
        self,
        inputs=None,
        cache=None,
        input_embeddings=None,
        *,
        causal=True,
        window_size=None,
    ):
        x = self.embed_tokens(inputs) if input_embeddings is None else input_embeddings
        if cache is None:
            cache = [None] * len(self.layers)
        if len(cache) != len(self.layers):
            raise ValueError("Cache must contain one entry per layer")
        for layer, c in zip(self.layers, cache):
            window = (
                self.args.sliding_window
                if layer.layer_type == "sliding_attention"
                else window_size
            )
            mask = (
                create_attention_mask(x, c, window_size=window)
                if causal and layer.is_attention_layer
                else None
            )
            x = layer(x, mask=mask, cache=c)
        return self.embedding_norm(x)


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = Lfm3Model(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs=None, cache=None, input_embeddings=None):
        h = self.model(inputs, cache=cache, input_embeddings=input_embeddings)
        return (
            self.model.embed_tokens.as_linear(h)
            if self.args.tie_word_embeddings
            else self.lm_head(h)
        )

    def make_cache(self):
        return self.model.make_cache()

    @property
    def layers(self):
        return self.model.layers
