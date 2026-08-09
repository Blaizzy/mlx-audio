# Copyright © 2025 Apple Inc.
# Vendored verbatim from mlx-lm v0.31.3 (ed1fca4cef15a824c5f1702c80f70b4cffc8e4dd),
# mlx_lm/models/lfm2.py. MIT licensed.
from dataclasses import dataclass
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

from .activations import swiglu
from .base import (
    BaseModelArgs,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from .cache import ArraysCache, KVCache


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int
    norm_eps: float
    conv_bias: bool
    conv_L_cache: int
    block_dim: int
    block_ff_dim: int
    block_multiple_of: int
    block_ffn_dim_multiplier: float
    block_auto_adjust_ff_dim: bool
    rope_theta: float = 1000000.0
    rope_parameters: Optional[dict] = None
    full_attn_idxs: Optional[List[int]] = None
    layer_types: Optional[List[str]] = None

    def __post_init__(self):
        if self.rope_parameters is not None and "rope_theta" in self.rope_parameters:
            self.rope_theta = self.rope_parameters["rope_theta"]
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.full_attn_idxs is None:
            self.full_attn_idxs = [
                i
                for i, layer_type in enumerate(self.layer_types)
                if layer_type == "full_attention"
            ]


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads

        self.head_dim = head_dim = args.hidden_size // n_heads

        self.scale = head_dim**-0.5

        self.q_layernorm = nn.RMSNorm(head_dim, eps=args.norm_eps)
        self.k_layernorm = nn.RMSNorm(head_dim, eps=args.norm_eps)

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        self.rope = nn.RoPE(
            self.head_dim,
            base=args.rope_theta,
            traditional=False,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = self.q_layernorm(queries.reshape(B, L, self.n_heads, -1)).transpose(
            0, 2, 1, 3
        )
        keys = self.k_layernorm(keys.reshape(B, L, self.n_kv_heads, -1)).transpose(
            0, 2, 1, 3
        )
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, mask=mask, scale=self.scale
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.out_proj(output)


class ShortConv(nn.Module):
    def __init__(
        self,
        args: ModelArgs,
        layer_idx: int,
    ):
        super().__init__()
        self.args = args
        self.layer_idx = layer_idx
        self.L_cache = args.conv_L_cache
        self.bias = args.conv_bias

        self.conv = nn.Conv1d(
            in_channels=args.hidden_size,
            out_channels=args.hidden_size,
            kernel_size=self.L_cache,
            groups=args.hidden_size,
            bias=self.bias,
        )
        self.in_proj = nn.Linear(args.hidden_size, 3 * args.hidden_size, bias=self.bias)
        self.out_proj = nn.Linear(args.hidden_size, args.hidden_size, bias=self.bias)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ):
        BCx = self.in_proj(x)
        B, C, x = mx.split(BCx, 3, axis=-1)
        Bx = B * x
        if mask is not None:
            Bx = mx.where(mask[..., None], Bx, 0)

        if cache is not None:
            if cache[0] is None:
                state = mx.zeros(
                    (Bx.shape[0], self.L_cache - 1, self.args.hidden_size),
                    dtype=Bx.dtype,
                )
            else:
                state = cache[0]
            Bx = mx.concatenate([state, Bx], axis=1)
            n_keep = self.L_cache - 1
            t = x.shape[1]
            if cache.lengths is not None:
                ends = mx.clip(cache.lengths, 0, t)
                positions = (ends[:, None] + mx.arange(n_keep))[..., None]
                cache[0] = mx.take_along_axis(Bx, positions, axis=1)
            else:
                cache[0] = Bx[:, -n_keep:, :]
            cache.advance(t)
        else:
            Bx = mx.pad(Bx, [(0, 0), (self.L_cache - 1, 0), (0, 0)])

        conv_out = self.conv(Bx)

        y = C * conv_out
        return self.out_proj(y)


class MLP(nn.Module):
    def __init__(
        self,
        dim: int,
        ff_dim: int,
        multiple_of: int,
        auto_adjust_ff_dim: bool,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        if auto_adjust_ff_dim:
            ff_dim = int(2 * ff_dim / 3)
            if ffn_dim_multiplier is not None:
                ff_dim = int(ffn_dim_multiplier * ff_dim)
            ff_dim = multiple_of * ((ff_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, ff_dim, bias=False)
        self.w3 = nn.Linear(dim, ff_dim, bias=False)
        self.w2 = nn.Linear(ff_dim, dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.w2(swiglu(self.w1(x), self.w3(x)))


class Lfm2DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.is_attention_layer = layer_idx in args.full_attn_idxs

        if self.is_attention_layer:
            self.self_attn = Attention(args)
        else:
            self.conv = ShortConv(args, layer_idx)
        self.feed_forward = MLP(
            dim=args.block_dim,
            ff_dim=args.block_ff_dim,
            multiple_of=args.block_multiple_of,
            auto_adjust_ff_dim=args.block_auto_adjust_ff_dim,
            ffn_dim_multiplier=args.block_ffn_dim_multiplier,
        )

        self.operator_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_eps)
        self.ffn_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:

        if self.is_attention_layer:
            r = self.self_attn(self.operator_norm(x), mask=mask, cache=cache)
        else:
            r = self.conv(
                self.operator_norm(x),
                mask=mask,
                cache=cache,
            )
        h = x + r
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Lfm2Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            Lfm2DecoderLayer(args, layer_idx=i) for i in range(args.num_hidden_layers)
        ]

        self.embedding_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_eps)

        self.fa_idx = args.full_attn_idxs[0]
        self.conv_idx = 0
        for i in range(args.num_hidden_layers):
            if i in args.full_attn_idxs:
                self.conv_idx += 1
            else:
                break

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        attn_mask = create_attention_mask(h, cache[self.fa_idx])
        conv_mask = create_ssm_mask(h, cache[self.conv_idx])

        for layer, c in zip(self.layers, cache):
            mask = attn_mask if layer.is_attention_layer else conv_mask
            h = layer(h, mask, cache=c)

        return self.embedding_norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Lfm2Model(args)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        out = self.model(inputs, cache, input_embeddings)
        return self.model.embed_tokens.as_linear(out)

    def sanitize(self, weights):
        sanitized_weights = {}
        for name, param in weights.items():
            if "conv.weight" in name:
                if param.shape[-1] > param.shape[1]:
                    param = param.transpose(0, 2, 1)

            sanitized_weights[name] = param
        return sanitized_weights

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [
            KVCache() if l.is_attention_layer else ArraysCache(size=1)
            for l in self.layers
        ]
