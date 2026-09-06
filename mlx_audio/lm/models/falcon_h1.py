# Copyright © 2025 Apple Inc.
# Vendored from mlx-lm 0.31.3.

from dataclasses import dataclass, field
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn

from .activations import swiglu
from .base import (
    BaseModelArgs,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from .cache import ArraysCache, CacheList, KVCache
from .rope_utils import initialize_rope
from .ssm import ssm_update


@dataclass
class ModelArgs(BaseModelArgs):
    attention_bias: bool = False
    attention_in_multiplier: float = 1.0
    attention_out_multiplier: float = 0.9375
    embedding_multiplier: float = 5.656854249492381
    head_dim: int = 64
    hidden_size: int = 1024
    initializer_range: float = 0.02
    intermediate_size: int = 2048
    key_multiplier: float = 0.390625
    lm_head_multiplier: float = 0.0390625
    mamba_chunk_size: int = 128
    mamba_conv_bias: bool = True
    mamba_d_conv: int = 4
    mamba_d_head: int = 64
    mamba_d_ssm: int = 1536
    mamba_d_state: int = 128
    mamba_expand: int = 2
    mamba_n_groups: int = 1
    mamba_n_heads: int = 24
    mamba_norm_before_gate: bool = False
    mamba_proj_bias: bool = False
    mamba_rms_norm: bool = False
    mamba_use_mlp: bool = True
    max_position_embeddings: int = 131072
    mlp_bias: bool = False
    mlp_expansion_factor: int = 8
    mlp_multipliers: List[float] = field(
        default_factory=lambda: [0.8838834764831844, 0.5859375]
    )
    model_type: str = "falcon_h1"
    num_attention_heads: int = 8
    num_hidden_layers: int = 36
    num_key_value_heads: int = 2
    projectors_bias: bool = False
    rms_norm_eps: float = 1e-05
    rope_traditional: bool = False
    rope_scaling: Optional[float] = None
    rope_theta: float = 100000000000.0
    ssm_in_multiplier: float = 1.25
    ssm_multipliers: List[float] = field(
        default_factory=lambda: [
            0.3535533905932738,
            0.25,
            0.3535533905932738,
            0.5,
            0.3535533905932738,
        ]
    )
    ssm_out_multiplier: float = 0.23570226039551587
    vocab_size: int = 32784
    tie_word_embeddings: bool = True


class FalconH1RMSNormGated(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, n_groups=1, norm_before_gate=True):
        super().__init__()
        self.weight = mx.ones((hidden_size,))
        self.variance_epsilon = eps
        self.n_groups = n_groups
        self.norm_before_gate = norm_before_gate

    def __call__(self, hidden_states, gate=None):
        if not self.norm_before_gate and gate is not None:
            hidden_states = swiglu(gate, hidden_states)

        hidden_states = mx.fast.rms_norm(
            hidden_states, self.weight, self.variance_epsilon
        )

        if self.norm_before_gate and gate is not None:
            hidden_states = swiglu(gate, hidden_states)
        return hidden_states


def compute_mup_vector(args):
    intermediate_size = args.mamba_d_ssm
    groups_time_state_size = args.mamba_n_groups * args.mamba_d_state
    num_heads = args.mamba_n_heads
    sizes = [
        intermediate_size,
        intermediate_size,
        groups_time_state_size,
        groups_time_state_size,
        num_heads,
    ]
    return mx.concatenate(
        [
            mx.broadcast_to(mx.array(m), (s,))
            for s, m in zip(sizes, args.ssm_multipliers)
        ]
    )


class FalconH1Attention(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.hidden_size = args.hidden_size
        self.num_heads = args.num_attention_heads
        self.num_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=args.attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=args.attention_bias
        )

        self.rope = initialize_rope(
            self.head_dim,
            args.rope_theta,
            args.rope_traditional,
            args.rope_scaling,
            args.max_position_embeddings,
        )

    def __call__(self, x, mask=None, cache=None):
        B, L, _ = x.shape

        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        queries = queries.reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.num_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, mask=mask, scale=self.scale, cache=cache
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.o_proj(output)


class FalconH1Mixer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_heads = args.mamba_n_heads
        self.hidden_size = args.hidden_size
        self.ssm_state_size = args.mamba_d_state
        self.conv_kernel_size = args.mamba_d_conv
        self.intermediate_size = args.mamba_d_ssm
        self.use_conv_bias = args.mamba_conv_bias

        self.layer_norm_epsilon = args.rms_norm_eps
        self.groups_time_state_size = args.mamba_n_groups * self.ssm_state_size

        self.n_groups = args.mamba_n_groups
        self.head_dim = args.mamba_d_head
        self.chunk_size = args.mamba_chunk_size

        self.time_step_limit = (0.0, float("inf"))
        self.time_step_min = 0.001
        self.time_step_max = 0.1

        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=self.use_conv_bias,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
        )

        projection_size = self.intermediate_size + self.conv_dim + self.num_heads
        self.in_proj = nn.Linear(
            self.hidden_size,
            projection_size,
            bias=args.mamba_proj_bias,
        )

        self.dt_bias = mx.ones(self.num_heads)

        A = mx.arange(1, self.num_heads + 1)
        self.A_log = mx.log(A)

        self.mamba_rms_norm = args.mamba_rms_norm
        if self.mamba_rms_norm:
            self.norm = FalconH1RMSNormGated(
                self.intermediate_size,
                eps=self.layer_norm_epsilon,
                n_groups=self.n_groups,
                norm_before_gate=args.mamba_norm_before_gate,
            )

        self.D = mx.ones(self.num_heads)

        self.out_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=args.projectors_bias
        )

    def _conv(
        self,
        conv_input: mx.array,
        cache: Optional[ArraysCache],
        mask: Optional[mx.array],
    ) -> mx.array:
        if mask is not None:
            conv_input = mx.where(mask[..., None], conv_input, 0)

        if cache is not None:
            if cache[0] is None:
                conv_state = mx.zeros(
                    (conv_input.shape[0], self.conv_kernel_size - 1, self.conv_dim),
                    dtype=conv_input.dtype,
                )
            else:
                conv_state = cache[0]
            padded_input = mx.concatenate([conv_state, conv_input], axis=1)
            n_keep = self.conv_kernel_size - 1
            if cache.lengths is not None:
                t = padded_input.shape[1]
                ends = mx.clip(cache.lengths, 0, t - n_keep)
                positions = (ends[:, None] + mx.arange(n_keep))[..., None]
                cache[0] = mx.take_along_axis(padded_input, positions, axis=1)
            else:
                cache[0] = padded_input[:, -n_keep:, :]
        else:
            padded_input = mx.pad(
                conv_input, [(0, 0), (self.conv_kernel_size - 1, 0), (0, 0)]
            )

        conv_output = self.conv1d(padded_input)
        return nn.silu(conv_output)

    def _ssm(
        self,
        hidden_states: mx.array,
        B: mx.array,
        C: mx.array,
        dt: mx.array,
        cache: Optional[ArraysCache],
        mask: Optional[mx.array],
    ) -> mx.array:
        batch_size, seq_len, _ = hidden_states.shape
        hidden_states = hidden_states.reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        B = B.reshape(batch_size, seq_len, self.n_groups, self.ssm_state_size)
        C = C.reshape(batch_size, seq_len, self.n_groups, self.ssm_state_size)
        if cache:
            state = cache[1]
            lengths = cache.lengths
        else:
            state, lengths = None, None
        y, state = ssm_update(
            hidden_states,
            self.A_log,
            B,
            C,
            self.D,
            dt,
            self.dt_bias,
            state,
            self.time_step_limit,
            mask,
            lengths,
        )
        if cache:
            cache[1] = state
        return y.reshape(batch_size, seq_len, self.intermediate_size)

    def __call__(self, input_states, cache=None, mask: Optional[mx.array] = None):
        projected_states = self.in_proj(input_states)

        gate, conv_input, dt = mx.split(
            projected_states,
            [self.intermediate_size, self.intermediate_size + self.conv_dim],
            axis=-1,
        )

        conv_output = self._conv(conv_input, cache, mask)

        hidden_states, B, C = mx.split(
            conv_output,
            [
                self.intermediate_size,
                self.intermediate_size + self.n_groups * self.ssm_state_size,
            ],
            axis=-1,
        )

        y = self._ssm(hidden_states, B, C, dt, cache, mask=mask)
        if cache:
            cache.advance(y.shape[1])

        if self.mamba_rms_norm:
            y = self.norm(y, gate)
        else:
            y = swiglu(gate, y)

        return self.out_proj(y)


class FalconH1MLP(nn.Module):

    def __init__(self, args):
        super().__init__()

        hidden_size = args.hidden_size
        intermediate_size = args.intermediate_size

        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=args.mlp_bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=args.mlp_bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=args.mlp_bias)

    def __call__(self, x):
        y = swiglu(self.gate_proj(x), self.up_proj(x))
        y = self.down_proj(y)
        return y


class FalconH1DecoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.feed_forward = FalconH1MLP(args)

        head_dim = args.head_dim
        self.channels_attn = (
            args.num_attention_heads * head_dim
            + 2 * args.num_key_value_heads * head_dim
        )

        self.mamba = FalconH1Mixer(args=args)

        self.self_attn = FalconH1Attention(args)

        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.pre_ff_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        h: mx.array,
        cache,
        attn_mask: Optional[mx.array],
        mamba_mask: Optional[mx.array],
    ) -> mx.array:

        residual = h
        h = self.input_layernorm(h)

        mamba_h = self.mamba(input_states=h, cache=cache[0], mask=mamba_mask)

        attn_h = self.self_attn(
            h,
            mask=attn_mask,
            cache=cache[1],
        )

        h = residual + mamba_h + attn_h

        residual = h
        h = self.pre_ff_layernorm(h)
        h = self.feed_forward(h)
        return residual + h


class FalconH1Model(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.vocab_size = args.vocab_size
        self.hidden_size = args.hidden_size

        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)

        self._mup_vector = compute_mup_vector(args)
        self.layers = [
            FalconH1DecoderLayer(args) for _ in range(args.num_hidden_layers)
        ]
        self.final_layernorm = nn.RMSNorm(self.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, inputs, cache=None):

        h = self.embed_tokens(inputs)

        h = h

        if cache is None:
            cache = [(None, None) * len(self.layers)]

        mamba_mask = create_ssm_mask(h, cache[0][0])
        attn_mask = create_attention_mask(h, cache[0][1])

        for layer, c in zip(self.layers, cache):
            h = layer(
                h,
                cache=c,
                attn_mask=attn_mask,
                mamba_mask=mamba_mask,
            )

        return self.final_layernorm(h)


class Model(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = FalconH1Model(args=args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs, cache=None):
        hidden_states = self.model(inputs, cache=cache)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(hidden_states)
            return out * (self.args.lm_head_multiplier / self.args.embedding_multiplier)
        else:
            return self.lm_head(hidden_states)

    def sanitize(self, weights):
        # Check if needs sanitization
        c1d = weights["model.layers.0.mamba.conv1d.weight"]
        if c1d.shape[-1] <= c1d.shape[1]:
            return weights

        sanitized_weights = {}
        args = self.args

        for name, param in weights.items():
            # Fold-in multipliers
            if name.endswith("embed_tokens.weight"):
                param *= args.embedding_multiplier
            elif name.endswith("lm_head.weight"):
                param *= args.lm_head_multiplier
            elif name.endswith("q_proj.weight") or name.endswith("k_proj.weight"):
                param *= args.attention_in_multiplier
            elif name.endswith("key_proj.weight"):
                param *= args.attention_in_multiplier * args.key_multiplier
            elif name.endswith("o_proj.weight"):
                param *= args.attention_out_multiplier
            elif name.endswith("out_proj.weight"):
                param *= args.ssm_out_multiplier
            elif name.endswith("gate_proj.weight"):
                param *= args.mlp_multipliers[0]
            elif name.endswith("down_proj.weight"):
                param *= args.mlp_multipliers[1]
            elif name.endswith("in_proj.weight"):
                param *= (
                    args.ssm_in_multiplier
                    * self.model._mup_vector.astype(param.dtype)[:, None]
                )
            elif "conv1d.weight" in name:
                param = param.transpose(0, 2, 1)
            sanitized_weights[name] = param
        return sanitized_weights

    def make_cache(self):
        return [
            CacheList(ArraysCache(size=2), KVCache())
            for _ in range(self.args.num_hidden_layers)
        ]

    @property
    def layers(self):
        return self.model.layers
