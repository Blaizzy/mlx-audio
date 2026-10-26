# Copyright 2026 MLX Audio contributors.
# Architecture adapted from inclusionAI's Apache-2.0 BailingMoE V3 implementation.
"""Ming 3.0's hybrid Kimi Delta Attention / MLA mixture-of-experts backbone."""

import re
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask
from .cache import ArraysCache, KVCache
from .gated_delta import gated_delta_kernel, gated_delta_ops
from .switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "bailing_moe_v3"
    hidden_size: int = 1536
    intermediate_size: int = 4608
    num_hidden_layers: int = 24
    vocab_size: int = 157184
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    head_dim: int = 128
    layer_group_size: int = 4
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_interleave: bool = True
    q_lora_rank: int | None = 256
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    short_conv_kernel_size: int = 4
    no_kda_lora: bool = True
    kda_safe_gate: bool = True
    kda_lower_bound: float = -5.0
    num_experts: int = 128
    num_shared_experts: int = 1
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 512
    moe_shared_expert_intermediate_size: int = 512
    first_k_dense_replace: int = 1
    n_group: int = 8
    topk_group: int = 4
    routed_scaling_factor: float = 2.5
    gated_attention_proj_granularity_type: str | None = "head_wise"
    expert_swiglu_limit_list: list[float] | None = None
    share_expert_swiglu_limit_list: list[float] | None = None
    max_position_embeddings: int = 8192
    tie_word_embeddings: bool = False
    eos_token_id: int | list[int] = 156892

    def __post_init__(self):
        if self.layer_group_size < 1:
            raise ValueError("layer_group_size must be positive")
        if self.num_experts % self.n_group or not 0 < self.topk_group <= self.n_group:
            raise ValueError("Invalid Ming expert grouping")
        if not 0 < self.num_experts_per_tok <= self.num_experts:
            raise ValueError("Invalid number of active experts")

    def uses_mla(self, index):
        return (index + 1) % self.layer_group_size == 0 or index >= (
            self.num_hidden_layers // self.layer_group_size * self.layer_group_size
        )


def _l2_normalize(x, eps=1e-6):
    x = x.astype(mx.float32)
    return x * mx.rsqrt(mx.sum(x * x, axis=-1, keepdims=True) + eps)


class KimiDeltaAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        channels = args.num_attention_heads * args.head_dim
        for name in ("q_proj", "k_proj", "v_proj", "b_proj"):
            width = args.num_attention_heads if name == "b_proj" else channels
            setattr(self, name, nn.Linear(args.hidden_size, width, bias=False))
        for name in ("f", "g"):
            if args.no_kda_lora:
                setattr(
                    self, name + "_proj", nn.Linear(args.hidden_size, channels, False)
                )
            else:
                setattr(
                    self,
                    name + "_a_proj",
                    nn.Linear(args.hidden_size, args.head_dim, False),
                )
                setattr(
                    self, name + "_b_proj", nn.Linear(args.head_dim, channels, False)
                )
        for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
            setattr(
                self,
                name,
                nn.Conv1d(
                    channels,
                    channels,
                    args.short_conv_kernel_size,
                    groups=channels,
                    bias=False,
                ),
            )
        self.A_log = mx.zeros((args.num_attention_heads,), dtype=mx.float32)
        self.dt_bias = mx.zeros((channels,), dtype=mx.float32)
        self.o_norm = nn.RMSNorm(args.head_dim, args.rms_norm_eps)
        self.o_proj = nn.Linear(channels, args.hidden_size, bias=False)

    def _projection(self, name, x):
        if self.args.no_kda_lora:
            return getattr(self, name + "_proj")(x)
        return getattr(self, name + "_b_proj")(getattr(self, name + "_a_proj")(x))

    def __call__(self, x, cache=None):
        b, length, _ = x.shape
        a = self.args
        projected = []
        keep = a.short_conv_kernel_size - 1
        for index, name in enumerate(("q", "k", "v")):
            value = getattr(self, name + "_proj")(x)
            previous = cache[index] if cache is not None else None
            if previous is None:
                previous = mx.zeros((b, keep, value.shape[-1]), dtype=value.dtype)
            value = mx.concatenate([previous, value], axis=1)
            if cache is not None:
                cache[index] = value[:, -keep:, :] if keep else value[:, :0, :]
            value = nn.silu(getattr(self, name + "_conv1d")(value))
            projected.append(
                value.reshape(b, length, a.num_attention_heads, a.head_dim)
            )
        q, k, v = projected
        q = _l2_normalize(q) * a.head_dim**-0.5
        k = _l2_normalize(k)
        gate = self._projection("f", x).astype(mx.float32)
        gate = (gate + self.dt_bias).reshape(
            b, length, a.num_attention_heads, a.head_dim
        )
        rate = mx.exp(self.A_log.astype(mx.float32))[:, None]
        if a.kda_safe_gate:
            gate = a.kda_lower_bound * mx.sigmoid(rate * gate)
        else:
            gate = -rate * nn.softplus(gate)
        decay = mx.exp(gate)
        beta = mx.sigmoid(self.b_proj(x).astype(mx.float32))
        state = cache[3] if cache is not None else None
        if state is None:
            state = mx.zeros(
                (b, a.num_attention_heads, a.head_dim, a.head_dim), mx.float32
            )
        if (
            mx.default_device() == mx.gpu
            and mx.metal.is_available()
            and a.head_dim % 32 == 0
        ):
            out, state = gated_delta_kernel(q, k, v, decay, beta, state)
        else:
            out, state = gated_delta_ops(q, k, v, decay, beta, state)
        if cache is not None:
            cache[3] = state
        # FLA rounds the recurrence output to the value dtype before gated RMSNorm.
        out = out.astype(v.dtype).astype(mx.float32)
        out = mx.fast.rms_norm(
            out, self.o_norm.weight.astype(mx.float32), a.rms_norm_eps
        )
        gate = self._projection("g", x).reshape(out.shape).astype(mx.float32)
        out = (out * mx.sigmoid(gate)).astype(x.dtype).reshape(b, length, -1)
        return self.o_proj(out)


class MultiLatentAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        heads = args.num_attention_heads
        q_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        if args.q_lora_rank is None:
            self.q_proj = nn.Linear(args.hidden_size, heads * q_dim, False)
        else:
            self.q_a_proj = nn.Linear(args.hidden_size, args.q_lora_rank, False)
            self.q_a_layernorm = nn.RMSNorm(args.q_lora_rank, args.rms_norm_eps)
            self.q_b_proj = nn.Linear(args.q_lora_rank, heads * q_dim, False)
        self.kv_a_proj_with_mqa = nn.Linear(
            args.hidden_size, args.kv_lora_rank + args.qk_rope_head_dim, False
        )
        self.kv_a_layernorm = nn.RMSNorm(args.kv_lora_rank, args.rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            args.kv_lora_rank, heads * (args.qk_nope_head_dim + args.v_head_dim), False
        )
        granularity = args.gated_attention_proj_granularity_type
        if granularity not in (None, "head_wise", "element_wise"):
            raise ValueError(f"Unsupported attention gating: {granularity}")
        self.g_proj = (
            None
            if granularity is None
            else nn.Linear(
                args.hidden_size,
                heads * (args.v_head_dim if granularity == "element_wise" else 1),
                False,
            )
        )
        self.dense = nn.Linear(heads * args.v_head_dim, args.hidden_size, False)
        self.rope = nn.RoPE(
            args.qk_rope_head_dim,
            traditional=args.rope_interleave,
            base=args.rope_theta,
        )
        self.scale = q_dim**-0.5

    def __call__(self, x, cache=None):
        a = self.args
        b, length, _ = x.shape
        q = (
            self.q_proj(x)
            if a.q_lora_rank is None
            else self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))
        )
        q = q.reshape(b, length, a.num_attention_heads, -1).transpose(0, 2, 1, 3)
        q_nope, q_rope = mx.split(q, [a.qk_nope_head_dim], axis=-1)
        kv, k_rope = mx.split(self.kv_a_proj_with_mqa(x), [a.kv_lora_rank], axis=-1)
        kv = self.kv_b_proj(self.kv_a_layernorm(kv))
        kv = kv.reshape(b, length, a.num_attention_heads, -1).transpose(0, 2, 1, 3)
        k_nope, values = mx.split(kv, [a.qk_nope_head_dim], axis=-1)
        k_rope = k_rope[:, None, :, :]
        offset = cache.offset if cache is not None else 0
        q_rope = self.rope(q_rope, offset=offset)
        k_rope = self.rope(k_rope, offset=offset)
        keys = mx.concatenate([k_nope, mx.broadcast_to(k_rope, q_rope.shape)], axis=-1)
        queries = mx.concatenate([q_nope, q_rope], axis=-1)
        mask = create_attention_mask(x, cache)
        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)
        out = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        out = out.transpose(0, 2, 1, 3)
        if self.g_proj is not None:
            gate = mx.sigmoid(self.g_proj(x).astype(mx.float32)).astype(x.dtype)
            out = out * gate.reshape(b, length, a.num_attention_heads, -1)
        return self.dense(out.reshape(b, length, -1))


class SwiGLU(nn.Module):
    def __init__(self, limit=None):
        super().__init__()
        self.limit = limit

    def __call__(self, up, gate):
        gate = nn.silu(gate)
        if self.limit:
            gate = mx.minimum(gate, self.limit)
            up = mx.clip(up, -self.limit, self.limit)
        return gate * up


class MLP(nn.Module):
    def __init__(self, hidden, intermediate, limit=None):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, False)
        self.up_proj = nn.Linear(hidden, intermediate, False)
        self.down_proj = nn.Linear(intermediate, hidden, False)
        self.activation = SwiGLU(limit)

    def __call__(self, x):
        return self.down_proj(self.activation(self.up_proj(x), self.gate_proj(x)))


class Gate(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.weight = nn.Linear(args.hidden_size, args.num_experts, bias=False).weight
        self.expert_bias = mx.zeros((args.num_experts,), mx.float32)

    def __call__(self, x):
        a = self.args
        scores = mx.sigmoid(x.astype(mx.float32) @ self.weight.astype(mx.float32).T)
        adjusted = scores + self.expert_bias
        if a.n_group > 1:
            grouped = adjusted.reshape(*scores.shape[:-1], a.n_group, -1)
            group_scores = mx.topk(grouped, min(2, grouped.shape[-1]), axis=-1).sum(-1)
            chosen = mx.argpartition(group_scores, kth=-a.topk_group, axis=-1)[
                ..., -a.topk_group :
            ]
            allowed = mx.any(mx.arange(a.n_group) == chosen[..., None], axis=-2)
            adjusted = mx.where(allowed[..., :, None], grouped, -float("inf")).reshape(
                scores.shape
            )
        selected = mx.argpartition(adjusted, kth=-a.num_experts_per_tok, axis=-1)[
            ..., -a.num_experts_per_tok :
        ]
        weights = mx.take_along_axis(scores, selected, axis=-1)
        if a.num_experts_per_tok > 1:
            weights = weights / (weights.sum(-1, keepdims=True) + 1e-20)
        return selected, weights * a.routed_scaling_factor


class SparseMoE(nn.Module):
    def __init__(self, args, index):
        super().__init__()

        def limit(values):
            return values[index] if values and index < len(values) else None

        self.gate = Gate(args)
        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            args.moe_intermediate_size,
            args.num_experts,
            activation=SwiGLU(limit(args.expert_swiglu_limit_list)),
        )
        self.shared_experts = (
            MLP(
                args.hidden_size,
                args.moe_shared_expert_intermediate_size * args.num_shared_experts,
                limit(args.share_expert_swiglu_limit_list),
            )
            if args.num_shared_experts
            else None
        )

    def __call__(self, x):
        indices, weights = self.gate(x)
        routed = self.switch_mlp(x, indices)
        out = (routed.astype(mx.float32) * weights[..., None]).sum(-2).astype(x.dtype)
        return out + self.shared_experts(x) if self.shared_experts is not None else out


class DecoderLayer(nn.Module):
    def __init__(self, args, index):
        super().__init__()
        self.attention = (
            MultiLatentAttention(args)
            if args.uses_mla(index)
            else KimiDeltaAttention(args)
        )
        self.input_layernorm = nn.RMSNorm(args.hidden_size, args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, args.rms_norm_eps)
        self.mlp = (
            SparseMoE(args, index)
            if index >= args.first_k_dense_replace
            else MLP(args.hidden_size, args.intermediate_size)
        )

    def __call__(self, x, cache=None):
        x = x + self.attention(self.input_layernorm(x), cache)
        return x + self.mlp(self.post_attention_layernorm(x))


class BailingMoeV3Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.word_embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, args.rms_norm_eps)

    def __call__(self, inputs=None, cache=None, inputs_embeds=None):
        x = self.word_embeddings(inputs) if inputs_embeds is None else inputs_embeds
        if cache is None:
            cache = [None] * len(self.layers)
        if len(cache) != len(self.layers):
            raise ValueError("Cache length must match the number of decoder layers")
        for layer, state in zip(self.layers, cache):
            x = layer(x, state)
        return self.norm(x)

    def make_cache(self):
        return [
            KVCache() if self.args.uses_mla(i) else ArraysCache(4)
            for i in range(len(self.layers))
        ]


def sanitize_backbone(weights, args, prefix="model."):
    """Pack routed experts and remove checkpoint-only speculative MTP layers."""
    weights = dict(weights)
    for key in list(weights):
        match = re.match(re.escape(prefix) + r"layers\.(\d+)\.", key)
        if (match and int(match[1]) >= args.num_hidden_layers) or key.endswith(
            "rotary_emb.inv_freq"
        ):
            weights.pop(key)
        elif (
            key.endswith(("q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight"))
            and weights[key].shape[-1] != 1
        ):
            weights[key] = weights[key].transpose(0, 2, 1)
    for index in range(args.first_k_dense_replace, args.num_hidden_layers):
        stem = f"{prefix}layers.{index}.mlp."
        for projection in ("gate_proj", "up_proj", "down_proj"):
            for suffix in ("weight", "scales", "biases"):
                first = f"{stem}experts.0.{projection}.{suffix}"
                if first in weights:
                    keys = [
                        f"{stem}experts.{expert}.{projection}.{suffix}"
                        for expert in range(args.num_experts)
                    ]
                    if not all(key in weights for key in keys):
                        raise ValueError(
                            f"Incomplete routed experts for {stem}{projection}"
                        )
                    weights[f"{stem}switch_mlp.{projection}.{suffix}"] = mx.stack(
                        [weights.pop(key) for key in keys]
                    )
    return weights


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = BailingMoeV3Model(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, False)

    def __call__(self, inputs=None, cache=None, inputs_embeds=None):
        return self.lm_head(self.model(inputs, cache, inputs_embeds))

    def make_cache(self):
        return self.model.make_cache()

    def sanitize(self, weights):
        return sanitize_backbone(weights, self.args)
