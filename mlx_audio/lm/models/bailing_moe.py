# Copyright © 2025 Apple Inc.
# Vendored verbatim from mlx-lm v0.31.3 (ed1fca4cef15a824c5f1702c80f70b4cffc8e4dd),
# mlx_lm/models/bailing_moe.py. MIT licensed.

from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from .activations import swiglu
from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .rope_utils import initialize_rope
from .switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    intermediate_size: int
    max_position_embeddings: int
    moe_intermediate_size: int
    num_experts: int
    num_shared_experts: int
    norm_topk_prob: bool
    num_attention_heads: int
    num_experts_per_tok: int
    num_hidden_layers: int
    num_key_value_heads: int
    rms_norm_eps: float
    rope_theta: float
    vocab_size: int
    first_k_dense_replace: int
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    use_bias: bool = False
    use_qkv_bias: bool = False
    norm_head: bool = False
    norm_softmax: bool = False
    use_qk_norm: bool = False
    tie_word_embeddings: bool = False
    partial_rotary_factor: float = 1.0
    rotary_dim: Optional[int] = None
    moe_router_enable_expert_bias: bool = False
    moe_router_enable_routed_scaling: bool = True
    routed_scaling_factor: float = 1.0
    score_function: str = "softmax"
    n_group: int = 1
    topk_group: int = 4
    moe_shared_expert_intermediate_size: Optional[int] = None
    moe_router_enable_shared_expert: bool = True


@partial(mx.compile, shapeless=True)
def aggregate_expert_outputs(expert_outputs, scores):
    return (
        (expert_outputs * scores[..., None]).sum(axis=-2).astype(expert_outputs.dtype)
    )


class BailingMoeMLP(nn.Module):
    def __init__(self, args: ModelArgs, intermediate_size: Optional[int] = None):
        super().__init__()
        self.intermediate_size = (
            intermediate_size
            if intermediate_size is not None
            else args.intermediate_size
        )

        self.gate_proj = nn.Linear(
            args.hidden_size, self.intermediate_size, bias=args.use_bias
        )
        self.down_proj = nn.Linear(
            self.intermediate_size, args.hidden_size, bias=args.use_bias
        )
        self.up_proj = nn.Linear(
            args.hidden_size, self.intermediate_size, bias=args.use_bias
        )

    def __call__(self, x) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class BailingMoeAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.use_qk_norm = args.use_qk_norm
        self.num_attention_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.head_dim = args.hidden_size // self.num_attention_heads
        self.scale = self.head_dim**-0.5

        self.query_key_value = nn.Linear(
            args.hidden_size,
            (self.num_attention_heads + 2 * self.num_key_value_heads) * self.head_dim,
            bias=args.use_qkv_bias,
        )
        self.dense = nn.Linear(
            self.num_attention_heads * self.head_dim,
            args.hidden_size,
            bias=args.use_bias,
        )

        if args.use_qk_norm:
            self.key_layernorm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
            self.query_layernorm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)

        if (rope_dim := args.rotary_dim) is None:
            rope_dim = int(self.head_dim * args.partial_rotary_factor)
        self.rope = initialize_rope(
            rope_dim,
            args.rope_theta,
            traditional=False,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        qkv = self.query_key_value(x)

        q_size = self.num_attention_heads * self.head_dim
        kv_size = self.num_key_value_heads * self.head_dim
        q, k, v = mx.split(qkv, [q_size, q_size + kv_size], axis=-1)

        queries = q.reshape(B, L, self.num_attention_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        keys = k.reshape(B, L, self.num_key_value_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        values = v.reshape(B, L, self.num_key_value_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )

        if self.use_qk_norm:
            queries = self.query_layernorm(queries)
            keys = self.key_layernorm(keys)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.dense(output)


@mx.compile
def group_expert_select(
    gates,
    e_score_correction_bias,
    top_k,
    n_group,
    topk_group,
    routed_scaling_factor,
    norm_topk_prob,
    score_function,
):

    in_type = gates.dtype
    if score_function == "sigmoid":
        scores = mx.sigmoid(gates.astype(mx.float32))
    else:
        scores = mx.softmax(gates.astype(mx.float32), axis=-1)
    orig_scores = scores
    if e_score_correction_bias is not None:
        scores = scores + e_score_correction_bias
    if n_group > 1:
        scores = mx.unflatten(scores, axis=-1, shape=(n_group, -1))
        group_scores = mx.topk(scores, 2, axis=-1).sum(axis=-1, keepdims=True)
        k = n_group - topk_group
        group_idx = mx.argpartition(group_scores, kth=k - 1, axis=-2)[..., :k, :]
        scores = mx.put_along_axis(
            scores, mx.stop_gradient(group_idx), mx.array(0.0, scores.dtype), axis=-2
        )
        scores = mx.flatten(scores, -2, -1)

    k = top_k
    inds = mx.argpartition(scores, kth=-k, axis=-1)[..., -k:]
    scores = mx.take_along_axis(orig_scores, inds, axis=-1)
    if top_k > 1 and norm_topk_prob:
        denominator = scores.sum(axis=-1, keepdims=True) + 1e-20
        scores = scores / denominator
    scores = scores * routed_scaling_factor

    return inds, scores.astype(in_type)


class BailingMoeGate(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.norm_topk_prob = args.norm_topk_prob

        self.top_k = args.num_experts_per_tok
        self.n_group = args.n_group
        self.topk_group = args.topk_group
        self.routed_scaling_factor = args.routed_scaling_factor
        self.enable_routed_scaling = args.moe_router_enable_routed_scaling

        self.gate_proj = nn.Linear(args.hidden_size, args.num_experts, bias=False)
        self.expert_bias = (
            mx.zeros((args.num_experts,))
            if args.moe_router_enable_expert_bias
            else None
        )
        self.score_function = args.score_function

    def __call__(self, x):
        return group_expert_select(
            self.gate_proj(x),
            self.expert_bias,
            self.top_k,
            self.n_group,
            self.topk_group,
            self.routed_scaling_factor,
            self.norm_topk_prob,
            self.score_function,
        )


class BailingMoeSparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.num_experts_per_tok = args.num_experts_per_tok
        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            args.moe_intermediate_size,
            args.num_experts,
            bias=args.use_bias,
        )
        self.gate = BailingMoeGate(args)
        shared_dim = (
            args.moe_shared_expert_intermediate_size or args.moe_intermediate_size
        )
        self.shared_experts = (
            BailingMoeMLP(
                args=args,
                intermediate_size=shared_dim * args.num_shared_experts,
            )
            if args.num_shared_experts > 0 and args.moe_router_enable_shared_expert
            else None
        )

    def __call__(self, x):
        topk_idx, topk_weight = self.gate(x)
        out = self.switch_mlp(x, topk_idx)
        out = aggregate_expert_outputs(out, topk_weight)
        if self.shared_experts is not None:
            out = out + self.shared_experts(x)
        return out


class BailingMoeDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.attention = BailingMoeAttention(args)

        self.mlp = (
            BailingMoeSparseMoeBlock(args)
            if (
                args.num_experts is not None and layer_idx >= args.first_k_dense_replace
            )
            else BailingMoeMLP(args)
        )
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = self.attention(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class BailingMoeModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.word_embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            BailingMoeDecoderLayer(args, layer_idx=i)
            for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ):
        h = self.word_embeddings(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        mask = create_attention_mask(h, cache[0])

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.norm_head = args.norm_head
        self.model_type = args.model_type
        self.model = BailingMoeModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        out = self.model(inputs, cache)
        if self.args.tie_word_embeddings:
            out = self.model.word_embeddings.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    def sanitize(self, weights):
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        if self.norm_head:
            w = weights["lm_head.weight"]
            dtype = w.dtype
            weight_norm = (
                mx.linalg.norm(w.astype(mx.float32), axis=0, keepdims=True) + 1e-7
            )
            weights["lm_head.weight"] = (w / weight_norm).astype(dtype)

        for l in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{l}"

            if l >= self.args.first_k_dense_replace:
                for m in ["gate_proj", "down_proj", "up_proj"]:
                    for k in ["weight", "scales", "biases"]:
                        if f"{prefix}.mlp.experts.0.{m}.{k}" in weights:
                            to_join = [
                                weights.pop(f"{prefix}.mlp.experts.{e}.{m}.{k}")
                                for e in range(self.args.num_experts)
                            ]
                            weights[f"{prefix}.mlp.switch_mlp.{m}.{k}"] = mx.stack(
                                to_join
                            )

                if f"{prefix}.mlp.gate.weight" in weights:
                    gate_weight = weights.pop(f"{prefix}.mlp.gate.weight")
                    weights[f"{prefix}.mlp.gate.gate_proj.weight"] = gate_weight

                if f"{prefix}.mlp.gate.bias" in weights:
                    gate_bias = weights.pop(f"{prefix}.mlp.gate.bias")
                    weights[f"{prefix}.mlp.gate.gate_proj.bias"] = gate_bias

        return weights

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if path.endswith("mlp.gate.gate_proj"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate

    @property
    def cast_predicate(self):
        def predicate(k):
            return "expert_bias" not in k

        return predicate

    @property
    def layers(self):
        return self.model.layers
