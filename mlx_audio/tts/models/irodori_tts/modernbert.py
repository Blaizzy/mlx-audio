from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.tts.models.base import BaseModelArgs

# ---------------------------------------------------------------------------
# ModernBERT encoder (MLX port of transformers' ModernBertModel).
#
# Irodori-TTS v4 conditions on a frozen-architecture pretrained encoder
# (sbintuitions/modernbert-ja-310m) whose weights are bundled in the
# checkpoint, so only the bidirectional encoder stack is needed here.
# Parameter names mirror the HuggingFace module tree to keep conversion
# a straight rename.
# ---------------------------------------------------------------------------


@dataclass
class ModernBertConfig(BaseModelArgs):
    vocab_size: int = 102400
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 25
    num_attention_heads: int = 12
    hidden_activation: str = "gelu"
    norm_eps: float = 1e-5
    norm_bias: bool = False
    attention_bias: bool = False
    mlp_bias: bool = False
    global_attn_every_n_layers: int = 3
    local_attention: int = 128
    global_rope_theta: float = 160000.0
    local_rope_theta: float = 10000.0
    max_position_embeddings: int = 8192
    pad_token_id: int = 3
    bos_token_id: int = 1
    eos_token_id: int = 2

    @classmethod
    def from_dict(cls, params: dict) -> "ModernBertConfig":
        params = dict(params)
        # transformers >= 5 nests the per-layer-type RoPE settings; earlier
        # releases exposed flat global_/local_rope_theta keys.
        rope_parameters = params.pop("rope_parameters", None)
        if isinstance(rope_parameters, dict):
            full = rope_parameters.get("full_attention")
            if isinstance(full, dict) and "rope_theta" in full:
                params.setdefault("global_rope_theta", full["rope_theta"])
            sliding = rope_parameters.get("sliding_attention")
            if isinstance(sliding, dict) and "rope_theta" in sliding:
                params.setdefault("local_rope_theta", sliding["rope_theta"])
        return super().from_dict(params)

    def is_global_layer(self, layer_idx: int) -> bool:
        if not self.global_attn_every_n_layers:
            return True
        return layer_idx % self.global_attn_every_n_layers == 0

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------


class LayerNorm(nn.Module):
    """LayerNorm with optional bias, accumulating in float32."""

    def __init__(self, dims: int, eps: float, bias: bool = False):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dims,))
        self.use_bias = bias
        if bias:
            self.bias = mx.zeros((dims,))

    def __call__(self, x: mx.array) -> mx.array:
        x_dtype = x.dtype
        x = x.astype(mx.float32)
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        x = (x - mean) * mx.rsqrt(var + self.eps)
        x = x * self.weight.astype(mx.float32)
        if self.use_bias:
            x = x + self.bias.astype(mx.float32)
        return x.astype(x_dtype)


def rope_cos_sin(
    head_dim: int, seq_len: int, theta: float
) -> Tuple[mx.array, mx.array]:
    """HuggingFace-style (rotate_half) RoPE tables of shape (seq_len, head_dim)."""
    inv_freq = 1.0 / (
        theta ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / float(head_dim))
    )
    positions = mx.arange(seq_len, dtype=mx.float32)
    freqs = mx.outer(positions, inv_freq)
    emb = mx.concatenate([freqs, freqs], axis=-1)
    return mx.cos(emb), mx.sin(emb)


def _rotate_half(x: mx.array) -> mx.array:
    half = x.shape[-1] // 2
    return mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def _apply_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """x: (B, H, S, D); cos/sin: (S, D)."""
    cos = cos[None, None].astype(x.dtype)
    sin = sin[None, None].astype(x.dtype)
    return x * cos + _rotate_half(x) * sin


class ModernBertAttention(nn.Module):
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.Wqkv = nn.Linear(
            config.hidden_size, 3 * config.hidden_size, bias=config.attention_bias
        )
        self.Wo = nn.Linear(
            config.hidden_size, config.hidden_size, bias=config.attention_bias
        )

    def __call__(
        self,
        x: mx.array,
        cos: mx.array,
        sin: mx.array,
        attn_mask: Optional[mx.array],
    ) -> mx.array:
        bsz, seq_len = x.shape[:2]
        qkv = self.Wqkv(x).reshape(bsz, seq_len, 3, self.num_heads, self.head_dim)
        # (B, S, 3, H, D) -> (3, B, H, S, D)
        qkv = mx.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        out = mx.fast.scaled_dot_product_attention(
            q=q, k=k, v=v, scale=self.scale, mask=attn_mask
        )
        out = mx.transpose(out, (0, 2, 1, 3)).reshape(bsz, seq_len, -1)
        return self.Wo(out)


class ModernBertMLP(nn.Module):
    """GeGLU MLP with a single fused input projection (Wi -> 2 * intermediate)."""

    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.Wi = nn.Linear(
            config.hidden_size, 2 * config.intermediate_size, bias=config.mlp_bias
        )
        self.Wo = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=config.mlp_bias
        )
        activation = str(config.hidden_activation).lower()
        if activation not in ("gelu", "gelu_new", "gelu_pytorch_tanh"):
            raise ValueError(
                f"Unsupported ModernBERT hidden_activation={config.hidden_activation!r}"
            )
        self.exact_gelu = activation == "gelu"

    def __call__(self, x: mx.array) -> mx.array:
        gated = self.Wi(x)
        half = gated.shape[-1] // 2
        act = nn.gelu if self.exact_gelu else nn.gelu_approx
        return self.Wo(act(gated[..., :half]) * gated[..., half:])


class ModernBertEncoderLayer(nn.Module):
    def __init__(self, config: ModernBertConfig, layer_idx: int):
        super().__init__()
        # The first layer consumes the already-normalised embedding output.
        self.attn_norm = (
            nn.Identity()
            if layer_idx == 0
            else LayerNorm(
                config.hidden_size, eps=config.norm_eps, bias=config.norm_bias
            )
        )
        self.attn = ModernBertAttention(config)
        self.mlp_norm = LayerNorm(
            config.hidden_size, eps=config.norm_eps, bias=config.norm_bias
        )
        self.mlp = ModernBertMLP(config)

    def __call__(
        self,
        x: mx.array,
        cos: mx.array,
        sin: mx.array,
        attn_mask: Optional[mx.array],
    ) -> mx.array:
        x = x + self.attn(self.attn_norm(x), cos, sin, attn_mask)
        return x + self.mlp(self.mlp_norm(x))


class ModernBertEmbeddings(nn.Module):
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.norm = LayerNorm(
            config.hidden_size, eps=config.norm_eps, bias=config.norm_bias
        )

    def __call__(self, input_ids: mx.array) -> mx.array:
        return self.norm(self.tok_embeddings(input_ids))


class ModernBertEncoder(nn.Module):
    """Bidirectional ModernBERT encoder returning last_hidden_state."""

    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.config = config
        self.embeddings = ModernBertEmbeddings(config)
        self.layers = [
            ModernBertEncoderLayer(config, i) for i in range(config.num_hidden_layers)
        ]
        self.final_norm = LayerNorm(
            config.hidden_size, eps=config.norm_eps, bias=config.norm_bias
        )

    def _attention_masks(
        self, mask: Optional[mx.array], seq_len: int, dtype: mx.Dtype
    ) -> Tuple[Optional[mx.array], mx.array]:
        """Return (global_mask, local_mask) as additive (B, 1, S, S) masks."""
        window = self.config.local_attention // 2
        positions = mx.arange(seq_len)
        distance = mx.abs(positions[:, None] - positions[None, :])
        within_window = distance <= window

        if mask is None:
            global_mask = None
            local_bool = mx.broadcast_to(
                within_window[None, None], (1, 1, seq_len, seq_len)
            )
        else:
            key_mask = mask.astype(mx.bool_)[:, None, None, :]
            global_bool = mx.broadcast_to(
                key_mask, (mask.shape[0], 1, seq_len, seq_len)
            )
            global_mask = _additive_mask(global_bool, dtype)
            # A padding query far from every real token would otherwise have an
            # all-masked row, whose softmax is NaN and would poison later layers.
            # Always leaving the diagonal open keeps those rows finite without
            # changing any non-padding output.
            local_bool = (global_bool & within_window[None, None]) | (distance == 0)

        return global_mask, _additive_mask(local_bool, dtype)

    def __call__(
        self, input_ids: mx.array, mask: Optional[mx.array] = None
    ) -> mx.array:
        x = self.embeddings(input_ids)
        seq_len = input_ids.shape[1]

        global_cos, global_sin = rope_cos_sin(
            self.config.head_dim, seq_len, self.config.global_rope_theta
        )
        local_cos, local_sin = rope_cos_sin(
            self.config.head_dim, seq_len, self.config.local_rope_theta
        )
        global_mask, local_mask = self._attention_masks(mask, seq_len, x.dtype)

        for i, layer in enumerate(self.layers):
            if self.config.is_global_layer(i):
                x = layer(x, global_cos, global_sin, global_mask)
            else:
                x = layer(x, local_cos, local_sin, local_mask)

        return self.final_norm(x)


def _additive_mask(bool_mask: mx.array, dtype: mx.Dtype) -> mx.array:
    """Additive attention mask in the attention compute dtype. Uses a finite
    floor (as transformers does with ``finfo.min``) rather than -inf."""
    return mx.where(
        bool_mask,
        mx.zeros(bool_mask.shape, dtype=dtype),
        mx.full(bool_mask.shape, mx.finfo(dtype).min, dtype=dtype),
    )
