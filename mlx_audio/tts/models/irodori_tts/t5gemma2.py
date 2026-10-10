from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.tts.models.base import BaseModelArgs

# ---------------------------------------------------------------------------
# T5Gemma2 text encoder (MLX port of transformers' T5Gemma2TextEncoder).
#
# Irodori-TTS v4-Large conditions on a frozen-architecture pretrained encoder
# (google/t5gemma-2-1b-1b) whose weights are bundled in the checkpoint, so
# only the bidirectional text encoder half of the (otherwise
# encoder-decoder, multimodal) T5Gemma2 model is needed here -- no decoder,
# no vision tower. Parameter names mirror the HuggingFace module tree
# (embed_tokens, layers.N.self_attn/mlp/pre_self_attn_layernorm/...) to keep
# conversion a straight rename, same as modernbert.py.
#
# Architecturally this is Gemma3's text backbone (same sandwich of 4 RMSNorms
# per block, same GQA + per-head q/k-norm, same alternating sliding/full
# attention with per-type RoPE) adapted to run bidirectionally instead of
# causally, plus T5Gemma2's scaled word embedding with a learned "end of
# image" token override (inert for text-only input, kept for fidelity).
# ---------------------------------------------------------------------------


@dataclass
class T5Gemma2Config(BaseModelArgs):
    vocab_size: int = 262144
    hidden_size: int = 1152
    intermediate_size: int = 6912
    num_hidden_layers: int = 26
    num_attention_heads: int = 4
    num_key_value_heads: int = 1
    head_dim: int = 256
    hidden_activation: str = "gelu_pytorch_tanh"
    rms_norm_eps: float = 1e-6
    query_pre_attn_scalar: float = 256.0
    sliding_window: int = 512
    sliding_window_pattern: int = 6
    max_position_embeddings: int = 32768
    pad_token_id: int = 0
    eoi_token_index: int = 256_000
    full_attention_rope_theta: float = 1_000_000.0
    full_attention_rope_factor: float = 1.0
    sliding_attention_rope_theta: float = 10_000.0
    sliding_attention_rope_factor: float = 1.0

    @classmethod
    def from_dict(cls, params: dict) -> "T5Gemma2Config":
        params = dict(params)
        rope_parameters = params.pop("rope_parameters", None)
        if isinstance(rope_parameters, dict):
            full = rope_parameters.get("full_attention")
            if isinstance(full, dict):
                if "rope_theta" in full:
                    params.setdefault("full_attention_rope_theta", full["rope_theta"])
                if (
                    str(full.get("rope_type", "default")).lower() == "linear"
                    and "factor" in full
                ):
                    params.setdefault("full_attention_rope_factor", full["factor"])
            sliding = rope_parameters.get("sliding_attention")
            if isinstance(sliding, dict):
                if "rope_theta" in sliding:
                    params.setdefault(
                        "sliding_attention_rope_theta", sliding["rope_theta"]
                    )
                if (
                    str(sliding.get("rope_type", "default")).lower() == "linear"
                    and "factor" in sliding
                ):
                    params.setdefault(
                        "sliding_attention_rope_factor", sliding["factor"]
                    )
        return super().from_dict(params)

    def is_global_layer(self, layer_idx: int) -> bool:
        return (layer_idx + 1) % self.sliding_window_pattern == 0


# ---------------------------------------------------------------------------
# Norm / RoPE helpers
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Gemma-style RMSNorm: (1 + weight) * normalize(x)."""

    def __init__(self, dims: int, eps: float):
        super().__init__()
        self.weight = mx.zeros((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, 1.0 + self.weight, self.eps)


def rope_cos_sin(
    head_dim: int, seq_len: int, theta: float, position_scale: float = 1.0
) -> Tuple[mx.array, mx.array]:
    """HuggingFace-style (rotate_half) RoPE tables of shape (seq_len, head_dim).

    ``position_scale`` implements HF's "linear" RoPE scaling: dividing
    inv_freq by `factor` is equivalent to multiplying position by `1/factor`
    (freq * position is commutative under scalar scaling), so pass
    ``position_scale = 1 / factor``.
    """
    inv_freq = 1.0 / (
        theta ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / float(head_dim))
    )
    positions = mx.arange(seq_len, dtype=mx.float32) * position_scale
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


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------


class T5Gemma2Attention(nn.Module):
    def __init__(self, config: T5Gemma2Config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = config.query_pre_attn_scalar**-0.5

        dim = config.hidden_size
        self.q_proj = nn.Linear(dim, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        cos: mx.array,
        sin: mx.array,
        attn_mask: Optional[mx.array],
    ) -> mx.array:
        bsz, seq_len = x.shape[:2]
        q = self.q_proj(x).reshape(bsz, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(bsz, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(bsz, seq_len, self.num_kv_heads, self.head_dim)
        q, k, v = (mx.transpose(t, (0, 2, 1, 3)) for t in (q, k, v))

        q = self.q_norm(q)
        k = self.k_norm(k)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        out = mx.fast.scaled_dot_product_attention(
            q=q, k=k, v=v, scale=self.scale, mask=attn_mask
        )
        out = mx.transpose(out, (0, 2, 1, 3)).reshape(bsz, seq_len, -1)
        return self.o_proj(out)


class T5Gemma2MLP(nn.Module):
    def __init__(self, config: T5Gemma2Config):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )
        activation = str(config.hidden_activation).lower()
        if activation not in ("gelu_pytorch_tanh", "gelu_new"):
            raise ValueError(
                f"Unsupported T5Gemma2 hidden_activation={config.hidden_activation!r}"
            )

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.gelu_approx(self.gate_proj(x)) * self.up_proj(x))


@partial(mx.compile, shapeless=True)
def _clip_residual(x: mx.array, y: mx.array) -> mx.array:
    """Gemma's activations routinely exceed fp16 range in deep residual
    stacks; add in fp32 and clip back rather than silently overflowing to
    inf (which immediately turns into NaN through the next RMSNorm)."""
    if x.dtype != mx.float16:
        return x + y
    bound = mx.finfo(mx.float16).max
    return mx.clip(x.astype(mx.float32) + y.astype(mx.float32), -bound, bound).astype(
        mx.float16
    )


class T5Gemma2EncoderLayer(nn.Module):
    """Pre+post-norm sandwich around both self-attention and the MLP."""

    def __init__(self, config: T5Gemma2Config, layer_idx: int):
        super().__init__()
        self.is_global = config.is_global_layer(layer_idx)
        self.self_attn = T5Gemma2Attention(config)
        self.pre_self_attn_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_self_attn_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = T5Gemma2MLP(config)
        self.pre_feedforward_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        cos: mx.array,
        sin: mx.array,
        attn_mask: Optional[mx.array],
    ) -> mx.array:
        residual = x
        h = self.pre_self_attn_layernorm(x)
        h = self.self_attn(h, cos, sin, attn_mask)
        h = self.post_self_attn_layernorm(h)
        x = _clip_residual(residual, h)

        residual = x
        h = self.pre_feedforward_layernorm(x)
        h = self.mlp(h)
        h = self.post_feedforward_layernorm(h)
        return _clip_residual(residual, h)


class T5Gemma2ScaledWordEmbedding(nn.Module):
    """Embedding scaled by sqrt(hidden_size), with the "end of image" token's
    embedding overridden by a separate learned vector. Only the override is
    checkpoint-visible (the scale is a non-persistent HF buffer); inert for
    plain-text input since eoi_token_index is a reserved multimodal id."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        embed_scale: float,
        eoi_token_index: int,
    ):
        super().__init__()
        self.weight = mx.zeros((vocab_size, hidden_size))
        self.eoi_embedding = mx.zeros((hidden_size,))
        self._embed_scale = float(embed_scale)
        self._eoi_token_index = int(eoi_token_index)

    def __call__(self, input_ids: mx.array) -> mx.array:
        embeddings = self.weight[input_ids] * self._embed_scale
        is_eoi = input_ids == self._eoi_token_index
        if bool(mx.any(is_eoi)):
            eoi = self.eoi_embedding.astype(embeddings.dtype)
            embeddings = mx.where(is_eoi[..., None], eoi, embeddings)
        return embeddings


class T5Gemma2TextEncoder(nn.Module):
    """Bidirectional T5Gemma2 text encoder returning last_hidden_state."""

    def __init__(self, config: T5Gemma2Config):
        super().__init__()
        self.config = config
        self.embed_tokens = T5Gemma2ScaledWordEmbedding(
            config.vocab_size,
            config.hidden_size,
            embed_scale=config.hidden_size**0.5,
            eoi_token_index=config.eoi_token_index,
        )
        self.layers = [
            T5Gemma2EncoderLayer(config, i) for i in range(config.num_hidden_layers)
        ]
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def _attention_masks(
        self, mask: Optional[mx.array], seq_len: int, dtype: mx.Dtype
    ) -> Tuple[Optional[mx.array], mx.array]:
        """Return (global_mask, sliding_mask) as additive (B, 1, S, S) masks.

        Both are bidirectional. The sliding window is centered on the query
        position (``sliding_window_mask_function(is_causal=False)`` in
        transformers): a query at ``q`` attends to keys within
        ``[q - ceil(window/2), q + floor(window/2)]``.
        """
        window = self.config.sliding_window
        left = (window + 1) // 2
        right = window // 2 + 1
        positions = mx.arange(seq_len)
        dist = positions[:, None] - positions[None, :]
        within_window = ((dist >= 0) & (dist < left)) | ((dist < 0) & (-dist < right))

        if mask is None:
            global_bool = mx.ones((1, 1, seq_len, seq_len), dtype=mx.bool_)
            sliding_bool = mx.broadcast_to(
                within_window[None, None], (1, 1, seq_len, seq_len)
            )
            return None, _additive_mask(sliding_bool, dtype)

        key_mask = mask.astype(mx.bool_)[:, None, None, :]
        global_bool = mx.broadcast_to(key_mask, (mask.shape[0], 1, seq_len, seq_len))
        global_mask = _additive_mask(global_bool, dtype)
        # A padding query far from every real token would otherwise have an
        # all-masked sliding-window row, whose softmax is NaN and would
        # poison later layers (the window is larger than max_text_length in
        # practice, so this is a defensive guard rather than the common case).
        sliding_bool = (global_bool & within_window[None, None]) | (dist == 0)[
            None, None
        ]
        return global_mask, _additive_mask(sliding_bool, dtype)

    def __call__(
        self, input_ids: mx.array, mask: Optional[mx.array] = None
    ) -> mx.array:
        x = self.embed_tokens(input_ids)
        seq_len = input_ids.shape[1]

        global_cos, global_sin = rope_cos_sin(
            self.config.head_dim,
            seq_len,
            self.config.full_attention_rope_theta,
            position_scale=1.0 / self.config.full_attention_rope_factor,
        )
        sliding_cos, sliding_sin = rope_cos_sin(
            self.config.head_dim,
            seq_len,
            self.config.sliding_attention_rope_theta,
            position_scale=1.0 / self.config.sliding_attention_rope_factor,
        )
        global_mask, sliding_mask = self._attention_masks(mask, seq_len, x.dtype)

        for layer in self.layers:
            if layer.is_global:
                x = layer(x, global_cos, global_sin, global_mask)
            else:
                x = layer(x, sliding_cos, sliding_sin, sliding_mask)

        return self.norm(x)


def _additive_mask(bool_mask: mx.array, dtype: mx.Dtype) -> mx.array:
    """Additive attention mask in the attention compute dtype. Uses a finite
    floor (as transformers does with ``finfo.min``) rather than -inf."""
    return mx.where(
        bool_mask,
        mx.zeros(bool_mask.shape, dtype=dtype),
        mx.full(bool_mask.shape, mx.finfo(dtype).min, dtype=dtype),
    )
