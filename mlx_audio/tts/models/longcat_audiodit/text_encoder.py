"""UMT5 encoder for AudioDiT text conditioning.

Structurally identical to T5 v1.1 encoder with relative position bias.
Returns both last_hidden_state and initial embeddings for text_add_embed.
"""

import math
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .config import TextEncoderConfig


class T5LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        variance = mx.mean(x.astype(mx.float32) ** 2, axis=-1, keepdims=True)
        x = x * mx.rsqrt(variance + self.eps)
        return self.weight * x


class T5DenseGatedActDense(nn.Module):
    def __init__(self, config: TextEncoderConfig):
        super().__init__()
        self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        hidden_gelu = nn.gelu(self.wi_0(x))
        hidden_linear = self.wi_1(x)
        return self.wo(hidden_gelu * hidden_linear)


class T5DenseActDense(nn.Module):
    def __init__(self, config: TextEncoderConfig):
        super().__init__()
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.wo(nn.gelu(self.wi(x)))


class T5Attention(nn.Module):
    def __init__(
        self, config: TextEncoderConfig, has_relative_attention_bias: bool = False
    ):
        super().__init__()
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.n_heads = config.num_heads
        self.d_kv = config.d_kv
        self.inner_dim = self.n_heads * self.d_kv

        self.q = nn.Linear(config.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(config.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(config.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, config.d_model, bias=False)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(
                self.relative_attention_num_buckets, self.n_heads
            )

    @staticmethod
    def _relative_position_bucket(
        relative_position: mx.array,
        bidirectional: bool = True,
        num_buckets: int = 32,
        max_distance: int = 128,
    ) -> mx.array:
        relative_buckets = mx.zeros(relative_position.shape, dtype=mx.int32)
        if bidirectional:
            num_buckets //= 2
            relative_buckets = (
                relative_buckets
                + (relative_position > 0).astype(mx.int32) * num_buckets
            )
            relative_position = mx.abs(relative_position)
        else:
            relative_position = -mx.minimum(
                relative_position, mx.zeros_like(relative_position)
            )

        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        relative_position_if_large = max_exact + (
            mx.log(relative_position.astype(mx.float32) / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).astype(mx.int32)
        relative_position_if_large = mx.minimum(
            relative_position_if_large,
            mx.full(relative_position_if_large.shape, num_buckets - 1, dtype=mx.int32),
        )
        relative_buckets = relative_buckets + mx.where(
            is_small, relative_position.astype(mx.int32), relative_position_if_large
        )
        return relative_buckets

    def compute_bias(self, query_length: int, key_length: int) -> mx.array:
        context_position = mx.arange(query_length)[:, None]
        memory_position = mx.arange(key_length)[None, :]
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=True,
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(relative_position_bucket)
        values = mx.transpose(values, (2, 0, 1))[None, :, :, :]
        return values

    def __call__(
        self,
        hidden_states: mx.array,
        mask: Optional[mx.array] = None,
        position_bias: Optional[mx.array] = None,
    ) -> Tuple[mx.array, Optional[mx.array]]:
        B, L, _ = hidden_states.shape

        q = (
            self.q(hidden_states)
            .reshape(B, L, self.n_heads, self.d_kv)
            .transpose(0, 2, 1, 3)
        )
        k = (
            self.k(hidden_states)
            .reshape(B, L, self.n_heads, self.d_kv)
            .transpose(0, 2, 1, 3)
        )
        v = (
            self.v(hidden_states)
            .reshape(B, L, self.n_heads, self.d_kv)
            .transpose(0, 2, 1, 3)
        )

        scores = q @ k.transpose(0, 1, 3, 2)

        if position_bias is None:
            if self.has_relative_attention_bias:
                position_bias = self.compute_bias(L, L)
            else:
                position_bias = mx.zeros((1, self.n_heads, L, L))

        scores = scores + position_bias
        if mask is not None:
            scores = scores + mask

        attn_weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(
            scores.dtype
        )
        attn_output = (
            (attn_weights @ v).transpose(0, 2, 1, 3).reshape(B, L, self.inner_dim)
        )
        return self.o(attn_output), position_bias


class T5Block(nn.Module):
    def __init__(
        self, config: TextEncoderConfig, has_relative_attention_bias: bool = False
    ):
        super().__init__()
        self.SelfAttention = T5Attention(
            config, has_relative_attention_bias=has_relative_attention_bias
        )
        self.layer_norm_sa = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        if config.is_gated_act:
            self.DenseReluDense = T5DenseGatedActDense(config)
        else:
            self.DenseReluDense = T5DenseActDense(config)
        self.layer_norm_ff = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

    def __call__(
        self,
        hidden_states: mx.array,
        mask: Optional[mx.array] = None,
        position_bias: Optional[mx.array] = None,
    ) -> Tuple[mx.array, Optional[mx.array]]:
        normed = self.layer_norm_sa(hidden_states)
        attn_output, position_bias = self.SelfAttention(
            normed, mask=mask, position_bias=position_bias
        )
        hidden_states = hidden_states + attn_output
        normed = self.layer_norm_ff(hidden_states)
        ff_output = self.DenseReluDense(normed)
        hidden_states = hidden_states + ff_output
        return hidden_states, position_bias


class UMT5Encoder(nn.Module):
    """UMT5 encoder model. Returns (last_hidden_state, initial_embedding)."""

    def __init__(self, config: TextEncoderConfig):
        super().__init__()
        self.config = config
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        # UMT5 has relative_attention_bias in EVERY block (unlike T5 which only has it in block 0)
        self.block = [
            T5Block(config, has_relative_attention_bias=True)
            for i in range(config.num_layers)
        ]
        self.final_layer_norm = T5LayerNorm(
            config.d_model, eps=config.layer_norm_epsilon
        )

    def __call__(
        self, input_ids: mx.array, attention_mask: Optional[mx.array] = None
    ) -> Tuple[mx.array, mx.array]:
        hidden_states = self.shared(input_ids)
        initial_embedding = hidden_states

        extended_mask = None
        if attention_mask is not None:
            extended_mask = attention_mask[:, None, None, :]
            extended_mask = (1.0 - extended_mask) * -1e9

        for block in self.block:
            # UMT5: each block has its own relative_attention_bias, so pass None each time
            hidden_states, _ = block(
                hidden_states, mask=extended_mask, position_bias=None
            )

        hidden_states = self.final_layer_norm(hidden_states)
        return hidden_states, initial_embedding
