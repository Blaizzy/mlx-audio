from __future__ import annotations

import math
from typing import Sequence

import mlx.core as mx
import mlx.nn as nn

from .layers import Attention, FeedForward, rms_norm
from .rope import LTXRopeType, precompute_freqs_cis


def norm_and_concat_per_token_rms(
    encoded_text: mx.array,
    attention_mask: mx.array,
    eps: float = 1e-6,
) -> mx.array:
    """Match LTX FeatureExtractorV2 per-token RMS normalization.

    Args:
        encoded_text: Hidden states with shape ``[B, T, D, L]``.
        attention_mask: Binary valid-token mask with shape ``[B, T]``.
    """
    batch, seq_len, hidden_dim, num_layers = encoded_text.shape
    variance = mx.mean(mx.square(encoded_text), axis=2, keepdims=True)
    normed = encoded_text * mx.rsqrt(variance + eps)
    normed = normed.reshape(batch, seq_len, hidden_dim * num_layers)
    mask = attention_mask.astype(mx.bool_)[:, :, None]
    return mx.where(mask, normed, mx.zeros_like(normed))


def stack_hidden_states(hidden_states: Sequence[mx.array] | mx.array) -> mx.array:
    if isinstance(hidden_states, mx.array):
        return hidden_states
    return mx.stack(list(hidden_states), axis=-1)


def rescale_norm(x: mx.array, target_dim: int, source_dim: int) -> mx.array:
    return x * math.sqrt(target_dim / source_dim)


class FeatureExtractorV2(nn.Module):
    """Dramabox Gemma hidden-state projector for audio context features."""

    def __init__(
        self,
        embedding_dim: int = 3840,
        audio_inner_dim: int = 2048,
        num_layers: int = 49,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.audio_inner_dim = audio_inner_dim
        self.num_layers = num_layers
        self.audio_aggregate_embed = nn.Linear(
            embedding_dim * num_layers, audio_inner_dim, bias=True
        )

    def __call__(
        self,
        hidden_states: Sequence[mx.array] | mx.array,
        attention_mask: mx.array,
    ) -> mx.array:
        encoded = stack_hidden_states(hidden_states)
        normed = norm_and_concat_per_token_rms(encoded, attention_mask)
        scaled = rescale_norm(
            normed.astype(encoded.dtype), self.audio_inner_dim, self.embedding_dim
        )
        return self.audio_aggregate_embed(scaled)


def binary_to_additive_attention_mask(
    attention_mask: mx.array,
    dtype=mx.float32,
) -> mx.array:
    return (attention_mask.astype(mx.int64) - 1).astype(dtype).reshape(
        attention_mask.shape[0], 1, -1, attention_mask.shape[-1]
    ) * mx.finfo(dtype).max


def _to_binary_mask(
    encoded: mx.array,
    encoded_mask: mx.array,
) -> tuple[mx.array, mx.array]:
    binary_mask = (encoded_mask < 0.000001).astype(mx.int64)
    binary_mask = binary_mask.reshape(encoded.shape[0], encoded.shape[1], 1)
    return encoded * binary_mask, binary_mask


class BasicTransformerBlock1D(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        rope_type: LTXRopeType | str = LTXRopeType.INTERLEAVED,
        apply_gated_attention: bool = False,
    ):
        super().__init__()
        self.attn1 = Attention(
            query_dim=dim,
            heads=heads,
            dim_head=dim_head,
            rope_type=rope_type,
            apply_gated_attention=apply_gated_attention,
        )
        self.ff = FeedForward(dim, dim_out=dim)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
        pe: tuple[mx.array, mx.array] | None = None,
    ) -> mx.array:
        hidden_states = hidden_states + self.attn1(
            rms_norm(hidden_states),
            mask=attention_mask,
            pe=pe,
        )
        hidden_states = hidden_states + self.ff(rms_norm(hidden_states))
        return hidden_states


class Embeddings1DConnector(nn.Module):
    def __init__(
        self,
        attention_head_dim: int = 64,
        num_attention_heads: int = 32,
        num_layers: int = 8,
        positional_embedding_theta: float = 10000.0,
        positional_embedding_max_pos: list[int] | None = None,
        num_learnable_registers: int | None = 128,
        rope_type: LTXRopeType | str = LTXRopeType.SPLIT,
        double_precision_rope: bool = True,
        apply_gated_attention: bool = True,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.inner_dim = num_attention_heads * attention_head_dim
        self.positional_embedding_theta = positional_embedding_theta
        self.positional_embedding_max_pos = positional_embedding_max_pos or [4096]
        self.num_learnable_registers = num_learnable_registers
        self.rope_type = rope_type
        self.double_precision_rope = double_precision_rope
        self.transformer_1d_blocks = [
            BasicTransformerBlock1D(
                dim=self.inner_dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                rope_type=rope_type,
                apply_gated_attention=apply_gated_attention,
            )
            for _ in range(num_layers)
        ]
        if num_learnable_registers:
            self.learnable_registers = mx.random.uniform(
                low=-1.0,
                high=1.0,
                shape=(num_learnable_registers, self.inner_dim),
            ).astype(mx.bfloat16)

    def _replace_padded_with_learnable_registers(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array,
    ) -> tuple[mx.array, mx.array]:
        if not self.num_learnable_registers:
            return hidden_states, attention_mask
        if hidden_states.shape[1] % self.num_learnable_registers != 0:
            raise ValueError(
                f"Hidden states sequence length {hidden_states.shape[1]} must be "
                f"divisible by num_learnable_registers {self.num_learnable_registers}."
            )

        binary = (attention_mask.squeeze(1).squeeze(1) >= -9000.0).astype(
            hidden_states.dtype
        )
        positions = mx.arange(hidden_states.shape[1], dtype=hidden_states.dtype)[None]
        order_keys = mx.where(binary > 0, positions - hidden_states.shape[1], positions)
        order = mx.argsort(order_keys, axis=1)
        hidden_states = mx.take_along_axis(hidden_states, order[:, :, None], axis=1)
        valid_counts = mx.sum(binary, axis=1, keepdims=True)
        front_mask = (positions < valid_counts).astype(hidden_states.dtype)

        registers = mx.tile(
            self.learnable_registers,
            (hidden_states.shape[1] // self.num_learnable_registers, 1),
        ).astype(hidden_states.dtype)
        registers = mx.broadcast_to(registers[None, :, :], hidden_states.shape)
        hidden_states = hidden_states * front_mask[:, :, None] + registers * (
            1.0 - front_mask[:, :, None]
        )
        return hidden_states, mx.zeros_like(attention_mask)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
    ) -> tuple[mx.array, mx.array | None]:
        if self.num_learnable_registers:
            if attention_mask is None:
                attention_mask = mx.zeros(
                    (
                        hidden_states.shape[0],
                        1,
                        hidden_states.shape[1],
                        hidden_states.shape[1],
                    ),
                    dtype=hidden_states.dtype,
                )
            hidden_states, attention_mask = (
                self._replace_padded_with_learnable_registers(
                    hidden_states, attention_mask
                )
            )

        indices_grid = mx.arange(hidden_states.shape[1], dtype=mx.float32)[
            None, None, :
        ]
        pe = precompute_freqs_cis(
            indices_grid=indices_grid,
            dim=self.inner_dim,
            out_dtype=hidden_states.dtype,
            theta=self.positional_embedding_theta,
            max_pos=self.positional_embedding_max_pos,
            num_attention_heads=self.num_attention_heads,
            rope_type=self.rope_type,
            double_precision=self.double_precision_rope,
        )

        for block in self.transformer_1d_blocks:
            hidden_states = block(hidden_states, attention_mask=attention_mask, pe=pe)
        return rms_norm(hidden_states), attention_mask


class DramaboxTextConditioner(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 3840,
        audio_inner_dim: int = 2048,
        num_gemma_layers: int = 49,
        connector_layers: int = 8,
        connector_heads: int = 32,
        connector_head_dim: int = 64,
        connector_num_learnable_registers: int | None = 128,
    ):
        super().__init__()
        self.feature_extractor = FeatureExtractorV2(
            embedding_dim=embedding_dim,
            audio_inner_dim=audio_inner_dim,
            num_layers=num_gemma_layers,
        )
        self.audio_connector = Embeddings1DConnector(
            attention_head_dim=connector_head_dim,
            num_attention_heads=connector_heads,
            num_layers=connector_layers,
            num_learnable_registers=connector_num_learnable_registers,
        )

    def __call__(
        self,
        hidden_states: Sequence[mx.array] | mx.array,
        attention_mask: mx.array,
    ) -> tuple[mx.array, mx.array]:
        audio_features = self.feature_extractor(hidden_states, attention_mask)
        additive_mask = binary_to_additive_attention_mask(
            attention_mask,
            dtype=audio_features.dtype,
        )
        audio_encoded, encoded_mask = self.audio_connector(
            audio_features, additive_mask
        )
        if encoded_mask is None:
            return audio_encoded, attention_mask
        audio_encoded, binary_mask = _to_binary_mask(audio_encoded, encoded_mask)
        return audio_encoded, binary_mask.squeeze(-1)
