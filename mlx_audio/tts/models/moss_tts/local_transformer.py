"""Local transformer used by MOSS-TTS-Local (no positional embeddings)."""

from __future__ import annotations

from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.activations import swiglu
from mlx_lm.models.base import create_causal_mask

from .config import MossQwen3Config


class MossTTSAttentionWithoutPositionalEmbedding(nn.Module):
    """Qwen3 attention without RoPE for short local channel sequences."""

    def __init__(self, config: MossQwen3Config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            config.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        batch_size, sequence_length, _ = hidden_states.shape

        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        queries = self.q_norm(
            queries.reshape(batch_size, sequence_length, self.num_heads, self.head_dim)
        ).transpose(0, 2, 1, 3)
        keys = self.k_norm(
            keys.reshape(
                batch_size,
                sequence_length,
                self.num_kv_heads,
                self.head_dim,
            )
        ).transpose(0, 2, 1, 3)
        values = values.reshape(
            batch_size,
            sequence_length,
            self.num_kv_heads,
            self.head_dim,
        ).transpose(0, 2, 1, 3)

        attended = mx.fast.scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=self.scale,
            mask=mask,
        )
        attended = attended.transpose(0, 2, 1, 3).reshape(
            batch_size, sequence_length, -1
        )
        return self.o_proj(attended)


class MossTTSLocalMLP(nn.Module):
    def __init__(self, config: MossQwen3Config):
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

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return self.down_proj(
            swiglu(self.gate_proj(hidden_states), self.up_proj(hidden_states))
        )


class MossTTSLocalLayer(nn.Module):
    def __init__(self, config: MossQwen3Config):
        super().__init__()
        self.self_attn = MossTTSAttentionWithoutPositionalEmbedding(config)
        self.mlp = MossTTSLocalMLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(self, hidden_states: mx.array, mask: Optional[mx.array]) -> mx.array:
        residual = hidden_states
        hidden_states = self.self_attn(self.input_layernorm(hidden_states), mask=mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        hidden_states = residual + hidden_states
        return hidden_states


class MossTTSLocalTransformer(nn.Module):
    def __init__(self, config: MossQwen3Config):
        super().__init__()
        self.config = config
        self.layers = [
            MossTTSLocalLayer(config) for _ in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, input_embeddings: mx.array) -> mx.array:
        sequence_length = input_embeddings.shape[1]
        mask = None
        if sequence_length > 1:
            mask = create_causal_mask(sequence_length)

        hidden_states = input_embeddings
        for layer in self.layers:
            hidden_states = layer(hidden_states, mask=mask)
        return self.norm(hidden_states)


__all__ = ["MossTTSLocalTransformer", "MossTTSAttentionWithoutPositionalEmbedding"]
