"""Qwen3-style global backbone for MOSS-TTS Local/Delay models."""

from __future__ import annotations

from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.activations import swiglu
from mlx_lm.models.base import create_attention_mask, scaled_dot_product_attention
from mlx_lm.models.cache import KVCache
from mlx_lm.models.rope_utils import initialize_rope

from .config import MossQwen3Config


class MossTTSBackboneAttention(nn.Module):
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

        # Qwen3-style QK-norm.
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.rope = initialize_rope(
            self.head_dim,
            base=config.rope_theta,
            traditional=False,
            scaling_config=config.rope_scaling,
            max_position_embeddings=config.max_position_embeddings,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
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

        if cache is None:
            queries = self.rope(queries)
            keys = self.rope(keys)
        else:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)

        attended = scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=cache,
            scale=self.scale,
            mask=mask,
        )
        attended = attended.transpose(0, 2, 1, 3).reshape(
            batch_size, sequence_length, -1
        )
        return self.o_proj(attended)


class MossTTSBackboneMLP(nn.Module):
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


class MossTTSBackboneLayer(nn.Module):
    def __init__(self, config: MossQwen3Config):
        super().__init__()
        self.self_attn = MossTTSBackboneAttention(config)
        self.mlp = MossTTSBackboneMLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(
        self,
        hidden_states: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        residual = hidden_states
        hidden_states = self.self_attn(self.input_layernorm(hidden_states), mask, cache)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        hidden_states = residual + hidden_states
        return hidden_states


class MossTTSBackbone(nn.Module):
    """Global backbone that consumes fused multimodal embeddings."""

    def __init__(self, config: MossQwen3Config):
        super().__init__()
        self.config = config
        self.layers = [
            MossTTSBackboneLayer(config) for _ in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        input_embeddings: mx.array,
        cache: Optional[List[Optional[KVCache]]] = None,
    ) -> mx.array:
        if cache is None:
            cache = [None] * len(self.layers)
        mask = create_attention_mask(input_embeddings, cache[0])

        hidden_states = input_embeddings
        for layer, layer_cache in zip(self.layers, cache):
            hidden_states = layer(hidden_states, mask=mask, cache=layer_cache)
        return self.norm(hidden_states)

    def make_cache(self) -> List[KVCache]:
        return [KVCache() for _ in self.layers]


__all__ = ["MossTTSBackbone"]
