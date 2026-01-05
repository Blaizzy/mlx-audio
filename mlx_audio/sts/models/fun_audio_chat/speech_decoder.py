# Copyright (c) 2025 Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

from .config import CRQTransformerConfig, FunAudioChatAudioEncoderConfig


class CRQAttention(nn.Module):

    def __init__(self, config: CRQTransformerConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        # Q and K norms (like Qwen3)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.rope = nn.RoPE(
            dims=self.head_dim,
            traditional=False,
            base=config.rope_theta,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        batch_size, seq_len, _ = hidden_states.shape

        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        # Reshape to (batch, seq_len, num_heads, head_dim)
        queries = queries.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        keys = keys.reshape(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        values = values.reshape(batch_size, seq_len, self.num_key_value_heads, self.head_dim)

        queries = self.q_norm(queries)
        keys = self.k_norm(keys)

        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)


        attn_output = mx.fast.scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=self.scale,
            mask=attention_mask,
        )

        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(
            batch_size, -1, self.num_heads * self.head_dim,
        )

        return self.o_proj(attn_output)


class CRQMLP(nn.Module):

    def __init__(self, config: CRQTransformerConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class CRQDecoderLayer(nn.Module):

    def __init__(self, config: CRQTransformerConfig, layer_idx: int = 0):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = CRQAttention(config, layer_idx)
        self.mlp = CRQMLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask=attention_mask, cache=cache)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class CRQTransformer(nn.Module):

    def __init__(self, config: CRQTransformerConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [CRQDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        cache: Optional[List[Any]] = None,
    ) -> mx.array:
        if inputs_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = inputs_embeds

        if cache is None:
            cache = [None] * len(self.layers)

        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, attention_mask=attention_mask, cache=cache[i])

        return self.norm(hidden_states)


class FunAudioChatDecoder(nn.Module):


    def __init__(self, config: FunAudioChatAudioEncoderConfig):
        super().__init__()
        self.config = config

        if config.crq_transformer_config is None:
            raise ValueError("crq_transformer_config is required for FunAudioChatDecoder")

        crq_config = config.crq_transformer_config
        self.group_size = config.group_size

        # Pre-matching: project from LM dim to upsampled representation
        # This upsamples by group_size (5) to go from 5Hz to 25Hz
        self.pre_matching = nn.Linear(config.output_dim, crq_config.hidden_size * self.group_size)

        # CRQ Transformer
        self.crq_transformer = CRQTransformer(crq_config)

        # Input/output matching projections
        self.input_matching = nn.Linear(crq_config.hidden_size, crq_config.hidden_size, bias=False)
        self.output_matching = nn.Linear(crq_config.hidden_size, crq_config.hidden_size, bias=False)

        # LM head for speech token prediction
        self.lm_head = nn.Linear(crq_config.hidden_size, config.codebook_size, bias=False)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        cache: Optional[List[Any]] = None,
    ) -> mx.array:
       
        batch_size, seq_len, _ = hidden_states.shape
        hidden_states = self.pre_matching(hidden_states)
        hidden_states = hidden_states.reshape(batch_size, seq_len * self.group_size, -1)
        hidden_states = self.input_matching(hidden_states)
        hidden_states = self.crq_transformer(inputs_embeds=hidden_states, attention_mask=attention_mask, cache=cache)
        hidden_states = self.output_matching(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits
