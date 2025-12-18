# Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

from dataclasses import dataclass
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


@dataclass
class GPT2Config:
    """Configuration for GPT2 model."""

    vocab_size: int = 50276
    n_positions: int = 8196
    n_embd: int = 1024
    n_layer: int = 24
    n_head: int = 16
    n_inner: Optional[int] = None  # defaults to 4 * n_embd
    activation_function: str = "gelu_new"
    resid_pdrop: float = 0.1
    embd_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    layer_norm_epsilon: float = 1e-5

    @property
    def hidden_size(self) -> int:
        return self.n_embd

    @property
    def num_attention_heads(self) -> int:
        return self.n_head

    @property
    def num_hidden_layers(self) -> int:
        return self.n_layer


def gelu_new(x: mx.array) -> mx.array:
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Also see the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """
    return (
        0.5
        * x
        * (1.0 + mx.tanh(mx.sqrt(2.0 / mx.pi) * (x + 0.044715 * mx.power(x, 3.0))))
    )


class GPT2Attention(nn.Module):
    """GPT2 multi-head attention."""

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.embed_dim = config.n_embd
        self.num_heads = config.n_head
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5

        # Combined QKV projection
        self.c_attn = nn.Linear(self.embed_dim, 3 * self.embed_dim)
        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array]]]:
        B, T, C = hidden_states.shape

        # QKV projection
        qkv = self.c_attn(hidden_states)
        q, k, v = mx.split(qkv, 3, axis=-1)

        # Reshape to (B, num_heads, T, head_dim)
        q = q.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Handle KV cache
        if cache is not None:
            k_cache, v_cache = cache
            k = mx.concatenate([k_cache, k], axis=2)
            v = mx.concatenate([v_cache, v], axis=2)

        new_cache = (k, v)

        # Attention
        attn_weights = (q @ k.transpose(0, 1, 3, 2)) * self.scale

        # Causal mask
        if attention_mask is None:
            # Create causal mask
            query_len = q.shape[2]
            key_len = k.shape[2]
            causal_mask = mx.triu(
                mx.full((query_len, key_len), float("-inf")), k=key_len - query_len + 1
            )
            attn_weights = attn_weights + causal_mask
        else:
            attn_weights = attn_weights + attention_mask

        attn_weights = mx.softmax(attn_weights, axis=-1)

        # Apply attention to values
        attn_output = attn_weights @ v

        # Reshape back
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(B, T, C)

        # Output projection
        attn_output = self.c_proj(attn_output)

        return attn_output, new_cache


class GPT2MLP(nn.Module):
    """GPT2 MLP (feed-forward) layer."""

    def __init__(self, config: GPT2Config):
        super().__init__()
        inner_dim = config.n_inner if config.n_inner is not None else 4 * config.n_embd
        self.c_fc = nn.Linear(config.n_embd, inner_dim)
        self.c_proj = nn.Linear(inner_dim, config.n_embd)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.c_fc(x)
        x = gelu_new(x)
        x = self.c_proj(x)
        return x


class GPT2Block(nn.Module):
    """GPT2 transformer block."""

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(config)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array]]]:
        # Self-attention
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output, new_cache = self.attn(
            hidden_states,
            attention_mask=attention_mask,
            cache=cache,
        )
        hidden_states = residual + attn_output

        # MLP
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)

        return hidden_states, new_cache


class GPT2Model(nn.Module):
    """GPT2 base model (without LM head)."""

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config

        # Token embeddings (not used when inputs_embeds is provided)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)

        # Transformer blocks
        self.h = [GPT2Block(config) for _ in range(config.n_layer)]

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        cache: Optional[list] = None,
    ) -> Tuple[mx.array, list]:
        """
        Forward pass of GPT2.

        Args:
            input_ids: Token IDs (B, T)
            inputs_embeds: Pre-computed embeddings (B, T, D) - takes precedence over input_ids
            attention_mask: Optional attention mask
            cache: Optional list of KV caches for each layer

        Returns:
            hidden_states: Output hidden states (B, T, D)
            new_cache: Updated KV caches
        """
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        elif input_ids is not None:
            hidden_states = self.wte(input_ids)
        else:
            raise ValueError("Either input_ids or inputs_embeds must be provided")

        B, T, _ = hidden_states.shape

        # Add positional embeddings
        if cache is not None and len(cache) > 0 and cache[0] is not None:
            past_length = cache[0][0].shape[2]
        else:
            past_length = 0

        position_ids = mx.arange(past_length, past_length + T)
        position_embeds = self.wpe(position_ids)
        hidden_states = hidden_states + position_embeds

        # Initialize cache if not provided
        if cache is None:
            cache = [None] * len(self.h)

        new_cache = []

        # Forward through transformer blocks
        for i, block in enumerate(self.h):
            layer_cache = cache[i] if i < len(cache) else None
            hidden_states, new_layer_cache = block(
                hidden_states,
                attention_mask=attention_mask,
                cache=layer_cache,
            )
            new_cache.append(new_layer_cache)

        # Final layer norm
        hidden_states = self.ln_f(hidden_states)

        return hidden_states, new_cache


def create_gpt2_config() -> GPT2Config:
    """Create GPT2 Medium config for T3 Turbo."""
    return GPT2Config(
        vocab_size=50276,
        n_positions=8196,
        n_embd=1024,
        n_layer=24,
        n_head=16,
        activation_function="gelu_new",
        layer_norm_epsilon=1e-5,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
    )
