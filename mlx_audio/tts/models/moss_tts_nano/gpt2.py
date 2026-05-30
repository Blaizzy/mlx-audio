from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from .config import GPT2Config


@dataclass
class KVCache:
    keys: mx.array | None = None
    values: mx.array | None = None

    @property
    def offset(self) -> int:
        return 0 if self.keys is None else int(self.keys.shape[2])

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        if self.keys is None:
            self.keys = keys
            self.values = values
        else:
            self.keys = mx.concatenate([self.keys, keys], axis=2)
            self.values = mx.concatenate([self.values, values], axis=2)
        return self.keys, self.values


def gelu_new(x: mx.array) -> mx.array:
    return (
        0.5
        * x
        * (1.0 + mx.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * mx.power(x, 3))))
    )


def rotate_half(x: mx.array) -> mx.array:
    even = x[..., ::2]
    odd = x[..., 1::2]
    return mx.stack([-odd, even], axis=-1).reshape(x.shape)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE head_dim must be even, got {dim}")
        self.dim = int(dim)
        self.base = float(base)

    def __call__(self, x: mx.array, offset: int = 0) -> mx.array:
        seq_len = x.shape[1]
        positions = mx.arange(offset, offset + seq_len, dtype=mx.float32)
        inv_freq = 1.0 / (
            self.base ** (mx.arange(0, self.dim, 2, dtype=mx.float32) / self.dim)
        )
        freqs = positions[:, None] * inv_freq[None, :]
        cos = mx.repeat(mx.cos(freqs), 2, axis=-1)[None, :, None, :].astype(x.dtype)
        sin = mx.repeat(mx.sin(freqs), 2, axis=-1)[None, :, None, :].astype(x.dtype)
        return (x * cos) + (rotate_half(x) * sin)


def _causal_additive_mask(
    query_length: int,
    key_length: int,
    dtype: mx.Dtype,
    attention_mask: mx.array | None = None,
) -> mx.array:
    query_positions = mx.arange(query_length) + max(key_length - query_length, 0)
    key_positions = mx.arange(key_length)
    visible = key_positions[None, :] <= query_positions[:, None]
    mask = mx.where(visible, 0.0, mx.finfo(dtype).min).astype(dtype)
    mask = mask[None, None, :, :]
    if attention_mask is None:
        return mask
    key_mask = attention_mask.astype(mx.bool_)[:, None, None, :]
    key_additive = mx.where(key_mask, 0.0, mx.finfo(dtype).min).astype(dtype)
    return mask + key_additive


class Attention(nn.Module):
    def __init__(self, config: GPT2Config, layer_idx: int):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError(
                f"n_embd={config.n_embd} must be divisible by n_head={config.n_head}"
            )
        self.embed_dim = int(config.n_embd)
        self.num_heads = int(config.n_head)
        self.head_dim = self.embed_dim // self.num_heads
        self.layer_idx = int(layer_idx)
        self.scale = self.head_dim**-0.5 if config.scale_attn_weights else 1.0
        if config.scale_attn_by_inverse_layer_idx:
            self.scale /= float(self.layer_idx + 1)
        self.position_embedding_type = str(config.position_embedding_type).lower()
        self.c_attn = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=True)
        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.rotary_emb = (
            RotaryEmbedding(self.head_dim, base=config.rope_base)
            if self.position_embedding_type == "rope"
            else None
        )

    def __call__(
        self,
        x: mx.array,
        attention_mask: mx.array | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        batch_size, query_length, _ = x.shape
        qkv = self.c_attn(x)
        queries, keys, values = mx.split(qkv, 3, axis=-1)

        queries = queries.reshape(
            batch_size, query_length, self.num_heads, self.head_dim
        )
        keys = keys.reshape(batch_size, query_length, self.num_heads, self.head_dim)
        values = values.reshape(batch_size, query_length, self.num_heads, self.head_dim)

        offset = 0 if cache is None else cache.offset
        if self.rotary_emb is not None:
            queries = self.rotary_emb(queries, offset=offset)
            keys = self.rotary_emb(keys, offset=offset)

        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        key_length = keys.shape[2]
        mask = _causal_additive_mask(
            query_length,
            key_length,
            x.dtype,
            attention_mask=attention_mask,
        )
        output = mx.fast.scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=self.scale,
            mask=mask,
        )
        output = output.transpose(0, 2, 1, 3).reshape(
            batch_size, query_length, self.embed_dim
        )
        return self.c_proj(output)


class MLP(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        hidden_size = int(config.n_embd)
        inner_size = int(config.n_inner or 4 * hidden_size)
        self.fc_in = nn.Linear(hidden_size, inner_size, bias=True)
        self.fc_out = nn.Linear(inner_size, hidden_size, bias=True)
        self.activation_function = str(config.activation_function)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.fc_in(x)
        if self.activation_function == "gelu_new":
            x = gelu_new(x)
        else:
            x = nn.gelu(x)
        return self.fc_out(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: GPT2Config, layer_idx: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = Attention(config, layer_idx=layer_idx)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = MLP(config)

    def __call__(
        self,
        x: mx.array,
        attention_mask: mx.array | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        x = x + self.attn(self.ln_1(x), attention_mask=attention_mask, cache=cache)
        return x + self.mlp(self.ln_2(x))


class GPT2Model(nn.Module):
    def __init__(self, config: GPT2Config, *, use_token_embedding: bool = True):
        super().__init__()
        self.config = config
        self.use_token_embedding = bool(use_token_embedding)
        self.wte = (
            nn.Embedding(config.vocab_size, config.n_embd)
            if self.use_token_embedding
            else None
        )
        self.wpe = (
            nn.Embedding(config.n_positions, config.n_embd)
            if str(config.position_embedding_type).lower() == "absolute"
            else None
        )
        self.h = [TransformerBlock(config, layer_idx=i) for i in range(config.n_layer)]
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.h]

    def __call__(
        self,
        input_ids: mx.array | None = None,
        *,
        inputs_embeds: mx.array | None = None,
        attention_mask: mx.array | None = None,
        cache: list[KVCache] | None = None,
    ) -> mx.array:
        if inputs_embeds is None:
            if self.wte is None or input_ids is None:
                raise ValueError("inputs_embeds are required for this GPT2Model.")
            hidden_states = self.wte(input_ids)
        else:
            hidden_states = inputs_embeds

        if self.wpe is not None:
            offset = 0 if not cache else cache[0].offset
            position_ids = mx.arange(offset, offset + hidden_states.shape[1])
            hidden_states = hidden_states + self.wpe(position_ids)

        if cache is None:
            cache = [None] * len(self.h)
        for block, block_cache in zip(self.h, cache):
            hidden_states = block(
                hidden_states,
                attention_mask=attention_mask,
                cache=block_cache,
            )
        return self.ln_f(hidden_states)


def sanitize_gpt2_weights(weights: dict[str, Any], prefix: str) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in weights.items():
        if not key.startswith(prefix):
            continue
        sanitized[key] = value
    return sanitized
