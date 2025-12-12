import math
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .config import LMConfig, ModelArgs


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x):
        # x: (..., D)
        return mx.fast.rms_norm(x, self.weight, self.eps)


class MiniCPMLongRoPE(nn.Module):
    def __init__(self, config: LMConfig):
        super().__init__()
        self.config = config
        self.dim = config.hidden_size // config.num_attention_heads  # Head dim
        if config.num_key_value_heads:  # If GQA
            pass

        self.base = config.rope_theta
        self.max_position_embeddings = config.max_position_embeddings
        self.original_max_position_embeddings = config.original_max_position_embeddings

        self.short_factor = mx.array(config.rope_short_factor)
        self.long_factor = mx.array(config.rope_long_factor)

        scale = self.max_position_embeddings / self.original_max_position_embeddings
        self.scaling_factor = math.sqrt(
            1
            + math.log(max(scale, 1.0))
            / math.log(self.original_max_position_embeddings)
        )

        # inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        half_dim = self.dim // 2
        exponents = mx.arange(0, half_dim, dtype=mx.float32) / half_dim
        inv_freq = 1.0 / (self.base**exponents)
        self.inv_freq = inv_freq  # (dim/2,)

    def __call__(self, position_ids):
        # position_ids: (N,) or (N, L)
        # We need to construct cos/sin

        seq_len = position_ids.max().item() + 1

        # Decide factors
        factors = (
            self.long_factor
            if seq_len > self.original_max_position_embeddings
            else self.short_factor
        )

        t = mx.arange(seq_len, dtype=mx.float32)

        # freqs = outer(t, 1/factors) * inv_freq
        # (L, 1) * (D/2,) * (D/2,) -> (L, D/2)

        freqs = (t[:, None] * (1.0 / factors[None, :])) * self.inv_freq[None, :]

        # emb = cat(freqs, freqs) -> (L, D)
        emb = mx.concatenate([freqs, freqs], axis=-1)

        cos = mx.cos(emb) * self.scaling_factor
        sin = mx.sin(emb) * self.scaling_factor

        return cos[position_ids], sin[position_ids]


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    # q: (B, L, H, D) or (B, H, L, D)?

    # Need to expand dims for H
    cos = cos[:, :, None, :]
    sin = sin[:, :, None, :]

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Attention(nn.Module):
    def __init__(self, config: LMConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // self.num_heads

        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=False
        )

    def __call__(self, x, cos, sin, mask=None, cache=None):
        B, L, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.reshape(B, L, self.num_heads, self.head_dim)
        k = k.reshape(B, L, self.num_kv_heads, self.head_dim)
        v = v.reshape(B, L, self.num_kv_heads, self.head_dim)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # KV Cache Logic - concatenate BEFORE transpose
        # Cache format: (B, L_past, H_kv, D)
        if cache is not None:
            k_cache, v_cache = cache
            k = mx.concatenate([k_cache, k], axis=1)
            v = mx.concatenate([v_cache, v], axis=1)

        # Store cache in (B, L, H_kv, D) format for next iteration
        new_cache = (k, v)

        # Transpose to (B, H, L, D) for attention
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=1 / math.sqrt(self.head_dim), mask=mask
        )

        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out), new_cache


class MLP(nn.Module):
    def __init__(self, config: LMConfig):
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

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class MiniCPMDecoderLayer(nn.Module):
    def __init__(self, config: LMConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.scale_depth = config.scale_depth
        self.num_hidden_layers = config.num_hidden_layers
        self.use_mup = config.use_mup

    def __call__(self, x, cos, sin, mask=None, cache=None):
        r = x
        x = self.input_layernorm(x)
        h, new_cache = self.self_attn(x, cos, sin, mask, cache)

        if self.use_mup:
            x = r + h * (self.scale_depth / math.sqrt(self.num_hidden_layers))
        else:
            x = r + h

        r = x
        x = self.post_attention_layernorm(x)
        h = self.mlp(x)

        if self.use_mup:
            x = r + h * (self.scale_depth / math.sqrt(self.num_hidden_layers))
        else:
            x = r + h

        return x, new_cache


class MiniCPMModel(nn.Module):
    def __init__(self, config: LMConfig):
        super().__init__()
        self.config = config

        if config.vocab_size > 0:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        self.layers = [
            MiniCPMDecoderLayer(config) for _ in range(config.num_hidden_layers)
        ]
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rope = MiniCPMLongRoPE(config)

    def __call__(
        self, inputs_embeds=None, input_ids=None, mask=None, cache=None, is_causal=True
    ):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        B, L, D = inputs_embeds.shape

        # Position IDs
        # If cache is present, we need to offset
        offset = 0
        if cache is not None:
            # cache is list of (k, v)
            # k shape (B, L_past, H_kv, D) - note: cache format is (B, L, H, D)
            offset = cache[0][0].shape[1]

        position_ids = mx.arange(offset, offset + L).astype(mx.int32)
        # expand to batch? RoPE expects (L) usually if broadcastable.

        cos, sin = self.rope(position_ids)
        # cos: (L, D)
        # resize for batch? apply_rotary_pos_emb handles broadcasting
        cos = cos[None, :, :]
        sin = sin[None, :, :]

        # Generate attention mask if is_causal=True and no explicit mask provided
        if mask is None and is_causal and L > 1:
            # Create causal mask: (1, 1, L, L) with -inf above diagonal
            causal_mask = mx.triu(mx.full((L, L), float("-inf")), k=1)
            mask = causal_mask[None, None, :, :]  # (1, 1, L, L)
        # If is_causal=False (like for DiT), mask stays None for full attention

        h = inputs_embeds
        new_caches = []

        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            h, c = layer(h, cos, sin, mask=mask, cache=layer_cache)
            new_caches.append(c)

        h = self.norm(h)
        return h, new_caches
