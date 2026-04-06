import math
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps)


class RotaryEmbedding:
    def __init__(self, dim: int, base: float = 500000.0):
        self._dim = dim
        self._base = base

    def __call__(self, positions: mx.array) -> Tuple[mx.array, mx.array]:
        inv_freq = 1.0 / (
            self._base ** (mx.arange(0, self._dim, 2, dtype=mx.float32) / self._dim)
        )
        t = positions.astype(mx.float32)
        freqs = mx.outer(t, inv_freq)
        emb = mx.concatenate([freqs, freqs], axis=-1)
        return mx.cos(emb), mx.sin(emb)


def rotate_half(x: mx.array) -> mx.array:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(
    q: mx.array, k: mx.array, cos: mx.array, sin: mx.array
) -> Tuple[mx.array, mx.array]:
    cos = cos[None, :, None, :]  # (1, L, 1, D)
    sin = sin[None, :, None, :]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

    def __call__(
        self,
        x: mx.array,
        cos: mx.array,
        sin: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        B, L, _ = x.shape

        q = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(B, L, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(B, L, self.num_kv_heads, self.head_dim)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if cache is not None:
            k_cache, v_cache = cache
            k = mx.concatenate([k_cache, k], axis=1)
            v = mx.concatenate([v_cache, v], axis=1)

        new_cache = (k, v)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out), new_cache


class MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        rms_norm_eps: float,
    ):
        super().__init__()
        self.self_attn = Attention(hidden_size, num_heads, num_kv_heads, head_dim)
        self.mlp = MLP(hidden_size, intermediate_size)
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        cos: mx.array,
        sin: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        residual = x
        x = self.input_layernorm(x)
        h, new_cache = self.self_attn(x, cos, sin, mask, cache)
        x = residual + h

        residual = x
        x = self.post_attention_layernorm(x)
        x = residual + self.mlp(x)

        return x, new_cache


class LlamaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            DecoderLayer(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                intermediate_size=config.intermediate_size,
                rms_norm_eps=config.rms_norm_eps,
            )
            for _ in range(config.num_hidden_layers)
        ]
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(dim=config.head_dim, base=config.rope_theta)

    def __call__(
        self,
        inputs_embeds: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[List[Tuple[mx.array, mx.array]]] = None,
    ) -> Tuple[mx.array, List[Tuple[mx.array, mx.array]]]:
        B, L, _ = inputs_embeds.shape

        offset = 0
        if cache is not None and cache[0] is not None:
            offset = cache[0][0].shape[1]

        positions = mx.arange(offset, offset + L, dtype=mx.int32)
        cos, sin = self.rotary_emb(positions)

        if mask is None and L > 1:
            k_len = offset + L
            q_pos = mx.expand_dims(
                mx.arange(offset, offset + L, dtype=mx.int32), axis=1
            )
            k_pos = mx.expand_dims(mx.arange(0, k_len, dtype=mx.int32), axis=0)
            allow = q_pos >= k_pos
            neg_inf = mx.array(float("-inf"), dtype=mx.float32)
            mask = mx.where(allow, mx.array(0.0, dtype=mx.float32), neg_inf)
            mask = mask[None, None, :, :]

        h = inputs_embeds
        new_caches = []

        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            h, c = layer(h, cos, sin, mask=mask, cache=layer_cache)
            new_caches.append(c)

        h = self.norm(h)
        return h, new_caches
