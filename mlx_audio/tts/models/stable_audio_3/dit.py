"""Diffusion Transformer (DiT) for Stable Audio 3.

20-layer transformer with AdaLN conditioning, self-attention with RoPE,
cross-attention for text tokens, and optional local additive conditioning.
"""

import math
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


@dataclass
class DiTConfig:
    io_channels: int = 256
    embed_dim: int = 1024
    depth: int = 20
    num_heads: int = 16
    cond_token_dim: int = 768
    global_cond_dim: int = 768
    local_add_cond_dim: int = 257
    num_memory_tokens: int = 64
    ff_mult: float = 4.0


class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gamma = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        x_f32 = x.astype(mx.float32)
        rms = mx.sqrt(mx.mean(x_f32 * x_f32, axis=-1, keepdims=True) + 1e-6)
        return (self.gamma * (x_f32 / rms)).astype(x.dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.inv_freq = 1.0 / (
            10000.0 ** (mx.arange(0, dim, 2).astype(mx.float32) / dim)
        )

    def __call__(self, seq_len: int) -> mx.array:
        t = mx.arange(seq_len, dtype=mx.float32)
        return mx.outer(t, self.inv_freq)


def rotate_half(x: mx.array) -> mx.array:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rotary_emb(x: mx.array, freqs: mx.array) -> mx.array:
    # freqs has dim/2 entries; only first rope_dim dimensions get rotated
    rope_dim = freqs.shape[-1] * 2
    cos = mx.cos(freqs)
    sin = mx.sin(freqs)
    cos = mx.concatenate([cos, cos], axis=-1)
    sin = mx.concatenate([sin, sin], axis=-1)

    while cos.ndim < x.ndim:
        cos = mx.expand_dims(cos, axis=0)
        sin = mx.expand_dims(sin, axis=0)

    x_rope = x[..., :rope_dim]
    x_pass = x[..., rope_dim:]
    x_rotated = x_rope * cos + rotate_half(x_rope) * sin
    return mx.concatenate([x_rotated, x_pass], axis=-1)


class ExpoFourierFeatures(nn.Module):
    def __init__(
        self, dim: int = 256, min_freq: float = 0.5, max_freq: float = 10000.0
    ):
        super().__init__()
        self._dim = dim
        self._log_min = math.log(min_freq)
        self._log_max = math.log(max_freq)

    def __call__(self, t: mx.array) -> mx.array:
        if t.ndim == 1:
            t = mx.expand_dims(t, -1)
        half = self._dim // 2
        ramp = mx.linspace(0.0, 1.0, half)
        freqs = mx.exp(ramp * (self._log_max - self._log_min) + self._log_min)
        args = t * freqs * (2.0 * math.pi)
        return mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)


class _GatedProj(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.proj(x)
        x, gate = mx.split(x, 2, axis=-1)
        return x * nn.silu(gate)


class GLU(nn.Module):
    def __init__(self, dim: int, mult: float = 4.0):
        super().__init__()
        inner = int(dim * mult)
        self.ff = [
            _GatedProj(dim, inner),
            None,
            nn.Linear(inner, dim),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        x = self.ff[0](x)
        x = self.ff[2](x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.scale = self.head_dim**-0.5

    def __call__(self, x: mx.array, freqs: mx.array) -> mx.array:
        B, L, _ = x.shape
        qkv = self.to_qkv(x)
        q, k, v = mx.split(qkv, 3, axis=-1)

        q = q.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if freqs is not None:
            q = apply_rotary_emb(q, freqs)
            k = apply_rotary_emb(k, freqs)

        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        attn = mx.softmax(attn, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.to_out(out)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, cond_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(cond_dim, dim * 2, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.scale = self.head_dim**-0.5

    def __call__(
        self,
        x: mx.array,
        context: mx.array,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        B, L, _ = x.shape
        _, S, _ = context.shape

        q = self.to_q(x)
        kv = self.to_kv(context)
        k, v = mx.split(kv, 2, axis=-1)

        q = q.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, S, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, S, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        q = self.q_norm(q)
        k = self.k_norm(k)

        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        if mask is not None:
            attn = attn + mask
        attn = mx.softmax(attn, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.to_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, config: DiTConfig):
        super().__init__()
        dim = config.embed_dim

        self.to_scale_shift_gate = mx.zeros((6 * dim,))

        self.pre_norm = RMSNorm(dim)
        self.self_attn = SelfAttention(dim, config.num_heads)

        self.cross_attend_norm = RMSNorm(dim)
        self.cross_attn = CrossAttention(dim, config.num_heads, dim)

        if config.local_add_cond_dim > 0:
            self.to_local_embed = [
                nn.Linear(config.local_add_cond_dim, dim, bias=True),
                None,
                nn.Linear(dim, dim, bias=True),
            ]

        self.ff_norm = RMSNorm(dim)
        self.ff = GLU(dim, config.ff_mult)

    def __call__(
        self,
        x: mx.array,
        freqs: mx.array,
        cond_tokens: mx.array,
        adaln_params: mx.array,
        local_cond: Optional[mx.array] = None,
        cond_mask: Optional[mx.array] = None,
    ) -> mx.array:
        adaln_params = adaln_params + self.to_scale_shift_gate
        scale_self, shift_self, gate_self, scale_ff, shift_ff, gate_ff = mx.split(
            adaln_params, 6, axis=-1
        )

        residual = x
        x = self.pre_norm(x)
        x = x * (1 + scale_self) + shift_self
        x = self.self_attn(x, freqs)
        x = x * mx.sigmoid(1 - gate_self)
        x = x + residual

        x = x + self.cross_attn(
            self.cross_attend_norm(x), cond_tokens, mask=cond_mask
        )

        if local_cond is not None and hasattr(self, "to_local_embed"):
            lc = self.to_local_embed[0](local_cond)
            lc = nn.silu(lc)
            lc = self.to_local_embed[2](lc)
            x = x + lc

        residual = x
        x = self.ff_norm(x)
        x = x * (1 + scale_ff) + shift_ff
        x = self.ff(x)
        x = x * mx.sigmoid(1 - gate_ff)
        x = x + residual

        return x


class ContinuousTransformer(nn.Module):
    def __init__(self, config: DiTConfig):
        super().__init__()
        dim = config.embed_dim

        self.project_in = nn.Linear(config.io_channels, dim, bias=False)
        self.project_out = nn.Linear(dim, config.io_channels, bias=False)

        self.memory_tokens = mx.zeros((config.num_memory_tokens, dim))
        self.rotary_pos_emb = RotaryEmbedding(dim // config.num_heads // 2)

        self.global_cond_embedder = [
            nn.Linear(dim, dim, bias=True),
            None,
            nn.Linear(dim, dim * 6, bias=True),
        ]

        self.layers = [TransformerBlock(config) for _ in range(config.depth)]

    def __call__(
        self,
        x: mx.array,
        cond_tokens: mx.array,
        global_embed: mx.array,
        local_cond: Optional[mx.array] = None,
        cond_mask: Optional[mx.array] = None,
    ) -> mx.array:
        B = x.shape[0]
        x = self.project_in(x)

        mem = mx.broadcast_to(
            self.memory_tokens,
            (B, self.memory_tokens.shape[0], self.memory_tokens.shape[1]),
        )
        x = mx.concatenate([mem, x], axis=1)

        if local_cond is not None:
            mem_pad = mx.zeros((B, mem.shape[1], local_cond.shape[-1]))
            local_cond = mx.concatenate([mem_pad, local_cond], axis=1)

        freqs = self.rotary_pos_emb(x.shape[1])

        adaln = self.global_cond_embedder[0](global_embed)
        adaln = nn.silu(adaln)
        adaln = self.global_cond_embedder[2](adaln)
        adaln = mx.expand_dims(adaln, 1)

        for layer in self.layers:
            x = layer(x, freqs, cond_tokens, adaln, local_cond, cond_mask)

        x = x[:, mem.shape[1] :]
        x = self.project_out(x)
        return x


class DiffusionTransformer(nn.Module):
    def __init__(self, config: DiTConfig):
        super().__init__()

        self.preprocess_conv = nn.Conv1d(
            config.io_channels, config.io_channels, 1, bias=False
        )
        self.postprocess_conv = nn.Conv1d(
            config.io_channels, config.io_channels, 1, bias=False
        )

        self.timestep_features = ExpoFourierFeatures(256)
        self.to_timestep_embed = [
            nn.Linear(256, config.embed_dim, bias=True),
            None,
            nn.Linear(config.embed_dim, config.embed_dim, bias=True),
        ]

        self.to_cond_embed = [
            nn.Linear(config.cond_token_dim, config.embed_dim, bias=False),
            None,
            nn.Linear(config.embed_dim, config.embed_dim, bias=False),
        ]
        self.to_global_embed = [
            nn.Linear(config.global_cond_dim, config.embed_dim, bias=False),
            None,
            nn.Linear(config.embed_dim, config.embed_dim, bias=False),
        ]

        self.transformer = ContinuousTransformer(config)

    def __call__(
        self,
        x: mx.array,
        t: mx.array,
        cond_tokens: mx.array,
        global_cond: mx.array,
        local_cond: Optional[mx.array] = None,
        cond_mask: Optional[mx.array] = None,
    ) -> mx.array:
        x = x.transpose(0, 2, 1)
        x = self.preprocess_conv(x) + x

        t_emb = self.to_timestep_embed[0](self.timestep_features(t))
        t_emb = self.to_timestep_embed[2](nn.silu(t_emb))

        cond_tokens = self.to_cond_embed[0](cond_tokens)
        cond_tokens = self.to_cond_embed[2](nn.silu(cond_tokens))

        g_emb = self.to_global_embed[0](global_cond)
        g_emb = self.to_global_embed[2](nn.silu(g_emb))

        global_embed = t_emb + g_emb

        x = self.transformer(x, cond_tokens, global_embed, local_cond, cond_mask)
        x = self.postprocess_conv(x) + x

        return x.transpose(0, 2, 1)
