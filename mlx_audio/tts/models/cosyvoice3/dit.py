"""CosyVoice3 DiT flow-matching estimator.

The estimator accepts noised mels, conditioning mels, speaker embeddings, and
an optional prompt-mel condition, then returns a velocity field. It implements
full-context attention; streaming chunk masks are not supported.
"""

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

# ---------------------------------------------------------------------------
# Rotary position embedding
# ---------------------------------------------------------------------------
# Applies interleaved rotary frequencies to the flattened q/k representation.
# The first dim_head channels are rotated before q/k are split into heads.


class RotaryEmbedding:
    """Interleaved rotary embedding with a cached cosine/sine table."""

    def __init__(self, dim: int, theta: float = 10000.0):
        self.dim = dim
        self.theta = theta
        self._cos: Optional[mx.array] = None
        self._sin: Optional[mx.array] = None
        self._cached_len = 0

    def _build(self, seq_len: int):
        inv_freq = 1.0 / (
            self.theta ** (mx.arange(0, self.dim, 2).astype(mx.float32) / self.dim)
        )
        t = mx.arange(seq_len).astype(mx.float32)
        freqs = mx.outer(t, inv_freq)  # (N, dim/2)
        freqs = mx.repeat(freqs, 2, axis=-1)  # interleave: (N, dim)
        self._cos = mx.cos(freqs)
        self._sin = mx.sin(freqs)
        mx.eval(self._cos, self._sin)
        self._cached_len = seq_len

    def forward_from_seq_len(self, seq_len: int):
        if self._cos is None or seq_len > self._cached_len:
            self._build(seq_len)
        return self._cos[:seq_len], self._sin[:seq_len]


def _rotate_half(x: mx.array) -> mx.array:
    """Rotate adjacent channel pairs by 90 degrees."""
    shape = x.shape
    x = x.reshape(*shape[:-1], shape[-1] // 2, 2)
    x1, x2 = x[..., 0], x[..., 1]
    x = mx.stack([-x2, x1], axis=-1)
    return x.reshape(shape)


def apply_rotary_pos_emb(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Apply rotary embedding to the leading rotary dimensions of ``x``."""
    rot_dim = cos.shape[-1]
    x_rot, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    cos = cos[None]
    sin = sin[None]
    x_rot = (
        x_rot.astype(mx.float32) * cos + _rotate_half(x_rot).astype(mx.float32) * sin
    ).astype(x.dtype)
    return mx.concatenate([x_rot, x_pass], axis=-1)


# ---------------------------------------------------------------------------
# Time / conv-position / text-conv building blocks
# ---------------------------------------------------------------------------


class SinusPositionEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def __call__(self, x: mx.array, scale: float = 1000.0) -> mx.array:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = mx.exp(mx.arange(half_dim).astype(mx.float32) * -emb)
        emb = scale * x[:, None] * emb[None, :]
        return mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)


class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int, freq_embed_dim: int = 256):
        super().__init__()
        self.time_embed = SinusPositionEmbedding(freq_embed_dim)
        self.time_mlp = [nn.Linear(freq_embed_dim, dim), nn.Linear(dim, dim)]

    def __call__(self, timestep: mx.array) -> mx.array:
        x = self.time_embed(timestep)
        x = nn.silu(self.time_mlp[0](x))
        return self.time_mlp[1](x)


class CausalConvPositionEmbedding(nn.Module):
    """Two causal depthwise-ish Conv1d(groups=16) + Mish, left-padded."""

    def __init__(self, dim: int, kernel_size: int = 31, groups: int = 16):
        super().__init__()
        assert kernel_size % 2 != 0
        self.kernel_size = kernel_size
        # nn.Conv1d in MLX expects (B, L, C_in) and weight (C_out, k, C_in/groups)
        self.conv1 = nn.Conv1d(dim, dim, kernel_size, groups=groups)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size, groups=groups)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        # x: (B, N, D)  (channels-last, matches MLX conv)
        if mask is not None:
            x = mx.where(mask[..., None], x, 0.0)
        x = mx.pad(x, [(0, 0), (self.kernel_size - 1, 0), (0, 0)])
        x = nn.mish(self.conv1(x))
        x = mx.pad(x, [(0, 0), (self.kernel_size - 1, 0), (0, 0)])
        x = nn.mish(self.conv2(x))
        if mask is not None:
            x = mx.where(mask[..., None], x, 0.0)
        return x


class GRN(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gamma = mx.zeros((1, 1, dim))
        self.beta = mx.zeros((1, 1, dim))

    def __call__(self, x: mx.array) -> mx.array:
        gx = mx.sqrt(mx.sum(x * x, axis=1, keepdims=True))
        nx = gx / (mx.mean(gx, axis=-1, keepdims=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x


class ConvNeXtV2Block(nn.Module):
    """Depthwise Conv1d(k=7) + LN + pwconv + GELU + GRN + pwconv (+ residual)."""

    def __init__(self, dim: int, intermediate_dim: int, dilation: int = 1):
        super().__init__()
        padding = (dilation * (7 - 1)) // 2
        self.padding = padding
        self.dilation = dilation
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=7, padding=padding, groups=dim, dilation=dilation
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.grn = GRN(intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, N, D) channels-last
        residual = x
        x = self.dwconv(x)  # MLX conv is channels-last already
        x = self.norm(x)
        x = self.pwconv1(x)
        x = nn.gelu(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        return residual + x


# ---------------------------------------------------------------------------
# AdaLayerNorm (zero) modules
# ---------------------------------------------------------------------------


def _layer_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    xf = x.astype(mx.float32)
    mean = mx.mean(xf, axis=-1, keepdims=True)
    var = mx.var(xf, axis=-1, keepdims=True)
    return ((xf - mean) * mx.rsqrt(var + eps)).astype(x.dtype)


class AdaLayerNormZero(nn.Module):
    """SiLU->Linear(dim*6); returns modulated x + msa gate + mlp params."""

    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim * 6)

    def __call__(self, x: mx.array, emb: mx.array):
        emb = self.linear(nn.silu(emb))  # (B, dim*6)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mx.split(
            emb, 6, axis=-1
        )
        x = _layer_norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormZeroFinal(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim * 2)

    def __call__(self, x: mx.array, emb: mx.array) -> mx.array:
        emb = self.linear(nn.silu(emb))
        scale, shift = mx.split(emb, 2, axis=-1)
        return _layer_norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]


# ---------------------------------------------------------------------------
# Attention + FeedForward + DiTBlock
# ---------------------------------------------------------------------------


class Attention(nn.Module):
    """Self-attention with rotary position embeddings."""

    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64):
        super().__init__()
        self.heads = heads
        self.inner_dim = dim_head * heads
        self.to_q = nn.Linear(dim, self.inner_dim)
        self.to_k = nn.Linear(dim, self.inner_dim)
        self.to_v = nn.Linear(dim, self.inner_dim)
        self.to_out = [nn.Linear(self.inner_dim, dim)]

    def __call__(self, x: mx.array, mask: Optional[mx.array], rope=None) -> mx.array:
        B, N, _ = x.shape
        head_dim = self.inner_dim // self.heads

        q = self.to_q(x)  # (B, N, inner_dim), flat (heads not split yet)
        k = self.to_k(x)
        v = self.to_v(x)

        # Apply rotary frequencies before splitting into attention heads.
        if rope is not None:
            cos, sin = rope
            q = apply_rotary_pos_emb(q, cos, sin)
            k = apply_rotary_pos_emb(k, cos, sin)

        q = q.reshape(B, N, self.heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, N, self.heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, N, self.heads, head_dim).transpose(0, 2, 1, 3)

        scale = 1.0 / math.sqrt(head_dim)
        scores = (q @ k.transpose(0, 1, 3, 2)) * scale
        if mask is not None:
            # mask: (B, 1, N, N) or (B, N, N) boolean -> broadcast over heads
            if mask.ndim == 3:
                mask = mask[:, None]
            scores = mx.where(mask, scores, mx.array(float("-inf"), dtype=scores.dtype))
        weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
        out = (weights @ v).transpose(0, 2, 1, 3).reshape(B, N, self.inner_dim)
        return self.to_out[0](out)


class FeedForward(nn.Module):
    """Linear -> GELU(tanh) -> Linear feed-forward network."""

    def __init__(self, dim: int, mult: float = 2.0):
        super().__init__()
        inner_dim = int(dim * mult)
        self.ff = [nn.Linear(dim, inner_dim), nn.Linear(inner_dim, dim)]

    def __call__(self, x: mx.array) -> mx.array:
        return self.ff[1](nn.gelu_approx(self.ff[0](x)))


class DiTBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, ff_mult: float = 2.0):
        super().__init__()
        self.attn_norm = AdaLayerNormZero(dim)
        self.attn = Attention(dim=dim, heads=heads, dim_head=dim_head)
        self.ff = FeedForward(dim=dim, mult=ff_mult)

    def __call__(self, x: mx.array, t: mx.array, mask=None, rope=None) -> mx.array:
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t)
        attn_out = self.attn(norm, mask=mask, rope=rope)
        x = x + gate_msa[:, None] * attn_out

        ff_norm = _layer_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_out = self.ff(ff_norm)
        x = x + gate_mlp[:, None] * ff_out
        return x


# ---------------------------------------------------------------------------
# Input embedding (mel x + prompt cond + mu + spk) -> hidden
# ---------------------------------------------------------------------------


class InputEmbedding(nn.Module):
    def __init__(self, mel_dim: int, mu_dim: int, out_dim: int, spk_dim: int = 0):
        super().__init__()
        self.spk_dim = spk_dim
        self.proj = nn.Linear(mel_dim * 2 + mu_dim + spk_dim, out_dim)
        self.conv_pos_embed = CausalConvPositionEmbedding(out_dim)

    def __call__(
        self, x: mx.array, cond: mx.array, mu: mx.array, spks: Optional[mx.array]
    ) -> mx.array:
        # all inputs channels-last: (B, N, D)
        to_cat = [x, cond, mu]
        if self.spk_dim > 0 and spks is not None:
            spks = mx.broadcast_to(spks[:, None, :], (x.shape[0], x.shape[1], spks.shape[-1]))
            to_cat.append(spks)
        x = self.proj(mx.concatenate(to_cat, axis=-1))
        x = self.conv_pos_embed(x) + x
        return x


# ---------------------------------------------------------------------------
# DiT
# ---------------------------------------------------------------------------


class DiT(nn.Module):
    def __init__(
        self,
        dim: int = 1024,
        depth: int = 22,
        heads: int = 16,
        dim_head: int = 64,
        ff_mult: float = 2.0,
        mel_dim: int = 80,
        mu_dim: Optional[int] = 80,
        spk_dim: Optional[int] = 80,
        out_channels: int = 80,
        long_skip_connection: bool = False,
        static_chunk_size: int = 50,
        num_decoding_left_chunks: int = -1,
    ):
        super().__init__()
        if mu_dim is None:
            mu_dim = mel_dim
        self.out_channels = out_channels
        self.dim = dim
        self.depth = depth
        self.static_chunk_size = static_chunk_size
        self.num_decoding_left_chunks = num_decoding_left_chunks

        self.time_embed = TimestepEmbedding(dim)
        self.input_embed = InputEmbedding(mel_dim, mu_dim, dim, spk_dim or 0)
        self.rotary_embed = RotaryEmbedding(dim_head)

        self.transformer_blocks = [
            DiTBlock(dim=dim, heads=heads, dim_head=dim_head, ff_mult=ff_mult)
            for _ in range(depth)
        ]
        self.long_skip_connection = (
            nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None
        )
        self.norm_out = AdaLayerNormZeroFinal(dim)
        self.proj_out = nn.Linear(dim, mel_dim)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array,
        mu: mx.array,
        t: mx.array,
        spks: Optional[mx.array] = None,
        cond: Optional[mx.array] = None,
        streaming: bool = False,
    ) -> mx.array:
        # channels-first -> channels-last
        x = mx.transpose(x, (0, 2, 1))  # (B, N, mel)
        mu = mx.transpose(mu, (0, 2, 1))  # (B, N, mu)
        cond = mx.transpose(cond, (0, 2, 1))  # (B, N, mel)

        B, N = x.shape[0], x.shape[1]
        if t.ndim == 0:
            t = mx.broadcast_to(t, (B,))

        t = self.time_embed(t)  # (B, dim)
        x = self.input_embed(x, cond, mu, spks)  # (B, N, dim)

        rope = self.rotary_embed.forward_from_seq_len(N)

        residual = x if self.long_skip_connection is not None else None

        # Non-streaming: full-context attention gated by the padding mask.
        # mask: (B, 1, N) -> valid frames; build (B, N, N) key mask.
        if streaming:
            raise NotImplementedError("streaming chunk mask (add_optional_chunk_mask)")
        key_valid = mask[:, 0, :].astype(mx.bool_)  # (B, N)
        attn_mask = mx.broadcast_to(key_valid[:, None, :], (B, N, N))  # (B, N, N)

        for block in self.transformer_blocks:
            x = block(x, t, mask=attn_mask, rope=rope)

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(mx.concatenate([x, residual], axis=-1))

        x = self.norm_out(x, t)
        out = self.proj_out(x)  # (B, N, mel)
        return mx.transpose(out, (0, 2, 1))  # (B, mel, N)
