"""DiT transformer backbone for AudioDiT.

Implements the CrossDiT architecture with:
- Qwen2-style RoPE (base=100000)
- AdaLN (global or local) with per-block residual scale/shift
- Self-attention + cross-attention to text embeddings
- ConvNeXtV2 text processing blocks
- Long skip connection
"""

import math

import mlx.core as mx
import mlx.nn as nn

from .config import ModelConfig

# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        x_float = x.astype(mx.float32)
        normed = x_float * mx.rsqrt(
            mx.mean(x_float**2, axis=-1, keepdims=True) + self.eps
        )
        return normed.astype(x.dtype) * self.weight


# ---------------------------------------------------------------------------
# Rotary position embedding (Qwen2-style)
# ---------------------------------------------------------------------------


class RotaryEmbedding(nn.Module):
    def __init__(
        self, dim: int, max_position_embeddings: int = 2048, base: float = 100000.0
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self._cos = None
        self._sin = None
        self._cached_len = 0

    def _build(self, seq_len: int):
        inv_freq = 1.0 / (
            self.base ** (mx.arange(0, self.dim, 2).astype(mx.float32) / self.dim)
        )
        t = mx.arange(seq_len).astype(mx.float32)
        freqs = mx.outer(t, inv_freq)
        emb = mx.concatenate([freqs, freqs], axis=-1)
        self._cos = mx.cos(emb)
        self._sin = mx.sin(emb)
        self._cached_len = seq_len

    def __call__(self, seq_len: int):
        if self._cos is None or seq_len > self._cached_len:
            self._build(max(seq_len, self.max_position_embeddings))
        return self._cos[:seq_len], self._sin[:seq_len]


def _rotate_half(x: mx.array) -> mx.array:
    x1, x2 = mx.split(x, 2, axis=-1)
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rotary_emb(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    # x: (B, heads, L, dim), cos/sin: (L, dim)
    cos = cos[None, None]  # (1, 1, L, dim)
    sin = sin[None, None]
    return (
        x.astype(mx.float32) * cos + _rotate_half(x).astype(mx.float32) * sin
    ).astype(x.dtype)


# ---------------------------------------------------------------------------
# GRN + ConvNeXtV2 (text processing)
# ---------------------------------------------------------------------------


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
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        dilation: int = 1,
        kernel_size: int = 7,
        bias: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        padding = (dilation * (kernel_size - 1)) // 2
        self.dw_padding = padding
        self.dw_dilation = dilation
        self.channels = dim
        # Depthwise conv weight: (dim, kernel_size, 1) for groups=dim
        self.dwconv_weight = mx.zeros((dim, kernel_size, 1))
        self.dwconv_bias = mx.zeros((dim,))
        self.norm = nn.LayerNorm(dim, eps=eps)
        self.pwconv1 = nn.Linear(dim, intermediate_dim, bias=bias)
        self.grn = GRN(intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        # Depthwise conv: (B, L, C) with groups=C
        x = mx.conv1d(
            x,
            self.dwconv_weight,
            stride=1,
            padding=self.dw_padding,
            dilation=self.dw_dilation,
            groups=self.channels,
        )
        x = x + self.dwconv_bias
        x = self.norm(x)
        x = nn.silu(self.pwconv1(x))
        x = self.grn(x)
        x = self.pwconv2(x)
        return residual + x


# ---------------------------------------------------------------------------
# Embedders
# ---------------------------------------------------------------------------


class Embedder(nn.Module):
    """Linear -> SiLU -> Linear embedder (for input/text/latent)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = [nn.Linear(in_dim, out_dim), nn.Linear(out_dim, out_dim)]

    def __call__(self, x: mx.array, mask: mx.array = None) -> mx.array:
        if mask is not None:
            x = mx.where(mask[..., None], x, 0.0)
        x = nn.silu(self.proj[0](x))
        x = self.proj[1](x)
        if mask is not None:
            x = mx.where(mask[..., None], x, 0.0)
        return x


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
    def __init__(self, dim: int, freq_dim: int = 256):
        super().__init__()
        self.time_embed = SinusPositionEmbedding(freq_dim)
        self.time_mlp = [nn.Linear(freq_dim, dim), nn.Linear(dim, dim)]

    def __call__(self, timestep: mx.array) -> mx.array:
        x = self.time_embed(timestep)
        x = nn.silu(self.time_mlp[0](x))
        return self.time_mlp[1](x)


# ---------------------------------------------------------------------------
# AdaLN modules
# ---------------------------------------------------------------------------


class AdaLNMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.mlp = [nn.Linear(in_dim, out_dim, bias=bias)]

    def __call__(self, x: mx.array) -> mx.array:
        return self.mlp[0](nn.silu(x))


class AdaLayerNormZeroFinal(nn.Module):
    def __init__(self, dim: int, bias: bool = True, eps: float = 1e-6):
        super().__init__()
        self.linear = nn.Linear(dim, dim * 2, bias=bias)
        self.eps = eps

    def __call__(self, x: mx.array, emb: mx.array) -> mx.array:
        emb = self.linear(nn.silu(emb))
        scale, shift = mx.split(emb, 2, axis=-1)
        x = _layer_norm(x, self.eps)
        if scale.ndim == 2:
            x = x * (1 + scale)[:, None, :] + shift[:, None, :]
        else:
            x = x * (1 + scale) + shift
        return x


def _layer_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    """LayerNorm without affine parameters."""
    x_float = x.astype(mx.float32)
    mean = mx.mean(x_float, axis=-1, keepdims=True)
    var = mx.var(x_float, axis=-1, keepdims=True)
    normed = (x_float - mean) * mx.rsqrt(var + eps)
    return normed.astype(x.dtype)


def _modulate(
    x: mx.array, scale: mx.array, shift: mx.array, eps: float = 1e-6
) -> mx.array:
    """LayerNorm (no affine) + modulate."""
    x = _layer_norm(x, eps)
    if scale.ndim == 2:
        return x * (1 + scale[:, None]) + shift[:, None]
    return x * (1 + scale) + shift


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        bias: bool = True,
        qk_norm: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.heads = heads
        self.inner_dim = dim_head * heads
        self.to_q = nn.Linear(dim, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(dim, self.inner_dim, bias=bias)
        self.to_v = nn.Linear(dim, self.inner_dim, bias=bias)
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(self.inner_dim, eps=eps)
            self.k_norm = RMSNorm(self.inner_dim, eps=eps)
        self.to_out = nn.Linear(self.inner_dim, dim, bias=bias)

    def __call__(self, x: mx.array, mask: mx.array = None, rope=None) -> mx.array:
        B = x.shape[0]
        head_dim = self.inner_dim // self.heads

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = q.reshape(B, -1, self.heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, -1, self.heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, -1, self.heads, head_dim).transpose(0, 2, 1, 3)

        if rope is not None:
            cos, sin = rope
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)

        scale = math.sqrt(head_dim)
        scores = (q @ k.transpose(0, 1, 3, 2)) / scale

        if mask is not None:
            attn_mask = mask[:, None, None, :]  # (B, 1, 1, L)
            scores = mx.where(attn_mask, scores, mx.array(float("-inf")))

        weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
        out = (weights @ v).transpose(0, 2, 1, 3).reshape(B, -1, self.inner_dim)
        return self.to_out(out)


class CrossAttention(nn.Module):
    def __init__(
        self,
        q_dim: int,
        kv_dim: int,
        heads: int,
        dim_head: int,
        bias: bool = True,
        qk_norm: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.heads = heads
        self.inner_dim = dim_head * heads
        self.to_q = nn.Linear(q_dim, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(kv_dim, self.inner_dim, bias=bias)
        self.to_v = nn.Linear(kv_dim, self.inner_dim, bias=bias)
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(self.inner_dim, eps=eps)
            self.k_norm = RMSNorm(self.inner_dim, eps=eps)
        self.to_out = nn.Linear(self.inner_dim, q_dim, bias=bias)

    def __call__(
        self,
        x: mx.array,
        cond: mx.array,
        mask: mx.array = None,
        cond_mask: mx.array = None,
        rope=None,
        cond_rope=None,
    ) -> mx.array:
        B = x.shape[0]
        head_dim = self.inner_dim // self.heads

        q = self.to_q(x)
        k = self.to_k(cond)
        v = self.to_v(cond)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = q.reshape(B, -1, self.heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, -1, self.heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, -1, self.heads, head_dim).transpose(0, 2, 1, 3)

        if rope is not None:
            cos, sin = rope
            q = apply_rotary_emb(q, cos, sin)
        if cond_rope is not None:
            cos, sin = cond_rope
            k = apply_rotary_emb(k, cos, sin)

        scale = math.sqrt(head_dim)
        scores = (q @ k.transpose(0, 1, 3, 2)) / scale

        if cond_mask is not None:
            attn_mask = cond_mask[:, None, None, :]  # (B, 1, 1, cond_L)
            scores = mx.where(attn_mask, scores, mx.array(float("-inf")))

        weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
        out = (weights @ v).transpose(0, 2, 1, 3).reshape(B, -1, self.inner_dim)
        return self.to_out(out)


# ---------------------------------------------------------------------------
# Feed-forward
# ---------------------------------------------------------------------------


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: float = 4.0, bias: bool = True):
        super().__init__()
        inner_dim = int(dim * mult)
        self.ff = [
            nn.Linear(dim, inner_dim, bias=bias),
            nn.Linear(inner_dim, dim, bias=bias),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        return self.ff[1](nn.gelu_approx(self.ff[0](x)))


# ---------------------------------------------------------------------------
# DiT Block
# ---------------------------------------------------------------------------


class DiTBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        dim = config.dit_dim
        heads = config.dit_heads
        dim_head = dim // heads
        bias = config.dit_bias
        eps = config.dit_eps

        self.adaln_type = config.dit_adaln_type
        if config.dit_adaln_type == "local":
            self.adaln_mlp = AdaLNMLP(dim, dim * 6, bias=True)
        elif config.dit_adaln_type == "global":
            self.adaln_scale_shift = mx.zeros((dim * 6,))

        self.self_attn = SelfAttention(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            bias=bias,
            qk_norm=config.dit_qk_norm,
            eps=eps,
        )

        self.use_cross_attn = config.dit_cross_attn
        if config.dit_cross_attn:
            self.cross_attn = CrossAttention(
                q_dim=dim,
                kv_dim=dim,
                heads=heads,
                dim_head=dim_head,
                bias=bias,
                qk_norm=config.dit_qk_norm,
                eps=eps,
            )
            if config.dit_cross_attn_norm:
                self.cross_attn_norm = nn.LayerNorm(dim, eps=eps)
                self.cross_attn_norm_c = nn.LayerNorm(dim, eps=eps)
            else:
                self.cross_attn_norm = None
                self.cross_attn_norm_c = None

        self.ffn = FeedForward(dim=dim, mult=config.dit_ff_mult, bias=bias)

    def __call__(
        self,
        x: mx.array,
        t: mx.array,
        cond: mx.array,
        mask: mx.array = None,
        cond_mask: mx.array = None,
        rope=None,
        cond_rope=None,
        adaln_global_out: mx.array = None,
    ) -> mx.array:
        if self.adaln_type == "local" and adaln_global_out is None:
            if hasattr(self, "_adaln_use_text_cond") and self._adaln_use_text_cond:
                cond_mean = mx.sum(cond, axis=1) / mx.sum(
                    cond_mask.astype(mx.float32), axis=1, keepdims=True
                )
                norm_cond = t + cond_mean
            else:
                norm_cond = t
            adaln_out = self.adaln_mlp(norm_cond)
        else:
            adaln_out = adaln_global_out + self.adaln_scale_shift[None, :]

        gate_sa, scale_sa, shift_sa, gate_ffn, scale_ffn, shift_ffn = mx.split(
            adaln_out, 6, axis=-1
        )

        # Self-attention
        norm = _modulate(x, scale_sa, shift_sa)
        attn_output = self.self_attn(norm, mask=mask, rope=rope)
        if gate_sa.ndim == 2:
            gate_sa = gate_sa[:, None, :]
        x = x + gate_sa * attn_output

        # Cross-attention
        if self.use_cross_attn:
            x_norm = self.cross_attn_norm(x) if self.cross_attn_norm is not None else x
            c_norm = (
                self.cross_attn_norm_c(cond)
                if self.cross_attn_norm_c is not None
                else cond
            )
            cross_out = self.cross_attn(
                x=x_norm,
                cond=c_norm,
                mask=mask,
                cond_mask=cond_mask,
                rope=rope,
                cond_rope=cond_rope,
            )
            x = x + cross_out

        # FFN
        norm = _modulate(x, scale_ffn, shift_ffn)
        ff_output = self.ffn(norm)
        if gate_ffn.ndim == 2:
            gate_ffn = gate_ffn[:, None, :]
        x = x + gate_ffn * ff_output
        return x


# ---------------------------------------------------------------------------
# DiT Transformer
# ---------------------------------------------------------------------------


class AudioDiTTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        dim = config.dit_dim
        latent_dim = config.latent_dim
        text_dim = config.dit_text_dim
        dim_head = dim // config.dit_heads

        self.dim = dim
        self.depth = config.dit_depth
        self.long_skip = config.dit_long_skip
        self.adaln_type = config.dit_adaln_type
        self.adaln_use_text_cond = config.dit_adaln_use_text_cond

        self.time_embed = TimestepEmbedding(dim)
        self.input_embed = Embedder(latent_dim, dim)
        self.text_embed = Embedder(text_dim, dim)
        self.rotary_embed = RotaryEmbedding(dim_head, 2048, base=100000.0)

        self.blocks = [DiTBlock(config) for _ in range(config.dit_depth)]

        self.norm_out = AdaLayerNormZeroFinal(dim, bias=True, eps=config.dit_eps)
        self.proj_out = nn.Linear(dim, latent_dim)

        if config.dit_adaln_type == "global":
            self.adaln_global_mlp = AdaLNMLP(dim, dim * 6, bias=True)

        self.text_conv = config.dit_text_conv
        if config.dit_text_conv:
            self.text_conv_layer = [
                ConvNeXtV2Block(dim, dim * 2, bias=config.dit_bias, eps=config.dit_eps)
                for _ in range(4)
            ]

        self.use_latent_condition = config.dit_use_latent_condition
        if config.dit_use_latent_condition:
            self.latent_embed = Embedder(latent_dim, dim)
            self.latent_cond_embedder = Embedder(dim * 2, dim)

    def __call__(
        self,
        x: mx.array,
        text: mx.array,
        text_len: mx.array,
        time: mx.array,
        mask: mx.array = None,
        cond_mask: mx.array = None,
        return_ith_layer: int = None,
        latent_cond: mx.array = None,
    ) -> dict:
        batch = x.shape[0]
        text_seq_len = text.shape[1]
        if time.ndim == 0:
            time = mx.broadcast_to(time, (batch,))

        t = self.time_embed(time)
        text = self.text_embed(text, cond_mask)

        if self.text_conv:
            for block in self.text_conv_layer:
                text = block(text)
            text = mx.where(cond_mask[..., None], text, 0.0)

        x = self.input_embed(x, mask)

        if self.use_latent_condition and latent_cond is not None:
            latent_cond = self.latent_embed(latent_cond, mask)
            x = self.latent_cond_embedder(mx.concatenate([x, latent_cond], axis=-1))

        if self.long_skip:
            x_clone = mx.array(x)

        seq_len = x.shape[1]
        rope = self.rotary_embed(seq_len)
        cond_rope = self.rotary_embed(text_seq_len)

        if self.adaln_type == "global":
            if self.adaln_use_text_cond:
                text_mean = mx.sum(text, axis=1) / text_len[:, None].astype(text.dtype)
                norm_cond = t + text_mean
            else:
                norm_cond = t
            adaln_mlp_out = self.adaln_global_mlp(norm_cond)
        else:
            adaln_mlp_out = None
            norm_cond = None

        hidden_state = None
        for i, block in enumerate(self.blocks):
            x = block(
                x=x,
                t=t,
                cond=text,
                mask=mask,
                cond_mask=cond_mask,
                rope=rope,
                cond_rope=cond_rope,
                adaln_global_out=adaln_mlp_out,
            )
            if return_ith_layer == i + 1:
                hidden_state = mx.array(x)
                if self.long_skip:
                    x = x + x_clone

        if self.long_skip:
            x = x + x_clone

        x = self.norm_out(x, norm_cond if norm_cond is not None else t)
        output = self.proj_out(x)
        return {"last_hidden_state": output, "hidden_state": hidden_state}
