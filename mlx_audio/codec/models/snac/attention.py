import mlx.core as mx
import mlx.nn as nn
from einops.array_api import rearrange


class LocalMHA(nn.Module):
    def __init__(self, dim=1024, window_size=32, dim_head=64, use_rotary_pos_emb=True):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.heads = dim // dim_head
        self.window_size = window_size
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        if use_rotary_pos_emb:
            self.rel_pos = SinusoidalEmbeddings(dim_head, scale_base=window_size // 2)
        else:
            self.rel_pos = None
        self.to_out = nn.Linear(dim, dim, bias=False)

    def __call__(self, x):
        B, C, T = x.shape
        residual = x
        x = self.norm(x.moveaxis(1, 2))
        windows = T // self.window_size

        qkv = self.to_qkv(x)
        q, k, v = mx.split(qkv, 3, axis=-1)

        q = rearrange(q, "b (w n) (h d) -> b h w n d", w=windows, h=self.heads)
        k = rearrange(k, "b (w n) (h d) -> b h w n d", w=windows, h=self.heads)
        v = rearrange(v, "b (w n) (h d) -> b h w n d", w=windows, h=self.heads)

        if self.rel_pos is not None:
            pos_emb, scale = self.rel_pos(k)
            q, k = apply_rotary_pos_emb(q, k, pos_emb, scale)

        scale = mx.sqrt(mx.array(q.shape[-1], dtype=mx.float32))
        scores = mx.matmul(q, k.transpose(0, 1, 2, 4, 3)) / scale
        attn_weights = mx.softmax(scores, axis=-1)
        out = mx.matmul(attn_weights, v)

        out = rearrange(out, "b h w n d -> b (w n) (h d)")
        out = self.to_out(out)
        return out.moveaxis(1, 2) + residual


class SinusoidalEmbeddings(nn.Module):
    def __init__(self, dim, scale_base=None, use_xpos=False):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
        self.inv_freq = inv_freq

        # xpos related
        self.use_xpos = use_xpos
        self.scale_base = scale_base
        assert not (
            use_xpos and scale_base is None
        ), "scale base must be defined if using xpos"

        if use_xpos:
            scale = (mx.arange(0, dim, 2, dtype=mx.float32) + 0.4 * dim) / (1.4 * dim)
            self.scale = scale

    def __call__(self, x):
        seq_len = x.shape[-2]
        t = mx.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = mx.einsum("i,j->ij", t, self.inv_freq)
        freqs = mx.concatenate([freqs, freqs], axis=-1)

        if not self.use_xpos:
            return freqs, mx.ones((1,))

        power = (t - (seq_len // 2)) / self.scale_base
        power_reshaped = rearrange(power, "n -> n 1")
        scale = mx.power(self.scale, power_reshaped)
        scale = mx.concatenate([scale, scale], axis=-1)

        return freqs, scale


def rotate_half(x):
    x = rearrange(x, "b ... (r d) -> b ... r d", r=2)
    x1, x2 = mx.split(x, 2, axis=-2)
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(q, k, freqs, scale=1):
    q_len = q.shape[-2]
    q_freqs = freqs[..., -q_len:, :]
    inv_scale = mx.power(scale, -1)

    if scale.ndim == 2:
        scale = scale[-q_len:, :]

    q = (q * mx.cos(q_freqs) * scale) + (rotate_half(q) * mx.sin(q_freqs) * scale)
    k = (k * mx.cos(freqs) * inv_scale) + (rotate_half(k) * mx.sin(freqs) * inv_scale)

    return q, k
