import math

import mlx.core as mx
import mlx.nn as nn


def apply_rope(
    q: mx.array,
    k: mx.array,
    offset: int = 0,
    max_period: int | float = 10000,
):
    b, t, h, d = q.shape
    if d % 2 != 0:
        raise ValueError("RoPE requires an even head dimension.")
    half = d // 2
    freqs = mx.exp(mx.arange(half, dtype=mx.float32) * (-math.log(max_period) * 2 / d))
    ts = mx.arange(t, dtype=mx.float32) + float(offset)
    ts = ts[None, :, None, None]
    q = q.reshape(b, t, h, half, 2)
    k = k.reshape(b, t, h, half, 2)

    qr = q[..., 0].astype(mx.float32)
    qi = q[..., 1].astype(mx.float32)
    kr = k[..., 0].astype(mx.float32)
    ki = k[..., 1].astype(mx.float32)

    freqs = freqs[None, None, None, :]
    rotr = mx.cos(freqs * ts)
    roti = mx.sin(freqs * ts)
    qor = qr * rotr - qi * roti
    qoi = qr * roti + qi * rotr
    kor = kr * rotr - ki * roti
    koi = kr * roti + ki * rotr

    dtype = q.dtype
    q_out = mx.stack([qor.astype(dtype), qoi.astype(dtype)], axis=-1)
    k_out = mx.stack([kor.astype(dtype), koi.astype(dtype)], axis=-1)
    return q_out.reshape(b, t, h, d), k_out.reshape(b, t, h, d)


class RotaryEmbedding(nn.Module):
    def __init__(self, max_period: int | float = 10000.0):
        super().__init__()
        self.max_period = max_period

    def __call__(self, q: mx.array, k: mx.array, offset: int):
        return apply_rope(q, k, offset, self.max_period)
