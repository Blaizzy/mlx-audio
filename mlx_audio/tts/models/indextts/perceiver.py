from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.tts.models.indextts.attention import MultiHeadAttention


# gated gelu feedforward
class FeedForward(nn.Module):
    def __init__(self, dim: int, d_ff: int, use_bias: bool = True):
        super().__init__()
        self.w_1 = nn.Linear(dim, d_ff * 2, bias=use_bias)
        self.activation = nn.GELU()
        self.w_2 = nn.Linear(d_ff, dim, bias=use_bias)

    def __call__(self, x: mx.array) -> mx.array:
        x, gate = mx.split(self.w_1(x), 2, axis=-1)
        return self.w_2(self.activation(gate) * x)


class PerceiverResampler(nn.Module):
    def __init__(
        self,
        n_dim: int,
        n_depth=2,
        n_dim_context: Optional[int] = None,
        n_latents=32,
        n_dim_head=64,
        n_heads=8,
        n_ff_mult=4,
    ):
        super().__init__()

        n_dim_context = n_dim if n_dim_context is None else n_dim_context

        self.proj_context = (
            nn.Linear(n_dim_context, n_dim) if n_dim_context != n_dim else nn.Identity()
        )
        self.latents = mx.zeros((n_latents, n_dim))
        self.layers = [
            [
                MultiHeadAttention(n_heads, n_dim, False, n_dim_head),
                FeedForward(n_dim, (n_dim * n_ff_mult * 2) // 3),
            ]
            for _ in range(n_depth)
        ]
        self.norm = nn.RMSNorm(n_dim)

    def __call__(self, x, mask=None):
        B = x.shape[0]

        latents = mx.broadcast_to(self.latents, (B, *self.latents.shape))

        x = self.proj_context(x)

        for attn, ff in self.layers:
            kv = mx.concat([x, latents], axis=-2)
            latents += attn(latents, kv, kv, mask=mask)
            latents += ff(latents)

        return self.norm(latents)
