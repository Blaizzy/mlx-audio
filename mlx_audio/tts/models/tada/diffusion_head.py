import math

import mlx.core as mx
import mlx.nn as nn


def modulate(x: mx.array, shift: mx.array, scale: mx.array) -> mx.array:
    return x * (1 + scale) + shift


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = mx.ones((dim,))

    def _norm(self, x: mx.array) -> mx.array:
        return x * mx.rsqrt(mx.mean(x**2, axis=-1, keepdims=True) + self.eps)

    def __call__(self, x: mx.array) -> mx.array:
        output = self._norm(x.astype(mx.float32)).astype(x.dtype)
        if self.elementwise_affine:
            output = output * self.weight
        return output


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=False),
        )

    @staticmethod
    def timestep_embedding(t: mx.array, dim: int, max_period: int = 10000) -> mx.array:
        half = dim // 2
        freqs = mx.exp(
            -math.log(max_period) * mx.arange(0, half, dtype=mx.float32) / half
        )
        args = t[:, None].astype(mx.float32) * freqs[None, :]
        embedding = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
        if dim % 2:
            embedding = mx.concatenate(
                [embedding, mx.zeros_like(embedding[:, :1])], axis=-1
            )
        return embedding

    def __call__(self, t: mx.array) -> mx.array:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(embed_dim, ffn_dim, bias=False)
        self.up_proj = nn.Linear(embed_dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, embed_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class HeadLayer(nn.Module):
    def __init__(
        self, embed_dim: int, ffn_dim: int, cond_dim: int, norm_eps: float = 1e-5
    ):
        super().__init__()
        self.ffn = FeedForwardNetwork(embed_dim, ffn_dim)
        self.norm = RMSNorm(embed_dim, eps=norm_eps)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 3 * embed_dim, bias=False),
        )

    def __call__(self, x: mx.array, c: mx.array) -> mx.array:
        modulation = self.adaLN_modulation(c)
        shift_ffn, scale_ffn, gate_ffn = mx.split(modulation, 3, axis=-1)
        x = x + gate_ffn * self.ffn(modulate(self.norm(x), shift_ffn, scale_ffn))
        return x


class FinalLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        cond_size: int,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size, eps=norm_eps, elementwise_affine=False)
        self.linear = nn.Linear(hidden_size, output_size, bias=False)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_size, 2 * hidden_size, bias=False),
        )

    def __call__(self, x: mx.array, c: mx.array) -> mx.array:
        modulation = self.adaLN_modulation(c)
        shift, scale = mx.split(modulation, 2, axis=-1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class DiffusionHead(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        latent_size: int,
        head_layers: int,
        head_ffn_ratio: float,
        rms_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.cond_dim = hidden_size

        self.noisy_images_proj = nn.Linear(latent_size, hidden_size, bias=False)
        self.cond_proj = nn.Linear(hidden_size, self.cond_dim, bias=False)
        self.t_embedder = TimestepEmbedder(self.cond_dim)

        ffn_dim = int(hidden_size * head_ffn_ratio)

        self.layers = [
            HeadLayer(
                embed_dim=hidden_size,
                ffn_dim=ffn_dim,
                cond_dim=self.cond_dim,
                norm_eps=rms_norm_eps,
            )
            for _ in range(head_layers)
        ]

        self.final_layer = FinalLayer(
            hidden_size=hidden_size,
            output_size=latent_size,
            cond_size=self.cond_dim,
            norm_eps=rms_norm_eps,
        )

    def __call__(
        self,
        noisy_images: mx.array,
        timesteps: mx.array,
        condition: mx.array,
    ) -> mx.array:
        x = self.noisy_images_proj(noisy_images)
        t = self.t_embedder(timesteps)
        condition = self.cond_proj(condition)
        c = condition + t

        for layer in self.layers:
            x = layer(x, c)

        return self.final_layer(x, c)
