# Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)


import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .config import DiffusionHeadConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = mx.ones((dim,))
        else:
            self.weight = None

    def _norm(self, x: mx.array) -> mx.array:
        return x * mx.rsqrt(mx.mean(x**2, axis=-1, keepdims=True) + self.eps)

    def __call__(self, x: mx.array) -> mx.array:
        output = self._norm(x.astype(mx.float32)).astype(x.dtype)
        if self.weight is not None:
            output = output * self.weight
        return output


def modulate(x: mx.array, shift: mx.array, scale: mx.array) -> mx.array:
    """Apply adaptive layer normalization modulation."""
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

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
        """Create sinusoidal timestep embeddings.

        Args:
            t: 1D tensor of timestep indices
            dim: Embedding dimension
            max_period: Controls minimum frequency

        Returns:
            Positional embeddings of shape (N, dim)
        """
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
        t_emb = self.mlp(t_freq)
        return t_emb


class FeedForwardNetwork(nn.Module):
    """Feed-forward network with SwiGLU activation."""

    def __init__(self, embed_dim: int, ffn_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.gate_proj = nn.Linear(embed_dim, ffn_dim, bias=False)
        self.up_proj = nn.Linear(embed_dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, embed_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        gate = nn.silu(gate)
        return self.down_proj(gate * up)


class HeadLayer(nn.Module):
    """A layer in the diffusion head with adaptive layer norm."""

    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        cond_dim: int,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.cond_dim = cond_dim
        self.ffn_dim = ffn_dim

        self.ffn = FeedForwardNetwork(embed_dim, ffn_dim)
        self.norm = RMSNorm(embed_dim, eps=norm_eps)

        # AdaLN modulation: outputs shift, scale, gate
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 3 * embed_dim, bias=False),
        )

    def __call__(self, x: mx.array, c: mx.array) -> mx.array:
        # Get modulation parameters
        modulation = self.adaLN_modulation(c)
        shift_ffn, scale_ffn, gate_ffn = mx.split(modulation, 3, axis=-1)

        # Apply modulated FFN
        x = x + gate_ffn * self.ffn(modulate(self.norm(x), shift_ffn, scale_ffn))
        return x


class FinalLayer(nn.Module):
    """Final layer in the diffusion head."""

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

        # AdaLN modulation
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_size, 2 * hidden_size, bias=False),
        )

    def __call__(self, x: mx.array, c: mx.array) -> mx.array:
        modulation = self.adaLN_modulation(c)
        shift, scale = mx.split(modulation, 2, axis=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiffusionHead(nn.Module):
    """Diffusion prediction head for VibeVoice.

    This module predicts noise/velocity for the diffusion process.
    """

    def __init__(self, config: DiffusionHeadConfig):
        super().__init__()
        self.config = config
        self.cond_dim = config.hidden_size
        latent_size = config.latent_size

        # Input projections
        self.noisy_images_proj = nn.Linear(latent_size, config.hidden_size, bias=False)
        self.cond_proj = nn.Linear(config.hidden_size, self.cond_dim, bias=False)

        # Timestep embedder
        self.t_embedder = TimestepEmbedder(self.cond_dim)

        # FFN dimension
        ffn_dim = int(config.hidden_size * config.head_ffn_ratio)

        # Intermediate layers
        self.layers = [
            HeadLayer(
                embed_dim=config.hidden_size,
                ffn_dim=ffn_dim,
                cond_dim=self.cond_dim,
                norm_eps=config.rms_norm_eps,
            )
            for _ in range(config.head_layers)
        ]

        # Final layer
        self.final_layer = FinalLayer(
            hidden_size=config.hidden_size,
            output_size=latent_size,
            cond_size=self.cond_dim,
            norm_eps=config.rms_norm_eps,
        )

    def __call__(
        self,
        noisy_images: mx.array,
        timesteps: mx.array,
        condition: mx.array,
    ) -> mx.array:
        """Forward pass of the prediction head.

        Args:
            noisy_images: Noisy latents to denoise, shape (B, latent_size)
            timesteps: Diffusion timesteps, shape (B,)
            condition: Conditioning information, shape (B, hidden_size)

        Returns:
            Predicted noise/velocity, shape (B, latent_size)
        """
        x = self.noisy_images_proj(noisy_images)
        t = self.t_embedder(timesteps)
        condition = self.cond_proj(condition)
        c = condition + t

        for layer in self.layers:
            x = layer(x, c)

        x = self.final_layer(x, c)
        return x
