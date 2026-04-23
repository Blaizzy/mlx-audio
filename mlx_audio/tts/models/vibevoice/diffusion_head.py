# Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

import math
import time
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
        self._fused_gate_up_weight = None
        self._fused_gate_weight_ref = None
        self._fused_up_weight_ref = None

    def _get_fused_gate_up_weight(self) -> Optional[mx.array]:
        if type(self.gate_proj) is not nn.Linear or type(self.up_proj) is not nn.Linear:
            return None
        gate_weight = self.gate_proj.weight
        up_weight = self.up_proj.weight
        if (
            self._fused_gate_up_weight is None
            or self._fused_gate_weight_ref is not gate_weight
            or self._fused_up_weight_ref is not up_weight
        ):
            self._fused_gate_up_weight = mx.concatenate(
                [gate_weight, up_weight], axis=0
            )
            self._fused_gate_weight_ref = gate_weight
            self._fused_up_weight_ref = up_weight
        return self._fused_gate_up_weight

    def __call__(
        self,
        x: mx.array,
        timing_info: Optional[dict[str, float]] = None,
    ) -> mx.array:
        sync_timing = bool(timing_info is not None and timing_info.get("__sync__", False))
        gate_up_t0 = time.perf_counter() if timing_info is not None else None
        fused_weight = self._get_fused_gate_up_weight()
        if fused_weight is None:
            gate = self.gate_proj(x)
            up = self.up_proj(x)
        else:
            gate_up = mx.matmul(x, fused_weight.transpose())
            gate, up = mx.split(gate_up, [self.gate_proj.weight.shape[0]], axis=-1)
        if sync_timing:
            mx.eval(gate, up)
        if timing_info is not None:
            timing_info["ffn_gate_up"] = timing_info.get("ffn_gate_up", 0.0) + (
                time.perf_counter() - gate_up_t0
            )
        act_t0 = time.perf_counter() if timing_info is not None else None
        mixed = nn.silu(gate) * up
        if sync_timing:
            mx.eval(mixed)
        if timing_info is not None:
            timing_info["ffn_act"] = timing_info.get("ffn_act", 0.0) + (
                time.perf_counter() - act_t0
            )
        down_t0 = time.perf_counter() if timing_info is not None else None
        out = self.down_proj(mixed)
        if sync_timing:
            mx.eval(out)
        if timing_info is not None:
            timing_info["ffn_down"] = timing_info.get("ffn_down", 0.0) + (
                time.perf_counter() - down_t0
            )
        return out


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

    def __call__(
        self,
        x: mx.array,
        c: mx.array,
        timing_info: Optional[dict[str, float]] = None,
    ) -> mx.array:
        sync_timing = bool(timing_info is not None and timing_info.get("__sync__", False))
        # Get modulation parameters
        mod_t0 = time.perf_counter() if timing_info is not None else None
        modulation = self.adaLN_modulation(c)
        if sync_timing:
            mx.eval(modulation)
        if timing_info is not None:
            timing_info["adaln"] = timing_info.get("adaln", 0.0) + (
                time.perf_counter() - mod_t0
            )
        shift_ffn, scale_ffn, gate_ffn = mx.split(modulation, 3, axis=-1)

        # Apply modulated FFN
        ffn_out = self.ffn(
            modulate(self.norm(x), shift_ffn, scale_ffn),
            timing_info=timing_info,
        )
        x = x + gate_ffn * ffn_out
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

    def __call__(
        self,
        x: mx.array,
        c: mx.array,
        timing_info: Optional[dict[str, float]] = None,
    ) -> mx.array:
        sync_timing = bool(timing_info is not None and timing_info.get("__sync__", False))
        mod_t0 = time.perf_counter() if timing_info is not None else None
        modulation = self.adaLN_modulation(c)
        if sync_timing:
            mx.eval(modulation)
        if timing_info is not None:
            timing_info["final_adaln"] = timing_info.get("final_adaln", 0.0) + (
                time.perf_counter() - mod_t0
            )
        shift, scale = mx.split(modulation, 2, axis=-1)
        x = modulate(self.norm_final(x), shift, scale)
        final_t0 = time.perf_counter() if timing_info is not None else None
        x = self.linear(x)
        if sync_timing:
            mx.eval(x)
        if timing_info is not None:
            timing_info["final_linear"] = timing_info.get("final_linear", 0.0) + (
                time.perf_counter() - final_t0
            )
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
        c = self.condition_with_timestep(condition, timesteps)

        for layer in self.layers:
            x = layer(x, c)

        x = self.final_layer(x, c)
        return x

    def project_condition(self, condition: mx.array) -> mx.array:
        """Project conditioning once for reuse across diffusion steps."""
        return self.cond_proj(condition)

    def embed_timesteps(self, timesteps: mx.array) -> mx.array:
        """Embed timesteps for reuse across diffusion steps."""
        return self.t_embedder(timesteps)

    def condition_with_timestep(
        self,
        condition: mx.array,
        timesteps: mx.array,
    ) -> mx.array:
        """Combine raw condition with timestep embedding."""
        return self.project_condition(condition) + self.embed_timesteps(timesteps)

    def forward_with_condition(
        self,
        noisy_images: mx.array,
        condition_with_timestep: mx.array,
        timing_info: Optional[dict[str, float]] = None,
    ) -> mx.array:
        """Forward pass using precomputed condition+timestep embedding."""
        sync_timing = bool(timing_info is not None and timing_info.get("__sync__", False))
        proj_t0 = time.perf_counter() if timing_info is not None else None
        x = self.noisy_images_proj(noisy_images)
        if sync_timing:
            mx.eval(x)
        if timing_info is not None:
            timing_info["noisy_proj"] = timing_info.get("noisy_proj", 0.0) + (
                time.perf_counter() - proj_t0
            )
        c = condition_with_timestep

        for layer in self.layers:
            x = layer(x, c, timing_info=timing_info)

        x = self.final_layer(x, c, timing_info=timing_info)
        return x
