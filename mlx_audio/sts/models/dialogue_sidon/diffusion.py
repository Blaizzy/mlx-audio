"""AdaLN diffusion transformer and second-order DPM-Solver++ sampler."""

import math

import mlx.core as mx
import numpy as np
from mlx import nn

from .config import DiffusionConfig


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=False),
        )

    def __call__(self, timesteps):
        half = self.frequency_embedding_size // 2
        freqs = mx.exp(-math.log(10000) * mx.arange(half, dtype=mx.float32) / half)
        args = timesteps.astype(mx.float32)[:, None] * freqs[None, :]
        embedding = mx.concatenate((mx.cos(args), mx.sin(args)), axis=-1)
        if self.frequency_embedding_size % 2:
            embedding = mx.pad(embedding, ((0, 0), (0, 1)))
        return self.mlp(embedding.astype(self.mlp.layers[0].weight.dtype))


class DiTBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = dim // config.num_heads
        self.norm1 = nn.LayerNorm(dim, eps=1e-6, affine=False)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6, affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * config.ffn_ratio)),
            nn.GELU(approx="none"),
            nn.Linear(int(dim * config.ffn_ratio), dim),
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

    def __call__(self, x, conditioning):
        sa, ca, ga, sm, cm, gm = mx.split(
            self.adaLN_modulation(conditioning), 6, axis=-1
        )
        h = self.norm1(x) * (1 + ca) + sa
        batch, length, dim = h.shape
        q, k, v = [
            projection(h)
            .reshape(batch, length, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
            for projection in (self.q_proj, self.k_proj, self.v_proj)
        ]
        q = mx.fast.rope(
            q, dims=self.head_dim, traditional=False, base=10000, scale=1.0, offset=0
        )
        k = mx.fast.rope(
            k, dims=self.head_dim, traditional=False, base=10000, scale=1.0, offset=0
        )
        h = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.head_dim**-0.5)
        h = h.transpose(0, 2, 1, 3).reshape(batch, length, dim)
        x = x + ga * self.out_proj(h)
        return x + gm * self.mlp(self.norm2(x) * (1 + cm) + sm)


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, eps=1e-6, affine=False)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, hidden_size * 2)
        )
        self.linear = nn.Linear(hidden_size, output_size, bias=False)

    def __call__(self, x, conditioning):
        shift, scale = mx.split(self.adaLN_modulation(conditioning), 2, axis=-1)
        return self.linear(self.norm_final(x) * (1 + scale) + shift)


class DiffusionHead(nn.Module):
    def __init__(self, config: DiffusionConfig, latent_size: int, cond_size: int):
        super().__init__()
        self.t_embedder = TimestepEmbedder(
            config.hidden_size, config.frequency_embedding_size
        )
        self.latent_proj = nn.Linear(latent_size, config.hidden_size, bias=False)
        self.cond_proj = nn.Linear(cond_size, config.hidden_size, bias=False)
        self.blocks = [DiTBlock(config) for _ in range(config.num_layers)]
        self.final_layer = FinalLayer(config.hidden_size, latent_size)

    def __call__(self, noisy_latents, timesteps, conditioning):
        x = self.latent_proj(noisy_latents)
        c = self.cond_proj(conditioning) + self.t_embedder(timesteps)[:, None, :]
        for block in self.blocks:
            x = block(x, c)
        return self.final_layer(x, c)


class DPMSolver:
    """The reference's linear-beta, linspace, midpoint DPM-Solver++ path.

    Equivalent to Diffusers DPMSolverMultistepScheduler with solver_order=2,
    algorithm_type='dpmsolver++', final_sigmas_type='zero', thresholding=False.
    Sampler state is per call, so requests/chunks cannot share solver history.
    """

    def __init__(self, config: DiffusionConfig, num_steps: int):
        if (
            not isinstance(num_steps, int)
            or not 1 <= num_steps <= config.num_train_timesteps
        ):
            raise ValueError(
                f"num_steps must be between 1 and {config.num_train_timesteps}"
            )
        self.prediction_type = config.prediction_type
        betas = np.linspace(
            config.beta_start,
            config.beta_end,
            config.num_train_timesteps,
            dtype=np.float32,
        )
        cumulative = np.cumprod(1 - betas, dtype=np.float64).astype(np.float32)
        self.timesteps = (
            np.linspace(0, config.num_train_timesteps - 1, num_steps + 1)
            .round()
            .astype(np.int32)[::-1][:-1]
            .copy()
        )
        self.alpha = np.sqrt(cumulative[self.timesteps]).astype(np.float64).tolist() + [
            1.0
        ]
        self.sigma = np.sqrt(1 - cumulative[self.timesteps]).astype(
            np.float64
        ).tolist() + [0.0]
        self.lambdas = [
            math.log(a) - math.log(s) if s else math.inf
            for a, s in zip(self.alpha, self.sigma)
        ]
        self.previous = None
        self.index = 0

    def step(self, model_output, sample):
        i = self.index
        if i >= len(self.timesteps):
            raise ValueError("All diffusion steps have already been taken")
        if self.prediction_type == "v_prediction":
            x0 = self.alpha[i] * sample - self.sigma[i] * model_output
        elif self.prediction_type == "epsilon":
            x0 = (sample - self.sigma[i] * model_output) / self.alpha[i]
        else:
            raise ValueError(f"Unsupported prediction type: {self.prediction_type}")
        h = self.lambdas[i + 1] - self.lambdas[i]
        coefficient = self.alpha[i + 1] * math.expm1(-h)
        result = (self.sigma[i + 1] / self.sigma[i]) * sample - coefficient * x0
        if self.previous is not None and i < len(self.timesteps) - 1:
            r = (self.lambdas[i] - self.lambdas[i - 1]) / h
            result = result - 0.5 * coefficient * (x0 - self.previous) / r
        self.previous = x0
        self.index += 1
        return result
