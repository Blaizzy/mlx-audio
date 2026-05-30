import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .decoder_dit import DiT


class CausalConditionalCFM(nn.Module):
    def __init__(self, estimator: DiT, inference_cfg_rate: float = 0.7):
        super().__init__()
        self.estimator = estimator
        self.inference_cfg_rate = inference_cfg_rate
        self.out_channels = estimator.out_channels
        self.rand_noise = mx.random.normal((1, self.out_channels, 50 * 600))

    def solve_euler(
        self,
        x: mx.array,
        t_span: mx.array,
        mu: mx.array,
        mask: mx.array,
        spks: mx.array,
        cond: mx.array,
    ) -> mx.array:
        t = mx.expand_dims(t_span[0], 0)
        dt = t_span[1] - t_span[0]

        mask_in = mx.concatenate([mask, mask], axis=0)
        mu_in = mx.concatenate([mu, mx.zeros_like(mu)], axis=0)
        spks_in = mx.concatenate([spks, mx.zeros_like(spks)], axis=0)
        cond_in = mx.concatenate([cond, mx.zeros_like(cond)], axis=0)

        for step in range(1, len(t_span)):
            x_in = mx.concatenate([x, x], axis=0)
            t_in = mx.concatenate([t, t], axis=0)
            dphi_dt = self.estimator(x_in, mask_in, mu_in, t_in, spks_in, cond_in)
            dphi_dt, cfg_dphi_dt = mx.split(dphi_dt, 2, axis=0)
            dphi_dt = (
                1.0 + self.inference_cfg_rate
            ) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt
            x = x + dt * dphi_dt
            t = t + dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t
        return x

    def __call__(
        self,
        mu: mx.array,
        mask: mx.array,
        spks: mx.array,
        cond: mx.array,
        n_timesteps: int = 10,
        temperature: float = 1.0,
        noise: Optional[mx.array] = None,
    ) -> mx.array:
        if noise is None:
            noise = self.rand_noise[:, :, : mu.shape[2]]
        z = noise * temperature
        t_span = mx.linspace(0, 1, n_timesteps + 1, dtype=mu.dtype)
        t_span = 1 - mx.cos(t_span * 0.5 * math.pi)
        return self.solve_euler(z, t_span, mu, mask, spks, cond)
