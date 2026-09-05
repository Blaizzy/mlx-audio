"""Causal conditional flow matching (CFM) for CosyVoice3.

Uses a fixed-step Euler ODE solver over a rectified flow, with classifier-free
guidance and a cosine timestep schedule.

The estimator contract is:
    estimator(x, mask, mu, t, spks, cond) -> velocity

where all inputs are channels-first ``[B, mel, T]`` except ``spks``
(``[B, spk_dim]``) and ``t`` (``[B]``). During classifier-free guidance, the
conditional and unconditional inputs are batched together before one estimator
call, then combined as ``v = (1 + cfg) * v_cond - cfg * v_uncond``.
"""

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .dit import DiT


class CausalConditionalCFM(nn.Module):
    def __init__(
        self,
        estimator: DiT,
        inference_cfg_rate: float = 0.7,
        t_scheduler: str = "cosine",
        max_len: int = 50 * 300,
    ):
        super().__init__()
        self.estimator = estimator
        self.inference_cfg_rate = inference_cfg_rate
        self.t_scheduler = t_scheduler
        self.out_channels = estimator.out_channels
        # Keep the initial noise deterministic across model loads.
        self.rand_noise = mx.random.normal(
            (1, self.out_channels, max_len), key=mx.random.key(0)
        )

    def solve_euler(
        self,
        x: mx.array,
        t_span: mx.array,
        mu: mx.array,
        mask: mx.array,
        spks: mx.array,
        cond: mx.array,
        streaming: bool = False,
    ) -> mx.array:
        """Run fixed-step Euler integration with classifier-free guidance."""
        t = mx.expand_dims(t_span[0], 0)
        dt = t_span[1] - t_span[0]

        # CFG: unconditional branch zeroes mu / spks / cond.
        mask_in = mx.concatenate([mask, mask], axis=0)
        mu_in = mx.concatenate([mu, mx.zeros_like(mu)], axis=0)
        spks_in = mx.concatenate([spks, mx.zeros_like(spks)], axis=0)
        cond_in = mx.concatenate([cond, mx.zeros_like(cond)], axis=0)

        for step in range(1, len(t_span)):
            x_in = mx.concatenate([x, x], axis=0)
            t_in = mx.concatenate([t, t], axis=0)
            dphi_dt = self.estimator(
                x_in, mask_in, mu_in, t_in, spks_in, cond_in, streaming=streaming
            )
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
        streaming: bool = False,
    ) -> mx.array:
        """mu/cond: [B, mel, T]; mask: [B, 1, T]; spks: [B, spk_dim].

        Returns generated mel [B, mel, T].
        """
        if noise is None:
            noise = self.rand_noise[:, :, : mu.shape[2]]
        z = noise * temperature
        t_span = mx.linspace(0, 1, n_timesteps + 1, dtype=mu.dtype)
        if self.t_scheduler == "cosine":
            t_span = 1 - mx.cos(t_span * 0.5 * math.pi)
        return self.solve_euler(
            z, t_span, mu, mask, spks, cond, streaming=streaming
        )
