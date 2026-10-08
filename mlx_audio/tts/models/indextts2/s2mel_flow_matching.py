from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx

import mlx.nn as nn

from .s2mel_dit import DiT, DiTConfig


class CFM(nn.Module):
    def __init__(self, dit_cfg: DiTConfig):
        super().__init__()
        self.sigma_min = 1e-6
        self.in_channels = dit_cfg.in_channels
        self.estimator = DiT(dit_cfg)

    def inference(
        self,
        mu: mx.array,
        x_lens: mx.array,
        prompt: mx.array,
        style: mx.array,
        f0: Optional[mx.array],
        n_timesteps: int,
        temperature: float = 1.0,
        inference_cfg_rate: float = 0.7,
    ) -> mx.array:
        # mu: (B, T, 512)
        B, T, _ = mu.shape
        z = mx.random.normal((B, self.in_channels, T)).astype(mx.float32) * float(temperature)
        t_span = mx.linspace(0.0, 1.0, n_timesteps + 1)
        return self.solve_euler(z, x_lens, prompt, mu, style, f0, t_span, inference_cfg_rate)

    def solve_euler(
        self,
        x: mx.array,
        x_lens: mx.array,
        prompt: mx.array,
        mu: mx.array,
        style: mx.array,
        f0: Optional[mx.array],
        t_span: mx.array,
        inference_cfg_rate: float,
    ) -> mx.array:
        del f0
        prompt_len = prompt.shape[-1]

        prompt_x = mx.zeros_like(x)
        prompt_x[..., :prompt_len] = prompt[..., :prompt_len]

        if prompt_len > 0:
            x = mx.concatenate([mx.zeros_like(x[..., :prompt_len]), x[..., prompt_len:]], axis=-1)

        t = t_span[0]
        for step in range(1, t_span.shape[0]):
            dt = t_span[step] - t_span[step - 1]

            if inference_cfg_rate > 0:
                zeros_prompt = mx.zeros_like(prompt_x)
                zeros_style = mx.zeros_like(style)
                zeros_mu = mx.zeros_like(mu)

                stacked_prompt_x = mx.concatenate([prompt_x, zeros_prompt], axis=0)
                stacked_style = mx.concatenate([style, zeros_style], axis=0)
                stacked_mu = mx.concatenate([mu, zeros_mu], axis=0)
                stacked_x = mx.concatenate([x, x], axis=0)
                stacked_t = mx.concatenate([mx.array([t]), mx.array([t])], axis=0)
                stacked_x_lens = mx.concatenate([x_lens, x_lens], axis=0)

                stacked_d = self.estimator(
                    stacked_x,
                    stacked_prompt_x,
                    stacked_x_lens,
                    stacked_t,
                    stacked_style,
                    stacked_mu,
                )
                dphi_dt, cfg_dphi_dt = mx.split(stacked_d, 2, axis=0)
                dphi_dt = (1.0 + inference_cfg_rate) * dphi_dt - inference_cfg_rate * cfg_dphi_dt
            else:
                dphi_dt = self.estimator(x, prompt_x, x_lens, mx.array([t]), style, mu)

            x = x + dt * dphi_dt
            t = t + dt
            if prompt_len > 0:
                x = mx.concatenate([mx.zeros_like(x[..., :prompt_len]), x[..., prompt_len:]], axis=-1)

        return x
