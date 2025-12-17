# Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from tqdm import tqdm


class ConditionalCFM(nn.Module):
    """
    Conditional Flow Matching for speech generation.
    Uses Euler solver for ODE integration.
    """

    def __init__(
        self,
        in_channels: int = 240,
        n_spks: int = 1,
        spk_emb_dim: int = 80,
        sigma_min: float = 1e-6,
        t_scheduler: str = "cosine",
        inference_cfg_rate: float = 0.7,
        estimator: nn.Module = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.sigma_min = sigma_min
        self.t_scheduler = t_scheduler
        self.inference_cfg_rate = inference_cfg_rate
        self.estimator = estimator

    def __call__(
        self,
        mu: mx.array,
        mask: mx.array,
        n_timesteps: int,
        temperature: float = 1.0,
        spks: Optional[mx.array] = None,
        cond: Optional[mx.array] = None,
        noised_mels: Optional[mx.array] = None,
        meanflow: bool = False,
    ) -> Tuple[mx.array, None]:
        """
        Forward diffusion/flow matching.

        Args:
            mu: Encoder output (B, 80, T)
            mask: Output mask (B, 1, T)
            n_timesteps: Number of diffusion steps
            temperature: Temperature for noise scaling
            spks: Speaker embeddings (B, spk_emb_dim)
            cond: Conditioning (B, 80, T)
            noised_mels: Pre-noised mels for continuation
            meanflow: Whether to use meanflow mode

        Returns:
            Generated mel-spectrogram (B, 80, T)
        """
        B = mu.shape[0]

        # Initialize with random noise
        z = mx.random.normal(mu.shape) * temperature

        if noised_mels is not None:
            prompt_len = mu.shape[2] - noised_mels.shape[2]
            z = mx.concatenate([z[:, :, :prompt_len], noised_mels], axis=2)

        # Time steps for reverse diffusion
        t_span = mx.linspace(0, 1, n_timesteps + 1)
        if (not meanflow) and (self.t_scheduler == "cosine"):
            t_span = 1 - mx.cos(t_span * 0.5 * mx.pi)

        if meanflow:
            return self._basic_euler(z, t_span, mu, mask, spks, cond), None

        return self._solve_euler_cfg(z, t_span, mu, mask, spks, cond, meanflow), None

    def _basic_euler(
        self,
        x: mx.array,
        t_span: mx.array,
        mu: mx.array,
        mask: mx.array,
        spks: Optional[mx.array],
        cond: Optional[mx.array],
    ) -> mx.array:
        """Basic Euler solver without CFG (for meanflow distilled models)."""
        print("S3 Token -> Mel Inference...")

        for i in tqdm(range(len(t_span) - 1)):
            t = t_span[i : i + 1]
            r = t_span[i + 1 : i + 2]

            # Predict velocity
            dxdt = self.estimator(
                x=x,
                mask=mask,
                mu=mu,
                t=t,
                spks=spks,
                cond=cond,
                r=r,
            )

            # Euler step
            dt = r - t
            x = x + dt * dxdt

        return x

    def _solve_euler_cfg(
        self,
        x: mx.array,
        t_span: mx.array,
        mu: mx.array,
        mask: mx.array,
        spks: Optional[mx.array],
        cond: Optional[mx.array],
        meanflow: bool,
    ) -> mx.array:
        """Euler solver with classifier-free guidance."""
        B = mu.shape[0]
        T = x.shape[2]

        for i in tqdm(range(len(t_span) - 1), desc="CFM sampling"):
            t = t_span[i : i + 1]
            r = t_span[i + 1 : i + 2]

            # Duplicate for CFG: [cond, uncond]
            x_in = mx.concatenate([x, x], axis=0)
            mask_in = mx.concatenate([mask, mask], axis=0)
            mu_in = mx.concatenate([mu, mx.zeros_like(mu)], axis=0)
            t_in = mx.broadcast_to(t, (2 * B,))
            r_in = mx.broadcast_to(r, (2 * B,)) if meanflow else None

            if spks is not None:
                spks_in = mx.concatenate([spks, mx.zeros_like(spks)], axis=0)
            else:
                spks_in = None

            if cond is not None:
                cond_in = mx.concatenate([cond, mx.zeros_like(cond)], axis=0)
            else:
                cond_in = None

            # Predict velocity
            dxdt = self.estimator(
                x=x_in,
                mask=mask_in,
                mu=mu_in,
                t=t_in,
                spks=spks_in,
                cond=cond_in,
                r=r_in,
            )

            # CFG combination
            dxdt_cond, dxdt_uncond = mx.split(dxdt, 2, axis=0)
            dxdt = (
                1.0 + self.inference_cfg_rate
            ) * dxdt_cond - self.inference_cfg_rate * dxdt_uncond

            # Euler step
            dt = r - t
            x = x + dt * dxdt

        return x


class CausalConditionalCFM(ConditionalCFM):
    """Causal version of Conditional CFM for streaming."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(
        self,
        mu: mx.array,
        mask: mx.array,
        n_timesteps: int,
        temperature: float = 1.0,
        spks: Optional[mx.array] = None,
        cond: Optional[mx.array] = None,
        noised_mels: Optional[mx.array] = None,
        meanflow: bool = False,
    ) -> Tuple[mx.array, None]:
        """Forward with meanflow mode for distilled models."""
        B = mu.shape[0]

        # Initialize with random noise
        z = mx.random.normal(mu.shape)

        if noised_mels is not None:
            prompt_len = mu.shape[2] - noised_mels.shape[2]
            z = mx.concatenate([z[:, :, :prompt_len], noised_mels], axis=2)

        # Time steps
        t_span = mx.linspace(0, 1, n_timesteps + 1)
        if (not meanflow) and (self.t_scheduler == "cosine"):
            t_span = 1 - mx.cos(t_span * 0.5 * mx.pi)

        # Meanflow distilled models don't need CFG
        if meanflow:
            return self._basic_euler(z, t_span, mu, mask, spks, cond), None

        return self._solve_euler_cfg(z, t_span, mu, mask, spks, cond, meanflow), None
