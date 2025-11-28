# Ported from https://github.com/shivammehta25/Matcha-TTS
# Original: matcha/models/components/flow_matching.py

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass
class CFMParams:
    """Configuration for Conditional Flow Matching."""

    sigma_min: float = 1e-06
    solver: str = "euler"
    t_scheduler: str = "cosine"
    training_cfg_rate: float = 0.2
    inference_cfg_rate: float = 0.7
    reg_loss_type: str = "l1"


class BASECFM(nn.Module):
    """Base Conditional Flow Matching module."""

    def __init__(
        self,
        n_feats: int,
        cfm_params: CFMParams,
        n_spks: int = 1,
        spk_emb_dim: int = 128,
    ):
        super().__init__()
        self.n_feats = n_feats
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.solver = cfm_params.solver
        self.sigma_min = cfm_params.sigma_min
        self.estimator = None

    def __call__(
        self,
        mu: mx.array,
        mask: mx.array,
        n_timesteps: int,
        temperature: float = 1.0,
        spks: mx.array = None,
        cond: mx.array = None,
    ) -> mx.array:
        """
        Forward diffusion.

        Args:
            mu: Encoder output (B, n_feats, T)
            mask: Output mask (B, 1, T)
            n_timesteps: Number of diffusion steps
            temperature: Temperature for scaling noise
            spks: Speaker embeddings (B, spk_emb_dim)
            cond: Optional conditioning

        Returns:
            Generated mel-spectrogram (B, n_feats, T)
        """
        z = mx.random.normal(mu.shape) * temperature
        t_span = mx.linspace(0, 1, n_timesteps + 1)
        return self.solve_euler(
            z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond
        )

    def solve_euler(
        self,
        x: mx.array,
        t_span: mx.array,
        mu: mx.array,
        mask: mx.array,
        spks: mx.array,
        cond: mx.array,
    ) -> mx.array:
        """
        Fixed Euler solver for ODEs.

        Args:
            x: Random noise (B, n_feats, T)
            t_span: Time steps (n_timesteps + 1,)
            mu: Encoder output (B, n_feats, T)
            mask: Output mask (B, 1, T)
            spks: Speaker embeddings (B, spk_emb_dim)
            cond: Optional conditioning

        Returns:
            Solution at final timestep
        """
        t = t_span[0]
        dt = t_span[1] - t_span[0]

        sol = []
        for step in range(1, len(t_span)):
            dphi_dt = self.estimator(x, mask, mu, t, spks, cond)
            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)

            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        return sol[-1]
