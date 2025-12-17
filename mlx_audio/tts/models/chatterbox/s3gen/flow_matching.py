import math

import mlx.core as mx
import mlx.nn as nn

from .matcha.flow_matching import BASECFM, CFMParams

CFM_PARAMS = CFMParams()


class ConditionalCFM(BASECFM):

    def __init__(
        self,
        in_channels: int,
        cfm_params: CFMParams,
        n_spks: int = 1,
        spk_emb_dim: int = 64,
        estimator: nn.Module = None,
    ):
        super().__init__(
            n_feats=in_channels,
            cfm_params=cfm_params,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
        )
        self.t_scheduler = cfm_params.t_scheduler
        self.training_cfg_rate = cfm_params.training_cfg_rate
        self.inference_cfg_rate = cfm_params.inference_cfg_rate
        self.estimator = estimator

    def __call__(
        self,
        mu: mx.array,
        mask: mx.array,
        n_timesteps: int,
        temperature: float = 1.0,
        spks: mx.array = None,
        cond: mx.array = None,
        prompt_len: int = 0,
        flow_cache: mx.array = None,
    ) -> tuple:

        if flow_cache is None:
            flow_cache = mx.zeros((1, self.n_feats, 0, 2))

        z = mx.random.normal(mu.shape) * temperature
        cache_size = flow_cache.shape[2]

        if cache_size != 0:
            z = mx.concatenate([flow_cache[:, :, :, 0], z[:, :, cache_size:]], axis=2)
            mu = mx.concatenate([flow_cache[:, :, :, 1], mu[:, :, cache_size:]], axis=2)

        z_cache = mx.concatenate([z[:, :, :prompt_len], z[:, :, -34:]], axis=2)
        mu_cache = mx.concatenate([mu[:, :, :prompt_len], mu[:, :, -34:]], axis=2)
        flow_cache = mx.stack([z_cache, mu_cache], axis=-1)

        # Time span
        t_span = mx.linspace(0, 1, n_timesteps + 1)
        if self.t_scheduler == "cosine":
            t_span = 1 - mx.cos(t_span * 0.5 * math.pi)

        result = self.solve_euler(
            z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond
        )
        return result, flow_cache

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
        Euler solver with Classifier-Free Guidance.
        """
        t = mx.expand_dims(t_span[0], 0)
        dt = t_span[1] - t_span[0]

        sol = []

        # Prepare batch for CFG (defaults for when spks/cond are None)
        T_len = x.shape[2]
        spks_in = mx.zeros((2, self.spk_emb_dim))
        cond_in = mx.zeros((2, self.n_feats, T_len))

        for step in range(1, len(t_span)):
            # Prepare inputs for CFG
            x_in = mx.concatenate([x, x], axis=0)
            mask_in = mx.concatenate([mask, mask], axis=0)
            mu_in = mx.concatenate([mu, mx.zeros_like(mu)], axis=0)
            t_in = mx.concatenate([t, t], axis=0)
            if spks is not None:
                spks_in = mx.concatenate([spks, mx.zeros_like(spks)], axis=0)
            if cond is not None:
                cond_in = mx.concatenate([cond, mx.zeros_like(cond)], axis=0)

            # Forward estimator
            dphi_dt = self.estimator(x_in, mask_in, mu_in, t_in, spks_in, cond_in)

            # Split and apply CFG
            dphi_dt_cond = dphi_dt[: x.shape[0]]
            dphi_dt_uncond = dphi_dt[x.shape[0] :]
            dphi_dt = (
                1.0 + self.inference_cfg_rate
            ) * dphi_dt_cond - self.inference_cfg_rate * dphi_dt_uncond

            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)

            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        return sol[-1]


class CausalConditionalCFM(ConditionalCFM):

    MEL_CHANNELS = 80  # Important: must match PyTorch's hardcoded value

    def __init__(
        self,
        in_channels: int = 240,
        cfm_params: CFMParams = CFM_PARAMS,
        n_spks: int = 1,
        spk_emb_dim: int = 80,
        estimator: nn.Module = None,
    ):
        super().__init__(in_channels, cfm_params, n_spks, spk_emb_dim, estimator)

        mx.random.seed(0)  # Match PyTorch's deterministic seed
        self.rand_noise = mx.random.normal((1, self.MEL_CHANNELS, 50 * 300))

    def __call__(
        self,
        mu: mx.array,
        mask: mx.array,
        n_timesteps: int,
        temperature: float = 1.0,
        spks: mx.array = None,
        cond: mx.array = None,
        streaming: bool = False,
    ) -> tuple:

        T = mu.shape[2]
        z = self.rand_noise[:, :, :T] * temperature

        # Time span
        t_span = mx.linspace(0, 1, n_timesteps + 1)
        if self.t_scheduler == "cosine":
            t_span = 1 - mx.cos(t_span * 0.5 * math.pi)

        result = self.solve_euler(
            z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond
        )
        return result, None
