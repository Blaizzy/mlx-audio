"""Ming Tiny's native FP32 Euler sampler over the shared Ming DiT."""

import math

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.tts.models.bailingmm.bailingmm import DiT, get_epss_timesteps


class CFM(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model


class FlowLoss(nn.Module):
    def __init__(self, latent_dim, hidden_size, config):
        super().__init__()
        settings = {
            key: value
            for key, value in config.items()
            if key not in {"z_channels", "llm_cond_dim", "in_channels"}
        }
        self.cfm = CFM(
            DiT(in_channels=latent_dim, llm_cond_dim=hidden_size, **settings)
        )
        self.latent_dim = latent_dim

    def sample(
        self,
        hidden,
        history,
        *,
        cfg=2.0,
        patch_size=1,
        sigma=0.25,
        temperature=0.0,
        steps=10,
    ):
        if cfg < 0 or temperature < 0 or steps < 1 or sigma < 0:
            raise ValueError(
                "Flow CFG, sigma and temperature must be nonnegative and steps positive"
            )
        state = mx.random.normal(
            (hidden.shape[0], self.latent_dim, patch_size), dtype=mx.float32
        ).transpose(0, 2, 1)
        times = get_epss_timesteps(steps, mx.float32)
        times = times - (mx.cos(math.pi / 2 * times) - 1 + times)
        dtype = self.cfm.model.x_embedder.weight.dtype
        for index in range(steps):
            current = state.astype(dtype)
            cond = hidden.astype(dtype)
            past = history.astype(dtype)
            time = mx.full((current.shape[0],), times[index], dtype=dtype)
            if cfg == 1 or cfg < 1e-5:
                prediction = self.cfm.model(
                    current, time, cond if cfg == 1 else mx.zeros_like(cond), past
                )[:, -patch_size:]
            else:
                outputs = self.cfm.model.forward_with_cfg(
                    current, time, cond, cfg, past, patch_size
                )
                prediction, unconditional = mx.split(outputs, 2, axis=0)
                prediction = prediction + (prediction - unconditional) * cfg
            dt = times[index + 1] - times[index]
            state = state + dt * prediction.astype(mx.float32)
            # The upstream solver returns the final Euler value before the last noise draw.
            noise = mx.random.normal(state.shape, dtype=mx.float32)
            if index < steps - 1:
                state = (
                    state + sigma * math.sqrt(temperature) * mx.sqrt(mx.abs(dt)) * noise
                )
        return state
