"""Flow-matching Euler scheduler used by MiniMax Music 3."""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from .config import DIT_CFG_SCALE


def make_sigma_schedule(num_inference_steps: int) -> np.ndarray:
    if num_inference_steps < 1:
        raise ValueError("num_inference_steps must be at least one")
    raw = np.linspace(
        1.0,
        1.0 / num_inference_steps,
        num_inference_steps,
        dtype=np.float32,
    )
    return np.concatenate([1.0 - raw, np.array([1.0], dtype=np.float32)])


def denoise_chunk(
    transformer,
    latents: mx.array,
    condition: mx.array,
    num_inference_steps: int = 30,
    guidance_scale: float = DIT_CFG_SCALE,
    previous_latent: mx.array | None = None,
    previous_condition: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    overlap = 0
    if previous_latent is not None and previous_condition is not None:
        overlap = min(previous_latent.shape[-1], condition.shape[1])
        if overlap:
            condition = mx.concatenate(
                [previous_condition[:, :overlap], condition[:, overlap:]], axis=1
            )

    sigmas = make_sigma_schedule(num_inference_steps)
    x = latents
    noise_prompt = x[..., :overlap] if overlap else None
    zeros = mx.zeros_like(condition)
    for index in range(num_inference_steps):
        sigma = float(sigmas[index])
        sigma_next = float(sigmas[index + 1])
        if overlap:
            blend = (
                1.0 - (1.0 - 1e-6) * sigma
            ) * noise_prompt + sigma * previous_latent[..., :overlap]
            x = mx.concatenate([blend, x[..., overlap:]], axis=-1)
        timestep = mx.full((x.shape[0],), sigma, dtype=x.dtype)
        conditional = transformer(x, timestep, condition)
        if guidance_scale == 1.0:
            velocity = conditional
        else:
            unconditional = transformer(x, timestep, zeros)
            velocity = unconditional + guidance_scale * (conditional - unconditional)
        x = x + (sigma_next - sigma) * velocity
        mx.eval(x)

    if overlap:
        x = mx.concatenate([previous_latent[..., :overlap], x[..., overlap:]], axis=-1)
        mx.eval(x)
    return x, condition
