from __future__ import annotations

import math

import mlx.core as mx

BASE_SHIFT_ANCHOR = 1024
MAX_SHIFT_ANCHOR = 4096


def ltx2_sigmas(
    steps: int,
    latent: mx.array | None = None,
    max_shift: float = 2.05,
    base_shift: float = 0.95,
    stretch: bool = True,
    terminal: float = 0.1,
    default_number_of_tokens: int = MAX_SHIFT_ANCHOR,
) -> mx.array:
    tokens = (
        math.prod(latent.shape[2:]) if latent is not None else default_number_of_tokens
    )
    sigmas = mx.linspace(1.0, 0.0, steps + 1, dtype=mx.float32)

    slope = (max_shift - base_shift) / (MAX_SHIFT_ANCHOR - BASE_SHIFT_ANCHOR)
    intercept = base_shift - slope * BASE_SHIFT_ANCHOR
    sigma_shift = tokens * slope + intercept
    exp_shift = math.exp(sigma_shift)

    shifted = exp_shift / (exp_shift + (1 / sigmas - 1))
    sigmas = mx.where(sigmas != 0, shifted, mx.zeros_like(sigmas))

    if stretch:
        non_zero = sigmas[:-1]
        one_minus = 1.0 - non_zero
        scale_factor = one_minus[-1] / (1.0 - terminal)
        stretched = 1.0 - (one_minus / scale_factor)
        sigmas = mx.concatenate([stretched, sigmas[-1:]], axis=0)

    return sigmas.astype(mx.float32)


def euler_step(
    sample: mx.array,
    denoised_sample: mx.array,
    sigmas: mx.array,
    step_index: int,
) -> mx.array:
    sigma = sigmas[step_index]
    sigma_next = sigmas[step_index + 1]
    dt = sigma_next - sigma
    velocity = to_velocity(sample, sigma, denoised_sample)
    return (sample.astype(mx.float32) + velocity.astype(mx.float32) * dt).astype(
        sample.dtype
    )


def to_velocity(sample: mx.array, sigma: float | mx.array, denoised_sample: mx.array):
    if isinstance(sigma, mx.array) and sigma.size == 1 and float(sigma.item()) == 0.0:
        raise ValueError("Sigma can't be 0.0")
    if not isinstance(sigma, mx.array) and sigma == 0.0:
        raise ValueError("Sigma can't be 0.0")
    return (
        (sample.astype(mx.float32) - denoised_sample.astype(mx.float32)) / sigma
    ).astype(sample.dtype)


def to_denoised(sample: mx.array, velocity: mx.array, sigma: float | mx.array):
    return (sample.astype(mx.float32) - velocity.astype(mx.float32) * sigma).astype(
        sample.dtype
    )
