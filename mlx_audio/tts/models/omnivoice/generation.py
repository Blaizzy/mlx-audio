import math
from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn

if TYPE_CHECKING:
    from .omnivoice import Model


def _get_time_steps(num_step: int, t_shift: float = 0.1) -> list:
    """Cosine-shifted timestep schedule matching original OmniVoice."""
    n = num_step + 1
    ts = [i / num_step for i in range(n)]  # linspace 0..1
    # t_shift warp: t' = t_shift * t / (1 + (t_shift - 1) * t)
    return [t_shift * t / (1.0 + (t_shift - 1.0) * t) for t in ts]


def _gumbel_noise(x: mx.array, temperature: float) -> mx.array:
    """Add Gumbel noise: (x/temp) + Gumbel(0,1)."""
    u = mx.random.uniform(shape=x.shape)
    gumbel = -mx.log(-mx.log(u + 1e-10) + 1e-10)
    return x / temperature + gumbel


def _filter_top_k(log_probs: mx.array, ratio: float = 0.1) -> mx.array:
    """Keep only top-k entries (by ratio), set others to -inf."""
    V = log_probs.shape[-1]
    k = max(1, math.ceil(ratio * V))
    # sort descending, threshold at k-th value
    sorted_vals = mx.sort(log_probs, axis=-1)[..., ::-1]  # descending
    threshold = sorted_vals[..., k - 1 : k]  # [..., 1]
    return mx.where(log_probs >= threshold, log_probs, mx.array(-float("inf")))


def iterative_unmask(
    model: "Model",
    cond_input_ids: mx.array,
    cond_audio_mask: mx.array,
    T: int,
    num_steps: int = 32,
    guidance_scale: float = 2.0,
    class_temperature: float = 0.0,
    position_temperature: float = 5.0,
    layer_penalty_factor: float = 5.0,
    t_shift: float = 0.1,
) -> mx.array:
    C = model.config.num_audio_codebook
    mask_id = model.config.audio_mask_id
    V = model.config.audio_vocab_size

    c_len = cond_input_ids.shape[1]

    uncond_input_ids = cond_input_ids[:, -T:, :]
    uncond_audio_mask = cond_audio_mask[:, -T:]

    layer_ids = mx.arange(C, dtype=mx.float32)
    timesteps = _get_time_steps(num_steps, t_shift)
    total_mask = T * C

    mask_token_mask = mx.arange(V) == mask_id

    for step in range(num_steps):
        logits_all = model(cond_input_ids, cond_audio_mask)
        logits_cond = logits_all[:, c_len - T :, :, :]

        if guidance_scale != 0:
            logits_uncond_all = model(uncond_input_ids, uncond_audio_mask)
            logits_uncond = logits_uncond_all[:, :T, :, :]
            c_lp = nn.log_softmax(logits_cond, axis=-1)
            u_lp = nn.log_softmax(logits_uncond, axis=-1)
            log_probs = nn.log_softmax(c_lp + guidance_scale * (c_lp - u_lp), axis=-1)
        else:
            log_probs = nn.log_softmax(logits_cond, axis=-1)

        log_probs = mx.where(mask_token_mask, -float("inf"), log_probs)
        log_probs = log_probs[0]

        if class_temperature > 0.0:
            filtered = _filter_top_k(log_probs, ratio=0.1)
            new_tokens = mx.argmax(_gumbel_noise(filtered, class_temperature), axis=-1)
        else:
            new_tokens = mx.argmax(log_probs, axis=-1)

        confidence = mx.max(log_probs, axis=-1)
        confidence = confidence - layer_ids * layer_penalty_factor
        if position_temperature > 0.0:
            confidence = _gumbel_noise(confidence, position_temperature)

        dt = timesteps[step + 1] - timesteps[step]
        k = max(1, math.ceil(total_mask * dt))
        if step == num_steps - 1:
            k = total_mask

        current_tokens = cond_input_ids[0, c_len - T :, :]
        still_masked = current_tokens == mask_id
        score = mx.where(still_masked, confidence, mx.array(-float("inf")))
        flat_score = score.reshape(-1)
        rank = mx.argsort(mx.argsort(-flat_score))
        reveal = (rank < k).reshape(T, C) & still_masked

        updated = mx.where(reveal, new_tokens, current_tokens)
        mx.eval(updated)

        prefix = cond_input_ids[:, : c_len - T, :]
        cond_input_ids = mx.concatenate([prefix, updated[None]], axis=1)
        uncond_input_ids = updated[None]

    tokens = cond_input_ids[0, c_len - T :, :]
    tokens = mx.where(tokens == mask_id, mx.zeros_like(tokens), tokens)
    return tokens
