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
    cond_embeds: mx.array,  # [1, S_cond, D]
    uncond_embeds: mx.array,  # [1, S_uncond, D]  — empty for pure uncond
    T: int,
    num_steps: int = 32,
    guidance_scale: float = 2.0,
    class_temperature: float = 0.0,
    position_temperature: float = 5.0,
    layer_penalty_factor: float = 5.0,
    t_shift: float = 0.1,
) -> mx.array:  # [T, 8] in [0, 1023]
    """Iterative unmasking decode — matches original OmniVoice inference exactly.

    CFG is applied in log-prob space (not logit space).
    Position selection uses Gumbel noise + layer penalty.
    Class prediction is greedy (class_temperature=0) by default.
    """
    C = model.config.num_audio_codebook
    mask_id = model.config.audio_mask_id
    tokens = mx.full((T, C), mask_id, dtype=mx.int32)

    cond_prefix_len = cond_embeds.shape[1]
    uncond_prefix_len = uncond_embeds.shape[1]
    same_prefix = cond_prefix_len == uncond_prefix_len

    if same_prefix:
        prefix_batch = mx.concatenate([cond_embeds, uncond_embeds], axis=0)  # [2, S, D]

    # Layer IDs for penalty: shape [C] → broadcast over [T, C]
    layer_ids = mx.arange(C, dtype=mx.float32)  # [C]

    timesteps = _get_time_steps(num_steps, t_shift)
    total_mask = T * C

    for step in range(num_steps):
        # Build full inputs_embeds including current (partially-unmasked) tokens
        audio_embeds = sum(
            model.audio_embeddings[i](tokens[None, :, i]) for i in range(C)
        )  # [1, T, D]

        if same_prefix:
            audio_batch = mx.concatenate(
                [audio_embeds, audio_embeds], axis=0
            )  # [2, T, D]
            inputs_embeds = mx.concatenate(
                [prefix_batch, audio_batch], axis=1
            )  # [2, S+T, D]
            logits_batch = model(
                inputs_embeds, prefix_len=cond_prefix_len
            )  # [2, T, C, V]
            logits_cond = logits_batch[0:1]  # [1, T, C, V]
            logits_uncond = logits_batch[1:2]  # [1, T, C, V]
        else:
            # Different prefix lengths (voice cloning vs no-prefix uncond)
            inputs_cond = mx.concatenate(
                [cond_embeds, audio_embeds], axis=1
            )  # [1, S_cond+T, D]
            logits_cond = model(inputs_cond, prefix_len=cond_prefix_len)  # [1, T, C, V]

            if guidance_scale != 0:
                if uncond_prefix_len == 0:
                    logits_uncond = model(audio_embeds, prefix_len=0)  # [1, T, C, V]
                else:
                    inputs_uncond = mx.concatenate(
                        [uncond_embeds, audio_embeds], axis=1
                    )  # [1, S_uncond+T, D]
                    logits_uncond = model(
                        inputs_uncond, prefix_len=uncond_prefix_len
                    )  # [1, T, C, V]

        # CFG in log-prob space (matches original)
        if guidance_scale != 0:
            c_lp = nn.log_softmax(logits_cond, axis=-1)  # [1, T, C, V]
            u_lp = nn.log_softmax(logits_uncond, axis=-1)  # [1, T, C, V]
            log_probs = nn.log_softmax(
                c_lp + guidance_scale * (c_lp - u_lp), axis=-1
            )  # [1, T, C, V]
        else:
            log_probs = nn.log_softmax(logits_cond, axis=-1)

        # Mask out AUDIO_MASK_ID from predictions
        # Build a mask: True at index AUDIO_MASK_ID along the last axis
        V = log_probs.shape[-1]
        mask_token_mask = mx.arange(V) == mask_id  # [V]
        log_probs = mx.where(mask_token_mask, -float("inf"), log_probs)
        log_probs = log_probs[0]  # [T, C, V]

        # Predict tokens
        if class_temperature > 0.0:
            filtered = _filter_top_k(log_probs, ratio=0.1)
            new_tokens = mx.argmax(
                _gumbel_noise(filtered, class_temperature), axis=-1
            )  # [T, C]
        else:
            new_tokens = mx.argmax(log_probs, axis=-1)  # [T, C]

        # Confidence = max log-prob per position
        confidence = mx.max(log_probs, axis=-1)  # [T, C]

        # Layer penalty: encourage lower-index codebooks to unmask first
        # layer_ids [C] broadcast to [T, C]
        confidence = confidence - layer_ids * layer_penalty_factor

        # Position temperature: Gumbel noise on confidence scores
        if position_temperature > 0.0:
            confidence = _gumbel_noise(confidence, position_temperature)

        # How many positions to unmask this step
        dt = timesteps[step + 1] - timesteps[step]
        k = max(1, math.ceil(total_mask * dt))
        # On the last step, unmask everything still masked
        if step == num_steps - 1:
            k = total_mask

        # Only reveal still-masked positions
        still_masked = tokens == mask_id  # [T, C]
        # Set confidence to -inf for already-unmasked positions
        score = mx.where(still_masked, confidence, mx.array(-float("inf")))

        flat_score = score.reshape(-1)
        # Top-k positions to reveal
        rank = mx.argsort(mx.argsort(-flat_score))
        reveal_flat = rank < k
        reveal = reveal_flat.reshape(T, C)
        update = reveal & still_masked

        tokens = mx.where(update, new_tokens, tokens)
        mx.eval(tokens)

    # Safety: replace any remaining mask tokens with token 0
    tokens = mx.where(tokens == mask_id, mx.zeros_like(tokens), tokens)
    return tokens
