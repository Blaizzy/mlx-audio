from __future__ import annotations

from dataclasses import dataclass, field

import mlx.core as mx


@dataclass(frozen=True)
class MultiModalGuiderParams:
    cfg_scale: float = 1.0
    stg_scale: float = 0.0
    stg_blocks: list[int] = field(default_factory=list)
    rescale_scale: float = 0.0
    modality_scale: float = 1.0
    cfg_clamp_scale: float = 0.0


def auto_rescale_for_cfg(cfg: float) -> float:
    if cfg <= 2.0:
        return 0.0
    if cfg <= 3.0:
        return 0.6 * (cfg - 2.0)
    if cfg <= 4.0:
        return 0.6 + 0.2 * (cfg - 3.0)
    if cfg <= 8.0:
        return 0.8
    return min(1.0, 0.8 + 0.1 * (cfg - 8.0))


def calculate_guided_prediction(
    cond: mx.array,
    uncond_text: mx.array | float,
    uncond_perturbed: mx.array | float,
    uncond_modality: mx.array | float,
    params: MultiModalGuiderParams,
) -> mx.array:
    pred = (
        cond
        + (params.cfg_scale - 1) * (cond - uncond_text)
        + params.stg_scale * (cond - uncond_perturbed)
        + (params.modality_scale - 1) * (cond - uncond_modality)
    )

    if params.rescale_scale != 0:
        factor = mx.std(cond) / mx.std(pred)
        factor = params.rescale_scale * factor + (1 - params.rescale_scale)
        pred = pred * factor

    if params.cfg_clamp_scale > 0:
        cfg_delta = pred - cond
        delta_norm = mx.linalg.norm(cfg_delta, axis=-1, keepdims=True)
        cond_norm = mx.linalg.norm(cond, axis=-1, keepdims=True)
        max_norm = cond_norm * params.cfg_clamp_scale
        scale = mx.where(
            delta_norm > max_norm,
            max_norm / mx.maximum(delta_norm, 1e-8),
            mx.ones_like(delta_norm),
        )
        pred = cond + cfg_delta * scale

    return pred
