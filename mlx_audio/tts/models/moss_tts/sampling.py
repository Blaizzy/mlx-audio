"""Per-channel sampling utilities for MOSS-TTS local generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import mlx.core as mx
import numpy as np
from mlx_lm.sample_utils import apply_top_k, apply_top_p, categorical_sampling


@dataclass(frozen=True)
class ChannelSamplingConfig:
    do_sample: bool
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float = 1.0


def resolve_channel_sampling_configs(
    num_channels: int,
    *,
    default_temperature: float,
    default_top_p: float,
    default_top_k: int,
    default_repetition_penalty: float,
    do_samples: Optional[Sequence[bool]] = None,
    layers: Optional[Sequence[Dict[str, float]]] = None,
) -> List[ChannelSamplingConfig]:
    """
    Build per-channel sampling configs.

    Channel 0 (text/control) defaults to greedy decoding unless explicitly
    overridden.
    """

    configs: List[ChannelSamplingConfig] = []
    for channel_idx in range(num_channels):
        layer_cfg = (
            {} if layers is None or channel_idx >= len(layers) else layers[channel_idx]
        )
        do_sample = (
            do_samples[channel_idx]
            if do_samples is not None and channel_idx < len(do_samples)
            else channel_idx != 0
        )
        cfg = ChannelSamplingConfig(
            do_sample=bool(do_sample),
            temperature=float(layer_cfg.get("temperature", default_temperature)),
            top_p=float(layer_cfg.get("top_p", default_top_p)),
            top_k=int(layer_cfg.get("top_k", default_top_k)),
            repetition_penalty=float(
                layer_cfg.get("repetition_penalty", default_repetition_penalty)
            ),
        )
        configs.append(cfg)
    return configs


def apply_repetition_penalty(
    logits: mx.array,
    previous_tokens: mx.array,
    penalty: float,
    repetition_window: Optional[int] = None,
) -> mx.array:
    """Apply repetition penalty independently per batch row."""

    if penalty == 1.0:
        return logits
    # bfloat16 -> NumPy conversion can fail through the default bridge; run
    # math in float32 NumPy space and cast back to the original MLX dtype.
    logits_np = np.array(logits.astype(mx.float32), copy=True)
    history_np = np.array(previous_tokens.astype(mx.int32))

    if history_np.ndim == 1:
        history_np = history_np[None, :]
    if repetition_window is not None and int(repetition_window) > 0:
        history_np = history_np[:, -int(repetition_window) :]

    for row_idx in range(history_np.shape[0]):
        unique = np.unique(history_np[row_idx]).astype(np.int64)
        if unique.size == 0:
            continue
        token_logits = logits_np[row_idx, unique]
        positive = token_logits > 0
        token_logits[positive] = token_logits[positive] / penalty
        token_logits[~positive] = token_logits[~positive] * penalty
        logits_np[row_idx, unique] = token_logits

    return mx.array(logits_np, dtype=logits.dtype)


def sample_channel_token(
    logits: mx.array,
    config: ChannelSamplingConfig,
    previous_tokens: Optional[mx.array] = None,
    repetition_window: Optional[int] = None,
) -> mx.array:
    """
    Sample one token per batch row from channel logits.

    `logits` shape: [batch, vocab]
    """

    if previous_tokens is not None and config.repetition_penalty != 1.0:
        logits = apply_repetition_penalty(
            logits,
            previous_tokens,
            config.repetition_penalty,
            repetition_window=repetition_window,
        )

    if not config.do_sample or config.temperature <= 0:
        return mx.argmax(logits, axis=-1).astype(mx.int32)

    logprobs = logits.astype(mx.float32) / max(config.temperature, 1e-5)
    if config.top_k > 0:
        logprobs = apply_top_k(logprobs, config.top_k)
    if 0 < config.top_p < 1:
        logprobs = apply_top_p(logprobs, config.top_p)

    return categorical_sampling(logprobs, 1.0).astype(mx.int32)


__all__ = [
    "ChannelSamplingConfig",
    "apply_repetition_penalty",
    "resolve_channel_sampling_configs",
    "sample_channel_token",
]
