"""Fuse autoregressive hidden states onto the Flow-VAE timeline."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .config import ModelConfig


def nearest_interpolate_1d(x_ncl: mx.array, size: int) -> mx.array:
    length = x_ncl.shape[-1]
    if size == length:
        return x_ncl
    indices = (mx.arange(size) * (length / size)).astype(mx.int32)
    return x_ncl[:, :, mx.clip(indices, 0, length - 1)]


def latent_length_from_frames(num_frames: int, config: ModelConfig) -> int:
    return max(
        1,
        int(
            num_frames
            * config.output_sampling_rate
            / config.input_sampling_rate
            * config.input_hop_length
            / config.output_hop_length
        ),
    )


class ConditionEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.layer_weight_logits = mx.zeros((config.num_condition_layers,))
        self.layer_scale = mx.ones((1,))
        self.proj = nn.Conv1d(
            config.hidden_size,
            config.condition_out_dim,
            kernel_size=3,
            padding=1,
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        batch, num_frames, _ = hidden_states.shape
        hidden = hidden_states.reshape(
            batch,
            num_frames,
            self.config.num_condition_layers,
            self.config.hidden_size,
        ).transpose(0, 2, 3, 1)
        weights = mx.softmax(
            self.layer_weight_logits.astype(mx.float32), axis=0
        ).astype(hidden.dtype)
        hidden = (hidden * weights.reshape(1, -1, 1, 1)).sum(axis=1)
        hidden = self.layer_scale.astype(hidden.dtype) * hidden
        hidden = self.proj(hidden.transpose(0, 2, 1)).transpose(0, 2, 1)
        target = latent_length_from_frames(num_frames, self.config)
        return nearest_interpolate_1d(hidden, target).transpose(0, 2, 1)


def fuse_frame_hiddens(global_hidden: mx.array, depth_hiddens: mx.array) -> mx.array:
    return mx.concatenate([global_hidden, depth_hiddens], axis=-1)
