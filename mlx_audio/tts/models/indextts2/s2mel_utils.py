from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


def sequence_mask(lengths: mx.array, max_length: int | None = None) -> mx.array:
    # lengths: (B,)
    if max_length is None:
        max_length = int(mx.max(lengths).item())
    rng = mx.arange(max_length)
    return rng[None, :] < lengths[:, None]


def fused_add_tanh_sigmoid_multiply(
    input_a: mx.array, input_b: mx.array, n_channels: int
) -> mx.array:
    in_act = input_a + input_b
    t_act_part = in_act[:, :n_channels, :]
    s_act_part = in_act[:, n_channels:, :]
    return mx.tanh(t_act_part) * nn.sigmoid(s_act_part)
