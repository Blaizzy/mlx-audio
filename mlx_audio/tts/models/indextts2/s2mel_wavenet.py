from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .s2mel_utils import fused_add_tanh_sigmoid_multiply


def _conv1d_weightnorm_strip(weight_g: mx.array, weight_v: mx.array, eps: float = 1e-8) -> mx.array:
    # weightnorm: w = g * v / ||v||
    v = weight_v
    # Norm over (in, k) dims for conv1d v shaped (O, I, K) in torch; converter should transpose first.
    # Here we assume already MLX conv layout (O, K, I).
    norm = mx.sqrt(mx.sum(v * v, axis=(1, 2), keepdims=True) + eps)
    return v * (weight_g.reshape(-1, 1, 1) / norm)


class WNConv1d(nn.Module):
    """Conv1d wrapper that optionally accepts already-stripped weights."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, *, dilation: int = 1, padding: int = 0, bias: bool = True):
        super().__init__()
        self.conv = nn.Conv1d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            bias=bias,
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv(x)


class WN(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
        p_dropout: float = 0.0,
    ):
        super().__init__()
        assert kernel_size % 2 == 1
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.in_layers = []
        self.res_skip_layers = []

        if gin_channels != 0:
            self.cond_layer = nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1)

        for i in range(n_layers):
            dilation = dilation_rate**i
            padding = int((kernel_size * dilation - dilation) / 2)
            self.in_layers.append(
                nn.Conv1d(
                    hidden_channels,
                    2 * hidden_channels,
                    kernel_size,
                    dilation=dilation,
                    padding=padding,
                )
            )

            res_skip_channels = 2 * hidden_channels if i < n_layers - 1 else hidden_channels
            self.res_skip_layers.append(nn.Conv1d(hidden_channels, res_skip_channels, 1))

    def __call__(self, x: mx.array, x_mask: mx.array, g: Optional[mx.array] = None) -> mx.array:
        # x: (B, T, C), x_mask: (B, T, 1), g: (B, 1, gin)
        output = mx.zeros_like(x)
        n_ch = self.hidden_channels

        if g is not None and self.gin_channels != 0:
            g = self.cond_layer(g)

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            if g is not None:
                cond_offset = i * 2 * n_ch
                g_l = g[:, :, cond_offset : cond_offset + 2 * n_ch]
            else:
                g_l = mx.zeros_like(x_in)

            acts = fused_add_tanh_sigmoid_multiply(
                x_in.transpose(0, 2, 1), g_l.transpose(0, 2, 1), n_ch
            ).transpose(0, 2, 1)
            res_skip = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                res = res_skip[:, :, :n_ch]
                x = (x + res) * x_mask
                output = output + res_skip[:, :, n_ch:]
            else:
                output = output + res_skip

        return output * x_mask
