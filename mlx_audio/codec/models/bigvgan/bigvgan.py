from dataclasses import dataclass
from typing import Literal

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.codec.models.bigvgan.activation import Snake, SnakeBeta
from mlx_audio.codec.models.bigvgan.amp import AMPBlock1, AMPBlock2
from mlx_audio.codec.models.bigvgan.resample import Activation1d


def normalize_weight(x, except_dim=0):
    if x.ndim != 3:
        raise ValueError("Input tensor must have 3 dimensions")

    axes = tuple(i for i in range(x.ndim) if i != except_dim)
    return mx.sqrt(mx.sum(mx.power(x, 2), axis=axes, keepdims=True))


@dataclass
class BigVGANConfig:
    num_mels: int
    upsample_rates: list[int]
    upsample_kernel_sizes: list[int]
    upsample_initial_channel: int
    resblock: Literal["1", "2"]
    resblock_kernel_sizes: list[int]
    resblock_dilation_sizes: list[list[int]]
    activation: Literal["snakebeta", "snake"]
    snake_logscale: bool
    use_bias_at_final: bool = True  # compatability
    use_tanh_at_final: bool = True  # compatability


class BigVGAN(nn.Module):
    def __init__(self, config: BigVGANConfig):
        super().__init__()

        self.num_kernels = len(config.resblock_kernel_sizes)
        self.num_upsamples = len(config.upsample_rates)
        self.use_tanh_at_final = config.use_tanh_at_final

        self.conv_pre = nn.Conv1d(
            config.num_mels, config.upsample_initial_channel, 7, 1, 3
        )
        self.ups = [
            [
                nn.ConvTranspose1d(
                    config.upsample_initial_channel // (2**i),
                    config.upsample_initial_channel // (2 ** (i + 1)),
                    k,
                    u,
                    padding=(k - u) // 2,
                )
            ]
            for i, (u, k) in enumerate(
                zip(config.upsample_rates, config.upsample_kernel_sizes)
            )
        ]
        self.resblocks = [
            (
                AMPBlock1(
                    config.upsample_initial_channel // (2 ** (i + 1)),
                    config.snake_logscale,
                    config.activation,
                    k,
                    d,
                )
                if config.resblock == "1"
                else AMPBlock2(
                    config.upsample_initial_channel // (2 ** (i + 1)),
                    config.snake_logscale,
                    config.activation,
                    k,
                    d,
                )
            )
            for i in range(len(self.ups))
            for j, (k, d) in enumerate(
                zip(config.resblock_kernel_sizes, config.resblock_dilation_sizes)
            )
        ]
        self.activation_post = Activation1d(
            Snake(
                config.upsample_initial_channel // (2 ** len(self.ups)),
                alpha_logscale=config.snake_logscale,
            )
            if config.activation == "snake"
            else SnakeBeta(
                config.upsample_initial_channel // (2 ** len(self.ups)),
                alpha_logscale=config.snake_logscale,
            )
        )
        self.conv_post = nn.Conv1d(
            config.upsample_initial_channel // (2 ** len(self.ups)),
            1,
            7,
            1,
            padding=3,
            bias=config.use_bias_at_final,
        )

    def __call__(
        self, x: mx.array, *args, **kwargs
    ) -> mx.array:  # (batch, num_mels, seq)
        x = x.transpose(0, 2, 1)

        x = self.conv_pre(x)

        for step in range(self.num_upsamples):
            for idx in range(len(self.ups[step])):
                x = self.ups[step][idx](x)

            xs = self.resblocks[step * self.num_kernels](x)
            for idx in range(1, self.num_kernels):
                xs += self.resblocks[step * self.num_kernels + idx](x)

            x = xs / self.num_kernels

        x = self.activation_post(x)
        x = self.conv_post(x)

        if self.use_tanh_at_final:
            x = mx.tanh(x)
        else:
            x = mx.clip(x, -1.0, 1.0)

        return x.transpose(0, 2, 1)

    def sanitize(self, weights: dict[str, mx.array]):
        new_weights = {}

        for key, value in weights.items():
            if "num_batches_tracked" in key:
                continue

            if "conv" in key or "lowpass.filter" in key or "upsample.filter" in key:
                if value.ndim == 3:
                    value = value.transpose(0, 2, 1)
                elif value.ndim == 4:
                    value = value.transpose(0, 2, 3, 1)

            if "ups." in key:
                if value.ndim == 3:
                    value = value.transpose(1, 2, 0)

            if key.endswith("weight_g"):
                new_weights[key.replace("_g", "")] = (
                    new_weights.get(key.replace("_g", ""), 1) * value
                )
                continue

            if key.endswith("weight_v"):
                v_norm = (
                    normalize_weight(value, 0)
                    if "ups." not in key
                    else normalize_weight(value, 2)
                )
                new_weights[key.replace("_v", "")] = (
                    new_weights.get(key.replace("_v", ""), 1) * value / v_norm
                )
                continue

            new_weights[key] = value

        return new_weights
