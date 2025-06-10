from dataclasses import dataclass
from typing import Literal

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_audio.codec.models.bigvgan.activation import Snake, SnakeBeta
from mlx_audio.codec.models.bigvgan.amp import AMPBlock1, AMPBlock2
from mlx_audio.codec.models.bigvgan.conv import WNConv1d, WNConvTranspose1d
from mlx_audio.codec.models.bigvgan.resample import Activation1d


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

        self.conv_pre = WNConv1d(
            config.num_mels, config.upsample_initial_channel, 7, 1, 3
        )
        self.ups = [
            [
                WNConvTranspose1d(
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
        self.conv_post = WNConv1d(
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
        curr_weights = dict(tree_flatten(self.parameters()))

        for key, value in weights.items():
            if "num_batches_tracked" in key:
                continue

            if "conv" in key or "lowpass.filter" in key or "upsample.filter" in key:
                if value.ndim == 3:
                    if value.shape != curr_weights[key].shape:
                        value = value.transpose(0, 2, 1)
                elif value.ndim == 4:
                    if value.shape != curr_weights[key].shape:
                        value = value.transpose(0, 2, 3, 1)

            if "ups." in key:
                if value.ndim == 3:
                    if value.shape != curr_weights[key].shape:
                        value = value.transpose(1, 2, 0)

            new_weights[key] = value

        del curr_weights

        return new_weights
