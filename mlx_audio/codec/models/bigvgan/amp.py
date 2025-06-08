import mlx.core as mx
import mlx.nn as nn
from typing_extensions import Literal

from mlx_audio.codec.models.bigvgan.activation import Snake, SnakeBeta
from mlx_audio.codec.models.bigvgan.conv import WNConv1d
from mlx_audio.codec.models.bigvgan.resample import Activation1d


class AMPBlock1(nn.Module):
    def __init__(
        self,
        channels: int,
        snake_logscale: bool,
        activation: Literal["snake", "snakebeta"],
        kernel_size=3,
        dilation: list[int] = [1, 3, 5],
    ):
        super().__init__()

        self.convs1 = [
            WNConv1d(
                channels,
                channels,
                kernel_size,
                stride=1,
                dilation=d,
                padding=((kernel_size - 1) * d) // 2,
            )
            for d in dilation
        ]
        self.convs2 = [
            WNConv1d(
                channels,
                channels,
                kernel_size,
                stride=1,
                dilation=1,
                padding=(kernel_size - 1) // 2,
            )
            for _ in dilation
        ]
        self.activations = [
            Activation1d(
                Snake(channels, alpha_logscale=snake_logscale)
                if activation == "snake"
                else SnakeBeta(channels, alpha_logscale=snake_logscale)
            )
            for _ in range(len(dilation) * 2)
        ]

    def __call__(self, x: mx.array):
        for conv1, conv2, activation1, activation2 in zip(
            self.convs1, self.convs2, self.activations[::2], self.activations[1::2]
        ):
            x = x + conv2(activation2(conv1(activation1(x))))

        return x


class AMPBlock2(nn.Module):
    def __init__(
        self,
        channels: int,
        snake_logscale: bool,
        activation: Literal["snake", "snakebeta"],
        kernel_size=3,
        dilation: list[int] = [1, 3, 5],
    ):
        super().__init__()

        self.convs = [
            WNConv1d(
                channels,
                channels,
                kernel_size,
                stride=1,
                dilation=d,
                padding=((kernel_size - 1) * d) // 2,
            )
            for d in dilation
        ]
        self.activations = [
            Activation1d(
                Snake(channels, alpha_logscale=snake_logscale)
                if activation == "snake"
                else SnakeBeta(channels, alpha_logscale=snake_logscale)
            )
            for _ in dilation
        ]

    def __call__(self, x: mx.array):
        for conv, activation in zip(self.convs, self.activations):
            x = x + conv(activation(x))

        return x
