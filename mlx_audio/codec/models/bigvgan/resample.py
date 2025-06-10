import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def sinc(x: mx.array):
    return mx.where(
        x == 0,
        mx.array(1.0, dtype=x.dtype),
        mx.sin(math.pi * x) / math.pi / x,
    )


def kaiser_sinc_filter1d(
    cutoff: float, half_width: float, kernel_size: int
) -> mx.array:  # return filter [1,kernel_size,1]
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2

    # For kaiser window
    delta_f = 4 * half_width
    A = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if A > 50.0:
        beta = 0.1102 * (A - 8.7)
    elif A >= 21.0:
        beta = 0.5842 * (A - 21) ** 0.4 + 0.07886 * (A - 21.0)
    else:
        beta = 0.0
    window = mx.array(np.kaiser(kernel_size, beta=beta))

    # ratio = 0.5/cutoff -> 2 * cutoff = 1 / ratio
    if even:
        time = mx.arange(-half_size, half_size) + 0.5
    else:
        time = mx.arange(kernel_size) - half_size
    if cutoff == 0:
        filter = mx.zeros_like(time).reshape(1, kernel_size, 1)
    else:
        filter_ = 2 * cutoff * window * sinc(2 * cutoff * time)
        filter_ /= filter_.sum()
        filter = filter_.reshape(1, kernel_size, 1)

    return filter


class LowPassFilter1d(nn.Module):
    def __init__(
        self,
        cutoff: float = 0.5,
        half_width: float = 0.6,
        stride: int = 1,
        padding: bool = True,
        padding_mode: str = "edge",
        kernel_size: int = 12,
    ):
        super().__init__()

        if cutoff < -0.0:
            raise ValueError("Minimum cutoff must be larger than zero.")
        if cutoff > 0.5:
            raise ValueError("A cutoff above 0.5 does not make sense.")

        self.even = kernel_size % 2 == 0
        self.stride = stride

        self.pad_left = kernel_size // 2 - int(self.even)
        self.pad_right = kernel_size // 2
        self.padding = padding
        self.padding_mode = padding_mode

        self.filter = kaiser_sinc_filter1d(
            cutoff, half_width, kernel_size
        )  # (1, kernel_size, 1)
        mx.eval(self.filter)

    def __call__(self, x: mx.array):  # (b, t, c)
        _, _, C = x.shape

        if self.padding:
            x = mx.pad(
                x,
                ((0, 0), (self.pad_left, self.pad_right), (0, 0)),
                mode=self.padding_mode,
            )

        expanded_filter = mx.broadcast_to(self.filter, (C, *self.filter.shape[1:]))

        out = mx.conv1d(
            x,
            expanded_filter,
            stride=self.stride,
            groups=C,
        )

        return out


class UpSample1d(nn.Module):
    def __init__(self, ratio: int = 2, kernel_size: Optional[int] = None):
        super().__init__()

        self.ratio = ratio
        self.kernel_size = (
            int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        )
        self.stride = ratio

        self.pad = self.kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
        self.pad_right = (
            self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
        )

        self.filter = kaiser_sinc_filter1d(
            cutoff=0.5 / ratio, half_width=0.6 / ratio, kernel_size=self.kernel_size
        )
        mx.eval(self.filter)

    def __call__(self, x: mx.array) -> mx.array:  # (b, t, c)
        _, _, C = x.shape

        x = mx.pad(x, ((0, 0), (self.pad, self.pad), (0, 0)), mode="edge")

        expanded_filter = mx.broadcast_to(self.filter, (C, *self.filter.shape[1:]))

        x = self.ratio * mx.conv_transpose1d(
            x,
            expanded_filter,
            stride=self.stride,
            groups=C,
        )

        return x[:, self.pad_left : -self.pad_right, :]


class DownSample1d(nn.Module):
    def __init__(self, ratio: int = 2, kernel_size: Optional[int] = None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = (
            int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        )
        self.lowpass = LowPassFilter1d(
            cutoff=0.5 / ratio,
            half_width=0.6 / ratio,
            stride=ratio,
            kernel_size=self.kernel_size,
        )

    def __call__(self, x: mx.array) -> mx.array:  # (b, t, c)
        return self.lowpass(x)


class Activation1d(nn.Module):
    def __init__(
        self,
        activation: nn.Module,
        up_ratio: int = 2,
        down_ratio: int = 2,
        up_kernel_size: int = 12,
        down_kernel_size: int = 12,
    ):
        super().__init__()
        self.up_ratio = up_ratio
        self.down_ratio = down_ratio
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    def __call__(self, x):
        x = self.upsample(x)
        x = self.act(x)
        x = self.downsample(x)
        return x
