import math

import mlx.core as mx
import mlx.nn as nn


def normalize_weight(x, except_dim=0):
    if x.ndim != 3:
        raise ValueError("Input tensor must have 3 dimensions")

    axes = tuple(i for i in range(x.ndim) if i != except_dim)
    return mx.sqrt(mx.sum(mx.power(x, 2), axis=axes, keepdims=True))


class WNConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = True,
        groups: int = 1,
    ):
        super().__init__()

        if bias:
            self.bias = mx.zeros((out_channels,))

        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation = dilation
        self.stride = stride
        self.groups = groups

        scale = math.sqrt(1 / (in_channels * kernel_size))
        weight_init = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(out_channels, kernel_size, in_channels),
        )
        self.weight_g = normalize_weight(weight_init)
        self.weight_v = weight_init / (self.weight_g + 1e-12)

    def _extra_repr(self):
        return (
            f"in_channels={self.weight_v.shape[2]}, out_channels={self.weight_v.shape[0]}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, dilation={self.dilation}, "
            f"bias={'bias' in self}"
        )

    def __call__(self, x):
        weight = self.weight_g * self.weight_v / normalize_weight(self.weight_v)
        y = mx.conv1d(x, weight, self.stride, self.padding, self.dilation, self.groups)
        if "bias" in self:
            y = y + self.bias
        return y


class WNConvTranspose1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()

        self.bias = mx.zeros((out_channels,)) if bias else None

        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation = dilation
        self.stride = stride
        self.groups = groups

        scale = math.sqrt(1 / (in_channels * kernel_size))
        weight_init = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(out_channels, kernel_size, in_channels // groups),
        )
        self.weight_g = normalize_weight(weight_init, except_dim=2)
        self.weight_v = weight_init / (self.weight_g + 1e-12)

    def _extra_repr(self):
        return (
            f"in_channels={self.weight_v.shape[2] * self.groups}, out_channels={self.weight_v.shape[0]}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, dilation={self.dilation}, "
            f"groups={self.groups}, bias={'bias' in self}"
        )

    def __call__(self, x):
        weight = (
            self.weight_g
            * self.weight_v
            / normalize_weight(self.weight_v, except_dim=2)
        )
        y = mx.conv_transpose1d(
            x, weight, self.stride, self.padding, self.dilation, self.groups
        )
        if self.bias is not None:
            y = y + self.bias
        return y


def snake(x, alpha):
    recip = mx.reciprocal(alpha + 1e-9)
    x = x + recip * mx.power(mx.sin(alpha * x), 2)
    return x


class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = mx.ones((1, 1, channels))

    def __call__(self, x):
        x = snake(x, self.alpha)
        return x
