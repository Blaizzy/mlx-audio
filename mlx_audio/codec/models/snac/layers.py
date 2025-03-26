import math

import mlx.core as mx
import mlx.nn as nn

from .attention import LocalMHA


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
        groups: int = 1,
        bias: bool = True,
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
            shape=(out_channels, kernel_size, in_channels // groups),
        )
        self.weight_g = normalize_weight(weight_init)
        self.weight_v = weight_init / (self.weight_g + 1e-12)

    def _extra_repr(self):
        return (
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, dilation={self.dilation}, "
            f"groups={self.groups}, bias={'bias' in self}"
        )

    def __call__(self, x):
        weight = self.weight_g * self.weight_v / normalize_weight(self.weight_v)
        y = mx.conv1d(x, weight, self.stride, self.padding, self.dilation, self.groups)
        if hasattr(self, "bias"):
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
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()

        self.bias = mx.zeros((out_channels,)) if bias else None

        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation = dilation
        self.stride = stride
        self.output_padding = output_padding
        self.groups = groups

        scale = math.sqrt(1 / (in_channels * kernel_size))
        weight_init = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(in_channels, kernel_size, out_channels // groups),
        )
        self.weight_g = normalize_weight(weight_init, except_dim=0)
        self.weight_v = weight_init / (self.weight_g + 1e-12)

    def _extra_repr(self):
        return (
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, dilation={self.dilation}, "
            f"output_padding={self.output_padding}, "
            f"groups={self.groups}, bias={self.bias is not None}"
        )

    def __call__(self, x):
        weight = (
            self.weight_g
            * self.weight_v
            / normalize_weight(self.weight_v, except_dim=0)
        )
        y = mx.conv_transpose1d(
            x,
            weight.swapaxes(0, 2),  # mlx uses (out_channels, kernel_size, in_channels)
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        if self.bias is not None:
            y = y + self.bias
        return y


def snake(x, alpha):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    recip = mx.reciprocal(alpha + 1e-9)
    x = x + recip * mx.power(mx.sin(alpha * x), 2)
    x = x.reshape(shape)
    return x


class Encoder(nn.Module):
    def __init__(
        self,
        d_model=64,
        strides=[3, 3, 7, 7],
        depthwise=False,
        attn_window_size=32,
    ):
        super().__init__()
        layers = [WNConv1d(1, d_model, kernel_size=7, padding=3)]
        for stride in strides:
            d_model *= 2
            groups = d_model // 2 if depthwise else 1
            layers += [EncoderBlock(output_dim=d_model, stride=stride, groups=groups)]
        if attn_window_size is not None:
            layers += [LocalMHA(dim=d_model, window_size=attn_window_size)]
        groups = d_model if depthwise else 1
        layers += [
            WNConv1d(d_model, d_model, kernel_size=7, padding=3, groups=groups),
        ]
        self.block = nn.Sequential(*layers)

    def __call__(self, x):
        return self.block(x).moveaxis(1, 2)


class Decoder(nn.Module):
    def __init__(
        self,
        input_channel,
        channels,
        rates,
        noise=False,
        depthwise=False,
        attn_window_size=32,
        d_out=1,
    ):
        super().__init__()
        if depthwise:
            layers = [
                WNConv1d(
                    input_channel,
                    input_channel,
                    kernel_size=7,
                    padding=3,
                    groups=input_channel,
                ),
                WNConv1d(input_channel, channels, kernel_size=1),
            ]
        else:
            layers = [WNConv1d(input_channel, channels, kernel_size=7, padding=3)]

        if attn_window_size is not None:
            layers += [LocalMHA(dim=channels, window_size=attn_window_size)]

        for i, stride in enumerate(rates):
            input_dim = channels // (2**i)
            output_dim = channels // (2 ** (i + 1))
            groups = output_dim if depthwise else 1
            layers.append(
                DecoderBlock(input_dim, output_dim, stride, noise, groups=groups)
            )

        layers += [
            Snake1d(output_dim),
            WNConv1d(output_dim, d_out, kernel_size=7, padding=3),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*layers)

    def __call__(self, x):
        x = self.model(x)
        return x


class ResidualUnit(nn.Module):
    def __init__(self, dim=16, dilation=1, kernel=7, groups=1):
        super().__init__()
        pad = ((kernel - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            WNConv1d(
                dim,
                dim,
                kernel_size=kernel,
                dilation=dilation,
                padding=pad,
                groups=groups,
            ),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def __call__(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


class EncoderBlock(nn.Module):
    def __init__(self, output_dim=16, input_dim=None, stride=1, groups=1):
        super().__init__()
        input_dim = input_dim or output_dim // 2
        self.block = nn.Sequential(
            ResidualUnit(input_dim, dilation=1, groups=groups),
            ResidualUnit(input_dim, dilation=3, groups=groups),
            ResidualUnit(input_dim, dilation=9, groups=groups),
            Snake1d(input_dim),
            WNConv1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def __call__(self, x):
        return self.block(x)


class NoiseBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = WNConv1d(dim, dim, kernel_size=1, bias=False)

    def __call__(self, x):
        B, C, T = x.shape
        noise = mx.random.normal((B, 1, T))
        h = self.linear(x)
        n = noise * h
        x = x + n
        return x


class DecoderBlock(nn.Module):
    def __init__(self, input_dim=16, output_dim=8, stride=1, noise=False, groups=1):
        super().__init__()

        layers = [
            Snake1d(input_dim),
            WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
                output_padding=stride % 2,
            ),
        ]
        if noise:
            layers.append(NoiseBlock(output_dim))
        layers.extend(
            [
                ResidualUnit(output_dim, dilation=1, groups=groups),
                ResidualUnit(output_dim, dilation=3, groups=groups),
                ResidualUnit(output_dim, dilation=9, groups=groups),
            ]
        )
        self.block = nn.Sequential(*layers)

    def __call__(self, x):
        return self.block(x)


class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = mx.ones((1, channels, 1))

    def __call__(self, x):
        return snake(x, self.alpha.swapaxes(1, 2))
