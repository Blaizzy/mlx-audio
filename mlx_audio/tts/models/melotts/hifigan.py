"""HiFi-GAN vocoder for MeloTTS."""

from typing import List

import mlx.core as mx
import mlx.nn as nn


LRELU_SLOPE = 0.1


def get_padding(kernel_size, dilation=1):
    return (kernel_size * dilation - dilation) // 2


class Conv1dPT(nn.Module):
    """Conv1d accepting PyTorch-style (B, C, T) input."""

    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, groups=1
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding,
            dilation=dilation, bias=bias, groups=groups,
        )

    def __call__(self, x):
        x = x.transpose(0, 2, 1)
        x = self.conv(x)
        return x.transpose(0, 2, 1)


class ConvTranspose1dPT(nn.Module):
    """ConvTranspose1d accepting PyTorch-style (B, C, T) input."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv_t = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )

    def __call__(self, x):
        x = x.transpose(0, 2, 1)
        x = self.conv_t(x)
        return x.transpose(0, 2, 1)


class ResBlock1(nn.Module):
    """Multi-receptive-field residual block (type 1)."""

    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = [
            Conv1dPT(
                channels,
                channels,
                kernel_size,
                dilation=d,
                padding=get_padding(kernel_size, d),
            )
            for d in dilation
        ]
        self.convs2 = [
            Conv1dPT(
                channels,
                channels,
                kernel_size,
                dilation=1,
                padding=get_padding(kernel_size, 1),
            )
            for _ in dilation
        ]

    def __call__(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = nn.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = nn.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x


class ResBlock2(nn.Module):
    """Multi-receptive-field residual block (type 2)."""

    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super().__init__()
        self.convs = [
            Conv1dPT(
                channels,
                channels,
                kernel_size,
                dilation=d,
                padding=get_padding(kernel_size, d),
            )
            for d in dilation
        ]

    def __call__(self, x):
        for c in self.convs:
            xt = nn.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x


class Generator(nn.Module):
    """HiFi-GAN generator: upsamples latent features to audio waveform."""

    def __init__(
        self,
        initial_channel: int,
        resblock: str,
        resblock_kernel_sizes: List[int],
        resblock_dilation_sizes: List[List[int]],
        upsample_rates: List[int],
        upsample_initial_channel: int,
        upsample_kernel_sizes: List[int],
        gin_channels: int = 0,
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        self.conv_pre = Conv1dPT(
            initial_channel, upsample_initial_channel, 7, padding=3
        )

        ResBlockClass = ResBlock1 if resblock == "1" else ResBlock2

        self.ups = []
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            self.ups.append(
                ConvTranspose1dPT(
                    upsample_initial_channel // (2**i),
                    ch,
                    k,
                    stride=u,
                    padding=(k - u) // 2,
                )
            )

        self.resblocks = []
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(ResBlockClass(ch, k, d))

        self.conv_post = Conv1dPT(ch, 1, 7, padding=3, bias=False)

        if gin_channels != 0:
            self.cond = Conv1dPT(gin_channels, upsample_initial_channel, 1)
        else:
            self.cond = None

    def __call__(self, x, g=None):
        x = self.conv_pre(x)

        if g is not None and self.cond is not None:
            x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = nn.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)

            xs = None
            for j in range(self.num_kernels):
                block_out = self.resblocks[i * self.num_kernels + j](x)
                xs = block_out if xs is None else xs + block_out
            x = xs / self.num_kernels

        x = nn.leaky_relu(x, LRELU_SLOPE)
        x = self.conv_post(x)
        x = mx.tanh(x)
        return x
