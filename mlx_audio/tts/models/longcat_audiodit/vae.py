"""WAV-VAE audio autoencoder for AudioDiT.

Encodes raw 24kHz audio to latent space (dim=64) and decodes back.
All convolutions use pre-reconstructed weight-normalized weights (handled in sanitize).
Data flows in channels-last format: (B, L, C).
"""

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .config import VaeConfig

# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------


class SnakeBeta(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.alpha = mx.zeros((channels,))
        self.beta = mx.zeros((channels,))

    def __call__(self, x: mx.array) -> mx.array:
        alpha = mx.exp(self.alpha)
        beta = mx.exp(self.beta)
        return x + (1.0 / (beta + 1e-9)) * mx.power(mx.sin(x * alpha), 2)


class ELU(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        return nn.elu(x)


class Identity(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        return x


def _get_activation(use_snake: bool, channels: int = 0):
    return SnakeBeta(channels) if use_snake else ELU()


# ---------------------------------------------------------------------------
# Conv layers (weights pre-reconstructed from weight_norm in sanitize)
# ---------------------------------------------------------------------------


class Conv1d(nn.Module):
    """Conv1d operating on channels-last (B, L, C) data."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.weight = mx.zeros((out_channels, kernel_size, in_channels))
        if bias:
            self.bias = mx.zeros((out_channels,))

    def __call__(self, x: mx.array) -> mx.array:
        y = mx.conv1d(
            x,
            self.weight,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )
        if "bias" in self:
            y = y + self.bias
        return y


class ConvTranspose1d(nn.Module):
    """ConvTranspose1d operating on channels-last (B, L, C) data."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.weight = mx.zeros((out_channels, kernel_size, in_channels))
        if bias:
            self.bias = mx.zeros((out_channels,))

    def __call__(self, x: mx.array) -> mx.array:
        y = mx.conv_transpose1d(
            x, self.weight, stride=self.stride, padding=self.padding
        )
        if "bias" in self:
            y = y + self.bias
        return y


# ---------------------------------------------------------------------------
# Pixel shuffle / unshuffle (channels-last)
# ---------------------------------------------------------------------------


def pixel_unshuffle_1d(x: mx.array, factor: int) -> mx.array:
    """(B, W, C) -> (B, W//factor, C*factor)"""
    B, W, C = x.shape
    x = x.reshape(B, W // factor, factor, C)
    x = x.transpose(0, 1, 3, 2)  # (B, W//f, C, f)
    return x.reshape(B, W // factor, C * factor)


def pixel_shuffle_1d(x: mx.array, factor: int) -> mx.array:
    """(B, W, C) -> (B, W*factor, C//factor)"""
    B, W, C = x.shape
    c = C // factor
    x = x.reshape(B, W, c, factor)
    x = x.transpose(0, 1, 3, 2)  # (B, W, f, c)
    return x.reshape(B, W * factor, c)


# ---------------------------------------------------------------------------
# Shortcut modules
# ---------------------------------------------------------------------------


class DownsampleShortcut(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, factor: int):
        super().__init__()
        self.factor = factor
        self.group_size = in_channels * factor // out_channels
        self.out_channels = out_channels

    def __call__(self, x: mx.array) -> mx.array:
        x = pixel_unshuffle_1d(x, self.factor)
        B, N, _ = x.shape
        return x.reshape(B, N, self.out_channels, self.group_size).mean(axis=3)


class UpsampleShortcut(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, factor: int):
        super().__init__()
        self.factor = factor
        self.repeats = out_channels * factor // in_channels

    def __call__(self, x: mx.array) -> mx.array:
        x = mx.repeat(x, self.repeats, axis=2)
        return pixel_shuffle_1d(x, self.factor)


# ---------------------------------------------------------------------------
# Residual unit
# ---------------------------------------------------------------------------


class VaeResidualUnit(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation: int,
        kernel_size: int = 7,
        use_snake: bool = True,
    ):
        super().__init__()
        padding = (dilation * (kernel_size - 1)) // 2
        self.layers = [
            _get_activation(use_snake, out_channels),
            Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                dilation=dilation,
                padding=padding,
            ),
            _get_activation(use_snake, out_channels),
            Conv1d(out_channels, out_channels, kernel_size=1),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        h = x
        for layer in self.layers:
            h = layer(h)
        return x + h


# ---------------------------------------------------------------------------
# Encoder / Decoder blocks
# ---------------------------------------------------------------------------


class VaeEncoderBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int,
        use_snake: bool = True,
        downsample_shortcut: str = "none",
    ):
        super().__init__()
        self.layers = [
            VaeResidualUnit(in_ch, in_ch, dilation=1, use_snake=use_snake),
            VaeResidualUnit(in_ch, in_ch, dilation=3, use_snake=use_snake),
            VaeResidualUnit(in_ch, in_ch, dilation=9, use_snake=use_snake),
            _get_activation(use_snake, in_ch),
            Conv1d(
                in_ch,
                out_ch,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        ]
        self.res = (
            DownsampleShortcut(in_ch, out_ch, stride)
            if downsample_shortcut == "averaging"
            else None
        )

    def __call__(self, x: mx.array) -> mx.array:
        h = x
        for layer in self.layers:
            h = layer(h)
        if self.res is not None:
            return h + self.res(x)
        return h


class VaeDecoderBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int,
        use_snake: bool = True,
        upsample_shortcut: str = "none",
    ):
        super().__init__()
        self.layers = [
            _get_activation(use_snake, in_ch),
            ConvTranspose1d(
                in_ch,
                out_ch,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
            VaeResidualUnit(out_ch, out_ch, dilation=1, use_snake=use_snake),
            VaeResidualUnit(out_ch, out_ch, dilation=3, use_snake=use_snake),
            VaeResidualUnit(out_ch, out_ch, dilation=9, use_snake=use_snake),
        ]
        self.res = (
            UpsampleShortcut(in_ch, out_ch, stride)
            if upsample_shortcut == "duplicating"
            else None
        )

    def __call__(self, x: mx.array) -> mx.array:
        h = x
        for layer in self.layers:
            h = layer(h)
        if self.res is not None:
            return h + self.res(x)
        return h


# ---------------------------------------------------------------------------
# Encoder / Decoder
# ---------------------------------------------------------------------------


class VaeEncoder(nn.Module):
    def __init__(self, config: VaeConfig):
        super().__init__()
        c_mults = [1] + config.c_mults
        ch = config.channels
        self.layers = [
            Conv1d(config.in_channels, c_mults[0] * ch, kernel_size=7, padding=3)
        ]
        for i in range(len(c_mults) - 1):
            self.layers.append(
                VaeEncoderBlock(
                    c_mults[i] * ch,
                    c_mults[i + 1] * ch,
                    config.strides[i],
                    use_snake=config.use_snake,
                    downsample_shortcut=config.downsample_shortcut,
                )
            )
        self.layers.append(
            Conv1d(
                c_mults[-1] * ch, config.encoder_latent_dim, kernel_size=3, padding=1
            )
        )

        if config.out_shortcut == "averaging":
            self.shortcut = DownsampleShortcut(
                c_mults[-1] * ch, config.encoder_latent_dim, 1
            )
        else:
            self.shortcut = None

    def __call__(self, x: mx.array) -> mx.array:
        if self.shortcut is None:
            for layer in self.layers:
                x = layer(x)
            return x
        for layer in self.layers[:-1]:
            x = layer(x)
        return self.layers[-1](x) + self.shortcut(x)


class VaeDecoder(nn.Module):
    def __init__(self, config: VaeConfig):
        super().__init__()
        c_mults = [1] + config.c_mults
        ch = config.channels

        if config.in_shortcut == "duplicating":
            self.shortcut = UpsampleShortcut(config.latent_dim, c_mults[-1] * ch, 1)
        else:
            self.shortcut = None

        self.layers = [
            Conv1d(config.latent_dim, c_mults[-1] * ch, kernel_size=7, padding=3)
        ]
        for i in range(len(c_mults) - 1, 0, -1):
            self.layers.append(
                VaeDecoderBlock(
                    c_mults[i] * ch,
                    c_mults[i - 1] * ch,
                    config.strides[i - 1],
                    use_snake=config.use_snake,
                    upsample_shortcut=config.upsample_shortcut,
                )
            )
        self.layers.append(_get_activation(config.use_snake, c_mults[0] * ch))
        self.layers.append(
            Conv1d(
                c_mults[0] * ch,
                config.in_channels,
                kernel_size=7,
                padding=3,
                bias=False,
            )
        )
        self.layers.append(Identity())  # placeholder for final_tanh=False

    def __call__(self, x: mx.array) -> mx.array:
        if self.shortcut is None:
            for layer in self.layers:
                x = layer(x)
            return x
        x_short = self.shortcut(x) + self.layers[0](x)
        for layer in self.layers[1:]:
            x_short = layer(x_short)
        return x_short


# ---------------------------------------------------------------------------
# WAV-VAE
# ---------------------------------------------------------------------------


class AudioDiTVae(nn.Module):
    def __init__(self, config: VaeConfig):
        super().__init__()
        self.config = config
        self.encoder = VaeEncoder(config)
        self.decoder = VaeDecoder(config)
        self.scale = config.scale
        self.downsampling_ratio = config.downsampling_ratio

    def encode(self, audio: mx.array) -> mx.array:
        """Encode audio (B, L, 1) to latent (B, T, latent_dim)."""
        latents = self.encoder(audio)
        mean, scale_param = mx.split(latents, 2, axis=-1)
        stdev = nn.softplus(scale_param) + 1e-4
        latents = mx.random.normal(mean.shape) * stdev + mean
        return latents / self.scale

    def decode(self, latents: mx.array) -> mx.array:
        """Decode latent (B, T, latent_dim) to audio (B, L, 1)."""
        z = latents * self.scale
        return self.decoder(z)
