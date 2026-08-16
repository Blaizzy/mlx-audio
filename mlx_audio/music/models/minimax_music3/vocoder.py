"""DAC-style Flow-VAE decoder for MiniMax Music 3."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from .config import ModelConfig


class Snake1d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.alpha = mx.ones((1, channels, 1))

    def __call__(self, x_ncl: mx.array) -> mx.array:
        alpha = self.alpha.astype(x_ncl.dtype)
        return x_ncl + mx.sin(alpha * x_ncl) ** 2 / (alpha + 1e-9)


class ResidualUnit(nn.Module):
    def __init__(self, dim: int, dilation: int):
        super().__init__()
        padding = (7 - 1) * dilation // 2
        self.snake1 = Snake1d(dim)
        self.conv1 = nn.Conv1d(
            dim, dim, kernel_size=7, dilation=dilation, padding=padding
        )
        self.snake2 = Snake1d(dim)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=1)

    def __call__(self, x_ncl: mx.array) -> mx.array:
        y = self.conv1(self.snake1(x_ncl).transpose(0, 2, 1))
        y = self.conv2(self.snake2(y.transpose(0, 2, 1)).transpose(0, 2, 1))
        return x_ncl + y.transpose(0, 2, 1)


class VocoderBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, stride: int):
        super().__init__()
        self.snake1 = Snake1d(input_dim)
        self.conv_t1 = nn.ConvTranspose1d(
            input_dim,
            output_dim,
            kernel_size=2 * stride,
            stride=stride,
            padding=math.ceil(stride / 2),
        )
        self.res_unit1 = ResidualUnit(output_dim, dilation=1)
        self.res_unit2 = ResidualUnit(output_dim, dilation=3)
        self.res_unit3 = ResidualUnit(output_dim, dilation=9)

    def __call__(self, x_ncl: mx.array) -> mx.array:
        y = self.conv_t1(self.snake1(x_ncl).transpose(0, 2, 1)).transpose(0, 2, 1)
        return self.res_unit3(self.res_unit2(self.res_unit1(y)))


class Vocoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        latent_half = config.dit_in_channels // 2
        self.dec_in_proj = nn.Conv1d(
            latent_half, config.vocoder_input_dim, kernel_size=1
        )
        self.conv_in = nn.Conv1d(
            config.vocoder_input_dim,
            config.vocoder_hidden_dim,
            kernel_size=7,
            padding=3,
        )
        blocks = []
        output_dim = config.vocoder_hidden_dim
        for index, stride in enumerate(config.vocoder_upsampling_ratios):
            input_dim = config.vocoder_hidden_dim // (2**index)
            output_dim = config.vocoder_hidden_dim // (2 ** (index + 1))
            blocks.append(VocoderBlock(input_dim, output_dim, stride))
        self.blocks = blocks
        self.snake_out = Snake1d(output_dim)
        self.conv_out = nn.Conv1d(output_dim, 1, kernel_size=7, padding=3)

    def __call__(self, latents: mx.array) -> mx.array:
        batch, _, length = latents.shape
        half = self.config.dit_in_channels // 2
        hidden = latents.reshape(batch * 2, half, length)
        hidden = self.dec_in_proj(hidden.transpose(0, 2, 1))
        hidden = self.conv_in(hidden).transpose(0, 2, 1)
        for block in self.blocks:
            hidden = block(hidden)
        wave = mx.tanh(
            self.conv_out(self.snake_out(hidden).transpose(0, 2, 1))
        ).transpose(0, 2, 1)
        return wave.reshape(batch, 2, -1)
