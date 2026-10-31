# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
# Adapted for MLX from NVIDIA-NemotronLabs-RE-USE.
"""Channels-last convolutional encoder and magnitude/phase decoders."""

import mlx.core as mx
import mlx.nn as nn


def _apply(layers, x):
    for layer in layers:
        x = layer(x)
    return x


def _normalized(conv, dim):
    return [conv, nn.InstanceNorm(dim, affine=True), nn.PReLU(dim)]


class DenseBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dense_block = [
            _normalized(
                nn.Conv2d(
                    dim * (i + 1), dim, (3, 3), padding=(2**i, 1), dilation=(2**i, 1)
                ),
                dim,
            )
            for i in range(4)
        ]

    def __call__(self, x):
        skip = x
        for layers in self.dense_block:
            x = _apply(layers, skip)
            skip = mx.concatenate([x, skip], axis=-1)
        return x


class DenseEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config.hid_feature
        self.dense_conv_1 = _normalized(nn.Conv2d(config.input_channel, dim, 1), dim)
        self.dense_block = DenseBlock(dim)
        self.dense_conv_2 = _normalized(nn.Conv2d(dim, dim, (1, 3), stride=(4, 2)), dim)

    def __call__(self, x):
        x = self.dense_block(_apply(self.dense_conv_1, x))
        return _apply(self.dense_conv_2, x)


class SubpixelConv2d(nn.Module):
    def __init__(self, dim, factor):
        super().__init__()
        self.factor = factor
        self.conv = nn.Conv2d(dim, dim * factor, (1, 3), padding=(0, 1))

    def __call__(self, x):
        x = self.conv(x)
        batch, height, width, channels = x.shape
        # Upstream channel order is [factor, channels], not [channels, factor].
        x = x.reshape(batch, height, width, self.factor, channels // self.factor)
        return x.reshape(batch, height, width * self.factor, channels // self.factor)


class Decoder(nn.Module):
    def __init__(self, config, phase=False):
        super().__init__()
        dim = config.hid_feature
        self.phase = phase
        self.dense_block = DenseBlock(dim)
        self.up_conv1 = _normalized(SubpixelConv2d(dim, 2), dim)
        self.up_conv2 = _normalized(SubpixelConv2d(dim, 4), dim)
        if phase:
            self.phase_conv_r = nn.Conv2d(dim, config.output_channel, 1)
            self.phase_conv_i = nn.Conv2d(dim, config.output_channel, 1)
        else:
            self.final_conv = nn.Conv2d(dim, config.output_channel, 1)

    def __call__(self, x):
        x = _apply(self.up_conv1, self.dense_block(x))
        x = _apply(self.up_conv2, x.transpose(0, 2, 1, 3)).transpose(0, 2, 1, 3)
        if self.phase:
            return mx.arctan2(self.phase_conv_i(x), self.phase_conv_r(x))
        return self.final_conv(x)
