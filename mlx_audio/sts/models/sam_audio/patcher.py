# Copyright (c) 2025 Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

import math
from typing import Tuple

import mlx.core as mx
import mlx.nn as nn


def pad1d(
    x: mx.array,
    paddings: Tuple[int, int],
    mode: str = "constant",
    value: float = 0.0,
) -> mx.array:
    """
    1D padding with support for reflect mode on small inputs.

    Args:
        x: Input tensor of shape (..., length)
        paddings: (left_pad, right_pad)
        mode: Padding mode ('constant' or 'reflect')
        value: Value for constant padding

    Returns:
        Padded tensor
    """
    length = x.shape[-1]
    padding_left, padding_right = paddings

    assert padding_left >= 0 and padding_right >= 0

    if mode == "reflect":
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = mx.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, extra_pad)])

        # Apply reflect padding
        padded = _reflect_pad_1d(x, padding_left, padding_right)

        if extra_pad > 0:
            padded = padded[..., : padded.shape[-1] - extra_pad]
        return padded
    else:
        # Constant padding
        pad_width = [(0, 0)] * (x.ndim - 1) + [(padding_left, padding_right)]
        return mx.pad(x, pad_width, constant_values=value)


def _reflect_pad_1d(x: mx.array, left: int, right: int) -> mx.array:
    """Apply reflect padding along the last dimension."""
    length = x.shape[-1]

    if left == 0 and right == 0:
        return x

    # Build reflected indices
    left_indices = mx.arange(left, 0, -1)
    middle_indices = mx.arange(length)
    right_indices = mx.arange(length - 2, length - 2 - right, -1)

    indices = mx.concatenate([left_indices, middle_indices, right_indices])
    return x[..., indices]


def get_extra_padding_for_conv1d(
    x: mx.array, kernel_size: int, stride: int, padding_total: int = 0
) -> int:
    """Calculate extra padding needed for conv1d to handle odd strides."""
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return int(ideal_length - length)


class Conv1d(nn.Module):
    """
    Conv1d with asymmetric padding to handle variable-length inputs.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

        # Weight shape: (out_channels, kernel_size, in_channels)
        scale = math.sqrt(1.0 / (in_channels * kernel_size))
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(out_channels, kernel_size, in_channels),
        )
        self.bias = mx.zeros((out_channels,)) if bias else None

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass with automatic padding.

        Args:
            x: Input of shape (batch, in_channels, length)

        Returns:
            Output of shape (batch, out_channels, new_length)
        """
        # Calculate effective kernel size with dilation
        effective_kernel = (self.kernel_size - 1) * self.dilation + 1
        padding_total = effective_kernel - self.stride

        extra_padding = get_extra_padding_for_conv1d(
            x, effective_kernel, self.stride, padding_total
        )

        # Asymmetric padding for odd strides
        padding_right = padding_total // 2
        padding_left = padding_total - padding_right

        x = pad1d(x, (padding_left, padding_right + extra_padding))

        # Transpose for MLX conv1d: (batch, length, channels)
        x = mx.transpose(x, (0, 2, 1))

        # Apply convolution
        out = mx.conv1d(
            x,
            self.weight,
            stride=self.stride,
            dilation=self.dilation,
        )

        # Transpose back: (batch, channels, length)
        out = mx.transpose(out, (0, 2, 1))

        if self.bias is not None:
            out = out + self.bias[:, None]

        return out


class ConvBlock1d(nn.Module):
    """Convolution block with GroupNorm and SiLU activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        num_groups: int = 8,
    ):
        super().__init__()
        self.groupnorm = nn.GroupNorm(num_groups, in_channels)
        self.activation = nn.SiLU()
        self.project = Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
        )

    def __call__(self, x: mx.array) -> mx.array:
        # GroupNorm expects (batch, length, channels)
        x_t = mx.transpose(x, (0, 2, 1))
        x_t = self.groupnorm(x_t)
        x = mx.transpose(x_t, (0, 2, 1))

        x = self.activation(x)
        return self.project(x)


class ResnetBlock1d(nn.Module):
    """Residual block with two ConvBlocks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        num_groups: int = 8,
    ):
        super().__init__()
        self.block1 = ConvBlock1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            num_groups=num_groups,
        )
        self.block2 = ConvBlock1d(
            in_channels=out_channels,
            out_channels=out_channels,
            num_groups=num_groups,
        )

        # Residual projection if dimensions differ
        if in_channels != out_channels:
            self.to_out = Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
            )
        else:
            self.to_out = None

    def __call__(self, x: mx.array) -> mx.array:
        h = self.block1(x)
        h = self.block2(h)

        if self.to_out is not None:
            x = self.to_out(x)

        return h + x


class Patcher(nn.Module):
    """
    Patcher module that applies ResNet block and reshapes output.

    This is used in the DiT to process the input features before
    passing through the transformer layers.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int,
    ):
        super().__init__()
        assert (
            out_channels % patch_size == 0
        ), f"out_channels ({out_channels}) must be divisible by patch_size ({patch_size})"
        self.patch_size = patch_size
        self.block = ResnetBlock1d(
            in_channels=in_channels,
            out_channels=out_channels // patch_size,
            num_groups=1,
        )

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, channels, length)

        Returns:
            Output tensor of shape (batch, channels * patch_size, length // patch_size)
        """
        x = self.block(x)
        # Rearrange: (b, c, l*p) -> (b, c*p, l)
        batch, channels, length = x.shape
        assert (
            length % self.patch_size == 0
        ), f"Length ({length}) must be divisible by patch_size ({self.patch_size})"
        new_length = length // self.patch_size

        # Reshape: (b, c, l, p) -> (b, c*p, l)
        x = x.reshape(batch, channels, new_length, self.patch_size)
        x = mx.transpose(x, (0, 1, 3, 2))  # (b, c, p, l)
        x = x.reshape(batch, channels * self.patch_size, new_length)

        return x
