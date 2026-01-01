# Original: ClearerVoice-Studio (github.com/modelscope/ClearerVoice-Studio)
# Copyright (c) Speech Lab, Alibaba Group
# Licensed under Apache License 2.0
# MLX port by Dmitry Starkov

import mlx.core as mx
import mlx.nn as nn

from .depthwise_conv1d_kernel import depthwise_conv1d


class ConvModule(nn.Module):
    """
    MLX implementation of ConvModule.

    This class provides identical behavior to the PyTorch version.

    ConvModule structure:
    - Transpose (1, 2): (B, T, C) -> (B, C, T)
    - DepthwiseConv1d: channels -> same channels with depthwise convolution
    - Transpose back + residual connection

    Args:
        in_channels (int): Number of input channels
        kernel_size (int): Convolution kernel size (default: 17)
        expansion_factor (int): Expansion factor (default: 2, currently unused)
        dropout_p (float): Dropout probability (default: 0.1, currently unused)
    """

    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 17,
        expansion_factor: int = 2,
        dropout_p: float = 0.1,
    ):
        super().__init__()

        # Validate inputs like PyTorch version
        assert (
            kernel_size - 1
        ) % 2 == 0, "kernel_size should be a odd number for 'SAME' padding"
        assert expansion_factor == 2, "Currently, Only Supports expansion_factor 2"

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2

        # Simple weight storage
        # Shape: (out_channels, kernel_size, in_channels) for MLX conv1d
        # For depthwise conv, out_channels = in_channels
        self.weight = mx.zeros((in_channels, kernel_size, 1))

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass for ConvModule exactly matching PyTorch behavior:
        inputs + self.sequential(inputs).transpose(1, 2)

        Where sequential is:
        1. Transpose(1, 2): (B, T, C) → (B, C, T)
        2. DepthwiseConv1d: convolution on (B, C, T)

        Args:
            x: Input tensor of shape (B, T, C)
        Returns:
            Output tensor of shape (B, T, C)
        """
        residual = x  # (B, T, C)

        conv_out = depthwise_conv1d(
            x, self.weight, stride=1, padding=self.padding, groups=self.in_channels
        )  # Output: (B, T, C)

        return residual + conv_out
