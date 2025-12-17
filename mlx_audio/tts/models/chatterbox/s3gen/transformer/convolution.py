from typing import Tuple

import mlx.core as mx
import mlx.nn as nn


class ConvolutionModule(nn.Module):
    """Convolution module for Conformer."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 15,
        activation: nn.Module = None,
        norm: str = "batch_norm",
        causal: bool = False,
        bias: bool = True,
    ):
        """
        Args:
            channels: Number of channels
            kernel_size: Kernel size of depthwise convolution
            activation: Activation function (defaults to SiLU)
            norm: Normalization type ('batch_norm' or 'layer_norm')
            causal: Whether to use causal convolution
            bias: Whether to use bias
        """
        super().__init__()

        # Pointwise expansion
        self.pointwise_conv1 = nn.Conv1d(
            channels,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )

        # Causal vs symmetric padding
        if causal:
            padding = 0
            self.lorder = kernel_size - 1
        else:
            assert (
                kernel_size - 1
            ) % 2 == 0, "kernel_size must be odd for symmetric conv"
            padding = (kernel_size - 1) // 2
            self.lorder = 0

        # Depthwise convolution (groups=channels for depthwise)
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=padding,
            groups=channels,
            bias=bias,
        )
        self.channels = channels
        self.kernel_size = kernel_size

        # Normalization
        assert norm in ["batch_norm", "layer_norm"]
        if norm == "batch_norm":
            self.use_layer_norm = False
            self.norm = nn.BatchNorm(channels)
        else:
            self.use_layer_norm = True
            self.norm = nn.LayerNorm(channels)

        # Pointwise compression
        self.pointwise_conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )

        self.activation = activation if activation is not None else nn.SiLU()

    def __call__(
        self,
        x: mx.array,
        mask_pad: mx.array = None,
        cache: mx.array = None,
    ) -> Tuple[mx.array, mx.array]:
        """
        Compute convolution module.

        Args:
            x: Input tensor (B, T, C)
            mask_pad: Padding mask (B, 1, T)
            cache: Left context cache for causal conv (B, C, cache_t)

        Returns:
            output: (B, T, C)
            new_cache: (B, C, lorder) for causal, or empty
        """
        # Transpose to (B, C, T)
        x = mx.swapaxes(x, 1, 2)

        # Apply mask
        if mask_pad is not None and mask_pad.shape[2] > 0:
            x = mx.where(mask_pad, x, 0.0)

        # Handle causal convolution caching
        if self.lorder > 0:
            if cache is None or cache.shape[2] == 0:
                # Pad on left for causal conv
                x = mx.pad(x, [(0, 0), (0, 0), (self.lorder, 0)])
            else:
                # Use cache
                x = mx.concatenate([cache, x], axis=2)
            new_cache = x[:, :, -self.lorder :]
        else:
            new_cache = mx.zeros((0, 0, 0))

        # GLU mechanism: pointwise expansion + gated linear unit
        x = self.pointwise_conv1(x)  # (B, 2C, T)
        x = nn.glu(x, axis=1)  # (B, C, T)

        # Depthwise convolution
        x = self.depthwise_conv(x)

        # Normalization
        if self.use_layer_norm:
            x = mx.swapaxes(x, 1, 2)  # (B, T, C)
            x = self.norm(x)
            x = self.activation(x)
            x = mx.swapaxes(x, 1, 2)  # (B, C, T)
        else:
            x = self.norm(x)
            x = self.activation(x)

        # Pointwise compression
        x = self.pointwise_conv2(x)

        # Apply mask again
        if mask_pad is not None and mask_pad.shape[2] > 0:
            x = mx.where(mask_pad, x, 0.0)

        # Transpose back to (B, T, C)
        return mx.swapaxes(x, 1, 2), new_cache
