"""SemanticEncoder for HiggsAudioV2 — processes HuBERT output before fusion.

Architecture: Conv1d + N blocks of (2 ResidualUnits + Conv1d).
All convolutions use ELU activation. Strides are typically [1,1] (no downsampling).

Weight keys: encoder_semantic.conv.*, encoder_semantic.conv_blocks.N.*
"""

from typing import List

import mlx.core as mx
import mlx.nn as nn


class SemanticResidualUnit(nn.Module):
    """ELU + dilated Conv1d + ELU + 1x1 Conv1d, with skip connection."""

    def __init__(self, dim: int, dilation: int = 1, kernel_size: int = 3):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.conv1: nn.Conv1d = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=pad,
            dilation=dilation,
            bias=False,
        )
        self.conv2: nn.Conv1d = nn.Conv1d(dim, dim, kernel_size=1, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        y = nn.elu(x)
        y = self.conv1(y)
        y = nn.elu(y)
        y = self.conv2(y)
        return x + y


class SemanticConvBlock(nn.Module):
    """2 residual units followed by a strided convolution."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        stride: int = 1,
        dilation: int = 1,
        kernel_size: int = 3,
        unit_kernel_size: int = 3,
    ):
        super().__init__()
        self.res_units: List[SemanticResidualUnit] = [
            SemanticResidualUnit(
                in_dim, dilation=dilation, kernel_size=unit_kernel_size
            ),
            SemanticResidualUnit(
                in_dim, dilation=dilation, kernel_size=unit_kernel_size
            ),
        ]
        pad = (kernel_size - 1) // 2
        self.conv: nn.Conv1d = nn.Conv1d(
            in_dim,
            out_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=pad,
            bias=True,
        )

    def __call__(self, x: mx.array) -> mx.array:
        for ru in self.res_units:
            x = ru(x)
        x = self.conv(x)
        return x


class SemanticEncoder(nn.Module):
    """Processes HuBERT semantic features before fusion with acoustic features.

    Input:  [B, T, hidden_size]  (channels-last, MLX convention)
    Output: [B, T, hidden_size]  (channels-last)
    """

    def __init__(
        self,
        hidden_size: int = 768,
        strides: List[int] | None = None,
        dilations: List[int] | None = None,
        channel_ratios: List[int] | None = None,
        kernel_size: int = 3,
        unit_kernel_size: int = 3,
    ):
        super().__init__()
        if strides is None:
            strides = [1, 1]
        if dilations is None:
            dilations = [1, 1]
        if channel_ratios is None:
            channel_ratios = [1, 1]
        pad = (kernel_size - 1) // 2
        self.conv: nn.Conv1d = nn.Conv1d(
            hidden_size, hidden_size, kernel_size=kernel_size, padding=pad, bias=False
        )
        self.conv_blocks: List[SemanticConvBlock] = [
            SemanticConvBlock(
                in_dim=hidden_size * ratio,
                out_dim=hidden_size * ratio,
                stride=s,
                dilation=d,
                kernel_size=kernel_size,
                unit_kernel_size=unit_kernel_size,
            )
            for s, d, ratio in zip(strides, dilations, channel_ratios)
        ]

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv(x)
        for block in self.conv_blocks:
            x = block(x)
        return x
