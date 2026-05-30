from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .config import FSMNEncoderConfig


class FSMNBlock(nn.Module):
    """
    FSMN memory block: causal depthwise conv + residual (inside block).

    FunASR logic: output = input + conv_left(padded_input)
    """

    def __init__(self, proj_dim: int, lorder: int, lstride: int = 1):
        super().__init__()
        self.proj_dim = proj_dim
        self.lorder = lorder
        self.lstride = lstride
        self.pad_left = (lorder - 1) * lstride
        self.conv_left = nn.Conv1d(
            in_channels=proj_dim,
            out_channels=proj_dim,
            kernel_size=lorder,
            stride=1,
            groups=proj_dim,
            bias=False,
        )

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: [batch, time, proj_dim]
        Returns:
            x + memory: [batch, time, proj_dim]
        """
        # [batch, time, proj_dim] → [batch, proj_dim, time] for padding
        x_t = mx.transpose(x, axes=(0, 2, 1))
        # Causal left padding
        x_padded = mx.pad(x_t, pad_width=[(0, 0), (0, 0), (self.pad_left, 0)])
        # [batch, proj_dim, time+pad] → [batch, time+pad, proj_dim] for Conv1d
        x_padded = mx.transpose(x_padded, axes=(0, 2, 1))
        # Depthwise conv: [batch, time, proj_dim]
        y_left = self.conv_left(x_padded)
        # Residual inside block: output = input + conv(input)
        return x + y_left


class FSMNLayer(nn.Module):
    """
    One FSMN BasicBlock (from FunASR):
        x1 = linear(input)           # project down, no bias
        x2 = fsmn_block(x1)          # x1 + conv(x1), residual inside
        x3 = affine(x2)              # project up, with bias
        x4 = relu(x3)
        return x4                    # NO outer skip connection
    """

    def __init__(self, linear_dim: int, proj_dim: int, lorder: int, lstride: int = 1):
        super().__init__()
        self.linear = nn.Linear(linear_dim, proj_dim, bias=False)
        self.fsmn_block = FSMNBlock(proj_dim, lorder, lstride)
        self.affine = nn.Linear(proj_dim, linear_dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: [batch, time, linear_dim]
        Returns:
            [batch, time, linear_dim]
        """
        x1 = self.linear(x)
        x2 = self.fsmn_block(x1)  # includes internal residual
        x3 = self.affine(x2)
        x4 = nn.relu(x3)
        return x4


class FSMNEncoder(nn.Module):
    """
    Full FSMN encoder (matches FunASR FSMN class):

        x = in_linear1(input)         # AffineTransform, NO relu
        x = in_linear2(x)             # AffineTransform, NO relu
        x = relu(x)                   # single relu
        x = fsmn_stack(x)             # 4x BasicBlock
        x = out_linear1(x)            # AffineTransform, NO relu
        x = out_linear2(x)            # AffineTransform, NO relu
        x = softmax(x)
    """

    def __init__(self, config: FSMNEncoderConfig):
        super().__init__()
        self.config = config

        self.in_linear1 = nn.Linear(
            config.input_dim, config.input_affine_dim, bias=True
        )
        self.in_linear2 = nn.Linear(
            config.input_affine_dim, config.linear_dim, bias=True
        )

        self.fsmn = [
            FSMNLayer(config.linear_dim, config.proj_dim, config.lorder, config.lstride)
            for _ in range(config.fsmn_layers)
        ]

        self.out_linear1 = nn.Linear(
            config.linear_dim, config.output_affine_dim, bias=True
        )
        self.out_linear2 = nn.Linear(
            config.output_affine_dim, config.output_dim, bias=True
        )

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: [batch, time, input_dim] (400-dim)
        Returns:
            [batch, time, output_dim] (248-dim softmax)
        """
        x = self.in_linear1(x)  # no relu
        x = self.in_linear2(x)  # no relu
        x = nn.relu(x)  # single relu after in_linear2

        for layer in self.fsmn:
            x = layer(x)

        x = self.out_linear1(x)  # no relu
        x = self.out_linear2(x)  # no relu
        x = mx.softmax(x, axis=-1)
        return x
