# Original: ClearerVoice-Studio (github.com/modelscope/ClearerVoice-Studio)
# Copyright (c) Speech Lab, Alibaba Group
# Licensed under Apache License 2.0
# MLX port by Dmitry Starkov

import mlx.core as mx
import mlx.nn as nn

from .convmodule import ConvModule
from .scalenorm import ScaleNorm


class FFConvM(nn.Module):
    """
    MLX implementation of FFConvM.

    This class provides identical behavior to the PyTorch version.

    FFConvM structure: norm -> linear -> silu -> conv_module -> dropout

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension
        norm_klass: Normalization class (default: nn.LayerNorm)
        dropout (float): Dropout probability (default: 0.1)
    """

    def __init__(
        self, dim_in: int, dim_out: int, norm_klass=nn.LayerNorm, dropout: float = 0.1
    ):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

        # Sequential structure matching PyTorch: norm, linear, silu, conv_module, dropout
        # Index 0: normalization
        if norm_klass == nn.LayerNorm:
            self.norm = nn.LayerNorm(dim_in)
        elif norm_klass == ScaleNorm:
            self.norm = ScaleNorm(dim_in)
        else:
            # Default to ScaleNorm since that's what the weights expect
            self.norm = ScaleNorm(dim_in)

        # Index 1: linear transformation
        self.linear = nn.Linear(dim_in, dim_out)

        # Index 2: SiLU activation (no parameters)

        # Index 3: ConvModule
        self.conv_module = ConvModule(dim_out)

        # Index 4: Dropout (not used)
        self.dropout_rate = dropout
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass for FFConvM.

        Args:
            x (mx.array): Input tensor of shape (batch, time, dim_in)

        Returns:
            mx.array: Output tensor of shape (batch, time, dim_out)
        """
        # Follow PyTorch FFConvM.mdl sequential structure
        x = self.norm(x)  # Index 0
        x = self.linear(x)  # Index 1

        x = nn.silu(x)  # Index 2
        x = self.conv_module(x)  # Index 3
        # x = self.dropout(x)       # Index 4 - Commented out for inference only

        return x

    def eval(self):
        """Set model to evaluation mode."""
        # Disable dropout
        self.dropout.p = 0.0
        # Also set eval mode for submodules if they have it
        if hasattr(self.conv_module, "eval"):
            self.conv_module.eval()
        return self

    def train(self):
        """Set model to training mode."""
        # Restore dropout rate
        self.dropout.p = self.dropout_rate if hasattr(self, "dropout_rate") else 0.1
        # Also set train mode for submodules if they have it
        if hasattr(self.conv_module, "train"):
            self.conv_module.train()
        return self
