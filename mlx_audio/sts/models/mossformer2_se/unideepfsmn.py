# Original: ClearerVoice-Studio (github.com/modelscope/ClearerVoice-Studio)
# Copyright (c) Speech Lab, Alibaba Group
# Licensed under Apache License 2.0
# MLX port by Dmitry Starkov

from typing import Optional

import mlx.core as mx
import mlx.nn as nn


class UniDeepFsmn(nn.Module):
    """
    MLX implementation of UniDeepFsmn with 1:1 mathematical equivalence to PyTorch.

    UniDeepFsmn is a neural network module that implements a single-deep feedforward
    sequence memory network (FSMN) for temporal sequence modeling.

    Args:
        input_dim (int): Dimension of the input features
        output_dim (int): Dimension of the output features
        lorder (int): Length order for convolution layers (memory span)
        hidden_size (int): Number of hidden units in the linear layer

    Inputs: input
        - **input** (batch, time, input_dim): Tensor containing input sequences (MLX format)

    Returns: output
        - **output** (batch, time, output_dim): Enhanced tensor with temporal memory
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        lorder: Optional[int] = None,
        hidden_size: Optional[int] = None,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        if lorder is None:
            # Early return for incomplete initialization (matching PyTorch behavior)
            return

        self.lorder = lorder
        self.hidden_size = hidden_size if hidden_size is not None else output_dim

        # Initialize layers (matching PyTorch architecture)
        self.linear = nn.Linear(input_dim, self.hidden_size)
        self.project = nn.Linear(self.hidden_size, output_dim, bias=False)

        # Conv2d matching PyTorch exactly
        # Weight shape (256, 39, 1, 1) with groups=256 means:
        # - out_channels = 256
        # - in_channels_per_group = 39
        # - kernel_size = (1, 1)
        # - total_in_channels = 256 * 39 = 9984

        # But we only have 256 input channels, so we need groups=1 and kernel_size=(39,1)
        kernel_size = (lorder + lorder - 1, 1)  # (39, 1)
        self.kernel_size = kernel_size

        # Use MLX's Conv2d to match PyTorch's Conv2d with grouped convolution
        # MLX expects NHWC format while PyTorch uses NCHW
        self.conv1 = nn.Conv2d(
            in_channels=output_dim,  # 256
            out_channels=output_dim,  # 256
            kernel_size=kernel_size,  # (39, 1)
            stride=(1, 1),
            padding=(0, 0),
            groups=output_dim,  # groups=output_dim for depthwise convolution
            bias=False,
        )

    def __call__(self, input_tensor: mx.array) -> mx.array:
        """
        Forward pass for the UniDeepFsm model.

        Args:
            input_tensor (mx.array): Input tensor of shape (batch, time, input_dim)

        Returns:
            mx.array: Output tensor of shape (batch, time, output_dim)
        """
        batch_size, time_steps, input_dim = input_tensor.shape

        # 1. Linear transformation with ReLU activation
        f1 = mx.maximum(self.linear(input_tensor), 0)  # ReLU

        # 2. Project to output dimension
        p1 = self.project(f1)  # Shape: (batch, time, output_dim)

        # 3. Apply grouped convolution (equivalent to PyTorch Conv2d with groups=output_dim)
        # PyTorch uses Conv2d with kernel [lorder+lorder-1, 1] where the convolution is on time dimension

        # Add width dimension (not height) to match PyTorch's 4D tensor
        # PyTorch: (batch, channels, time, 1) -> permute to (batch, time, 1, channels)
        x = mx.expand_dims(p1, axis=2)  # Shape: (batch, time, 1, output_dim)

        # MLX Conv2d expects NHWC format where:
        # N=batch, H=time (where convolution happens), W=1, C=channels

        # Causal padding: pad with (lorder-1) on both sides for the time dimension
        padding_left = self.lorder - 1
        padding_right = self.lorder - 1

        # Pad the H dimension (axis=1) which is the time dimension in our NHWC format
        y = mx.pad(x, [(0, 0), (padding_left, padding_right), (0, 0), (0, 0)])

        # Apply Conv2d (MLX expects NHWC)
        out = self.conv1(y)

        # Add residual connection: x + conv_output
        out = x + out

        # Remove the width dimension to return to 3D
        enhanced_features = mx.squeeze(out, axis=2)  # Shape: (batch, time, output_dim)

        # 5. Final residual connection with original input
        # This only works if input_dim == output_dim, as in the PyTorch version
        if self.input_dim == self.output_dim:
            return input_tensor + enhanced_features
        else:
            # If dimensions don't match, return enhanced features only
            return enhanced_features
