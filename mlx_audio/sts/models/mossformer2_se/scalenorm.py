# Original: ClearerVoice-Studio (github.com/modelscope/ClearerVoice-Studio)
# Copyright (c) Speech Lab, Alibaba Group
# Licensed under Apache License 2.0
# MLX port by Dmitry Starkov

import mlx.core as mx
import mlx.nn as nn


class ScaleNorm(nn.Module):
    """MLX implementation of ScaleNorm.

    ScaleNorm implements a scaled normalization technique for neural network layers.
    It computes the L2 norm along the last dimension and applies learnable scaling.

    Arguments:
        dim: Dimension of the input features (used to calculate scale factor)
        eps: Small value to prevent division by zero (default: 1e-8)
    """

    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.dim = dim
        self.scale = dim**-0.5  # Calculate scale factor: 1/sqrt(dim)
        self.eps = eps  # Small value for numerical stability

        # Initialize learnable scaling parameter
        self.g = mx.ones((1,))

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass for the ScaleNorm layer.

        Args:
            x: Input tensor of any shape where the last dimension has size `dim`

        Returns:
            Scaled and normalized output tensor of the same shape as input
        """
        # Compute L2 norm along the last dimension
        norm = mx.sqrt(mx.sum(x * x, axis=-1, keepdims=True)) * self.scale

        # Clamp norm to prevent division by zero
        norm = mx.maximum(norm, self.eps)

        # Normalize and scale with fused operation
        return x * (self.g / norm)
