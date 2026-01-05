from typing import List

import mlx.core as mx
import mlx.nn as nn


class OffsetScale(nn.Module):
    """MLX implementation of OffsetScale.

    OffsetScale applies learned offsets and scales to the input tensor for multiple heads.
    It performs element-wise scaling and offset operations and returns a list of tensors,
    one for each head.

    Arguments:
        dim: Dimension of the input features
        heads: Number of heads (default: 1)
    """

    def __init__(self, dim: int, heads: int = 1):
        super().__init__()
        self.dim = dim
        self.heads = heads

        # Initialize scale parameters (gamma) with normal distribution
        self.gamma = mx.random.normal((heads, dim), scale=0.02) + 1.0

        # Initialize offset parameters (beta) with zeros
        self.beta = mx.zeros((heads, dim))

    def __call__(self, x: mx.array) -> List[mx.array]:
        """Forward pass for the OffsetScale layer.

        Args:
            x: Input tensor of shape (..., dim)

        Returns:
            List of tensors with applied offsets and scales for each head.
            Each tensor has shape (..., dim)
        """
        # Apply scaling and offsets using einsum-like operation
        # PyTorch: einsum('... d, h d -> ... h d', x, self.gamma) + self.beta

        # Expand x to include head dimension: (..., dim) -> (..., 1, dim)
        x_expanded = mx.expand_dims(x, axis=-2)

        # Broadcast multiplication with gamma: (..., 1, dim) * (heads, dim) -> (..., heads, dim)
        scaled = x_expanded * self.gamma

        # Add beta offset: (..., heads, dim) + (heads, dim) -> (..., heads, dim)
        out = scaled + self.beta

        # Unbind along the head dimension to create a list of tensors
        # PyTorch: out.unbind(dim=-2)
        head_outputs = []
        for h in range(self.heads):
            head_output = out[..., h, :]  # Extract head h: (..., dim)
            head_outputs.append(head_output)

        return head_outputs
