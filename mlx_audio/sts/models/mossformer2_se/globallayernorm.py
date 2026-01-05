import mlx.core as mx
import mlx.nn as nn


class GlobalLayerNorm(nn.Module):
    """MLX implementation of Global Layer Normalization.

    This class calculates Global Layer Normalization, providing identical
    behavior to the PyTorch version for both 3D and 4D tensors.

    Arguments:
        dim: Input dimension size
        shape: Number of dimensions (3 or 4)
        eps: Small value for numerical stability (default: 1e-8)
        elementwise_affine: Whether to use learnable affine parameters (default: True)
    """

    def __init__(
        self, dim: int, shape: int, eps: float = 1e-8, elementwise_affine: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.shape = shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            if shape == 3:
                # Weight and bias for 3D tensors: initialize as [dim, 1] to match PyTorch
                # NPZ weights may be (dim,) or (dim, 1) - we'll handle both in loading
                self.weight = mx.ones((self.dim, 1))
                self.bias = mx.zeros((self.dim, 1))
            elif shape == 4:
                # Weight and bias for 4D tensors: initialize as [dim, 1, 1] to match PyTorch
                # NPZ weights may be (dim,) or (dim, 1) - we'll handle both in loading
                self.weight = mx.ones((self.dim, 1, 1))
                self.bias = mx.zeros((self.dim, 1, 1))
            else:
                raise ValueError(
                    f"Unsupported shape: {shape}. Only 3 and 4 are supported."
                )
        else:
            self.weight = None
            self.bias = None

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass for Global Layer Normalization.

        Args:
            x: Input tensor of size [N, C, L] for 3D or [N, C, K, S] for 4D

        Returns:
            Normalized tensor of the same shape as input
        """
        if x.ndim == 3:
            # For 3D tensors: normalize over dimensions (1, 2)
            # Input is [B, C, L] in PyTorch convention
            mean = mx.mean(x, axis=(1, 2), keepdims=True)
            var = mx.mean((x - mean) ** 2, axis=(1, 2), keepdims=True)

            if self.elementwise_affine:
                # Optimize broadcasting for weight and bias
                # Reshape weight and bias once for efficient broadcasting
                w = mx.reshape(
                    self.weight.squeeze() if self.weight.ndim > 1 else self.weight,
                    (1, -1, 1),
                )
                b = mx.reshape(
                    self.bias.squeeze() if self.bias.ndim > 1 else self.bias, (1, -1, 1)
                )
                # Fuse normalization and affine transformation
                x = w * (x - mean) / mx.sqrt(var + self.eps) + b
            else:
                x = (x - mean) / mx.sqrt(var + self.eps)

        elif x.ndim == 4:
            # For 4D tensors: normalize over dimensions (1, 2, 3)
            mean = mx.mean(x, axis=(1, 2, 3), keepdims=True)
            var = mx.mean((x - mean) ** 2, axis=(1, 2, 3), keepdims=True)

            if self.elementwise_affine:
                # Optimize broadcasting for weight and bias
                # Reshape weight and bias once for efficient broadcasting
                w = mx.reshape(
                    self.weight.squeeze() if self.weight.ndim > 1 else self.weight,
                    (1, -1, 1, 1),
                )
                b = mx.reshape(
                    self.bias.squeeze() if self.bias.ndim > 1 else self.bias,
                    (1, -1, 1, 1),
                )
                # Fuse normalization and affine transformation
                x = w * (x - mean) / mx.sqrt(var + self.eps) + b
            else:
                x = (x - mean) / mx.sqrt(var + self.eps)
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got {x.ndim}D tensor")

        return x
