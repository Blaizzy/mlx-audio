import mlx.core as mx
import mlx.nn as nn

from .gated_fsmn import Gated_FSMN


class CLayerNorm(nn.Module):
    """
    MLX implementation of CLayerNorm (Channel-wise Layer Normalization).

    This class applies layer normalization along the channel dimension.
    Unlike the PyTorch version which expects [N, C, T], this MLX version
    works directly with [N, T, C] format to avoid unnecessary transpositions.

    Args:
        normalized_shape: Input shape from last dimension
        eps: Small value for numerical stability (default: 1e-8)
        elementwise_affine: Whether to use learnable affine parameters (default: True)

    Shape:
        - Input: [batch_size, sequence_length, channels]
        - Output: [batch_size, sequence_length, channels]
    """

    def __init__(self, normalized_shape, eps=1e-8, elementwise_affine=True):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = mx.ones((normalized_shape,))
            self.bias = mx.zeros((normalized_shape,))
        else:
            self.weight = None
            self.bias = None

    def __call__(self, x):
        """Forward pass applying channel-wise layer normalization."""
        if x.ndim != 3:
            raise RuntimeError(
                f"CLayerNorm only accepts 3-D tensor as input, got {x.ndim}D"
            )

        # x is already in [N, T, C] format
        # Apply LayerNorm along the channel dimension (last dimension)
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        x = (x - mean) / mx.sqrt(var + self.eps)

        if self.elementwise_affine:
            x = x * self.weight + self.bias

        return x


class Gated_FSMN_Block(nn.Module):
    """
    MLX implementation of Gated_FSMN_Block with 1:1 mathematical equivalence to PyTorch.

    A 1-D convolutional block that incorporates a gated FSMN.
    This block consists of:
    1. Conv1d layer with PReLU activation
    2. CLayerNorm normalization
    3. Gated FSMN module
    4. Another CLayerNorm
    5. Final Conv1d projection
    6. Residual connection

    Args:
        dim (int): Dimensionality of the input/output
        inner_channels (int): Number of channels in the inner layers (default: 256)
        group_size (int): Size of the groups for normalization (default: 256)
        norm_type (str): Type of normalization to use ('scalenorm' or 'layernorm')

    Shape:
        - Input: [batch_size, seq_length, dim]
        - Output: [batch_size, seq_length, dim]
    """

    def __init__(self, dim, inner_channels=256, group_size=256, norm_type="scalenorm"):
        super().__init__()

        self.dim = dim
        self.inner_channels = inner_channels
        self.group_size = group_size
        self.norm_type = norm_type

        # First convolutional layer
        self.conv1 = nn.Conv1d(
            in_channels=dim, out_channels=inner_channels, kernel_size=1, bias=True
        )

        # PReLU activation
        self.prelu = nn.PReLU(num_parameters=1, init=0.25)

        # Normalization layers
        self.norm1 = CLayerNorm(inner_channels)
        self.norm2 = CLayerNorm(inner_channels)

        # Gated FSMN
        self.gated_fsmn = Gated_FSMN(
            in_channels=inner_channels,
            out_channels=inner_channels,
            lorder=20,
            hidden_size=inner_channels,
        )

        # Final convolutional layer
        self.conv2 = nn.Conv1d(
            in_channels=inner_channels, out_channels=dim, kernel_size=1, bias=True
        )

        # Track training mode
        self._training = True

    def eval(self):
        """Set model to evaluation mode."""
        self._training = False
        self.gated_fsmn.eval()
        return self

    def train(self):
        """Set model to training mode."""
        self._training = True
        self.gated_fsmn.train()
        return self

    def __call__(self, x):
        """
        Forward pass for the Gated FSMN Block.

        Args:
            x: Input tensor of shape [batch_size, seq_length, dim]

        Returns:
            Output tensor of shape [batch_size, seq_length, dim]
        """
        residual = x

        # First convolution - input is already [B, T, D]
        x = self.conv1(x)

        # PReLU activation
        x = self.prelu(x)

        # First normalization (now accepts [B, T, C] directly)
        x = self.norm1(x)

        # Gated FSMN (expects [B, T, C])
        x = self.gated_fsmn(x)

        # Second normalization (now accepts [B, T, C] directly)
        x = self.norm2(x)

        # Final convolution
        x = self.conv2(x)

        # Residual connection
        return x + residual
