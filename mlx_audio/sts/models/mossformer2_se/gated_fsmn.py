import mlx.core as mx
import mlx.nn as nn

from .ffconvm import FFConvM
from .unideepfsmn import UniDeepFsmn


class Gated_FSMN(nn.Module):
    """
    MLX implementation of Gated_FSMN with mathematical equivalence to PyTorch.

    Gated Frequency Selective Memory Network (FSMN) that combines two feedforward
    convolutional networks with a frequency selective memory module.

    The gated FSMN uses a gating mechanism to selectively combine:
    1. A feedforward branch (to_u) processed through FSMN for temporal memory
    2. A feedforward branch (to_v) used as the gate
    3. The original input as a residual connection

    The operation is: output = gate * memory + input
    where gate = to_v(input) and memory = fsmn(to_u(input))

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        lorder (int): Order of the filter for FSMN (memory span)
        hidden_size (int): Number of hidden units in the network

    Inputs:
        - **x** (batch, time, in_channels): Input tensor

    Returns:
        - **output** (batch, time, out_channels): Output tensor after gated FSMN operations
    """

    def __init__(
        self, in_channels: int, out_channels: int, lorder: int, hidden_size: int
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lorder = lorder
        self.hidden_size = hidden_size

        # Feedforward network for the first branch (u)
        # This branch will be processed through FSMN for temporal memory
        self.to_u = FFConvM(
            dim_in=in_channels,
            dim_out=hidden_size,
            norm_klass=nn.LayerNorm,
            dropout=0.1,
        )

        # Feedforward network for the second branch (v)
        # This branch acts as the gate
        self.to_v = FFConvM(
            dim_in=in_channels,
            dim_out=hidden_size,
            norm_klass=nn.LayerNorm,
            dropout=0.1,
        )

        # Frequency selective memory network
        # Following PyTorch implementation exactly
        self.fsmn = UniDeepFsmn(in_channels, out_channels, lorder, hidden_size)

        # Track training state for dropout consistency
        self._training = True

    def eval(self):
        """Set model to evaluation mode."""
        self._training = False
        self.to_u.eval()
        self.to_v.eval()
        return self

    def train(self):
        """Set model to training mode."""
        self._training = True
        self.to_u.train()
        self.to_v.train()
        return self

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass for the Gated FSMN.

        The gated FSMN performs the following operations:
        1. Process input through first branch (to_u)
        2. Process input through second branch (to_v) - acts as gate
        3. Apply FSMN to first branch output for temporal memory
        4. Combine: output = gate * memory + input (residual connection)

        Args:
            x (mx.array): Input tensor of shape (batch, time, in_channels)

        Returns:
            mx.array: Output tensor of shape (batch, time, out_channels)
        """
        # Store original input for residual connection
        input_residual = x

        # Process input through both branches
        x_u = self.to_u(x)  # First branch - will be processed through FSMN

        x_v = self.to_v(x)  # Second branch - acts as gate

        # Apply FSMN to the first branch for temporal memory
        x_u = self.fsmn(x_u)

        # Gated combination with residual connection
        # Following PyTorch: x = x_v * x_u + input
        x = x_v * x_u + input_residual

        return x
