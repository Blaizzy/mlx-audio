from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .mossformerm import MossFormerM
from .mossformerm2 import MossFormerM2


class Computation_Block(nn.Module):
    """
    MLX implementation of Computation_Block for dual-path processing.

    This class implements the computation block that contains MossFormer layers
    with normalization and skip connections for intra-chunk processing.

    Args:
        num_blocks (int): Number of MossFormer blocks.
        out_channels (int): Dimensionality of model output.
        norm (str, optional): Normalization type ('ln' for layer norm, None for no norm). Default is 'ln'.
        skip_around_intra (bool, optional): Skip connection around the intra layer. Default is True.
        use_mossformer2 (bool, optional): Use MossFormerM2 instead of MossFormerM. Default is False.

    Shape:
        - Input: (batch_size, out_channels, sequence_length)
        - Output: (batch_size, out_channels, sequence_length)

    Example:
        >>> import mlx.core as mx
        >>> x = mx.random.normal((10, 64, 100))
        >>> comp_block = Computation_Block(num_blocks=2, out_channels=64)
        >>> output = comp_block(x)
        >>> print(output.shape)  # (10, 64, 100)
    """

    def __init__(
        self,
        num_blocks: int,
        out_channels: int,
        norm: Optional[str] = "ln",
        skip_around_intra: bool = True,
        use_mossformer2: bool = False,
    ):
        super().__init__()

        # Store parameters for weight loading
        self.num_blocks = num_blocks
        self.out_channels = out_channels
        self.norm = norm
        self.skip_around_intra = skip_around_intra
        self.use_mossformer2 = use_mossformer2

        # Initialize the appropriate MossFormer model
        if use_mossformer2:
            # Previous MossFormer model
            self.intra_mdl = MossFormerM2(num_blocks=num_blocks, d_model=out_channels)
        else:
            # Default MossFormer2 model
            self.intra_mdl = MossFormerM(num_blocks=num_blocks, d_model=out_channels)

        # Initialize normalization layer
        self.intra_norm = None
        if norm is not None:
            if norm == "ln":
                # In PyTorch, norm="ln" is GroupNorm(1, dim, eps=1e-8)
                self.intra_norm = nn.GroupNorm(
                    1, out_channels, eps=1e-8, affine=True, pytorch_compatible=True
                )
            # Add other normalization types as needed

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass through the Computation_Block.

        Args:
            x (mx.array): Input tensor of shape (batch_size, out_channels, sequence_length)

        Returns:
            mx.array: Output tensor of shape (batch_size, out_channels, sequence_length)
        """
        B, N, S = x.shape

        # Convert to NLC format for MossFormer processing
        intra = mx.transpose(x, (0, 2, 1))  # [B, S, N]

        # Apply MossFormer model (operates in NLC format)
        intra = self.intra_mdl(intra)

        # Apply normalization if specified (GroupNorm expects NLC format)
        if self.norm is not None and self.intra_norm is not None:
            intra = self.intra_norm(intra)

        # Convert back to NCL format
        intra = mx.transpose(intra, (0, 2, 1))  # [B, N, S]

        # Apply skip connection if enabled
        if self.skip_around_intra:
            intra = intra + x

        return intra

    def eval(self):
        """Set the model to evaluation mode."""
        # Set all sub-modules to eval mode
        if hasattr(self.intra_mdl, "eval"):
            self.intra_mdl.eval()
        return self

    def train(self):
        """Set the model to training mode."""
        # Set all sub-modules to train mode
        if hasattr(self.intra_mdl, "train"):
            self.intra_mdl.train()
        return self
