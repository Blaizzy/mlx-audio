from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .mossformerblock import MossFormerBlock


class MossFormerM2(nn.Module):
    """
    MLX implementation of MossFormerM2 transformer encoder.

    This class implements the transformer encoder using standard MossFormer blocks
    without the Gated FSMN components (previous version).

    Args:
        num_blocks (int): Number of mossformer blocks to include.
        d_model (int): The dimension of the input embedding.
        causal (bool, optional): True for causal / false for non causal. Default is False.
        group_size (int, optional): The chunk size. Default is 256.
        query_key_dim (int, optional): The attention vector dimension. Default is 128.
        expansion_factor (float, optional): The expansion factor for the linear projection in conv module. Default is 4.0.
        attn_dropout (float, optional): Dropout for the self-attention. Default is 0.1.

    Shape:
        - Input: (batch_size, sequence_length, d_model)
        - Output: (batch_size, sequence_length, d_model)

    Example:
        >>> import mlx.core as mx
        >>> x = mx.random.normal((8, 60, 512))
        >>> net = MossFormerM2(num_blocks=8, d_model=512)
        >>> output = net(x)
        >>> print(output.shape)  # (8, 60, 512)
    """

    def __init__(
        self,
        num_blocks: int,
        d_model: Optional[int] = None,
        causal: bool = False,
        group_size: int = 256,
        query_key_dim: int = 128,
        expansion_factor: float = 4.0,
        attn_dropout: float = 0.1,
    ):
        super().__init__()

        # Store parameters for weight loading
        self.num_blocks = num_blocks
        self.d_model = d_model
        self.causal = causal
        self.group_size = group_size
        self.query_key_dim = query_key_dim
        self.expansion_factor = expansion_factor
        self.attn_dropout = attn_dropout

        # Initialize the MossFormer blocks (without GFSMN)
        self.mossformerM = MossFormerBlock(
            dim=d_model,
            depth=num_blocks,
            group_size=group_size,
            query_key_dim=query_key_dim,
            expansion_factor=expansion_factor,
            causal=causal,
            attn_dropout=attn_dropout,
        )

        # Layer normalization
        self.norm = nn.LayerNorm(d_model, eps=1e-8)

    def __call__(self, src: mx.array) -> mx.array:
        """
        Forward pass through the MossFormerM2 model.

        Args:
            src (mx.array): Input tensor of shape (batch_size, sequence_length, d_model)

        Returns:
            mx.array: Output tensor of shape (batch_size, sequence_length, d_model)
        """
        # Apply MossFormer blocks
        output = self.mossformerM(src)

        # Apply layer normalization
        output = self.norm(output)

        return output
