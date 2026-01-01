# Original: ClearerVoice-Studio (github.com/modelscope/ClearerVoice-Studio)
# Copyright (c) Speech Lab, Alibaba Group
# Licensed under Apache License 2.0
# MLX port by Dmitry Starkov

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .mossformerblock_gfsmn import MossFormerBlock_GFSMN


class MossFormerM(nn.Module):
    """
    MLX implementation of MossFormerM transformer encoder based on MossFormer2 layers.

    This class implements the transformer encoder using MossFormer2 layers with
    Gated FSMN blocks for enhanced sequence processing.

    Args:
        num_blocks (int): Number of mossformer2 blocks to include.
        d_model (int): The dimension of the input embedding.
        causal (bool, optional): True for causal / false for non causal. Default is False.
        group_size (int, optional): The chunk size for segmenting sequence. Default is 256.
        query_key_dim (int, optional): The attention vector dimension. Default is 128.
        expansion_factor (float, optional): The expansion factor for the linear projection in conv module. Default is 4.0.
        attn_dropout (float, optional): Dropout for the self-attention. Default is 0.1.

    Shape:
        - Input: (batch_size, sequence_length, d_model)
        - Output: (batch_size, sequence_length, d_model)

    Example:
        >>> import mlx.core as mx
        >>> x = mx.random.normal((8, 60, 512))
        >>> net = MossFormerM(num_blocks=8, d_model=512)
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

        # Initialize the MossFormer blocks with GFSMN
        self.mossformerM = MossFormerBlock_GFSMN(
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
        Forward pass through the MossFormerM model.

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

    def eval(self):
        """Set the model to evaluation mode."""
        # Set MossFormerBlock to eval mode
        if hasattr(self.mossformerM, "eval"):
            self.mossformerM.eval()
        return self

    def train(self):
        """Set the model to training mode."""
        # Set MossFormerBlock to train mode
        if hasattr(self.mossformerM, "train"):
            self.mossformerM.train()
        return self
