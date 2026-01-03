from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .flash_sharea_ffconvm import FLASH_ShareA_FFConvM
from .scalenorm import ScaleNorm


class MossFormerBlock(nn.Module):
    """
    MLX implementation of MossFormer Block with attention mechanisms.

    This block is designed to process input sequences using attention
    layers and incorporates rotary positional embeddings. It allows
    for configurable normalization types and can handle causal
    attention.

    Args:
        dim (int): Dimensionality of the input.
        depth (int): Number of attention layers in the block.
        group_size (int, optional): Size of groups for normalization. Default is 256.
        query_key_dim (int, optional): Dimension of the query and key in attention. Default is 128.
        expansion_factor (float, optional): Expansion factor for feedforward layers. Default is 4.
        causal (bool, optional): If True, enables causal attention. Default is False.
        attn_dropout (float, optional): Dropout rate for attention layers. Default is 0.1.
        norm_type (str, optional): Type of normalization to use ('scalenorm' or 'layernorm'). Default is 'scalenorm'.
        shift_tokens (bool, optional): If True, shifts tokens in the attention layer. Default is True.
    """

    def __init__(
        self,
        *,
        dim: int,
        depth: int,
        group_size: int = 256,
        query_key_dim: int = 128,
        expansion_factor: float = 4.0,
        causal: bool = False,
        attn_dropout: float = 0.1,
        norm_type: str = "scalenorm",
        shift_tokens: bool = True,
    ):
        super().__init__()

        # Store parameters
        self.dim = dim
        self.depth = depth
        self.group_size = group_size
        self.query_key_dim = query_key_dim
        self.expansion_factor = expansion_factor
        self.causal = causal
        self.attn_dropout = attn_dropout
        self.norm_type = norm_type
        self.shift_tokens = shift_tokens

        # Ensure normalization type is valid
        assert norm_type in (
            "scalenorm",
            "layernorm",
        ), "norm_type must be one of scalenorm or layernorm"

        # Select normalization class based on the provided type
        if norm_type == "scalenorm":
            norm_klass = ScaleNorm
        elif norm_type == "layernorm":
            norm_klass = nn.LayerNorm

        # Rotary positional embedding for attention
        rotary_pos_emb = nn.RoPE(
            dims=min(32, query_key_dim), traditional=False, base=10000
        )

        # Create a list of attention layers using FLASH_ShareA_FFConvM
        self.layers = []
        for _ in range(depth):
            layer = FLASH_ShareA_FFConvM(
                dim=dim,
                group_size=group_size,
                query_key_dim=query_key_dim,
                expansion_factor=expansion_factor,
                causal=causal,
                dropout=attn_dropout,
                rotary_pos_emb=rotary_pos_emb,
                norm_klass=norm_klass,
                shift_tokens=shift_tokens,
            )
            self.layers.append(layer)

    def __call__(self, x: mx.array, *, mask: Optional[mx.array] = None) -> mx.array:
        """
        Forward pass for the MossFormer Block.

        Args:
            x (mx.array): Input tensor of shape (batch_size, seq_len, dim).
            mask (mx.array, optional): Mask tensor for attention operations.

        Returns:
            mx.array: Output tensor after processing through the block.
        """
        # Process input through each attention layer
        for layer in self.layers:
            x = layer(x, mask=mask)  # Apply attention layer with optional mask

        return x  # Return the final output tensor
