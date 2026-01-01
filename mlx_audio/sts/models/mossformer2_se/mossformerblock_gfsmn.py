import mlx.core as mx
import mlx.nn as nn

from .flash_sharea_ffconvm import FLASH_ShareA_FFConvM
from .gated_fsmn_block import Gated_FSMN_Block
from .scalenorm import ScaleNorm


class MossFormerBlock_GFSMN(nn.Module):
    """
    MLX implementation of MossFormerBlock_GFSMN with 1:1 mathematical equivalence to PyTorch.

    MossFormer Block with Gated FSMN combines attention mechanisms and gated FSMN layers
    to process input sequences.

    Args:
        dim (int): Dimensionality of the input
        depth (int): Number of layers in the block
        group_size (int): Size of the groups for normalization (default: 256)
        query_key_dim (int): Dimension of the query and key in attention (default: 128)
        expansion_factor (float): Expansion factor for feedforward layers (default: 4.0)
        causal (bool): If True, enables causal attention (default: False)
        attn_dropout (float): Dropout rate for attention layers (default: 0.1)
        norm_type (str): Type of normalization to use ('scalenorm' or 'layernorm') (default: 'scalenorm')
        shift_tokens (bool): If True, shifts tokens in the attention layer (default: True)

    Shape:
        - Input: (batch_size, sequence_length, dim)
        - Output: (batch_size, sequence_length, dim)
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

        assert norm_type in (
            "scalenorm",
            "layernorm",
        ), "norm_type must be one of scalenorm or layernorm"

        if norm_type == "scalenorm":
            norm_klass = ScaleNorm
        elif norm_type == "layernorm":
            norm_klass = nn.LayerNorm

        self.group_size = group_size
        self.depth = depth

        # Rotary positional embedding
        rotary_pos_emb = nn.RoPE(
            dims=min(32, query_key_dim), traditional=False, base=10000
        )

        # Create a list of Gated FSMN blocks
        self.fsmn = [
            Gated_FSMN_Block(
                dim=dim, inner_channels=256, group_size=group_size, norm_type=norm_type
            )
            for _ in range(depth)
        ]

        # Create a list of attention layers using FLASH_ShareA_FFConvM
        self.layers = [
            FLASH_ShareA_FFConvM(
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
            for _ in range(depth)
        ]

    def __call__(self, x, mask=None):
        """
        Forward pass for the MossFormer Block with Gated FSMN.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, dim)
            mask: Optional mask tensor for attention operations

        Returns:
            Output tensor after processing through the block
        """

        for i in range(self.depth):
            x = self.layers[i](x, mask=mask)

            x = self.fsmn[i](x)

        return x

    def eval(self):
        """Set the model to evaluation mode."""
        # Set all FLASH layers to eval mode
        for layer in self.layers:
            layer.eval()

        # Set all FSMN layers to eval mode
        for fsmn in self.fsmn:
            fsmn.eval()

        return self

    def train(self):
        """Set the model to training mode."""
        # Set all FLASH layers to train mode
        for layer in self.layers:
            layer.train()

        # Set all FSMN layers to train mode
        for fsmn in self.fsmn:
            fsmn.train()

        return self
