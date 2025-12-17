from typing import Tuple

import mlx.core as mx
import mlx.nn as nn


class BaseSubsampling(nn.Module):
    """Base class for subsampling modules."""

    def __init__(self):
        super().__init__()
        self.right_context = 0
        self.subsampling_rate = 1

    def position_encoding(self, offset: int, size: int) -> mx.array:
        """Get position encoding from pos_enc module."""
        return self.pos_enc.position_encoding(offset, size)


class LinearNoSubsampling(BaseSubsampling):
    """
    Linear transform the input without subsampling.

    Used in UpsampleConformerEncoder.
    """

    def __init__(
        self, idim: int, odim: int, dropout_rate: float, pos_enc_class: nn.Module
    ):
        """
        Args:
            idim: Input dimension
            odim: Output dimension
            dropout_rate: Dropout rate
            pos_enc_class: Positional encoding module
        """
        super().__init__()
        self.linear = nn.Linear(idim, odim)
        self.norm = nn.LayerNorm(odim, eps=1e-5)
        self.dropout_rate = dropout_rate
        self.pos_enc = pos_enc_class
        self.right_context = 0
        self.subsampling_rate = 1

    def __call__(
        self, x: mx.array, x_mask: mx.array, offset: int = 0
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """
        Apply linear transformation without subsampling.

        Args:
            x: Input tensor (B, T, idim)
            x_mask: Input mask (B, 1, T)
            offset: Position offset

        Returns:
            x: Output tensor (B, T, odim)
            pos_emb: Positional embeddings
            x_mask: Output mask (B, 1, T) - unchanged
        """
        x = self.linear(x)
        x = self.norm(x)

        if self.training and self.dropout_rate > 0:
            x = nn.Dropout(self.dropout_rate)(x)

        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask
