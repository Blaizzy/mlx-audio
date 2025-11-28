# Ported from https://github.com/resemble-ai/chatterbox

import math
from typing import Tuple, Union

import mlx.core as mx
import mlx.nn as nn


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding.

    PE(pos, 2i)   = sin(pos/(10000^(2i/d_model)))
    PE(pos, 2i+1) = cos(pos/(10000^(2i/d_model)))
    """

    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 5000):
        """
        Args:
            d_model: Embedding dimension
            dropout_rate: Dropout rate
            max_len: Maximum input length
        """
        super().__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout_rate = dropout_rate
        self.max_len = max_len

        # Precompute positional encodings
        self.pe = self._create_pe(max_len, d_model)

    def _create_pe(self, max_len: int, d_model: int) -> mx.array:
        """Create positional encoding matrix."""
        pe = mx.zeros((max_len, d_model))
        position = mx.arange(0, max_len, dtype=mx.float32)[:, None]
        div_term = mx.exp(
            mx.arange(0, d_model, 2, dtype=mx.float32) * -(math.log(10000.0) / d_model)
        )

        pe_sin = mx.sin(position * div_term)
        pe_cos = mx.cos(position * div_term)

        # Interleave sin and cos
        pe = mx.zeros((max_len, d_model))
        pe = pe.at[:, 0::2].add(pe_sin)
        pe = pe.at[:, 1::2].add(pe_cos)

        return mx.expand_dims(pe, 0)  # (1, max_len, d_model)

    def __call__(self, x: mx.array, offset: int = 0) -> Tuple[mx.array, mx.array]:
        """
        Add positional encoding.

        Args:
            x: Input tensor (B, T, D)
            offset: Position offset

        Returns:
            x: Encoded input (B, T, D)
            pos_emb: Positional embeddings (1, T, D)
        """
        pos_emb = self.position_encoding(offset, x.shape[1])
        x = x * self.xscale + pos_emb

        if self.training and self.dropout_rate > 0:
            x = nn.Dropout(self.dropout_rate)(x)
            pos_emb = nn.Dropout(self.dropout_rate)(pos_emb)

        return x, pos_emb

    def position_encoding(self, offset: int, size: int) -> mx.array:
        """Get positional encoding for a range."""
        assert offset + size <= self.max_len
        return self.pe[:, offset : offset + size, :]


class RelPositionalEncoding(PositionalEncoding):
    """
    Relative positional encoding module.

    See Appendix B in https://arxiv.org/abs/1901.02860

    Unlike PositionalEncoding, this does NOT add pos_emb to input.
    The pos_emb is returned separately for use in relative attention.
    """

    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 5000):
        super().__init__(d_model, dropout_rate, max_len)

    def __call__(self, x: mx.array, offset: int = 0) -> Tuple[mx.array, mx.array]:
        """
        Compute positional encoding.

        Args:
            x: Input tensor (B, T, D)
            offset: Position offset

        Returns:
            x: Scaled input (B, T, D) - NOT added to pos_emb
            pos_emb: Positional embeddings (1, T, D)
        """
        x = x * self.xscale
        pos_emb = self.position_encoding(offset, x.shape[1])

        if self.training and self.dropout_rate > 0:
            x = nn.Dropout(self.dropout_rate)(x)
            pos_emb = nn.Dropout(self.dropout_rate)(pos_emb)

        return x, pos_emb


class EspnetRelPositionalEncoding(nn.Module):
    """
    Relative positional encoding module (ESPnet implementation).

    This version computes both positive and negative position encodings
    for bidirectional relative attention.

    See https://github.com/espnet/espnet/pull/2816
    and Appendix B in https://arxiv.org/abs/1901.02860
    """

    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout_rate = dropout_rate
        self.max_len = max_len

        # Initialize PE - will be extended as needed
        self.pe = None
        self._extend_pe(max_len)

    def _extend_pe(self, size: int):
        """Create or extend positional encodings."""
        if self.pe is not None and self.pe.shape[1] >= size * 2 - 1:
            return

        # Create positive and negative position encodings
        position = mx.arange(0, size, dtype=mx.float32)[:, None]
        div_term = mx.exp(
            mx.arange(0, self.d_model, 2, dtype=mx.float32)
            * -(math.log(10000.0) / self.d_model)
        )

        # Positive positions
        pe_positive_sin = mx.sin(position * div_term)
        pe_positive_cos = mx.cos(position * div_term)
        pe_positive = mx.zeros((size, self.d_model))
        pe_positive = pe_positive.at[:, 0::2].add(pe_positive_sin)
        pe_positive = pe_positive.at[:, 1::2].add(pe_positive_cos)

        # Negative positions
        pe_negative_sin = mx.sin(-1 * position * div_term)
        pe_negative_cos = mx.cos(-1 * position * div_term)
        pe_negative = mx.zeros((size, self.d_model))
        pe_negative = pe_negative.at[:, 0::2].add(pe_negative_sin)
        pe_negative = pe_negative.at[:, 1::2].add(pe_negative_cos)

        # Reverse positive and concatenate
        # pe_positive: [0, 1, 2, ...] -> reversed: [..., 2, 1, 0]
        pe_positive = mx.expand_dims(mx.flip(pe_positive, axis=0), 0)
        # pe_negative: skip first (which is 0) -> [1, 2, 3, ...]
        pe_negative = mx.expand_dims(pe_negative[1:], 0)

        # Concatenate: [..., 2, 1, 0, 1, 2, ...]
        self.pe = mx.concatenate([pe_positive, pe_negative], axis=1)

    def __call__(self, x: mx.array, offset: int = 0) -> Tuple[mx.array, mx.array]:
        """
        Add positional encoding.

        Args:
            x: Input tensor (B, T, D)
            offset: Position offset (not used in this implementation)

        Returns:
            x: Scaled input (B, T, D)
            pos_emb: Positional embeddings (1, 2*T-1, D)
        """
        self._extend_pe(x.shape[1])
        x = x * self.xscale
        pos_emb = self.position_encoding(x.shape[1], offset)

        if self.training and self.dropout_rate > 0:
            x = nn.Dropout(self.dropout_rate)(x)
            pos_emb = nn.Dropout(self.dropout_rate)(pos_emb)

        return x, pos_emb

    def position_encoding(self, size: int, offset: int = 0) -> mx.array:
        """
        Get positional encoding for relative attention.

        Args:
            size: Required size
            offset: Position offset (not used)

        Returns:
            pos_emb: Positional embeddings (1, 2*size-1, d_model)
        """
        # Extract from center: positions from -(size-1) to +(size-1)
        center = self.pe.shape[1] // 2
        start = center - size + 1
        end = center + size
        return self.pe[:, start:end, :]


class NoPositionalEncoding(nn.Module):
    """No positional encoding - returns zeros."""

    def __init__(self, d_model: int, dropout_rate: float):
        super().__init__()
        self.d_model = d_model
        self.dropout_rate = dropout_rate

    def __call__(self, x: mx.array, offset: int = 0) -> Tuple[mx.array, mx.array]:
        """Return input unchanged with zero positional embeddings."""
        pos_emb = mx.zeros((1, x.shape[1], self.d_model))

        if self.training and self.dropout_rate > 0:
            x = nn.Dropout(self.dropout_rate)(x)

        return x, pos_emb

    def position_encoding(self, offset: int, size: int) -> mx.array:
        """Return zero positional encoding."""
        return mx.zeros((1, size, self.d_model))
