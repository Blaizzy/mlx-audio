from typing import Union

import mlx.core as mx
import mlx.nn as nn


class LearnedPositionEmbeddings(nn.Module):
    """Learned position embeddings for T3 model."""

    def __init__(self, seq_len: int, model_dim: int, init: float = 0.02):
        super().__init__()
        self.emb = nn.Embedding(seq_len, model_dim)
        # Initialize with normal distribution (GPT-2 style)
        self.emb.weight = mx.random.normal(shape=(seq_len, model_dim), scale=init)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Returns positional embeddings for index 0 up to the length of x.

        Args:
            x: Input tensor of shape (B, T, ...)

        Returns:
            Positional embeddings of shape (T, model_dim)
        """
        sl = x.shape[1]
        return self.emb(mx.arange(sl))

    def get_fixed_embedding(self, idx: Union[int, mx.array]) -> mx.array:
        """
        Get positional embeddings for specific indices.

        Args:
            idx: Scalar int or integer array of shape (T,) or (B, T)

        Returns:
            Positional embeddings of shape (B, T, dim), or (1, 1, dim) for int input
        """
        if isinstance(idx, int):
            idx = mx.array([[idx]])
        elif idx.ndim == 1:
            idx = mx.expand_dims(idx, 0)

        assert idx.ndim == 2, f"Expected 2D array, got shape {idx.shape}"
        return self.emb(idx)  # (B, T, dim)
