"""Shared utilities for Voxtral TTS components."""

import mlx.core as mx
import mlx.nn as nn


def pad_to_multiple(x: int, multiple: int) -> int:
    """Round up to the nearest multiple."""
    return ((x + multiple - 1) // multiple) * multiple


class FeedForward(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, dim: int, hidden_dim: int, bias: bool = False):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))
