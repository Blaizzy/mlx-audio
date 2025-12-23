# Copyright (c) 2025 Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) for transformer attention.

    This implementation precomputes the rotation matrices and applies them
    efficiently during the forward pass.
    """

    def __init__(
        self,
        theta: float,
        head_dim: int,
        max_seqlen: int = 1024,
        scale_factor: int = 1,
        low_freq_factor: int = 1,
        high_freq_factor: int = 32,
        old_context_len: int = 8192,
    ):
        super().__init__()
        self.theta = theta
        self.head_dim = head_dim
        self.max_seqlen = max_seqlen
        self.scale_factor = scale_factor
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor
        self.old_context_len = old_context_len

        if scale_factor != 1:
            self.low_freq_wavelen = old_context_len / low_freq_factor
            self.high_freq_wavelen = old_context_len / high_freq_factor
            assert self.low_freq_wavelen >= self.high_freq_wavelen

        # Precompute rotation matrices
        self._freqs_cis = None
        self.reset_parameters()

    def reset_parameters(self):
        """Precompute rotation matrices."""
        self._freqs_cis = self._precompute_freqs_cis(
            dim=self.head_dim,
            end=self.max_seqlen,
            theta=self.theta,
        )

    def _apply_scaling(self, freqs: mx.array) -> mx.array:
        """Apply frequency scaling for context extension."""
        if self.scale_factor == 1:
            return freqs

        new_freqs = []
        for i in range(freqs.shape[0]):
            freq = freqs[i].item()
            wavelen = 2 * math.pi / freq
            if wavelen < self.high_freq_wavelen:
                new_freqs.append(freq)
            elif wavelen > self.low_freq_wavelen:
                new_freqs.append(freq / self.scale_factor)
            else:
                smooth = (self.old_context_len / wavelen - self.low_freq_factor) / (
                    self.high_freq_factor - self.low_freq_factor
                )
                new_freqs.append(
                    (1 - smooth) * freq / self.scale_factor + smooth * freq
                )
        return mx.array(new_freqs, dtype=freqs.dtype)

    def _precompute_freqs_cis(
        self,
        dim: int,
        end: int,
        theta: float = 10000.0,
    ) -> mx.array:
        """
        Precompute frequency tensor for complex exponentials (cis) with rotation matrices.

        Uses the same 2x2 rotation matrix approach as PyTorch SAM-Audio.

        Args:
            dim: Dimension of the head
            end: Maximum sequence length
            theta: Base for the frequency computation

        Returns:
            Precomputed frequency tensor with shape (1, end, 1, dim//2, 2, 2)
        """
        freqs = 1.0 / (
            theta ** (mx.arange(0, dim, 2, dtype=mx.float32)[: (dim // 2)] / dim)
        )
        freqs = self._apply_scaling(freqs)

        t = mx.arange(end, dtype=mx.float32)
        freqs = mx.outer(t, freqs)  # (end, dim//2)

        cos = mx.cos(freqs)  # (end, dim//2)
        sin = mx.sin(freqs)  # (end, dim//2)

        # Create 2x2 rotation matrix: [[cos, -sin], [sin, cos]]
        # Stack as (cos, -sin, sin, cos) and reshape to (..., 2, 2)
        freqs_cis = mx.stack([cos, -sin, sin, cos], axis=-1)
        freqs_cis = freqs_cis.reshape(*freqs.shape, 2, 2)  # (end, dim//2, 2, 2)

        # Add batch and head dims: (end, dim//2, 2, 2) -> (1, end, 1, dim//2, 2, 2)
        freqs_cis = mx.expand_dims(mx.expand_dims(freqs_cis, 0), 2)

        return freqs_cis

    def __call__(
        self,
        x: mx.array,
        bhle: bool = False,
        offset: int = 0,
    ) -> mx.array:
        """
        Apply rotary embeddings to input tensor.

        Uses 2x2 rotation matrix multiplication matching PyTorch SAM-Audio.

        Args:
            x: Input tensor of shape (B, L, H, E) or (B, H, L, E) if bhle=True
            bhle: If True, input is in (B, H, L, E) format
            offset: Position offset for incremental decoding

        Returns:
            Rotary-embedded tensor of same shape as input
        """
        if bhle:
            # (B, H, L, E) -> (B, L, H, E)
            x = mx.transpose(x, (0, 2, 1, 3))

        batch_size, seq_len, num_heads, head_dim = x.shape

        # Reshape to adjacent pairs: (B, L, H, E) -> (B, L, H, E/2, 1, 2)
        x_ = x.reshape(batch_size, seq_len, num_heads, head_dim // 2, 1, 2)

        # Get rotation matrix for sequence length
        # _freqs_cis shape: (1, max_seqlen, 1, dim//2, 2, 2)
        freqs_cis = self._freqs_cis[:, offset : offset + seq_len]

        # Apply rotation matrix multiplication
        # x_ shape: (B, L, H, E/2, 1, 2)
        # freqs_cis shape: (1, L, 1, E/2, 2, 2)
        # Result: (B, L, H, E/2, 1, 2) * (1, L, 1, E/2, 2, 2) -> sum over last dim -> (B, L, H, E/2, 2)
        x_out = (x_ * freqs_cis).sum(axis=-1)  # (B, L, H, E/2, 2)
        x_out = x_out.reshape(batch_size, seq_len, num_heads, head_dim)

        if bhle:
            # (B, L, H, E) -> (B, H, L, E)
            x_out = mx.transpose(x_out, (0, 2, 1, 3))

        return x_out.astype(x.dtype)


def apply_rotary_emb(
    xq: mx.array,
    xk: mx.array,
    freqs_cos: mx.array,
    freqs_sin: mx.array,
) -> tuple:
    """
    Apply rotary embeddings to query and key tensors.

    Args:
        xq: Query tensor (B, L, H, D)
        xk: Key tensor (B, L, H, D)
        freqs_cos: Cosine frequencies (L, D//2)
        freqs_sin: Sine frequencies (L, D//2)

    Returns:
        Tuple of rotated (query, key) tensors
    """
    # Reshape frequencies for broadcasting
    cos = mx.expand_dims(mx.expand_dims(freqs_cos, 0), 2)  # (1, L, 1, D//2)
    sin = mx.expand_dims(mx.expand_dims(freqs_sin, 0), 2)  # (1, L, 1, D//2)

    head_dim = xq.shape[-1]

    # Split into halves
    xq1, xq2 = xq[..., : head_dim // 2], xq[..., head_dim // 2 :]
    xk1, xk2 = xk[..., : head_dim // 2], xk[..., head_dim // 2 :]

    # Apply rotation
    xq_out = mx.concatenate([xq1 * cos - xq2 * sin, xq1 * sin + xq2 * cos], axis=-1)
    xk_out = mx.concatenate([xk1 * cos - xk2 * sin, xk1 * sin + xk2 * cos], axis=-1)

    return xq_out.astype(xq.dtype), xk_out.astype(xk.dtype)
