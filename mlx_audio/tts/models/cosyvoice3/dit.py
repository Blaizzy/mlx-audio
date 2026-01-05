"""
CosyVoice3 DiT (Diffusion Transformer) implementation in MLX.

Based on: https://github.com/FunAudioLLM/CosyVoice
"""

import math
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


def sinusoidal_embedding(timesteps: mx.array, dim: int) -> mx.array:
    """Create sinusoidal timestep embeddings."""
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = mx.exp(mx.arange(half_dim) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)
    return emb


class TimestepEmbedding(nn.Module):
    """Timestep embedding module for diffusion models."""

    def __init__(self, dim: int, time_dim: int = 256):
        super().__init__()
        self.dim = dim
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def __call__(self, t: mx.array) -> mx.array:
        # t should be in [0, 1]
        t_emb = sinusoidal_embedding(t, self.time_dim)
        return self.time_mlp(t_emb)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        rms = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return x / rms * self.weight


class AdaLayerNorm(nn.Module):
    """Adaptive Layer Normalization for DiT blocks."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # Output: scale, shift for x; plus gate for the block output
        self.linear = nn.Linear(dim, dim * 6)

    def __call__(
        self, x: mx.array, t: mx.array
    ) -> Tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
        emb = self.linear(nn.silu(t))
        emb = emb[:, None, :]  # (B, 1, 6*dim)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mx.split(
            emb, 6, axis=-1
        )
        return shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormZeroFinal(nn.Module):
    """Final AdaLN-Zero layer for output normalization."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.linear = nn.Linear(dim, dim * 2)

    def __call__(self, x: mx.array, t: mx.array) -> mx.array:
        emb = self.linear(nn.silu(t))
        emb = emb[:, None, :]  # (B, 1, 2*dim)
        shift, scale = mx.split(emb, 2, axis=-1)

        # Layer norm
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        x = (x - mean) / mx.sqrt(var + 1e-6)

        return x * (1 + scale) + shift


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding."""

    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (theta ** (mx.arange(0, dim, 2).astype(mx.float32) / dim))
        self.inv_freq = inv_freq

    def __call__(self, seq_len: int) -> Tuple[mx.array, mx.array]:
        t = mx.arange(seq_len).astype(mx.float32)
        freqs = mx.outer(t, self.inv_freq)
        cos = mx.cos(freqs)
        sin = mx.sin(freqs)
        return cos, sin


def apply_rotary_emb(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Apply rotary embeddings to input tensor."""
    # x: (B, H, T, D)
    # cos, sin: (T, D//2)
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]

    cos = cos[None, None, :, :]  # (1, 1, T, D//2)
    sin = sin[None, None, :, :]

    # Rotate
    x_rot = mx.concatenate([-x2, x1], axis=-1)
    cos_full = mx.concatenate([cos, cos], axis=-1)
    sin_full = mx.concatenate([sin, sin], axis=-1)

    return x * cos_full + x_rot * sin_full


class CausalConvPositionEmbedding(nn.Module):
    """Causal convolutional position embedding."""

    def __init__(self, dim: int, kernel_size: int = 31):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        groups = dim // 64  # 1024 // 64 = 16 groups

        # Two conv layers with group conv
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=kernel_size, groups=groups)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=kernel_size, groups=groups)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, T, D) - MLX Conv1d expects NLC format

        # Causal padding on time dimension
        pad = self.kernel_size - 1
        x = mx.pad(x, [(0, 0), (pad, 0), (0, 0)])
        x = nn.silu(self.conv1(x))

        x = mx.pad(x, [(0, 0), (pad, 0), (0, 0)])
        x = nn.silu(self.conv2(x))

        return x


class Attention(nn.Module):
    """Multi-head attention with rotary embeddings."""

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head**-0.5
        inner_dim = heads * dim_head

        self.to_q = nn.Linear(dim, inner_dim)
        self.to_k = nn.Linear(dim, inner_dim)
        self.to_v = nn.Linear(dim, inner_dim)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        rope: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        B, T, _ = x.shape

        # Project to q, k, v
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        # Reshape to (B, H, T, D)
        q = q.reshape(B, T, self.heads, self.dim_head).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.heads, self.dim_head).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.heads, self.dim_head).transpose(0, 2, 1, 3)

        # Apply rotary embeddings
        if rope is not None:
            cos, sin = rope
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)

        # Attention
        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale

        if mask is not None:
            # mask: (B, 1, T, T) or (B, T, T)
            if mask.ndim == 3:
                mask = mask[:, None, :, :]
            attn = mx.where(mask, attn, mx.array(float("-inf")))

        attn = mx.softmax(attn, axis=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = attn @ v
        out = out.transpose(0, 2, 1, 3).reshape(B, T, -1)

        return self.to_out(out)


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        inner_dim = dim * mult
        self.ff = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.ff(x)


class DiTBlock(nn.Module):
    """DiT (Diffusion Transformer) block."""

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        ff_mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attn_norm = AdaLayerNorm(dim)
        self.attn = Attention(dim, heads, dim_head, dropout)
        self.ff = FeedForward(dim, ff_mult, dropout)

    def __call__(
        self,
        x: mx.array,
        t: mx.array,
        mask: Optional[mx.array] = None,
        rope: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        # Get modulation parameters
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(
            x, t
        )

        # Pre-norm with modulation for attention
        x_norm = mx.fast.layer_norm(x, None, None, eps=1e-6)
        x_mod = x_norm * (1 + scale_msa) + shift_msa

        # Attention with gating
        x = x + gate_msa * self.attn(x_mod, mask=mask, rope=rope)

        # Pre-norm with modulation for FF
        x_norm = mx.fast.layer_norm(x, None, None, eps=1e-6)
        x_mod = x_norm * (1 + scale_mlp) + shift_mlp

        # FF with gating
        x = x + gate_mlp * self.ff(x_mod)

        return x


class InputEmbedding(nn.Module):
    """Input embedding for DiT combining mel, condition, text, and speaker."""

    def __init__(
        self,
        mel_dim: int = 80,
        text_dim: int = 80,
        out_dim: int = 1024,
        spk_dim: int = 80,
    ):
        super().__init__()
        # mel * 2 (noised + condition) + text + speaker
        in_dim = mel_dim * 2 + text_dim + spk_dim
        self.proj = nn.Linear(in_dim, out_dim)
        self.conv_pos_embed = CausalConvPositionEmbedding(out_dim)

    def __call__(
        self,
        x: mx.array,
        cond: mx.array,
        text_embed: mx.array,
        spks: mx.array,
    ) -> mx.array:
        # x, cond, text_embed: (B, T, D)
        # spks: (B, D) -> repeat to (B, T, D)
        B, T, _ = x.shape
        spks = mx.broadcast_to(spks[:, None, :], (B, T, spks.shape[-1]))

        # Concatenate all inputs
        x = mx.concatenate([x, cond, text_embed, spks], axis=-1)
        x = self.proj(x)

        # Add positional embedding
        pos = self.conv_pos_embed(x)
        x = x + pos

        return x


class DiT(nn.Module):
    """Diffusion Transformer for audio generation."""

    def __init__(
        self,
        dim: int = 1024,
        depth: int = 22,
        heads: int = 16,
        dim_head: int = 64,
        ff_mult: int = 2,
        dropout: float = 0.0,
        mel_dim: int = 80,
        mu_dim: int = 80,
        spk_dim: int = 80,
        out_channels: int = 80,
        static_chunk_size: int = 50,
        num_decoding_left_chunks: int = -1,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.mel_dim = mel_dim
        self.out_channels = out_channels
        self.static_chunk_size = static_chunk_size
        self.num_decoding_left_chunks = num_decoding_left_chunks

        # Time embedding
        self.time_embed = TimestepEmbedding(dim)

        # Input embedding
        self.input_embed = InputEmbedding(mel_dim, mu_dim, dim, spk_dim)

        # Rotary embedding
        self.rotary_embed = RotaryEmbedding(dim_head)

        # Transformer blocks
        self.transformer_blocks = [
            DiTBlock(dim, heads, dim_head, ff_mult, dropout) for _ in range(depth)
        ]

        # Output layers
        self.norm_out = AdaLayerNormZeroFinal(dim)
        self.proj_out = nn.Linear(dim, mel_dim)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array,
        mu: mx.array,
        t: mx.array,
        spks: mx.array,
        cond: mx.array,
        streaming: bool = False,
    ) -> mx.array:
        """
        Forward pass of DiT.

        Args:
            x: Noised input (B, mel_dim, T)
            mask: Attention mask (B, T)
            mu: Mean/condition from token embedding (B, mel_dim, T)
            t: Timestep (B,) or scalar
            spks: Speaker embedding (B, spk_dim)
            cond: Condition mel (B, mel_dim, T)
            streaming: Whether to use streaming attention mask

        Returns:
            Output mel prediction (B, mel_dim, T)
        """
        # Transpose to (B, T, D)
        x = x.transpose(0, 2, 1)
        mu = mu.transpose(0, 2, 1)
        cond = cond.transpose(0, 2, 1)

        B, T, _ = x.shape

        # Expand scalar timestep
        if t.ndim == 0:
            t = mx.broadcast_to(t, (B,))

        # Time embedding
        t_emb = self.time_embed(t)

        # Input embedding
        x = self.input_embed(x, cond, mu, spks)

        # Get rotary embeddings
        cos, sin = self.rotary_embed(T)
        rope = (cos, sin)

        # Create attention mask
        if streaming:
            # Causal mask with chunk attention
            attn_mask = self._create_chunk_mask(B, T, mask)
        else:
            # Full attention with padding mask
            attn_mask = mask[:, None, :].astype(mx.bool_)
            attn_mask = mx.broadcast_to(attn_mask, (B, T, T))

        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x, t_emb, mask=attn_mask, rope=rope)

        # Output
        x = self.norm_out(x, t_emb)
        output = self.proj_out(x)

        return output.transpose(0, 2, 1)

    def _create_chunk_mask(
        self, batch_size: int, seq_len: int, padding_mask: mx.array
    ) -> mx.array:
        """Create chunk-based causal attention mask."""
        # Create causal mask
        causal = mx.tril(mx.ones((seq_len, seq_len), dtype=mx.bool_))

        # Chunk-based attention
        chunk_size = self.static_chunk_size
        if chunk_size > 0:
            # Allow attending to current and previous chunks
            chunk_mask = mx.zeros((seq_len, seq_len), dtype=mx.bool_)
            for i in range(seq_len):
                chunk_start = (i // chunk_size) * chunk_size
                if self.num_decoding_left_chunks < 0:
                    start = 0
                else:
                    start = max(
                        0, chunk_start - self.num_decoding_left_chunks * chunk_size
                    )
                chunk_mask = chunk_mask.at[i, start : i + 1].set(True)
            causal = causal & chunk_mask

        # Combine with padding mask
        causal = mx.broadcast_to(causal[None, :, :], (batch_size, seq_len, seq_len))

        return causal
