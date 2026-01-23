# Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

import math
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


def sinusoidal_embedding(timesteps: mx.array, dim: int, scale: float = 1000.0) -> mx.array:
    """Create sinusoidal timestep embeddings matching PyTorch SinusPositionEmbedding."""
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = mx.exp(mx.arange(half_dim).astype(mx.float32) * -emb)
    # PyTorch uses: scale * x.unsqueeze(1) * emb.unsqueeze(0)
    emb = scale * timesteps[:, None] * emb[None, :]
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
        # Match PyTorch order: scale, shift = chunk(emb, 2, dim=1)
        scale, shift = mx.split(emb, 2, axis=-1)

        # Layer norm (elementwise_affine=False)
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        x = (x - mean) / mx.sqrt(var + 1e-6)

        return x * (1 + scale) + shift


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding matching x_transformers implementation."""

    def __init__(self, dim: int, theta: float = 10000.0, use_xpos: bool = False):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.use_xpos = use_xpos
        inv_freq = 1.0 / (theta ** (mx.arange(0, dim, 2).astype(mx.float32) / dim))
        self.inv_freq = inv_freq

    def __call__(self, seq_len: int) -> Tuple[mx.array, float]:
        """Returns (freqs, scale) tuple matching x_transformers format.

        Returns:
            Tuple of (freqs, scale) where:
            - freqs: (1, T, dim) interleaved frequency tensor
            - scale: 1.0 (no xpos scaling)
        """
        t = mx.arange(seq_len).astype(mx.float32)
        # (T, dim//2)
        freqs = mx.outer(t, self.inv_freq)
        # Stack and interleave: [f0, f0, f1, f1, ...] -> (T, dim)
        # This matches x_transformers: stack((freqs, freqs), dim=-1).rearrange('... d r -> ... (d r)')
        freqs = mx.stack([freqs, freqs], axis=-1).reshape(seq_len, -1)
        # Add batch dimension: (1, T, dim)
        freqs = freqs[None, :, :]
        # Return tuple (freqs, scale) matching x_transformers
        return freqs, 1.0


def rotate_half(x: mx.array) -> mx.array:
    """Rotate half the hidden dims - interleaved version matching x_transformers."""
    # x: (..., d) where d is even
    # Reshape to (..., d//2, 2)
    x = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x[..., 0], x[..., 1]
    # Stack as (-x2, x1) pairs
    rotated = mx.stack([-x2, x1], axis=-1)
    # Flatten back to (..., d)
    return rotated.reshape(*rotated.shape[:-2], -1)


def apply_rotary_emb(x: mx.array, freqs: mx.array, scale: float = 1.0) -> mx.array:
    """Apply rotary embeddings to input tensor - matching x_transformers.

    This implements partial rotary embeddings: only the first rot_dim
    dimensions are rotated, the rest are left unchanged.

    Args:
        x: Input tensor of shape (B, T, D) or (B, H, T, D)
        freqs: Frequencies tensor of shape (1, T, rot_dim) with interleaved values
        scale: Scale factor for xpos (default 1.0)

    Returns:
        Rotated tensor of same shape as input
    """
    # Get the rotation dimension
    rot_dim = freqs.shape[-1]
    seq_len = x.shape[-2] if x.ndim == 4 else x.shape[1]

    # Slice freqs to match sequence length
    freqs = freqs[:, -seq_len:, :]

    # Handle different input dimensions
    if x.ndim == 4:
        # x is (B, H, T, D) - add head dimension to freqs
        freqs = freqs[:, None, :, :]  # (1, 1, T, rot_dim)
    # else x is (B, T, D) and freqs is (1, T, rot_dim) - broadcast works

    # Split into rotated and unrotated parts (partial rotary embeddings)
    t, t_unrotated = x[..., :rot_dim], x[..., rot_dim:]

    # Apply rotation: (t * cos(freqs) + rotate_half(t) * sin(freqs)) * scale
    # Matching x_transformers: t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
    t = (t * mx.cos(freqs) * scale) + (rotate_half(t) * mx.sin(freqs) * scale)

    # Concatenate back
    return mx.concatenate([t, t_unrotated], axis=-1)


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
        x = nn.mish(self.conv1(x))

        x = mx.pad(x, [(0, 0), (pad, 0), (0, 0)])
        x = nn.mish(self.conv2(x))

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
        rope: Optional[Tuple[mx.array, float]] = None,
        output_mask: Optional[mx.array] = None,
    ) -> mx.array:
        B, T, _ = x.shape

        # Project to q, k, v - shape (B, T, H*D)
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        # Apply rotary embeddings on FLATTENED (B, T, H*D) tensor BEFORE reshape
        # This matches PyTorch AttnProcessor which calls apply_rotary_pos_emb
        # on the (B, T, inner_dim) tensor before view/transpose.
        # With freqs shape (1, T, rot_dim=64), only the first 64 of 1024 dims are rotated.
        if rope is not None:
            freqs, xpos_scale = rope
            q_scale = xpos_scale if xpos_scale is not None else 1.0
            k_scale = (1.0 / xpos_scale) if xpos_scale is not None and xpos_scale != 1.0 else 1.0
            q = apply_rotary_emb(q, freqs, q_scale)
            k = apply_rotary_emb(k, freqs, k_scale)

        # THEN reshape to (B, T, H, D) and transpose to (B, H, T, D)
        q = q.reshape(B, T, self.heads, self.dim_head).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.heads, self.dim_head).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.heads, self.dim_head).transpose(0, 2, 1, 3)

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

        out = self.to_out(out)

        # Zero out padded positions in output (matching PyTorch AttnProcessor)
        if output_mask is not None:
            # output_mask: (B, T) -> (B, T, 1)
            out = mx.where(output_mask[:, :, None], out, mx.zeros_like(out))

        return out


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        inner_dim = dim * mult
        self.ff = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(approx="tanh"),  # Match PyTorch's approximate="tanh"
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
        rope: Optional[mx.array] = None,
        output_mask: Optional[mx.array] = None,
    ) -> mx.array:
        # Get modulation parameters
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(
            x, t
        )

        # Pre-norm with modulation for attention
        x_norm = mx.fast.layer_norm(x, None, None, eps=1e-6)
        x_mod = x_norm * (1 + scale_msa) + shift_msa

        # Attention with gating - pass output_mask for zeroing padded positions
        x = x + gate_msa * self.attn(x_mod, mask=mask, rope=rope, output_mask=output_mask)

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
            mask: Attention mask (B, T) or (B, 1, T)
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

        # Get rotary embeddings (single freqs tensor in x_transformers format)
        rope = self.rotary_embed(T)

        # Extract padding mask for output masking (B, T)
        # This is used to zero out padded positions after attention
        if mask.ndim == 2:
            output_mask = mask.astype(mx.bool_)
        else:
            # mask is (B, 1, T), squeeze to (B, T)
            output_mask = mask.squeeze(1).astype(mx.bool_)

        # Create attention mask
        if streaming:
            # Causal mask with chunk attention
            attn_mask = self._create_chunk_mask(B, T, mask)
        else:
            # Full attention with padding mask
            # Handle both (B, T) and (B, 1, T) mask formats
            if mask.ndim == 2:
                # mask is (B, T), add dim to (B, 1, T)
                mask_3d = mask[:, None, :]
            else:
                # mask is (B, 1, T)
                mask_3d = mask
            # Broadcast to (B, T, T) for attention
            # Each row (query position) can attend to all valid key positions
            attn_mask = mx.broadcast_to(mask_3d, (B, T, T))
            attn_mask = attn_mask.astype(mx.bool_)

        # Transformer blocks - pass output_mask to zero padded positions
        for block in self.transformer_blocks:
            x = block(x, t_emb, mask=attn_mask, rope=rope, output_mask=output_mask)

        # Output
        x = self.norm_out(x, t_emb)
        output = self.proj_out(x)

        return output.transpose(0, 2, 1)

    def _create_chunk_mask(
        self, batch_size: int, seq_len: int, padding_mask: mx.array
    ) -> mx.array:
        """Create chunk-based attention mask matching PyTorch subsequent_chunk_mask.

        PyTorch implementation:
            pos_idx = torch.arange(size)
            block_value = (torch.div(pos_idx, chunk_size, rounding_mode='trunc') + 1) * chunk_size
            ret = pos_idx.unsqueeze(0) < block_value.unsqueeze(1)

        This means position i can attend to positions j where:
            j < ((i // chunk_size) + 1) * chunk_size
        i.e., all positions up to the end of the current chunk.
        """
        chunk_size = self.static_chunk_size

        pos_idx = mx.arange(seq_len)
        # For each query position, compute the end of its chunk
        block_value = ((pos_idx // chunk_size) + 1) * chunk_size
        # Each query can attend to key positions < block_value
        # pos_idx[None, :] is key positions, block_value[:, None] is per-query limit
        chunk_mask = pos_idx[None, :] < block_value[:, None]  # (seq_len, seq_len)

        # Combine with padding mask
        # padding_mask: (B, T) or (B, 1, T)
        if padding_mask.ndim == 3:
            pad_mask = padding_mask  # (B, 1, T)
        else:
            pad_mask = padding_mask[:, None, :]  # (B, 1, T)

        # Broadcast: chunk_mask (1, T, T) & pad_mask (B, 1, T) -> (B, T, T)
        attn_mask = chunk_mask[None, :, :] & pad_mask.astype(mx.bool_)

        return attn_mask
