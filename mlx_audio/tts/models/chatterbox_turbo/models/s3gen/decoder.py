# Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

import math
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn


class Conv1dPT(nn.Module):
    """
    Conv1d wrapper that accepts PyTorch format (B, C, T) input.
    Internally transposes to MLX format (B, T, C), applies conv, transposes back.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, C, T) -> (B, T, C)
        x = x.transpose(0, 2, 1)
        x = self.conv(x)
        # (B, T, C) -> (B, C, T)
        return x.transpose(0, 2, 1)


class ConvTranspose1dPT(nn.Module):
    """
    ConvTranspose1d wrapper that accepts PyTorch format (B, C, T) input.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ):
        super().__init__()
        self.conv = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, C, T) -> (B, T, C)
        x = x.transpose(0, 2, 1)
        x = self.conv(x)
        # (B, T, C) -> (B, C, T)
        return x.transpose(0, 2, 1)


def sinusoidal_pos_emb(timesteps: mx.array, dim: int, scale: float = 1000) -> mx.array:
    """Sinusoidal positional embeddings for timesteps."""
    if timesteps.ndim == 0:
        timesteps = timesteps[None]

    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = mx.exp(mx.arange(half_dim, dtype=mx.float32) * -emb)
    emb = scale * timesteps[:, None] * emb[None, :]
    emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)
    return emb


class TimestepEmbedding(nn.Module):
    """MLP for timestep embedding."""

    def __init__(self, in_channels: int, time_embed_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.linear_1(x)
        x = nn.silu(x)
        x = self.linear_2(x)
        return x


class CausalConv1d(nn.Module):
    """Causal 1D convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.causal_padding = (kernel_size - 1) * dilation

        self.conv = Conv1dPT(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
        )

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, C, T)
        # Pad on left side only for causal (time is last dim in PyTorch format)
        x = mx.pad(x, [(0, 0), (0, 0), (self.causal_padding, 0)])
        return self.conv(x)


class Block1D(nn.Module):
    """Basic 1D block with conv, norm, and activation."""

    def __init__(self, dim: int, dim_out: int, groups: int = 8):
        super().__init__()
        self.conv = Conv1dPT(dim, dim_out, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        x = x * mask
        x = self.conv(x)
        # GroupNorm in MLX expects (B, ..., C), we have (B, C, T)
        # Transpose for norm
        x = x.transpose(0, 2, 1)  # (B, T, C)
        x = self.norm(x)
        x = x.transpose(0, 2, 1)  # (B, C, T)
        x = nn.mish(x)
        return x * mask


class CausalBlock1D(nn.Module):
    """Causal version of Block1D - matches PyTorch structure: Conv + LayerNorm + Mish."""

    def __init__(self, dim: int, dim_out: int, groups: int = 8):
        super().__init__()
        # PyTorch structure: block = Sequential(CausalConv, Transpose, LayerNorm, Transpose, Mish)
        # So the order is: Conv -> LayerNorm -> Mish
        self.block = [
            CausalConv1d(dim, dim_out, kernel_size=3),  # block.0
            nn.LayerNorm(
                dim_out
            ),  # block.2 (indices 1,3 are Transpose which we handle inline)
        ]

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        x = x * mask
        x = self.block[0](x)  # Conv
        # LayerNorm expects (B, T, C), we have (B, C, T)
        x = x.transpose(0, 2, 1)
        x = self.block[1](x)  # LayerNorm
        x = x.transpose(0, 2, 1)
        x = nn.mish(x)  # Mish comes AFTER LayerNorm
        return x * mask


class ResnetBlock1D(nn.Module):
    """Resnet block with time embedding - matches PyTorch mlp.1 naming."""

    def __init__(
        self,
        dim: int,
        dim_out: int,
        time_emb_dim: int,
        causal: bool = True,
        groups: int = 8,
    ):
        super().__init__()
        # PyTorch structure: mlp = [Mish (index 0), Linear (index 1)]
        # So mlp.1 is the Linear layer
        self.mlp = [
            # Mish at index 0 (we apply inline since it's an activation)
            nn.Linear(time_emb_dim, dim_out),  # mlp.1
        ]

        if causal:
            self.block1 = CausalBlock1D(dim, dim_out, groups)
            self.block2 = CausalBlock1D(dim_out, dim_out, groups)
        else:
            self.block1 = Block1D(dim, dim_out, groups)
            self.block2 = Block1D(dim_out, dim_out, groups)

        self.res_conv = Conv1dPT(dim, dim_out, kernel_size=1)

    def __call__(self, x: mx.array, mask: mx.array, time_emb: mx.array) -> mx.array:
        h = self.block1(x, mask)
        # Apply Mish then linear (mlp.1) - PyTorch uses Mish, not SiLU
        h = h + self.mlp[0](nn.mish(time_emb))[:, :, None]
        h = self.block2(h, mask)
        return h + self.res_conv(x * mask)


class Downsample1D(nn.Module):
    """1D downsampling layer."""

    def __init__(self, dim: int):
        super().__init__()
        self.conv = Conv1dPT(dim, dim, kernel_size=3, stride=2, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv(x)


class Upsample1D(nn.Module):
    """1D upsampling layer with conv transpose."""

    def __init__(self, dim: int):
        super().__init__()
        self.conv = ConvTranspose1dPT(dim, dim, kernel_size=4, stride=2, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv(x)


class SelfAttention1D(nn.Module):
    """Self-attention for 1D sequences - matches PyTorch attn1 structure.

    Note: PyTorch decoder uses BIDIRECTIONAL attention in transformer blocks,
    not causal attention. The causality is handled by the CausalConv1d layers.
    """

    def __init__(
        self, dim: int, num_heads: int = 8, head_dim: int = 64, dropout: float = 0.0
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim
        self.scale = head_dim**-0.5

        # Match PyTorch naming: attn1.to_q, attn1.to_k, attn1.to_v, attn1.to_out.0
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = [nn.Linear(inner_dim, dim)]  # to_out.0

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        B, T, C = x.shape

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        # Reshape for multi-head attention
        q = q.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Attention (bidirectional - no causal mask)
        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale

        # Apply padding mask if provided (for variable length sequences)
        if mask is not None:
            # mask: (B, T) -> (B, 1, 1, T)
            mask = mask[:, None, None, :]
            attn = mx.where(mask > 0, attn, -1e9)

        attn = mx.softmax(attn, axis=-1)
        out = attn @ v

        # Reshape back
        out = out.transpose(0, 2, 1, 3).reshape(B, T, -1)
        return self.to_out[0](out)


class GELU(nn.Module):
    """GELU activation with linear projection - matches diffusers GELU."""

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        # PyTorch names this 'proj'
        self.proj = nn.Linear(dim_in, dim_out)

    def __call__(self, x: mx.array) -> mx.array:
        # Note: PyTorch decoder uses act_fn="gelu", which applies gelu AFTER proj
        return nn.gelu(self.proj(x))


class FeedForward(nn.Module):
    """Feed-forward network - matches PyTorch ff.net structure with GELU activation."""

    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        inner_dim = dim * mult  # 256 * 4 = 1024
        # PyTorch structure: net = [GELU (index 0), Dropout (index 1), Linear (index 2)]
        # net.0.proj = GELU proj, net.2 = output Linear
        # Note: decoder uses act_fn="gelu" which uses GELU class from diffusers
        self.net = [
            GELU(dim, inner_dim),  # net.0.proj: Linear(256, 1024)
            # Dropout at index 1 (we skip it for inference)
            nn.Linear(inner_dim, dim),  # net.2 -> net.1: Linear(1024, 256)
        ]

    def __call__(self, x: mx.array) -> mx.array:
        x = self.net[0](x)  # GELU
        x = self.net[1](x)  # Output linear
        return x


class TransformerBlock(nn.Module):
    """Basic transformer block - matches PyTorch naming with attn1, norm1, norm3, ff."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        head_dim: int = 64,
        ff_mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        # PyTorch naming: attn1, norm1, norm3, ff
        self.attn1 = SelfAttention1D(dim, num_heads, head_dim, dropout)
        self.ff = FeedForward(dim, ff_mult, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)  # Note: norm3, not norm2 (to match PyTorch)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        x = x + self.attn1(self.norm1(x), mask)
        x = x + self.ff(self.norm3(x))
        return x


class DownBlock(nn.Module):
    """Down block containing ResNet, transformers, and downsample - for proper MLX parameter tracking."""

    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        time_embed_dim: int,
        causal: bool,
        n_blocks: int,
        num_heads: int,
        attention_head_dim: int,
        is_last: bool,
    ):
        super().__init__()
        self.resnet = ResnetBlock1D(
            input_channel, output_channel, time_embed_dim, causal
        )
        self.transformer_blocks = [
            TransformerBlock(output_channel, num_heads, attention_head_dim)
            for _ in range(n_blocks)
        ]
        if is_last:
            self.downsample = (
                CausalConv1d(output_channel, output_channel, 3)
                if causal
                else Conv1dPT(output_channel, output_channel, 3, padding=1)
            )
        else:
            self.downsample = Downsample1D(output_channel)


class MidBlock(nn.Module):
    """Mid block containing ResNet and transformers - for proper MLX parameter tracking."""

    def __init__(
        self,
        channels: int,
        time_embed_dim: int,
        causal: bool,
        n_blocks: int,
        num_heads: int,
        attention_head_dim: int,
    ):
        super().__init__()
        self.resnet = ResnetBlock1D(channels, channels, time_embed_dim, causal)
        self.transformer_blocks = [
            TransformerBlock(channels, num_heads, attention_head_dim)
            for _ in range(n_blocks)
        ]


class UpBlock(nn.Module):
    """Up block containing ResNet, transformers, and upsample - for proper MLX parameter tracking."""

    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        time_embed_dim: int,
        causal: bool,
        n_blocks: int,
        num_heads: int,
        attention_head_dim: int,
        is_last: bool,
    ):
        super().__init__()
        self.resnet = ResnetBlock1D(
            input_channel, output_channel, time_embed_dim, causal
        )
        self.transformer_blocks = [
            TransformerBlock(output_channel, num_heads, attention_head_dim)
            for _ in range(n_blocks)
        ]
        if is_last:
            self.upsample = (
                CausalConv1d(output_channel, output_channel, 3)
                if causal
                else Conv1dPT(output_channel, output_channel, 3, padding=1)
            )
        else:
            self.upsample = Upsample1D(output_channel)


class ConditionalDecoder(nn.Module):
    """
    Conditional decoder for flow matching.
    Takes noisy input and predicts the velocity field.
    """

    def __init__(
        self,
        in_channels: int = 320,
        out_channels: int = 80,
        causal: bool = True,
        channels: List[int] = [256],
        dropout: float = 0.0,
        attention_head_dim: int = 64,
        n_blocks: int = 4,
        num_mid_blocks: int = 12,
        num_heads: int = 8,
        meanflow: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.causal = causal
        self.meanflow = meanflow

        # Time embedding
        time_embed_dim = channels[0] * 4
        self.time_mlp = TimestepEmbedding(in_channels, time_embed_dim)

        # Down blocks
        self.down_blocks = []
        output_channel = in_channels
        for i, ch in enumerate(channels):
            input_channel = output_channel
            output_channel = ch
            is_last = i == len(channels) - 1

            self.down_blocks.append(
                DownBlock(
                    input_channel,
                    output_channel,
                    time_embed_dim,
                    causal,
                    n_blocks,
                    num_heads,
                    attention_head_dim,
                    is_last,
                )
            )

        # Mid blocks
        self.mid_blocks = []
        for _ in range(num_mid_blocks):
            self.mid_blocks.append(
                MidBlock(
                    channels[-1],
                    time_embed_dim,
                    causal,
                    n_blocks,
                    num_heads,
                    attention_head_dim,
                )
            )

        # Up blocks
        self.up_blocks = []
        channels_up = list(reversed(channels)) + [channels[0]]
        for i in range(len(channels_up) - 1):
            input_channel = channels_up[i] * 2  # Skip connection
            output_channel = channels_up[i + 1]
            is_last = i == len(channels_up) - 2

            self.up_blocks.append(
                UpBlock(
                    input_channel,
                    output_channel,
                    time_embed_dim,
                    causal,
                    n_blocks,
                    num_heads,
                    attention_head_dim,
                    is_last,
                )
            )

        # Final layers
        final_ch = channels_up[-1]
        self.final_block = (
            CausalBlock1D(final_ch, final_ch) if causal else Block1D(final_ch, final_ch)
        )
        self.final_proj = Conv1dPT(final_ch, out_channels, kernel_size=1)

        # Meanflow time mixing - PyTorch uses Linear without bias
        self.time_embed_mixer = None
        if meanflow:
            # PyTorch has Linear(2048, 1024) without bias
            self.time_embed_mixer = nn.Linear(
                time_embed_dim * 2, time_embed_dim, bias=False
            )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array,
        mu: mx.array,
        t: mx.array,
        spks: Optional[mx.array] = None,
        cond: Optional[mx.array] = None,
        r: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Forward pass.

        Args:
            x: Noisy input (B, 80, T)
            mask: Mask (B, 1, T)
            mu: Encoder output (B, 80, T)
            t: Timestep (B,)
            spks: Speaker embedding (B, 80)
            cond: Conditioning (B, 80, T)
            r: End time for meanflow (B,)

        Returns:
            Predicted velocity field (B, 80, T)
        """
        # Time embedding
        t_emb = sinusoidal_pos_emb(t, self.in_channels)
        t_emb = self.time_mlp(t_emb)

        if self.meanflow and r is not None:
            r_emb = sinusoidal_pos_emb(r, self.in_channels)
            r_emb = self.time_mlp(r_emb)
            concat_emb = mx.concatenate([t_emb, r_emb], axis=-1)
            t_emb = self.time_embed_mixer(concat_emb)

        # Concatenate inputs: x, mu, spks, cond
        inputs = [x, mu]
        if spks is not None:
            # Expand speaker embedding to time dimension
            spks_expanded = mx.broadcast_to(
                spks[:, :, None], (spks.shape[0], spks.shape[1], x.shape[2])
            )
            inputs.append(spks_expanded)
        if cond is not None:
            inputs.append(cond)

        x = mx.concatenate(inputs, axis=1)

        # Down path
        hiddens = []
        masks = [mask]
        for down_block in self.down_blocks:
            mask_down = masks[-1]
            x = down_block.resnet(x, mask_down, t_emb)

            # Transpose for transformer: (B, C, T) -> (B, T, C)
            x = x.transpose(0, 2, 1)
            mask_t = mask_down[:, 0, :]  # (B, T)
            for block in down_block.transformer_blocks:
                x = block(x, mask_t)
            x = x.transpose(0, 2, 1)  # Back to (B, C, T)

            hiddens.append(x)
            x = down_block.downsample(x * mask_down)
            # Downsample mask
            masks.append(mask_down[:, :, ::2])

        masks = masks[:-1]
        mask_mid = masks[-1]

        # Mid path
        for mid_block in self.mid_blocks:
            x = mid_block.resnet(x, mask_mid, t_emb)
            x = x.transpose(0, 2, 1)
            mask_t = mask_mid[:, 0, :]
            for block in mid_block.transformer_blocks:
                x = block(x, mask_t)
            x = x.transpose(0, 2, 1)

        # Up path
        for up_block in self.up_blocks:
            mask_up = masks.pop()
            skip = hiddens.pop()

            # Align sizes
            x = x[:, :, : skip.shape[2]]
            x = mx.concatenate([x, skip], axis=1)

            x = up_block.resnet(x, mask_up, t_emb)
            x = x.transpose(0, 2, 1)
            mask_t = mask_up[:, 0, :]
            for block in up_block.transformer_blocks:
                x = block(x, mask_t)
            x = x.transpose(0, 2, 1)

            x = up_block.upsample(x * mask_up)

        # Final
        x = self.final_block(x, mask_up)
        x = self.final_proj(x * mask_up)

        return x * mask
