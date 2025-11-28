import mlx.core as mx
import mlx.nn as nn

from .matcha.decoder import (
    Block1D,
    Downsample1D,
    ResnetBlock1D,
    SinusoidalPosEmb,
    TimestepEmbedding,
    Upsample1D,
)
from .matcha.transformer import BasicTransformerBlock


class CausalConv1d(nn.Module):
    """Causal 1D convolution with left padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        assert stride == 1, "CausalConv1d only supports stride=1"
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            bias=bias,
        )
        self.causal_padding = kernel_size - 1

    def __call__(self, x: mx.array) -> mx.array:
        x = mx.swapaxes(x, 1, 2)  # (B, C, T) -> (B, T, C)
        # Pad on the left for causal convolution (time axis is now 1)
        x = mx.pad(x, [(0, 0), (self.causal_padding, 0), (0, 0)])
        x = self.conv(x)
        x = mx.swapaxes(x, 1, 2)  # (B, T, C) -> (B, C, T)
        return x


class CausalBlock1D(nn.Module):
    """
    Causal 1D block with LayerNorm (matches original).

    Unlike Block1D which uses GroupNorm, CausalBlock1D uses LayerNorm
    as per the original Chatterbox implementation.
    """

    def __init__(self, dim: int, dim_out: int):
        super().__init__()
        self.conv = CausalConv1d(dim, dim_out, 3)
        self.norm = nn.LayerNorm(dim_out)

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        output = self.conv(x * mask)
        # Transpose to (B, T, C), apply LayerNorm, transpose back
        output = mx.swapaxes(output, 1, 2)  # (B, C, T) -> (B, T, C)
        output = self.norm(output)
        output = mx.swapaxes(output, 1, 2)  # (B, T, C) -> (B, C, T)
        output = nn.mish(output)
        return output * mask


class CausalResnetBlock1D(ResnetBlock1D):
    """Causal ResNet block."""

    def __init__(self, dim: int, dim_out: int, time_emb_dim: int, groups: int = 8):
        super().__init__(dim, dim_out, time_emb_dim, groups)
        self.block1 = CausalBlock1D(dim, dim_out)
        self.block2 = CausalBlock1D(dim_out, dim_out)


class DownBlock(nn.Module):
    """Container for down block components."""

    def __init__(self, resnet, transformer_blocks, downsample):
        super().__init__()
        self.resnet = resnet
        # Store transformer blocks with indexed names for proper weight loading
        for i, block in enumerate(transformer_blocks):
            setattr(self, f"transformer_{i}", block)
        self.n_transformer = len(transformer_blocks)
        self.downsample = downsample

    @property
    def transformer_blocks(self):
        return [getattr(self, f"transformer_{i}") for i in range(self.n_transformer)]


class MidBlock(nn.Module):
    """Container for mid block components."""

    def __init__(self, resnet, transformer_blocks):
        super().__init__()
        self.resnet = resnet
        for i, block in enumerate(transformer_blocks):
            setattr(self, f"transformer_{i}", block)
        self.n_transformer = len(transformer_blocks)

    @property
    def transformer_blocks(self):
        return [getattr(self, f"transformer_{i}") for i in range(self.n_transformer)]


class UpBlock(nn.Module):
    """Container for up block components."""

    def __init__(self, resnet, transformer_blocks, upsample):
        super().__init__()
        self.resnet = resnet
        for i, block in enumerate(transformer_blocks):
            setattr(self, f"transformer_{i}", block)
        self.n_transformer = len(transformer_blocks)
        self.upsample = upsample

    @property
    def transformer_blocks(self):
        return [getattr(self, f"transformer_{i}") for i in range(self.n_transformer)]


class ConditionalDecoder(nn.Module):
    """
    Conditional U-Net decoder for flow matching.

    This is the Chatterbox-specific decoder that uses causal blocks
    for streaming TTS generation.
    """

    def __init__(
        self,
        in_channels: int = 320,
        out_channels: int = 80,
        causal: bool = True,
        channels: list = [256],
        dropout: float = 0.0,
        attention_head_dim: int = 64,
        n_blocks: int = 4,
        num_mid_blocks: int = 12,
        num_heads: int = 8,
        act_fn: str = "gelu",
    ):
        super().__init__()
        channels = tuple(channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.causal = causal

        # Time embeddings
        self.time_embeddings = SinusoidalPosEmb(in_channels)
        time_embed_dim = channels[0] * 4
        self.time_mlp = TimestepEmbedding(
            in_channels=in_channels,
            time_embed_dim=time_embed_dim,
            act_fn="silu",
        )

        # Down blocks - use indexed attributes for proper weight loading
        output_channel = in_channels
        for i, ch in enumerate(channels):
            input_channel = output_channel
            output_channel = ch
            is_last = i == len(channels) - 1

            ResBlock = CausalResnetBlock1D if causal else ResnetBlock1D
            resnet = ResBlock(input_channel, output_channel, time_embed_dim)

            transformer_blocks = [
                BasicTransformerBlock(
                    output_channel,
                    num_heads,
                    attention_head_dim,
                    dropout,
                    act_fn,
                )
                for _ in range(n_blocks)
            ]

            if not is_last:
                downsample = Downsample1D(output_channel)
            else:
                downsample = (
                    CausalConv1d(output_channel, output_channel, 3)
                    if causal
                    else nn.Conv1d(output_channel, output_channel, 3, padding=1)
                )

            setattr(
                self,
                f"down_blocks_{i}",
                DownBlock(resnet, transformer_blocks, downsample),
            )
        self.n_down_blocks = len(channels)

        # Mid blocks
        for i in range(num_mid_blocks):
            ResBlock = CausalResnetBlock1D if causal else ResnetBlock1D
            resnet = ResBlock(channels[-1], channels[-1], time_embed_dim)
            transformer_blocks = [
                BasicTransformerBlock(
                    channels[-1],
                    num_heads,
                    attention_head_dim,
                    dropout,
                    act_fn,
                )
                for _ in range(n_blocks)
            ]
            setattr(self, f"mid_blocks_{i}", MidBlock(resnet, transformer_blocks))
        self.n_mid_blocks = num_mid_blocks

        # Up blocks
        channels_reversed = list(reversed(channels)) + [channels[0]]
        for i in range(len(channels_reversed) - 1):
            input_channel = channels_reversed[i] * 2
            output_channel = channels_reversed[i + 1]
            is_last = i == len(channels_reversed) - 2

            ResBlock = CausalResnetBlock1D if causal else ResnetBlock1D
            resnet = ResBlock(input_channel, output_channel, time_embed_dim)

            transformer_blocks = [
                BasicTransformerBlock(
                    output_channel,
                    num_heads,
                    attention_head_dim,
                    dropout,
                    act_fn,
                )
                for _ in range(n_blocks)
            ]

            if not is_last:
                upsample = Upsample1D(output_channel, use_conv_transpose=True)
            else:
                upsample = (
                    CausalConv1d(output_channel, output_channel, 3)
                    if causal
                    else nn.Conv1d(output_channel, output_channel, 3, padding=1)
                )

            setattr(
                self, f"up_blocks_{i}", UpBlock(resnet, transformer_blocks, upsample)
            )
        self.n_up_blocks = len(channels_reversed) - 1

        # Final layers
        FinalBlock = CausalBlock1D if causal else Block1D
        self.final_block = FinalBlock(channels_reversed[-1], channels_reversed[-1])
        self.final_proj = nn.Conv1d(channels_reversed[-1], out_channels, 1)

    @property
    def down_blocks(self):
        return [getattr(self, f"down_blocks_{i}") for i in range(self.n_down_blocks)]

    @property
    def mid_blocks(self):
        return [getattr(self, f"mid_blocks_{i}") for i in range(self.n_mid_blocks)]

    @property
    def up_blocks(self):
        return [getattr(self, f"up_blocks_{i}") for i in range(self.n_up_blocks)]

    def __call__(
        self,
        x: mx.array,
        mask: mx.array,
        mu: mx.array,
        t: mx.array,
        spks: mx.array = None,
        cond: mx.array = None,
    ) -> mx.array:
        """
        Forward pass.

        Args:
            x: Noisy input (B, in_channels, T)
            mask: Mask (B, 1, T)
            mu: Condition (B, channels, T)
            t: Timestep (B,)
            spks: Speaker embeddings (B, spk_dim)
            cond: Additional conditioning

        Returns:
            Predicted noise (B, out_channels, T)
        """
        # Time embedding
        t = self.time_embeddings(t)
        t = self.time_mlp(t)

        # Concatenate conditioning
        x = mx.concatenate([x, mu], axis=1)
        if spks is not None:
            spks_expanded = mx.broadcast_to(
                mx.expand_dims(spks, -1), (spks.shape[0], spks.shape[1], x.shape[2])
            )
            x = mx.concatenate([x, spks_expanded], axis=1)
        if cond is not None:
            x = mx.concatenate([x, cond], axis=1)

        # Down blocks
        hiddens = []
        masks = [mask]
        for down_block in self.down_blocks:
            mask_down = masks[-1]
            x = down_block.resnet(x, mask_down, t)

            # Transformer blocks
            x_t = mx.swapaxes(x, 1, 2)  # (B, C, T) -> (B, T, C)
            for transformer_block in down_block.transformer_blocks:
                x_t = transformer_block(x_t, attention_mask=None, timestep=t)
            x = mx.swapaxes(x_t, 1, 2)  # (B, T, C) -> (B, C, T)

            hiddens.append(x)
            x = down_block.downsample(x * mask_down)
            # Downsample mask
            masks.append(mask_down[:, :, ::2])

        masks = masks[:-1]
        mask_mid = masks[-1]

        # Mid blocks
        for mid_block in self.mid_blocks:
            x = mid_block.resnet(x, mask_mid, t)

            x_t = mx.swapaxes(x, 1, 2)
            for transformer_block in mid_block.transformer_blocks:
                x_t = transformer_block(x_t, attention_mask=None, timestep=t)
            x = mx.swapaxes(x_t, 1, 2)

        # Up blocks
        for up_block in self.up_blocks:
            mask_up = masks.pop()
            skip = hiddens.pop()
            # Truncate x to match skip length
            x = mx.concatenate([x[:, :, : skip.shape[-1]], skip], axis=1)
            x = up_block.resnet(x, mask_up, t)

            x_t = mx.swapaxes(x, 1, 2)
            for transformer_block in up_block.transformer_blocks:
                x_t = transformer_block(x_t, attention_mask=None, timestep=t)
            x = mx.swapaxes(x_t, 1, 2)

            x = up_block.upsample(x * mask_up)

        # Final layers
        x = self.final_block(x, mask_up)
        # final_proj: (B, C, T) -> transpose -> conv -> transpose back
        x_proj = mx.swapaxes(x * mask_up, 1, 2)  # (B, C, T) -> (B, T, C)
        output = self.final_proj(x_proj)
        output = mx.swapaxes(output, 1, 2)  # (B, T, C) -> (B, C, T)
        return output * mask
