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


def subsequent_chunk_mask(
    size: int,
    chunk_size: int,
    num_left_chunks: int = -1,
) -> mx.array:
    pos_idx = mx.arange(size)
    block_value = ((pos_idx // chunk_size) + 1) * chunk_size
    ret = mx.expand_dims(pos_idx, 0) < mx.expand_dims(block_value, 1)
    return ret


def add_optional_chunk_mask(
    xs: mx.array,
    masks: mx.array,
    use_dynamic_chunk: bool,
    use_dynamic_left_chunk: bool,
    decoding_chunk_size: int,
    static_chunk_size: int,
    num_decoding_left_chunks: int,
) -> mx.array:
    if use_dynamic_chunk:
        max_len = xs.shape[1]
        if decoding_chunk_size < 0:
            chunk_size = max_len
            num_left_chunks = -1
        elif decoding_chunk_size > 0:
            chunk_size = decoding_chunk_size
            num_left_chunks = num_decoding_left_chunks
        else:
            # For training, use full context
            chunk_size = max_len
            num_left_chunks = -1

        chunk_masks = subsequent_chunk_mask(xs.shape[1], chunk_size, num_left_chunks)
        chunk_masks = mx.expand_dims(chunk_masks, 0)  # (1, T, T)
        chunk_masks = masks & chunk_masks  # (B, T, T)
    elif static_chunk_size > 0:
        num_left_chunks = num_decoding_left_chunks
        chunk_masks = subsequent_chunk_mask(
            xs.shape[1], static_chunk_size, num_left_chunks
        )
        chunk_masks = mx.expand_dims(chunk_masks, 0)  # (1, T, T)
        chunk_masks = masks & chunk_masks  # (B, T, T)
    else:
        # Full attention - broadcast masks to (B, T, T)
        chunk_masks = mx.broadcast_to(masks, (masks.shape[0], xs.shape[1], xs.shape[1]))

    return chunk_masks


def mask_to_bias(mask: mx.array, dtype: mx.Dtype) -> mx.array:
    mask = mask.astype(dtype)
    bias = (1.0 - mask) * -1.0e10
    return bias


class CausalConv1d(nn.Module):

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
        x = mx.pad(x, [(0, 0), (self.causal_padding, 0), (0, 0)])
        x = self.conv(x)
        x = mx.swapaxes(x, 1, 2)  # (B, T, C) -> (B, C, T)
        return x


class CausalBlock1D(nn.Module):

    def __init__(self, dim: int, dim_out: int):
        super().__init__()
        self.conv = CausalConv1d(dim, dim_out, 3)
        self.norm = nn.LayerNorm(dim_out)

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        output = self.conv(x * mask)
        output = mx.swapaxes(output, 1, 2)  # (B, C, T) -> (B, T, C)
        output = self.norm(output)
        output = mx.swapaxes(output, 1, 2)  # (B, T, C) -> (B, C, T)
        output = nn.mish(output)
        return output * mask


class CausalResnetBlock1D(ResnetBlock1D):

    def __init__(self, dim: int, dim_out: int, time_emb_dim: int, groups: int = 8):
        super().__init__(dim, dim_out, time_emb_dim, groups)
        self.block1 = CausalBlock1D(dim, dim_out)
        self.block2 = CausalBlock1D(dim_out, dim_out)


class DownBlock(nn.Module):

    def __init__(self, resnet, transformer_blocks, downsample):
        super().__init__()
        self.resnet = resnet
        for i, block in enumerate(transformer_blocks):
            setattr(self, f"transformer_{i}", block)
        self.n_transformer = len(transformer_blocks)
        self.downsample = downsample

    @property
    def transformer_blocks(self):
        return [getattr(self, f"transformer_{i}") for i in range(self.n_transformer)]


class MidBlock(nn.Module):

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
        static_chunk_size: int = 50,
        num_decoding_left_chunks: int = 2,
    ):
        super().__init__()
        channels = tuple(channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.causal = causal
        self.static_chunk_size = static_chunk_size
        self.num_decoding_left_chunks = num_decoding_left_chunks

        self.time_embeddings = SinusoidalPosEmb(in_channels)
        time_embed_dim = channels[0] * 4
        self.time_mlp = TimestepEmbedding(
            in_channels=in_channels,
            time_embed_dim=time_embed_dim,
            act_fn="silu",
        )

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
        streaming: bool = False,
    ) -> mx.array:

        # Time embedding
        t_emb = self.time_embeddings(t)
        t_emb = self.time_mlp(t_emb)

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
            x = down_block.resnet(x, mask_down, t_emb)

            # Transformer blocks with attention mask
            x_t = mx.swapaxes(x, 1, 2)  # (B, C, T) -> (B, T, C)

            # Compute attention mask based on streaming mode
            if streaming:
                # Use chunk-based causal attention for streaming
                attn_mask = add_optional_chunk_mask(
                    x_t,
                    mask_down.astype(mx.bool_),
                    False,
                    False,
                    0,
                    self.static_chunk_size,
                    -1,
                )
            else:
                # Use full context attention (non-streaming)
                # Expand mask_down to (B, T, T) for full attention
                attn_mask = add_optional_chunk_mask(
                    x_t, mask_down.astype(mx.bool_), False, False, 0, 0, -1
                )
                # Repeat along query dimension for proper broadcasting
                attn_mask = mx.broadcast_to(
                    attn_mask, (attn_mask.shape[0], x_t.shape[1], attn_mask.shape[2])
                )

            # Convert boolean mask to additive bias
            attn_bias = mask_to_bias(attn_mask, x_t.dtype)

            for transformer_block in down_block.transformer_blocks:
                x_t = transformer_block(x_t, attention_mask=attn_bias, timestep=t_emb)
            x = mx.swapaxes(x_t, 1, 2)  # (B, T, C) -> (B, C, T)

            hiddens.append(x)
            x = down_block.downsample(x * mask_down)
            # Downsample mask
            masks.append(mask_down[:, :, ::2])

        masks = masks[:-1]
        mask_mid = masks[-1]

        # Mid blocks
        for mid_block in self.mid_blocks:
            x = mid_block.resnet(x, mask_mid, t_emb)

            x_t = mx.swapaxes(x, 1, 2)

            # Compute attention mask for mid blocks
            if streaming:
                attn_mask = add_optional_chunk_mask(
                    x_t,
                    mask_mid.astype(mx.bool_),
                    False,
                    False,
                    0,
                    self.static_chunk_size,
                    -1,
                )
            else:
                attn_mask = add_optional_chunk_mask(
                    x_t, mask_mid.astype(mx.bool_), False, False, 0, 0, -1
                )
                attn_mask = mx.broadcast_to(
                    attn_mask, (attn_mask.shape[0], x_t.shape[1], attn_mask.shape[2])
                )

            attn_bias = mask_to_bias(attn_mask, x_t.dtype)

            for transformer_block in mid_block.transformer_blocks:
                x_t = transformer_block(x_t, attention_mask=attn_bias, timestep=t_emb)
            x = mx.swapaxes(x_t, 1, 2)

        # Up blocks
        for up_block in self.up_blocks:
            mask_up = masks.pop()
            skip = hiddens.pop()
            # Truncate x to match skip length
            x = mx.concatenate([x[:, :, : skip.shape[-1]], skip], axis=1)
            x = up_block.resnet(x, mask_up, t_emb)

            x_t = mx.swapaxes(x, 1, 2)

            # Compute attention mask for up blocks
            if streaming:
                attn_mask = add_optional_chunk_mask(
                    x_t,
                    mask_up.astype(mx.bool_),
                    False,
                    False,
                    0,
                    self.static_chunk_size,
                    -1,
                )
            else:
                attn_mask = add_optional_chunk_mask(
                    x_t, mask_up.astype(mx.bool_), False, False, 0, 0, -1
                )
                attn_mask = mx.broadcast_to(
                    attn_mask, (attn_mask.shape[0], x_t.shape[1], attn_mask.shape[2])
                )

            attn_bias = mask_to_bias(attn_mask, x_t.dtype)

            for transformer_block in up_block.transformer_blocks:
                x_t = transformer_block(x_t, attention_mask=attn_bias, timestep=t_emb)
            x = mx.swapaxes(x_t, 1, 2)

            x = up_block.upsample(x * mask_up)

        # Final layers
        x = self.final_block(x, mask_up)

        x_proj = mx.swapaxes(x * mask_up, 1, 2)  # (B, C, T) -> (B, T, C)
        output = self.final_proj(x_proj)
        output = mx.swapaxes(output, 1, 2)  # (B, T, C) -> (B, C, T)
        return output * mask
