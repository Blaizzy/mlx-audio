from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.tts.models.indextts.attention import (
    MultiHeadAttention,
    RelPositionalEncoding,
    RelPositionMultiHeadAttention,
)


@dataclass
class ConformerArgs:
    input_size: int = 100
    output_size: int = 256
    num_blocks: int = 6
    linear_units: int = 2048
    attention_heads: int = 4
    pos_enc_layer_type: str = "rel_pos"
    input_layer: str = "conv2d"
    cnn_module_kernel: int = 15
    pos_emb_max_len: int = 2048
    causal_downsampling: bool = False
    use_bias: bool = True
    xscaling: bool = True
    macaron_style: bool = False
    pos_bias_u: mx.array | None = None
    pos_bias_v: mx.array | None = None
    perceiver_mult: int = 2


class FeedForward(nn.Module):
    def __init__(self, dim: int, d_ff: int, use_bias: bool = True):
        super().__init__()
        self.w_1 = nn.Linear(dim, d_ff, bias=use_bias)
        self.activation = nn.SiLU()
        self.w_2 = nn.Linear(d_ff, dim, bias=use_bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w_2(self.activation(self.w_1(x)))


class Convolution(nn.Module):
    def __init__(self, args: ConformerArgs):
        assert (args.cnn_module_kernel - 1) % 2 == 0
        super().__init__()

        self.pointwise_conv1 = nn.Conv1d(
            args.output_size,
            args.output_size * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=args.use_bias,
        )
        self.depthwise_conv = nn.Conv1d(
            args.output_size,
            args.output_size,
            kernel_size=args.cnn_module_kernel,
            stride=1,
            padding=(args.cnn_module_kernel - 1) // 2,
            groups=args.output_size,
            bias=args.use_bias,
        )
        self.norm = nn.LayerNorm(args.output_size)
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(
            args.output_size,
            args.output_size,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=args.use_bias,
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = self.pointwise_conv1(x)
        x = nn.glu(x, axis=2)

        x = self.depthwise_conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)

        return x


class ConformerBlock(nn.Module):
    def __init__(self, args: ConformerArgs):
        super().__init__()
        self.macaron_style = args.macaron_style
        self.ff_scale = 0.5 if self.macaron_style else 1
        if args.macaron_style:
            self.norm_ff_macaron = nn.LayerNorm(args.output_size)
            self.feed_forward_macaron = FeedForward(
                args.output_size, args.linear_units, args.use_bias
            )

        self.norm_mha = nn.LayerNorm(args.output_size)
        self.self_attn = (
            RelPositionMultiHeadAttention(
                args.attention_heads,
                args.output_size,
                bias=args.use_bias,
                pos_bias_u=args.pos_bias_u,
                pos_bias_v=args.pos_bias_v,
            )
            if args.pos_enc_layer_type == "rel_pos"
            else MultiHeadAttention(
                args.attention_heads,
                args.output_size,
                bias=True,
            )
        )

        self.norm_conv = nn.LayerNorm(args.output_size)
        self.conv_module = Convolution(args)

        self.norm_ff = nn.LayerNorm(args.output_size)
        self.feed_forward = FeedForward(
            args.output_size, args.linear_units, args.use_bias
        )

        self.norm_final = nn.LayerNorm(args.output_size)

    def __call__(
        self,
        x: mx.array,
        pos_emb: mx.array | None = None,
        mask: mx.array | None = None,
        cache=None,
    ) -> mx.array:
        if self.macaron_style:
            x += self.ff_scale * self.feed_forward_macaron(self.norm_ff_macaron(x))

        x_norm = self.norm_mha(x)
        x += self.self_attn(
            x_norm, x_norm, x_norm, mask=mask, pos_emb=pos_emb, cache=cache
        )

        x += self.conv_module(self.norm_conv(x))
        x += self.ff_scale * self.feed_forward(self.norm_ff(x))

        return self.norm_final(x)


class Conv2dSubsampling(nn.Module):
    CONV_LAYERS = {
        "conv2d2": [(3, 2)],
        "conv2d3": [(5, 3)],
        "conv2d4": [(3, 2), (3, 2)],
        "conv2d6": [(3, 2), (5, 3)],
        "conv2d8": [(3, 2), (3, 2), (3, 2)],
    }
    CONV_MASKS = {
        "conv2d2": [slice(2, None, 2)],
        "conv2d3": [slice(None, -2, 3)],
        "conv2d4": [slice(2, None, 2), slice(2, None, 2)],
        "conv2d6": [slice(2, None, 2), slice(4, None, 3)],
        "conv2d8": [slice(2, None, 2), slice(2, None, 2), slice(2, None, 2)],
    }

    def __init__(self, args: ConformerArgs):
        super().__init__()
        conv_layers = self.CONV_LAYERS[args.input_layer]

        self.mask_patterns = self.CONV_MASKS[args.input_layer]
        self.conv = []
        self.subsampling_rate = 0

        in_channels = 1
        out_freq = args.input_size
        for kernel_size, stride in conv_layers:
            self.conv.append(
                nn.Conv2d(
                    in_channels,
                    args.output_size,
                    kernel_size=kernel_size,
                    stride=stride,
                )
            )
            self.conv.append(nn.ReLU())

            in_channels = args.output_size
            out_freq = (out_freq - kernel_size + stride) // stride
            self.subsampling_rate *= stride

        self.out = [nn.Linear(args.output_size * out_freq, args.output_size)]

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None):
        x = x[:, :, :, None]

        for layer in self.conv:
            x = layer(x)

        x = x.swapaxes(2, 3).reshape(*x.shape[:2], -1)

        for layer in self.out:
            x = layer(x)

        if mask is not None:
            for pattern in self.mask_patterns:
                mask = mask[pattern]

        return x, mask


class Conformer(nn.Module):
    def __init__(self, args: ConformerArgs):
        super().__init__()

        if args.pos_enc_layer_type == "rel_pos":
            self.pos_enc = RelPositionalEncoding(
                d_model=args.output_size,
                max_len=args.pos_emb_max_len,
                scale_input=args.xscaling,
            )
        else:
            self.pos_enc = None

        self.embed = Conv2dSubsampling(args)
        self.encoders = [ConformerBlock(args) for _ in range(args.num_blocks)]
        self.after_norm = nn.LayerNorm(args.output_size, eps=1e-5)

    def __call__(
        self, x: mx.array, mask: Optional[mx.array] = None, cache=None
    ) -> mx.array:
        x, mask = self.embed(x, mask)

        if cache is None:
            cache = [None] * len(self.encoders)

        pos_emb = None
        if self.pos_enc is not None:
            x, pos_emb = self.pos_enc(
                x,
                offset=cache[0].offset if cache[0] is not None else 0,  # type: ignore
            )

        for layer, c in zip(self.encoders, cache):
            x = layer(x, pos_emb=pos_emb, cache=c, mask=mask)

        x = self.after_norm(x)

        return x
