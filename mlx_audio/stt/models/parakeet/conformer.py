import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.stt.models.parakeet.attention import (
    MultiHeadAttention,
    RelPositionalEncoding,
    RelPositionMultiHeadAttention,
)


@dataclass
class ConformerArgs:
    feat_in: int  # mel-log
    n_layers: int
    d_model: int
    n_heads: int
    ff_expansion_factor: int
    subsampling_factor: int
    self_attention_model: str
    subsampling: str
    conv_kernel_size: int
    subsampling_conv_channels: int
    pos_emb_max_len: int
    causal_downsampling: bool = False
    use_bias: bool = True
    xscaling: bool = False
    pos_bias_u: mx.array | None = None
    pos_bias_v: mx.array | None = None
    subsampling_conv_chunking_factor: int = 1


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, use_bias: bool = True):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=use_bias)
        self.activation = nn.SiLU()
        self.linear2 = nn.Linear(d_ff, d_model, bias=use_bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear2(self.activation(self.linear1(x)))


class Convolution(nn.Module):
    def __init__(self, args: ConformerArgs):
        assert (args.conv_kernel_size - 1) % 2 == 0
        super().__init__()

        self.pointwise_conv1 = nn.Conv1d(
            args.d_model,
            args.d_model * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=args.use_bias,
        )
        self.depthwise_conv = nn.Conv1d(
            args.d_model,
            args.d_model,
            kernel_size=args.conv_kernel_size,
            stride=1,
            padding=(args.conv_kernel_size - 1) // 2,
            groups=args.d_model,
            bias=args.use_bias,
        )
        self.batch_norm = nn.BatchNorm(args.d_model)
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(
            args.d_model,
            args.d_model,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=args.use_bias,
        )

    def __call__(self, x: mx.array) -> mx.array:
        # x = x.swapaxes(1, 2)

        x = self.pointwise_conv1(x)
        x = nn.glu(x, axis=2)  # might make it variable later

        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)

        return x


class ConformerBlock(nn.Module):
    def __init__(self, args: ConformerArgs):
        super().__init__()
        ff_hidden_dim = args.d_model * args.ff_expansion_factor

        self.norm_feed_forward1 = nn.LayerNorm(args.d_model)
        self.feed_forward1 = FeedForward(args.d_model, ff_hidden_dim, args.use_bias)

        self.norm_self_att = nn.LayerNorm(args.d_model)
        self.self_attn = (
            RelPositionMultiHeadAttention(
                args.n_heads,
                args.d_model,
                bias=args.use_bias,
                pos_bias_u=args.pos_bias_u,
                pos_bias_v=args.pos_bias_v,
            )
            if args.self_attention_model == "rel_pos"
            else MultiHeadAttention(
                args.n_heads,
                args.d_model,
                bias=True,
            )
        )

        self.norm_conv = nn.LayerNorm(args.d_model)
        self.conv = Convolution(args)

        self.norm_feed_forward2 = nn.LayerNorm(args.d_model)
        self.feed_forward2 = FeedForward(args.d_model, ff_hidden_dim, args.use_bias)

        self.norm_out = nn.LayerNorm(args.d_model)

    def __call__(
        self,
        x: mx.array,
        pos_emb: mx.array | None = None,
        mask: mx.array | None = None,
        cache=None,
    ) -> mx.array:
        x += 0.5 * self.feed_forward1(self.norm_feed_forward1(x))

        x_norm = self.norm_self_att(x)
        x += self.self_attn(
            x_norm, x_norm, x_norm, mask=mask, pos_emb=pos_emb, cache=cache
        )

        x += self.conv(self.norm_conv(x))
        x += 0.5 * self.feed_forward2(self.norm_feed_forward2(x))

        return self.norm_out(x)


class DwStridingSubsampling(nn.Module):
    def __init__(self, args: ConformerArgs):
        super().__init__()

        assert (
            args.subsampling_factor > 0
            and (args.subsampling_factor & (args.subsampling_factor - 1)) == 0
        )
        self.subsampling_conv_chunking_factor = args.subsampling_conv_chunking_factor
        self._conv_channels = args.subsampling_conv_channels
        self._sampling_num = int(math.log(args.subsampling_factor, 2))
        self._stride = 2
        self._kernel_size = 3
        self._padding = (self._kernel_size - 1) // 2

        in_channels = 1
        final_freq_dim = args.feat_in
        for _ in range(self._sampling_num):
            final_freq_dim = (
                math.floor(
                    (final_freq_dim + 2 * self._padding - self._kernel_size)
                    / self._stride
                )
                + 1
            )
            if final_freq_dim < 1:
                raise ValueError("Non-positive final frequency dimension!")

        self.conv = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=self._conv_channels,
                kernel_size=self._kernel_size,
                stride=self._stride,
                padding=self._padding,
            ),
            nn.ReLU(),
        ]
        in_channels = self._conv_channels

        for _ in range(self._sampling_num - 1):
            self.conv.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=self._kernel_size,
                    stride=self._stride,
                    padding=self._padding,
                    groups=in_channels,
                )
            )
            self.conv.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=self._conv_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    groups=1,
                )
            )
            self.conv.append(nn.ReLU())

        self.out = nn.Linear(self._conv_channels * final_freq_dim, args.d_model)

    def conv_forward(self, x: mx.array) -> mx.array:
        x = x.transpose((0, 2, 3, 1))
        for layer in self.conv:
            x = layer(x)
        return x.transpose((0, 3, 1, 2))

    def conv_split_by_batch(self, x: mx.array) -> tuple[mx.array, bool]:
        b = x.shape[0]
        if b == 1:
            return x, False

        if self.subsampling_conv_chunking_factor > 1:
            cf = self.subsampling_conv_chunking_factor
        else:
            x_ceil = 2**31 / self._conv_channels * self._stride * self._stride
            p = math.ceil(math.log(x.size / x_ceil, 2))
            cf: int = 2**p

        new_batch_size = b // cf
        if new_batch_size == 0:
            return x, False

        return (
            mx.concat(
                [self.conv_forward(chunk) for chunk in mx.split(x, new_batch_size, 0)]
            ),
            True,
        )

    def __call__(self, x: mx.array, lengths: mx.array) -> tuple[mx.array, mx.array]:
        for _ in range(self._sampling_num):
            lengths = (
                mx.floor(
                    (lengths + 2 * self._padding - self._kernel_size) / self._stride
                )
                + 1.0
            )
        lengths = lengths.astype(mx.int32)

        x = mx.expand_dims(x, axis=1)

        if self.subsampling_conv_chunking_factor != -1:
            if self.subsampling_conv_chunking_factor == 1:
                x_ceil = 2**31 / self._conv_channels * self._stride * self._stride
                need_to_split = x.size > x_ceil
            else:
                need_to_split = True

            if need_to_split:
                x, success = self.conv_split_by_batch(x)
                if not success:
                    # TODO: Add channel splitting
                    x = self.conv_forward(x)  # try anyways
            else:
                x = self.conv_forward(x)
        else:
            x = self.conv_forward(x)

        x = x.swapaxes(1, 2).reshape(x.shape[0], x.shape[2], -1)
        x = self.out(x)
        return x, lengths


class Conformer(nn.Module):
    def __init__(self, args: ConformerArgs):
        super().__init__()

        if args.self_attention_model == "rel_pos":
            self.pos_enc = RelPositionalEncoding(
                d_model=args.d_model,
                max_len=args.pos_emb_max_len,
                scale_input=args.xscaling,
            )
        else:
            self.pos_enc = None

        if args.subsampling_factor > 1:
            if args.subsampling == "dw_striding" and args.causal_downsampling is False:
                self.pre_encode = DwStridingSubsampling(args)
            else:
                self.pre_encode = nn.Identity()
                raise NotImplementedError(
                    "Other subsampling haven't been implemented yet!"
                )
        else:
            self.pre_encode = nn.Linear(args.feat_in, args.d_model)

        self.layers = [ConformerBlock(args) for _ in range(args.n_layers)]

    def __call__(
        self, x: mx.array, lengths: mx.array | None = None, cache=None
    ) -> tuple[mx.array, mx.array]:
        if lengths is None:
            lengths = mx.full(
                (x.shape[0],),
                x.shape[-2],
                dtype=mx.int64,
            )

        if isinstance(self.pre_encode, DwStridingSubsampling):
            x, out_lengths = self.pre_encode(x, lengths)
        elif isinstance(self.pre_encode, nn.Linear):
            x = self.pre_encode(x)
            out_lengths = lengths
        else:
            raise NotImplementedError("Unimplemented pre-encoding layer type!")

        if cache is None:
            cache = [None] * len(self.layers)

        pos_emb = None
        if self.pos_enc is not None:
            x, pos_emb = self.pos_enc(
                x,
                offset=cache[0].offset if cache[0] is not None else 0,  # type: ignore
            )

        for layer, c in zip(self.layers, cache):
            x = layer(x, pos_emb=pos_emb, cache=c)

        return x, out_lengths
