"""Cache-aware FastConformer encoder for Nemotron 3.5 ASR.

Faithful to NeMo's ``ConformerEncoder`` for this checkpoint:
  * causal ``dw_striding`` subsampling (CausalConv2D, asymmetric pad),
  * convolution module with ``layer_norm`` and a *causal* depthwise conv,
  * ``rel_pos`` attention with the ``chunked_limited`` look-ahead mask,
  * no bias in the conformer-block projections (``use_bias=False``).
"""

import math

import mlx.core as mx
import mlx.nn as nn

from .attention import RelPositionalEncoding, RelPositionMultiHeadAttention
from .config import ConformerArgs

NEG_INF = -1e30


def create_chunked_limited_mask(seq_len: int, left_context: int, right_context: int):
    """Additive (1, 1, T, T) attention mask for NeMo's ``chunked_limited`` style.

    Frames are grouped into non-overlapping chunks of ``right_context + 1`` frames.
    A frame may attend to its own chunk and the previous ``left_context // chunk``
    chunks. Returns 0.0 where visible and a large negative value where blocked.
    """
    chunk_size = right_context + 1
    left_chunks = left_context // chunk_size if left_context >= 0 else 10**8

    chunk_idx = mx.arange(seq_len, dtype=mx.int32) // chunk_size
    diff = mx.expand_dims(chunk_idx, 1) - mx.expand_dims(chunk_idx, 0)  # (T, T)
    visible = (diff >= 0) & (diff <= left_chunks)
    mask = mx.where(visible, 0.0, NEG_INF).astype(mx.float32)
    return mask[None, None]  # (1, 1, T, T)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, use_bias: bool):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=use_bias)
        self.activation = nn.SiLU()
        self.linear2 = nn.Linear(d_ff, d_model, bias=use_bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear2(self.activation(self.linear1(x)))


class ConformerConvolution(nn.Module):
    """Convolution module with causal depthwise conv and LayerNorm.

    NeMo names the normalization layer ``batch_norm`` even when it is a LayerNorm;
    the attribute name is kept so checkpoint keys line up.
    """

    def __init__(self, args: ConformerArgs):
        super().__init__()
        d_model = args.d_model
        self.kernel_size = args.conv_kernel_size

        # Resolve conv_context_size to [left, right] padding for the depthwise conv.
        ctx = args.conv_context_size
        if ctx == "causal":
            self.pad_left, self.pad_right = self.kernel_size - 1, 0
        else:
            self.pad_left, self.pad_right = int(ctx[0]), int(ctx[1])

        self.pointwise_conv1 = nn.Conv1d(
            d_model, d_model * 2, kernel_size=1, bias=args.use_bias
        )
        self.depthwise_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=self.kernel_size,
            groups=d_model,
            bias=args.use_bias,
        )
        if args.conv_norm_type != "layer_norm":
            raise NotImplementedError(
                f"conv_norm_type={args.conv_norm_type} not supported (expected layer_norm)"
            )
        self.batch_norm = nn.LayerNorm(d_model)
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(
            d_model, d_model, kernel_size=1, bias=args.use_bias
        )

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, T, C)
        x = self.pointwise_conv1(x)
        x = nn.glu(x, axis=-1)
        # causal depthwise conv: left-pad the time axis only.
        x = mx.pad(x, ((0, 0), (self.pad_left, self.pad_right), (0, 0)))
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        return x


class ConformerBlock(nn.Module):
    def __init__(self, args: ConformerArgs):
        super().__init__()
        d_ff = args.d_model * args.ff_expansion_factor

        self.norm_feed_forward1 = nn.LayerNorm(args.d_model)
        self.feed_forward1 = FeedForward(args.d_model, d_ff, args.use_bias)

        self.norm_self_att = nn.LayerNorm(args.d_model)
        self.self_attn = RelPositionMultiHeadAttention(
            args.n_heads, args.d_model, bias=args.use_bias
        )

        self.norm_conv = nn.LayerNorm(args.d_model)
        self.conv = ConformerConvolution(args)

        self.norm_feed_forward2 = nn.LayerNorm(args.d_model)
        self.feed_forward2 = FeedForward(args.d_model, d_ff, args.use_bias)

        self.norm_out = nn.LayerNorm(args.d_model)

    def __call__(self, x, pos_emb, mask=None):
        x = x + 0.5 * self.feed_forward1(self.norm_feed_forward1(x))
        x = x + self.self_attn(self.norm_self_att(x), pos_emb=pos_emb, mask=mask)
        x = x + self.conv(self.norm_conv(x))
        x = x + 0.5 * self.feed_forward2(self.norm_feed_forward2(x))
        return self.norm_out(x)


class CausalDwStridingSubsampling(nn.Module):
    """Depthwise-striding conv subsampling with causal (asymmetric) padding."""

    def __init__(self, args: ConformerArgs):
        super().__init__()
        ch = args.subsampling_conv_channels
        self.sampling_num = int(math.log(args.subsampling_factor, 2))
        self.kernel_size = 3
        self.stride = 2
        # CausalConv2D padding: left = k-1, right = stride-1 (same on time & freq).
        self.pad_left = self.kernel_size - 1
        self.pad_right = self.stride - 1

        # Output frequency dimension after `sampling_num` strided convs.
        freq = args.feat_in
        for _ in range(self.sampling_num):
            freq = (
                math.floor(
                    (freq + self.pad_left + self.pad_right - self.kernel_size)
                    / self.stride
                )
                + 1
            )

        # Build the conv stack matching NeMo indices (ReLU at 1/4/7 carry no params).
        conv = [
            nn.Conv2d(1, ch, kernel_size=3, stride=2, padding=0),  # 0
            nn.ReLU(),  # 1
        ]
        for _ in range(self.sampling_num - 1):
            conv.append(
                nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=0, groups=ch)
            )  # 2 / 5 depthwise
            conv.append(nn.Conv2d(ch, ch, kernel_size=1, stride=1, padding=0))  # 3 / 6
            conv.append(nn.ReLU())  # 4 / 7
        self.conv = conv
        # Indices of the stride-2 3x3 convs that need causal padding.
        self._strided_idx = {0} | {2 + 3 * i for i in range(self.sampling_num - 1)}

        self.out = nn.Linear(ch * freq, args.d_model)

    def _calc_length(self, length: int) -> int:
        for _ in range(self.sampling_num):
            length = (
                math.floor(
                    (length + self.pad_left + self.pad_right - self.kernel_size)
                    / self.stride
                )
                + 1
            )
        return length

    def __call__(self, x: mx.array, lengths: mx.array):
        # x: (B, T, F) -> (B, T, F, 1) channels-last for MLX conv2d.
        out_lengths = mx.array(
            [self._calc_length(int(l)) for l in lengths], dtype=mx.int32
        )
        x = mx.expand_dims(x, axis=-1)
        for i, layer in enumerate(self.conv):
            if i in self._strided_idx:
                x = mx.pad(
                    x,
                    (
                        (0, 0),
                        (self.pad_left, self.pad_right),  # time
                        (self.pad_left, self.pad_right),  # freq
                        (0, 0),
                    ),
                )
            x = layer(x)
        # (B, T', F', C) -> (B, T', C*F') with channel-major ordering (matches NeMo).
        b, t, f, c = x.shape
        x = x.transpose(0, 1, 3, 2).reshape(b, t, c * f)
        x = self.out(x)
        return x, out_lengths


class Conformer(nn.Module):
    def __init__(self, args: ConformerArgs):
        super().__init__()
        self.args = args
        self.pos_enc = RelPositionalEncoding(
            d_model=args.d_model,
            max_len=args.pos_emb_max_len,
            scale_input=args.xscaling,
        )
        self.pre_encode = CausalDwStridingSubsampling(args)
        self.layers = [ConformerBlock(args) for _ in range(args.n_layers)]

    def __call__(
        self, x: mx.array, lengths: mx.array | None = None, att_context_size=None
    ):
        if lengths is None:
            lengths = mx.full((x.shape[0],), x.shape[-2], dtype=mx.int32)

        x, out_lengths = self.pre_encode(x, lengths)
        x, pos_emb = self.pos_enc(x)

        if att_context_size is None:
            att_context_size = self.args.att_context_size[0]
        left_context, right_context = att_context_size

        mask = None
        if self.args.att_context_style == "chunked_limited":
            mask = create_chunked_limited_mask(x.shape[1], left_context, right_context)
            mask = mask.astype(x.dtype)

        for layer in self.layers:
            x = layer(x, pos_emb=pos_emb, mask=mask)

        return x, out_lengths
