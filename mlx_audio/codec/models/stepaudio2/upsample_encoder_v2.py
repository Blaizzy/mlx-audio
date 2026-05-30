from typing import Tuple

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.tts.models.chatterbox.s3gen.transformer.attention import (
    MultiHeadedAttention,
    RelPositionMultiHeadedAttention,
)
from mlx_audio.tts.models.chatterbox.s3gen.transformer.embedding import (
    EspnetRelPositionalEncoding,
    RelPositionalEncoding,
)
from mlx_audio.tts.models.chatterbox.s3gen.transformer.encoder_layer import (
    ConformerEncoderLayer,
)
from mlx_audio.tts.models.chatterbox.s3gen.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,
)
from mlx_audio.tts.models.chatterbox.s3gen.transformer.subsampling import (
    LinearNoSubsampling,
)


class Upsample1D(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: int,
        stride: int = 2,
        scale_factor: float | None = None,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels
        self.stride = stride
        self.scale_factor = (
            float(stride) if scale_factor is None else float(scale_factor)
        )
        if self.scale_factor != float(stride):
            raise ValueError(
                "StepAudio2 MLX upsampling currently requires integer scale"
            )
        self.conv = nn.Conv1d(channels, out_channels, stride * 2 + 1)

    def __call__(
        self, inputs: mx.array, input_lengths: mx.array
    ) -> Tuple[mx.array, mx.array]:
        outputs = mx.repeat(inputs, self.stride, axis=2)
        outputs = mx.pad(outputs, [(0, 0), (0, 0), (self.stride * 2, 0)])
        outputs = mx.transpose(outputs, (0, 2, 1))
        outputs = self.conv(outputs)
        outputs = mx.transpose(outputs, (0, 2, 1))
        return outputs, input_lengths * self.stride


class PreLookaheadLayer(nn.Module):
    def __init__(self, channels: int, pre_lookahead_len: int = 1):
        super().__init__()
        self.channels = channels
        self.pre_lookahead_len = pre_lookahead_len
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=pre_lookahead_len + 1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3)

    def __call__(self, inputs: mx.array) -> mx.array:
        outputs = mx.pad(inputs, [(0, 0), (0, self.pre_lookahead_len), (0, 0)])
        outputs = nn.leaky_relu(self.conv1(outputs))
        outputs = mx.pad(outputs, [(0, 0), (2, 0), (0, 0)])
        outputs = self.conv2(outputs)
        return outputs + inputs


def make_pad_mask(lengths: mx.array, max_len: int) -> mx.array:
    batch_size = lengths.shape[0]
    seq_range = mx.arange(max_len)
    seq_range = mx.broadcast_to(mx.expand_dims(seq_range, 0), (batch_size, max_len))
    return seq_range >= mx.expand_dims(lengths, -1)


class UpsampleConformerEncoderV2(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        input_layer: str = "linear",
        pre_lookahead_len: int = 3,
        num_blocks: int = 6,
        num_up_blocks: int = 4,
        up_stride: int = 2,
        up_scale_factor: float = 2,
        attention_heads: int = 4,
        pos_enc_layer_type: str = "rel_pos_espnet",
        selfattention_layer_type: str = "rel_selfattn",
        key_bias: bool = True,
        linear_units: int = 2048,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        normalize_before: bool = True,
        activation_type: str = "swish",
        **kwargs,
    ):
        super().__init__()
        if input_layer != "linear":
            raise ValueError(f"Unsupported input layer: {input_layer}")
        if activation_type != "swish":
            raise ValueError(f"Unsupported activation: {activation_type}")

        self._output_size = output_size
        pos_enc = self._positional_encoding(
            pos_enc_layer_type, output_size, positional_dropout_rate
        )
        self.embed = LinearNoSubsampling(input_size, output_size, dropout_rate, pos_enc)
        self.normalize_before = normalize_before
        self.after_norm = nn.LayerNorm(output_size, eps=1e-5)
        activation = nn.SiLU()

        self_attn_class = self._attention_class(selfattention_layer_type)
        self.pre_lookahead_layer = PreLookaheadLayer(
            channels=output_size, pre_lookahead_len=pre_lookahead_len
        )
        self.encoders = [
            ConformerEncoderLayer(
                output_size,
                self_attn_class(
                    attention_heads,
                    output_size,
                    attention_dropout_rate,
                    key_bias,
                ),
                PositionwiseFeedForward(
                    output_size, linear_units, dropout_rate, activation
                ),
                None,
                None,
                dropout_rate,
                normalize_before,
            )
            for _ in range(num_blocks)
        ]
        self.up_layer = Upsample1D(
            channels=output_size,
            out_channels=output_size,
            stride=up_stride,
            scale_factor=up_scale_factor,
        )
        up_pos_enc = self._positional_encoding(
            pos_enc_layer_type, output_size, positional_dropout_rate
        )
        self.up_embed = LinearNoSubsampling(
            input_size, output_size, dropout_rate, up_pos_enc
        )
        self.up_encoders = [
            ConformerEncoderLayer(
                output_size,
                self_attn_class(
                    attention_heads,
                    output_size,
                    attention_dropout_rate,
                    key_bias,
                ),
                PositionwiseFeedForward(
                    output_size, linear_units, dropout_rate, activation
                ),
                None,
                None,
                dropout_rate,
                normalize_before,
            )
            for _ in range(num_up_blocks)
        ]

    @staticmethod
    def _positional_encoding(name: str, size: int, dropout_rate: float) -> nn.Module:
        if name == "rel_pos_espnet":
            return EspnetRelPositionalEncoding(size, dropout_rate)
        if name == "rel_pos":
            return RelPositionalEncoding(size, dropout_rate)
        raise ValueError(f"Unsupported positional encoding: {name}")

    @staticmethod
    def _attention_class(name: str) -> type[nn.Module]:
        if name == "rel_selfattn":
            return RelPositionMultiHeadedAttention
        if name == "selfattn":
            return MultiHeadedAttention
        raise ValueError(f"Unsupported self-attention: {name}")

    def output_size(self) -> int:
        return self._output_size

    def _forward_impl_encoder(
        self, x: mx.array, mask: mx.array, pos_emb: mx.array
    ) -> mx.array:
        for layer in self.encoders:
            x, _, _, _ = layer(x, mask, pos_emb)
        return x

    def _forward_impl_up_encoder(
        self, x: mx.array, mask: mx.array, pos_emb: mx.array
    ) -> mx.array:
        for layer in self.up_encoders:
            x, _, _, _ = layer(x, mask, pos_emb)
        return x

    def __call__(self, xs: mx.array, xs_lens: mx.array) -> Tuple[mx.array, mx.array]:
        masks = mx.logical_not(make_pad_mask(xs_lens, xs.shape[1]))
        masks = mx.expand_dims(masks, 1)
        xs, pos_emb, masks = self.embed(xs, masks)
        xs = self.pre_lookahead_layer(xs)
        xs = self._forward_impl_encoder(xs, masks, pos_emb)

        xs = mx.transpose(xs, (0, 2, 1))
        xs, xs_lens = self.up_layer(xs, xs_lens)
        xs = mx.transpose(xs, (0, 2, 1))

        masks = mx.logical_not(make_pad_mask(xs_lens, xs.shape[1]))
        masks = mx.expand_dims(masks, 1)
        xs, pos_emb, masks = self.up_embed(xs, masks)
        xs = self._forward_impl_up_encoder(xs, masks, pos_emb)
        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, masks
