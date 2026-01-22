import math

import mlx.core as mx
import mlx.nn as nn

from .codec.mimi.modules.conv import (
    Conv1d,
    ConvDownsample1d,
    ConvTrUpsample1d,
    get_extra_padding_for_conv1d,
)
from .codec.mimi.modules.seanet import SeanetConfig, SeanetDecoder, SeanetEncoder
from .codec.mimi.modules.transformer import ProjectedTransformer, TransformerConfig
from .config import MimiConfig


def _reset_kv_cache(cache) -> None:
    cache.keys = None
    cache.values = None
    cache.offset = 0
    if hasattr(cache, "_idx"):
        cache._idx = 0


def pad_for_conv1d(x: mx.array, kernel_size: int, stride: int, padding_total: int = 0):
    extra_padding = get_extra_padding_for_conv1d(
        x, kernel_size=kernel_size, stride=stride, padding_total=padding_total
    )
    if extra_padding <= 0:
        return x
    return mx.pad(x, pad_width=((0, 0), (0, 0), (0, extra_padding)))


class DummyQuantizer(nn.Module):
    def __init__(self, dimension: int, output_dimension: int):
        super().__init__()
        self.output_proj = Conv1d(dimension, output_dimension, 1, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.output_proj(x)


class MimiAdapter(nn.Module):
    def __init__(
        self,
        encoder: SeanetEncoder,
        decoder: SeanetDecoder,
        quantizer: DummyQuantizer,
        frame_rate: float,
        encoder_frame_rate: float,
        sample_rate: int,
        channels: int,
        encoder_transformer: ProjectedTransformer,
        decoder_transformer: ProjectedTransformer,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_transformer = encoder_transformer
        self.decoder_transformer = decoder_transformer
        self.quantizer = quantizer
        self.frame_rate = frame_rate
        self.sample_rate = sample_rate
        self.channels = channels
        self.encoder_frame_rate = encoder_frame_rate
        self.dimension = encoder_transformer.transformer.cfg.d_model

        if encoder_frame_rate != frame_rate:
            if encoder_frame_rate <= frame_rate:
                raise ValueError("Cannot upsample with conv.")
            downsample_stride = encoder_frame_rate / frame_rate
            if downsample_stride != int(downsample_stride):
                raise ValueError(
                    f"Only integer strides are supported, got {downsample_stride}"
                )
            downsample_stride = int(downsample_stride)
            self.downsample = ConvDownsample1d(
                downsample_stride, dim=self.dimension, causal=True
            )
            self.upsample = ConvTrUpsample1d(
                downsample_stride, dim=self.dimension, causal=True
            )
        else:
            self.downsample = None
            self.upsample = None

        self.encoder_cache = self.encoder_transformer.make_cache()
        self.decoder_cache = self.decoder_transformer.make_cache()

    @property
    def frame_size(self) -> int:
        return int(self.sample_rate / self.frame_rate)

    def reset_state(self):
        self.encoder.reset_state()
        self.decoder.reset_state()
        if self.downsample is not None:
            self.downsample.reset_state()
        if self.upsample is not None:
            self.upsample.reset_state()
        for cache in self.encoder_cache:
            _reset_kv_cache(cache)
        for cache in self.decoder_cache:
            _reset_kv_cache(cache)

    def _to_framerate(self, x: mx.array):
        if self.encoder_frame_rate == self.frame_rate:
            return x
        if self.downsample is None:
            raise ValueError("Downsample module is missing.")
        return self.downsample(x)

    def _to_encoder_framerate(self, x: mx.array):
        if self.encoder_frame_rate == self.frame_rate:
            return x
        if self.upsample is None:
            raise ValueError("Upsample module is missing.")
        return self.upsample(x)

    def _to_encoder_framerate_step(self, x: mx.array):
        if self.encoder_frame_rate == self.frame_rate:
            return x
        if self.upsample is None:
            raise ValueError("Upsample module is missing.")
        return self.upsample.step(x)

    def encode_to_latent(self, x: mx.array) -> mx.array:
        if x.ndim != 3:
            raise ValueError(
                "MimiAdapter.encode_to_latent expects audio of shape [B, C, T]."
            )
        self.encoder.reset_state()
        for cache in self.encoder_cache:
            _reset_kv_cache(cache)
        if self.downsample is not None:
            self.downsample.reset_state()

        frame_size = self.frame_size
        x = pad_for_conv1d(x, frame_size, frame_size)
        emb = self.encoder(x)
        emb = self.encoder_transformer(emb, cache=self.encoder_cache)[0]
        return self._to_framerate(emb)

    def decode_from_latent(self, latent: mx.array) -> mx.array:
        self.decoder.reset_state()
        for cache in self.decoder_cache:
            _reset_kv_cache(cache)
        if self.upsample is not None:
            self.upsample.reset_state()

        emb = self._to_encoder_framerate(latent)
        emb = self.decoder_transformer(emb, cache=self.decoder_cache)[0]
        return self.decoder(emb)

    def decode_step(self, latent: mx.array) -> mx.array:
        emb = self._to_encoder_framerate_step(latent)
        emb = self.decoder_transformer(emb, cache=self.decoder_cache)[0]
        return self.decoder.step(emb)

    @classmethod
    def from_config(cls, config: MimiConfig) -> "MimiAdapter":
        seanet_cfg = SeanetConfig(
            dimension=config.seanet.dimension,
            channels=config.seanet.channels,
            causal=True,
            nfilters=config.seanet.n_filters,
            nresidual_layers=config.seanet.n_residual_layers,
            ratios=config.seanet.ratios,
            ksize=config.seanet.kernel_size,
            residual_ksize=config.seanet.residual_kernel_size,
            last_ksize=config.seanet.last_kernel_size,
            dilation_base=config.seanet.dilation_base,
            pad_mode=config.seanet.pad_mode,
            true_skip=True,
            compress=config.seanet.compress,
        )
        encoder = SeanetEncoder(seanet_cfg)
        decoder = SeanetDecoder(seanet_cfg)

        transformer_cfg = TransformerConfig(
            d_model=config.transformer.d_model,
            num_heads=config.transformer.num_heads,
            num_layers=config.transformer.num_layers,
            causal=True,
            norm_first=True,
            bias_ff=False,
            bias_attn=False,
            layer_scale=config.transformer.layer_scale,
            positional_embedding="rope",
            use_conv_block=False,
            cross_attention=False,
            conv_kernel_size=3,
            use_conv_bias=False,
            gating=False,
            norm="layer_norm",
            context=config.transformer.context,
            max_period=config.transformer.max_period,
            max_seq_len=8192,
            kv_repeat=1,
            dim_feedforward=config.transformer.dim_feedforward,
            conv_layout=True,
        )
        output_dims = list(config.transformer.output_dimensions)
        encoder_transformer = ProjectedTransformer(
            transformer_cfg,
            input_dim=config.transformer.input_dimension,
            output_dims=output_dims,
        )
        decoder_transformer = ProjectedTransformer(
            transformer_cfg,
            input_dim=config.transformer.input_dimension,
            output_dims=output_dims,
        )
        quantizer = DummyQuantizer(
            dimension=config.quantizer.dimension,
            output_dimension=config.quantizer.output_dimension,
        )
        encoder_frame_rate = config.sample_rate / math.prod(config.seanet.ratios)
        return cls(
            encoder=encoder,
            decoder=decoder,
            quantizer=quantizer,
            frame_rate=config.frame_rate,
            encoder_frame_rate=encoder_frame_rate,
            sample_rate=config.sample_rate,
            channels=config.channels,
            encoder_transformer=encoder_transformer,
            decoder_transformer=decoder_transformer,
        )
