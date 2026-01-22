import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import hf_hub_download

from .modules import (
    ConvDownsample1d,
    ConvTranspose1d,
    ConvTrUpsample1d,
    EuclideanCodebook,
    ProjectedTransformer,
    SeanetConfig,
    SeanetDecoder,
    SeanetEncoder,
    SplitResidualVectorQuantizer,
    TransformerConfig,
)


def _reset_kv_cache(cache) -> None:
    cache.keys = None
    cache.values = None
    cache.offset = 0
    if hasattr(cache, "_idx"):
        cache._idx = 0


@dataclass
class MimiConfig:
    channels: int
    sample_rate: float
    frame_rate: float
    renormalize: bool
    seanet: SeanetConfig
    transformer: TransformerConfig
    quantizer_nq: int
    quantizer_bins: int
    quantizer_dim: int


def mimi_202407(num_codebooks: int) -> MimiConfig:
    seanet = SeanetConfig(
        dimension=512,
        channels=1,
        causal=True,
        nfilters=64,
        nresidual_layers=1,
        ratios=[8, 6, 5, 4],
        ksize=7,
        residual_ksize=3,
        last_ksize=3,
        dilation_base=2,
        pad_mode="constant",
        true_skip=True,
        compress=2,
    )
    transformer = TransformerConfig(
        d_model=seanet.dimension,
        num_heads=8,
        num_layers=8,
        causal=True,
        norm_first=True,
        bias_ff=False,
        bias_attn=False,
        layer_scale=0.01,
        positional_embedding="rope",
        use_conv_bias=True,
        gating=False,
        norm="layer_norm",
        context=250,
        max_period=10000,
        max_seq_len=8192,
        kv_repeat=1,
        dim_feedforward=2048,
        conv_layout=True,
        use_conv_block=False,
        cross_attention=False,
        conv_kernel_size=3,
    )
    return MimiConfig(
        channels=1,
        sample_rate=24000,
        frame_rate=12.5,
        renormalize=True,
        seanet=seanet,
        transformer=transformer,
        quantizer_nq=num_codebooks,
        quantizer_bins=2048,
        quantizer_dim=256,
    )


class Mimi(nn.Module):
    def __init__(self, cfg: MimiConfig):
        super().__init__()
        dim = cfg.seanet.dimension
        self.cfg = cfg
        encoder_frame_rate = cfg.sample_rate / math.prod(cfg.seanet.ratios)
        downsample_stride = int(encoder_frame_rate / cfg.frame_rate)
        self.encoder = SeanetEncoder(cfg.seanet)
        self.decoder = SeanetDecoder(cfg.seanet)
        self.quantizer = SplitResidualVectorQuantizer(
            dim=cfg.quantizer_dim,
            input_dim=dim,
            output_dim=dim,
            nq=cfg.quantizer_nq,
            bins=cfg.quantizer_bins,
        )
        self.encoder_transformer = ProjectedTransformer(
            cfg.transformer,
            input_dim=dim,
            output_dims=[dim],
        )
        self.decoder_transformer = ProjectedTransformer(
            cfg.transformer,
            input_dim=dim,
            output_dims=[dim],
        )
        self.downsample = ConvDownsample1d(
            stride=downsample_stride,
            dim=dim,
            causal=True,
        )
        self.upsample = ConvTrUpsample1d(
            stride=downsample_stride,
            dim=dim,
            causal=True,
        )
        self.encoder_cache = self.encoder_transformer.make_cache()
        self.decoder_cache = self.decoder_transformer.make_cache()

    def reset_state(self):
        self.encoder.reset_state()
        self.decoder.reset_state()
        for c in self.decoder_cache:
            _reset_kv_cache(c)
        for c in self.encoder_cache:
            _reset_kv_cache(c)

    def encode(self, xs: mx.array) -> mx.array:
        self.encoder.reset_state()
        for c in self.encoder_cache:
            _reset_kv_cache(c)
        xs = self.encoder(xs)
        xs = self.encoder_transformer(xs, cache=self.encoder_cache)[0]
        xs = self.downsample(xs)
        return self.quantizer.encode(xs)

    def decode(self, xs: mx.array) -> mx.array:
        self.decoder.reset_state()
        for c in self.decoder_cache:
            _reset_kv_cache(c)
        xs = self.quantizer.decode(xs)
        xs = self.upsample(xs)
        xs = self.decoder_transformer(xs, cache=self.decoder_cache)[0]
        return self.decoder(xs)

    def encode_step(self, xs: mx.array) -> mx.array:
        xs = self.encoder.step(xs)
        xs = self.encoder_transformer(xs, cache=self.encoder_cache)[0]
        xs = self.downsample.step(xs)
        xs = self.quantizer.encode(xs)
        return xs

    def decode_step(self, xs: mx.array) -> mx.array:
        xs = self.quantizer.decode(xs)
        xs = self.upsample.step(xs)
        xs = self.decoder_transformer(xs, cache=self.decoder_cache)[0]
        xs = self.decoder.step(xs)
        return xs

    def warmup(self):
        pcm = mx.zeros((1, 1, 1920 * 4))
        codes = self.encode(pcm)
        pcm_out = self.decode(codes)
        mx.eval(pcm_out)

    @property
    def frame_rate(self) -> float:
        return self.cfg.frame_rate

    @property
    def sample_rate(self) -> float:
        return self.cfg.sample_rate


class MimiStreamingDecoder:
    """Incremental decoder wrapper for the Mimi codec.

    This helper keeps the internal state of the Mimi model across calls and
    decodes audio tokens frame by frame using ``decode_step``.
    """

    def __init__(self, mimi: "Mimi") -> None:  # noqa: F821 - Mimi defined below
        self._mimi = mimi
        self.reset()

    def reset(self) -> None:
        """Reset the underlying codec state."""
        self._mimi.decoder.reset_state()
        self._mimi.upsample.reset_state()
        for c in self._mimi.decoder_cache:
            _reset_kv_cache(c)

    def decode_frames(self, tokens: mx.array) -> mx.array:
        """Decode a sequence of audio tokens incrementally.

        Parameters
        ----------
        tokens:
            Array of shape ``(B, C, T)`` or ``(C, T)`` containing the audio
            tokens to decode. ``B`` is the batch dimension, ``C`` is the number
            of codebooks and ``T`` the number of frames.

        Returns
        -------
        mx.array
            The decoded waveform for the provided frames.
        """

        if tokens.ndim == 2:
            tokens = mx.expand_dims(tokens, 0)

        pcm = []
        for t in range(tokens.shape[-1]):
            step_tokens = tokens[:, :, t : t + 1]
            pcm.append(self._mimi.decode_step(step_tokens))

        return mx.concat(pcm, axis=-1)
