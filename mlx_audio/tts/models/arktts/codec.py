"""MLX port of modeling_arktts_codec.py (Audio8-TTS 44.1 kHz codec).

Layout note: MLX convs are channels-last, so every tensor here is (B, T, C)
where the reference uses (B, C, T). The reference's `channels_first` transposes
therefore become no-ops; time-axis pads/crops move from axis -1 to axis 1.
Everything else mirrors the reference file class-for-class, line-for-line.
Weight-norm folding and conv-layout transposes happen in Model.sanitize().
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


def _rope(length: int, head_dim: int, base: float) -> mx.array:
    frequencies = 1.0 / (
        base ** (mx.arange(0, head_dim, 2).astype(mx.float32) / head_dim)
    )
    phases = mx.arange(length).astype(mx.float32)[:, None] * frequencies[None, :]
    # reference builds torch.polar(...) then casts the (real, imag) stack to bf16
    return mx.stack((mx.cos(phases), mx.sin(phases)), axis=-1).astype(mx.bfloat16)


def _apply_rope(x: mx.array, values: mx.array) -> mx.array:
    # x: (B, T, H, D)
    shaped = x.astype(mx.float32).reshape(*x.shape[:-1], -1, 2)
    values = values.astype(mx.float32).reshape(
        1, shaped.shape[1], 1, shaped.shape[3], 2
    )
    output = mx.stack(
        (
            shaped[..., 0] * values[..., 0] - shaped[..., 1] * values[..., 1],
            shaped[..., 1] * values[..., 0] + shaped[..., 0] * values[..., 1],
        ),
        axis=-1,
    )
    return output.flatten(3).astype(x.dtype)


class ArkttsCodecRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = float(eps)
        self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        xf = x.astype(mx.float32)
        output = xf * mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + self.eps)
        return output.astype(x.dtype) * self.weight


class ArkttsCodecLayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float = 1e-2):
        super().__init__()
        self.gamma = init_values * mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        return x * self.gamma


@dataclass
class ArkttsCodecTransformerConfig:
    n_layer: int
    n_head: int
    dim: int
    intermediate_size: int
    n_local_heads: int = -1
    head_dim: int = 64
    rope_base: float = 10000
    norm_eps: float = 1e-5
    channels_first: bool = True

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head


class ArkttsCodecAttention(nn.Module):
    def __init__(self, config: ArkttsCodecTransformerConfig):
        super().__init__()
        total = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        self.wqkv = nn.Linear(config.dim, total, bias=False)
        self.wo = nn.Linear(config.head_dim * config.n_head, config.dim, bias=False)
        self.n_head = config.n_head
        self.n_local_heads = config.n_local_heads
        self.head_dim = config.head_dim

    def __call__(self, x: mx.array, rope_values: mx.array, mask: mx.array) -> mx.array:
        batch, length, _ = x.shape
        query_size = self.n_head * self.head_dim
        kv_size = self.n_local_heads * self.head_dim
        qkv = self.wqkv(x)
        query, key, value = mx.split(qkv, [query_size, query_size + kv_size], axis=-1)
        query = query.reshape(batch, length, self.n_head, self.head_dim)
        key = key.reshape(batch, length, self.n_local_heads, self.head_dim)
        value = value.reshape(batch, length, self.n_local_heads, self.head_dim)
        query = _apply_rope(query, rope_values).transpose(0, 2, 1, 3)
        key = _apply_rope(key, rope_values).transpose(0, 2, 1, 3)
        value = value.transpose(0, 2, 1, 3)
        output = mx.fast.scaled_dot_product_attention(
            query, key, value, scale=1.0 / math.sqrt(self.head_dim), mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(batch, length, query_size)
        return self.wo(output)


class ArkttsCodecFeedForward(nn.Module):
    def __init__(self, config: ArkttsCodecTransformerConfig):
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class ArkttsCodecTransformerBlock(nn.Module):
    def __init__(self, config: ArkttsCodecTransformerConfig):
        super().__init__()
        self.attention = ArkttsCodecAttention(config)
        self.feed_forward = ArkttsCodecFeedForward(config)
        self.ffn_norm = ArkttsCodecRMSNorm(config.dim, config.norm_eps)
        self.attention_norm = ArkttsCodecRMSNorm(config.dim, config.norm_eps)
        self.attention_layer_scale = ArkttsCodecLayerScale(config.dim)
        self.ffn_layer_scale = ArkttsCodecLayerScale(config.dim)

    def __call__(self, x: mx.array, rope_values: mx.array, mask: mx.array) -> mx.array:
        hidden = x + self.attention_layer_scale(
            self.attention(self.attention_norm(x), rope_values, mask)
        )
        return hidden + self.ffn_layer_scale(self.feed_forward(self.ffn_norm(hidden)))


class ArkttsCodecWindowTransformer(nn.Module):
    def __init__(
        self,
        config: ArkttsCodecTransformerConfig,
        input_dim: int,
        window_size: int | None,
        causal: bool = True,
    ):
        super().__init__()
        self.layers = [
            ArkttsCodecTransformerBlock(config) for _ in range(config.n_layer)
        ]
        self.norm = ArkttsCodecRMSNorm(config.dim, config.norm_eps)
        self.window_size = window_size
        self.causal = causal
        self.input_proj = (
            nn.Linear(input_dim, config.dim)
            if input_dim != config.dim
            else nn.Identity()
        )
        self.output_proj = (
            nn.Linear(config.dim, input_dim)
            if input_dim != config.dim
            else nn.Identity()
        )
        self.head_dim = config.head_dim
        self.rope_base = config.rope_base

    def __call__(self, x: mx.array) -> mx.array:
        # reference transposes channels-first input; MLX tensors are already (B, T, C)
        x = self.input_proj(x)
        length = x.shape[1]
        row = mx.arange(length)[:, None]
        column = mx.arange(length)[None, :]
        mask = column <= row
        if self.window_size is not None:
            mask = mx.logical_and(
                mask, column >= mx.maximum(row - self.window_size + 1, 0)
            )
        mask = mask[None, None]
        rope_values = _rope(length, self.head_dim, self.rope_base)
        for layer in self.layers:
            x = layer(x, rope_values, mask)
        return self.output_proj(self.norm(x))


def _extra_padding(
    x: mx.array, kernel_size: int, stride: int, padding_total: int = 0
) -> int:
    length = x.shape[1]
    frames = (length - kernel_size + padding_total) / stride + 1
    ideal = (math.ceil(frames) - 1) * stride + kernel_size - padding_total
    return ideal - length


class ArkttsCausalConv1d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, dilation=1, stride=1, groups=1
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
        )
        self.stride = stride
        self.kernel_size = (kernel_size - 1) * dilation + 1
        self.padding = self.kernel_size - self.stride

    def __call__(self, x: mx.array) -> mx.array:
        right = _extra_padding(x, self.kernel_size, self.stride, self.padding)
        x = mx.pad(x, ((0, 0), (self.padding, right), (0, 0)))
        return self.conv(x)


class ArkttsCausalConvTranspose1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super().__init__()
        self.conv = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride
        )
        self.stride = stride
        self.kernel_size = kernel_size

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv(x)
        crop = self.kernel_size - self.stride
        return x[:, : x.shape[1] - crop] if crop else x


class ArkttsSnake1d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        # reference alpha is (1, C, 1); sanitize reshapes it to (1, 1, C) for channels-last
        self.alpha = mx.ones((1, 1, channels))

    def __call__(self, x: mx.array) -> mx.array:
        return x + mx.reciprocal(self.alpha + 1e-9) * mx.square(mx.sin(self.alpha * x))


class ArkttsResidualUnit(nn.Module):
    def __init__(self, dim: int, dilation: int):
        super().__init__()
        self.block = [
            ArkttsSnake1d(dim),
            ArkttsCausalConv1d(dim, dim, kernel_size=7, dilation=dilation),
            ArkttsSnake1d(dim),
            ArkttsCausalConv1d(dim, dim, kernel_size=1),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        output = x
        for module in self.block:
            output = module(output)
        difference = x.shape[1] - output.shape[1]
        if difference > 0:
            x = x[:, :-difference]
        return x + output


class ArkttsEncoderBlock(nn.Module):
    def __init__(self, dim: int, stride: int, transformer_layers: int):
        super().__init__()
        modules = [
            ArkttsResidualUnit(dim // 2, 1),
            ArkttsResidualUnit(dim // 2, 3),
            ArkttsResidualUnit(dim // 2, 9),
            ArkttsSnake1d(dim // 2),
            ArkttsCausalConv1d(dim // 2, dim, kernel_size=2 * stride, stride=stride),
        ]
        if transformer_layers:
            config = ArkttsCodecTransformerConfig(
                n_layer=transformer_layers,
                n_head=dim // 64,
                dim=dim,
                intermediate_size=dim * 3,
            )
            modules.append(ArkttsCodecWindowTransformer(config, dim, window_size=512))
        else:
            modules.append(nn.Identity())
        self.block = modules

    def __call__(self, x: mx.array) -> mx.array:
        for module in self.block:
            x = module(x)
        return x


class ArkttsEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        dim = 64
        modules = [ArkttsCausalConv1d(1, dim, kernel_size=7)]
        for stride, transformer_layers in zip((2, 4, 8, 8), (0, 0, 0, 4)):
            dim *= 2
            modules.append(ArkttsEncoderBlock(dim, stride, transformer_layers))
        modules.extend(
            (ArkttsSnake1d(dim), ArkttsCausalConv1d(dim, 1024, kernel_size=3))
        )
        self.block = modules

    def __call__(self, x: mx.array) -> mx.array:
        for module in self.block:
            x = module(x)
        return x


class ArkttsDecoderBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, stride: int):
        super().__init__()
        self.block = [
            ArkttsSnake1d(input_dim),
            ArkttsCausalConvTranspose1d(
                input_dim, output_dim, kernel_size=2 * stride, stride=stride
            ),
            ArkttsResidualUnit(output_dim, 1),
            ArkttsResidualUnit(output_dim, 3),
            ArkttsResidualUnit(output_dim, 9),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        for module in self.block:
            x = module(x)
        return x


class ArkttsDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        channels = 1536
        modules = [ArkttsCausalConv1d(1024, channels, kernel_size=7)]
        for index, stride in enumerate((8, 8, 4, 2)):
            input_dim = channels // (2**index)
            output_dim = channels // (2 ** (index + 1))
            modules.append(ArkttsDecoderBlock(input_dim, output_dim, stride))
        modules.extend(
            (
                ArkttsSnake1d(output_dim),
                ArkttsCausalConv1d(output_dim, 1, kernel_size=7),
            )
        )
        self.model = modules

    def __call__(self, x: mx.array) -> mx.array:
        for module in self.model:
            x = module(x)
        return mx.tanh(x)


class ArkttsVectorQuantizer(nn.Module):
    def __init__(self, input_dim: int, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.codebook_size = int(codebook_size)
        self.codebook_dim = int(codebook_dim)
        # 1x1 convs in the reference; stored transposed to (O, 1, I) MLX layout
        self.in_proj = nn.Conv1d(input_dim, codebook_dim, kernel_size=1)
        self.out_proj = nn.Conv1d(codebook_dim, input_dim, kernel_size=1)
        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    def decode_code(self, indices: mx.array) -> mx.array:
        # (B, T) -> (B, T, D); the reference returns channels-first, we stay (B, T, D)
        return self.codebook(indices)

    def decode_latents(self, latents: mx.array):
        batch, length, _ = latents.shape
        flattened = latents.reshape(batch * length, -1)
        flattened = flattened / mx.maximum(
            mx.linalg.norm(flattened, axis=-1, keepdims=True), 1e-12
        )
        codebook = self.codebook.weight
        codebook = codebook / mx.maximum(
            mx.linalg.norm(codebook, axis=-1, keepdims=True), 1e-12
        )
        distances = (
            mx.sum(mx.square(flattened), axis=1, keepdims=True)
            - 2 * flattened @ codebook.T
            + mx.sum(mx.square(codebook), axis=1, keepdims=True).T
        )
        indices = mx.argmax(-distances, axis=1).reshape(batch, length)
        return self.decode_code(indices), indices

    def __call__(self, z: mx.array):
        projected = self.in_proj(z)
        quantized, indices = self.decode_latents(projected)
        return self.out_proj(quantized), indices, projected


class ArkttsResidualQuantizer(nn.Module):
    def __init__(
        self, input_dim: int, n_codebooks: int, codebook_size: int, codebook_dim: int
    ):
        super().__init__()
        self.n_codebooks = int(n_codebooks)
        self.codebook_size = int(codebook_size)
        self.quantizers = [
            ArkttsVectorQuantizer(input_dim, codebook_size, codebook_dim)
            for _ in range(n_codebooks)
        ]

    def __call__(self, z: mx.array):
        quantized_sum = 0.0
        residual = z
        codes = []
        for quantizer in self.quantizers:
            quantized, indices, _ = quantizer(residual)
            quantized_sum = quantized_sum + quantized
            residual = residual - quantized
            codes.append(indices)
        return quantized_sum, mx.stack(codes, axis=1)

    def from_codes(self, codes: mx.array) -> mx.array:
        output = 0.0
        for index in range(codes.shape[1]):
            projected = self.quantizers[index].decode_code(codes[:, index])
            output = output + self.quantizers[index].out_proj(projected)
        return output


class ArkttsConvNeXtBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = ArkttsCausalConv1d(dim, dim, kernel_size=7, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = 1e-6 * mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        x = self.dwconv(x)
        x = self.pwconv2(nn.gelu(self.pwconv1(self.norm(x))))
        return residual + self.gamma * x


class ArkttsDownsampleQuantizer(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.semantic_quantizer = ArkttsResidualQuantizer(1024, 1, 4096, 8)
        self.quantizer = ArkttsResidualQuantizer(1024, 9, 1024, 8)
        self.downsample = [
            [
                ArkttsCausalConv1d(1024, 1024, kernel_size=2, stride=2),
                ArkttsConvNeXtBlock(1024),
            ],
            [
                ArkttsCausalConv1d(1024, 1024, kernel_size=2, stride=2),
                ArkttsConvNeXtBlock(1024),
            ],
        ]
        self.upsample = [
            [
                ArkttsCausalConvTranspose1d(1024, 1024, kernel_size=2, stride=2),
                ArkttsConvNeXtBlock(1024),
            ],
            [
                ArkttsCausalConvTranspose1d(1024, 1024, kernel_size=2, stride=2),
                ArkttsConvNeXtBlock(1024),
            ],
        ]
        pre_transformer_config = ArkttsCodecTransformerConfig(
            n_layer=8, n_head=16, dim=1024, intermediate_size=3072
        )
        post_transformer_config = ArkttsCodecTransformerConfig(
            n_layer=int(getattr(config, "codec_post_n_layer", 8)),
            n_head=int(getattr(config, "codec_post_n_head", 16)),
            n_local_heads=int(getattr(config, "codec_post_n_local_heads", 8)),
            dim=1024,
            intermediate_size=int(
                getattr(config, "codec_post_intermediate_size", 1216)
            ),
        )
        self.pre_module = ArkttsCodecWindowTransformer(
            pre_transformer_config, 1024, window_size=128
        )
        self.post_module = ArkttsCodecWindowTransformer(
            post_transformer_config, 1024, window_size=128
        )

    def _run(self, stages, x: mx.array) -> mx.array:
        for stage in stages:
            for module in stage:
                x = module(x)
        return x

    def __call__(self, z: mx.array):
        original_length = z.shape[1]
        z = self.pre_module(self._run(self.downsample, z))
        semantic, semantic_codes = self.semantic_quantizer(z)
        residual, residual_codes = self.quantizer(z - semantic)
        z = self._run(self.upsample, self.post_module(semantic + residual))
        difference = original_length - z.shape[1]
        if difference > 0:
            z = mx.pad(z, ((0, 0), (difference, 0), (0, 0)))
        elif difference < 0:
            z = z[:, -difference:]
        return z, mx.concatenate((semantic_codes, residual_codes), axis=1)

    def decode(self, indices: mx.array) -> mx.array:
        semantic_indices = mx.clip(
            indices[:, :1], 0, self.semantic_quantizer.codebook_size - 1
        )
        residual_indices = mx.clip(indices[:, 1:], 0, self.quantizer.codebook_size - 1)
        semantic = self.semantic_quantizer.from_codes(semantic_indices)
        residual = self.quantizer.from_codes(residual_indices)
        return self._run(self.upsample, self.post_module(semantic + residual))


class ArkttsCodec(nn.Module):
    sample_rate = 44100
    hop_length = 512
    frame_length = 2048

    def __init__(self, config=None):
        super().__init__()
        self.encoder = ArkttsEncoder()
        self.quantizer = ArkttsDownsampleQuantizer(config)
        self.decoder = ArkttsDecoder()

    def encode(self, audio: mx.array, audio_lengths: mx.array | None = None):
        """audio: (B, samples) or (B, samples, 1) mono waveform in [-1, 1]."""
        if audio.ndim == 2:
            audio = audio[..., None]
        if audio.ndim != 3 or audio.shape[-1] != 1:
            raise ValueError("audio must have shape [B, samples, 1]")
        original_length = audio.shape[1]
        right = (
            math.ceil(original_length / self.frame_length) * self.frame_length
            - original_length
        )
        if right:
            audio = mx.pad(audio, ((0, 0), (0, right), (0, 0)))
        if audio_lengths is None:
            audio_lengths = mx.full((audio.shape[0],), original_length, dtype=mx.int64)
        encoded = self.encoder(audio)
        _, codes = self.quantizer(encoded)
        codes = codes.astype(mx.int64)
        code_lengths = mx.ceil(
            audio_lengths.astype(mx.float32) / self.frame_length
        ).astype(mx.int64)
        max_codes = codes.shape[-1]
        frame_positions = mx.arange(max_codes)[None, None, :]
        valid = frame_positions < code_lengths[:, None, None]
        padded = mx.where(valid, codes, -1)
        return padded, mx.minimum(code_lengths, max_codes)

    def decode(self, codes: mx.array) -> mx.array:
        """codes: (B, num_codebooks, T) -> waveform (B, samples)."""
        return self.decoder(self.quantizer.decode(codes.astype(mx.int64)))[..., 0]
