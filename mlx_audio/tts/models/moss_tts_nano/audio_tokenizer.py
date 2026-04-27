from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_audio.utils import resample_audio

DEFAULT_AUDIO_TOKENIZER_REPO = "mlx-community/MOSS-Audio-Tokenizer-Nano"


@dataclass
class AudioTokenizerConfig:
    sample_rate: int = 48000
    sampling_rate: int = 48000
    downsample_rate: int = 3840
    causal_transformer_context_duration: float = 10.0
    number_channels: int = 2
    enable_channel_interleave: bool = True
    encoder_kwargs: list[dict[str, Any]] | None = None
    decoder_kwargs: list[dict[str, Any]] | None = None
    quantizer_type: str = "rlfq"
    quantizer_kwargs: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AudioTokenizerConfig":
        return cls(
            sample_rate=int(data.get("sample_rate", data.get("sampling_rate", 48000))),
            sampling_rate=int(
                data.get("sampling_rate", data.get("sample_rate", 48000))
            ),
            downsample_rate=int(data.get("downsample_rate", 3840)),
            causal_transformer_context_duration=float(
                data.get("causal_transformer_context_duration", 10.0)
            ),
            number_channels=int(data.get("number_channels", 2)),
            enable_channel_interleave=bool(data.get("enable_channel_interleave", True)),
            encoder_kwargs=list(data.get("encoder_kwargs", [])),
            decoder_kwargs=list(data.get("decoder_kwargs", [])),
            quantizer_type=str(data.get("quantizer_type", "rlfq")),
            quantizer_kwargs=dict(data.get("quantizer_kwargs", {})),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "AudioTokenizerConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))


def _exact_gelu(x: mx.array) -> mx.array:
    return 0.5 * x * (1.0 + mx.erf(x / math.sqrt(2.0)))


def _normalize_weight_except_dim(weight: mx.array, except_dim: int) -> mx.array:
    axes = tuple(axis for axis in range(weight.ndim) if axis != except_dim)
    return mx.sqrt(mx.sum(mx.square(weight), axis=axes, keepdims=True))


def _l2_normalize(x: mx.array, axis: int = -1, eps: float = 1e-12) -> mx.array:
    return x / mx.maximum(mx.sqrt(mx.sum(mx.square(x), axis=axis, keepdims=True)), eps)


def _load_weights_from_dir(path: Path) -> dict[str, mx.array]:
    direct = path / "model.safetensors"
    if direct.exists():
        return mx.load(str(direct))

    index_path = path / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        filenames = sorted(set(index.get("weight_map", {}).values()))
    else:
        filenames = [item.name for item in sorted(path.glob("*.safetensors"))]
    if not filenames:
        raise FileNotFoundError(f"No safetensors weights found in {path}")

    weights: dict[str, mx.array] = {}
    for filename in filenames:
        weights.update(mx.load(str(path / filename)))
    return weights


def _has_safetensors_weights(path: Path) -> bool:
    return (
        (path / "model.safetensors").exists()
        or (path / "model.safetensors.index.json").exists()
        or bool(list(path.glob("*.safetensors")))
    )


def _is_audio_tokenizer_dir(path: Path) -> bool:
    config_path = path / "config.json"
    if not config_path.exists() or not _has_safetensors_weights(path):
        return False
    try:
        config = json.loads(config_path.read_text())
    except json.JSONDecodeError:
        return False
    return "encoder_kwargs" in config and "decoder_kwargs" in config


def _resolve_audio_tokenizer_dir(source: str | Path) -> Path:
    source_path = Path(source).expanduser()
    if source_path.exists():
        return source_path.resolve()

    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            str(source),
            allow_patterns=["*.json", "*.safetensors", "*.index.json", "*.md"],
        )
    )


class WNConv1d(nn.Module):
    """Weight-normalized 1D convolution for upstream Conv1d checkpoint weights.

    Upstream stores weights as `(out_channels, in_channels, kernel_size)`.
    MLX `conv1d` expects input `(batch, length, channels)` and weights shaped
    `(out_channels, kernel_size, in_channels)`, so this module transposes at
    call time after reconstructing the weight-normalized kernel.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.stride = 1
        self.padding = 0
        self.dilation = 1
        self.groups = 1
        self.parametrizations = {
            "weight": {
                "original0": mx.ones((out_channels, 1, 1)),
                "original1": mx.zeros((out_channels, in_channels, kernel_size)),
            }
        }
        self.bias = mx.zeros((out_channels,))

    def _source_layout_weight(self) -> mx.array:
        weight_g = self.parametrizations["weight"]["original0"]
        weight_v = self.parametrizations["weight"]["original1"]
        return weight_g * weight_v / _normalize_weight_except_dim(weight_v, 0)

    def __call__(self, x: mx.array) -> mx.array:
        weight = self._source_layout_weight().transpose(0, 2, 1)
        y = mx.conv1d(
            x.transpose(0, 2, 1),
            weight,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        return (y + self.bias).transpose(0, 2, 1)


class LayerScale(nn.Module):
    def __init__(self, channels: int, init: float):
        super().__init__()
        self.scale = mx.full((channels,), float(init))

    def __call__(self, x: mx.array) -> mx.array:
        return self.scale * x


class Identity(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        return x


def _apply_rope(
    q: mx.array,
    k: mx.array,
    *,
    max_period: float,
    offset: int = 0,
) -> tuple[mx.array, mx.array]:
    batch, heads, time, dim = q.shape
    del batch, heads
    if dim % 2 != 0:
        raise ValueError(f"RoPE requires an even head dimension, got {dim}")
    freqs = mx.exp(
        mx.arange(dim // 2, dtype=mx.float32)
        * (-math.log(float(max_period)) * 2.0 / dim)
    )
    positions = mx.arange(offset, offset + time, dtype=mx.float32)
    phase = positions[None, None, :, None] * freqs[None, None, None, :]
    cos = mx.cos(phase)
    sin = mx.sin(phase)

    q_pairs = q.astype(mx.float32).reshape(*q.shape[:-1], dim // 2, 2)
    k_pairs = k.astype(mx.float32).reshape(*k.shape[:-1], dim // 2, 2)
    qr, qi = q_pairs[..., 0], q_pairs[..., 1]
    kr, ki = k_pairs[..., 0], k_pairs[..., 1]

    q_out = mx.stack([qr * cos - qi * sin, qr * sin + qi * cos], axis=-1)
    k_out = mx.stack([kr * cos - ki * sin, kr * sin + ki * cos], axis=-1)
    return q_out.reshape(q.shape).astype(q.dtype), k_out.reshape(k.shape).astype(
        k.dtype
    )


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        causal: bool,
        context: int | None,
        max_period: float,
        use_rope: bool,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        self.causal = bool(causal)
        self.context = None if context is None else int(context)
        self.max_period = float(max_period)
        self.use_rope = bool(use_rope)
        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def _mask(self, input_lengths: mx.array, max_seqlen: int, dtype: mx.Dtype):
        positions = mx.arange(max_seqlen, dtype=mx.int32)
        valid_k = positions[None, None, None, :] < input_lengths[:, None, None, None]
        delta = positions[None, None, :, None] - positions[None, None, None, :]
        allowed = valid_k
        if self.causal:
            allowed = allowed & (delta >= 0)
        if self.context is not None:
            allowed = allowed & (delta < self.context)
        return mx.where(allowed, 0.0, mx.finfo(dtype).min).astype(dtype)

    def __call__(self, x: mx.array, input_lengths: mx.array) -> mx.array:
        batch, time, _ = x.shape
        qkv = self.in_proj(x)
        qkv = qkv.reshape(batch, time, 3, self.num_heads, self.head_dim)
        q = qkv[:, :, 0].transpose(0, 2, 1, 3)
        k = qkv[:, :, 1].transpose(0, 2, 1, 3)
        v = qkv[:, :, 2].transpose(0, 2, 1, 3)
        if self.use_rope:
            q, k = _apply_rope(q, k, max_period=self.max_period)

        mask = self._mask(input_lengths.astype(mx.int32), time, x.dtype)
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.head_dim**-0.5, mask=mask
        )
        valid_q = (
            mx.arange(time, dtype=mx.int32)[None, None, :, None]
            < input_lengths[:, None, None, None]
        )
        out = mx.where(valid_q, out, mx.zeros((), dtype=out.dtype))
        out = out.transpose(0, 2, 1, 3).reshape(batch, time, self.embed_dim)
        return self.out_proj(out)


class TransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        *,
        causal: bool,
        context: int | None,
        positional_embedding: str,
        max_period: float,
        layer_scale: float | None,
        norm: str,
    ):
        super().__init__()
        if norm != "layer_norm":
            raise ValueError(f"Unsupported MOSS audio tokenizer norm: {norm}")
        self.self_attn = MultiheadAttention(
            d_model,
            num_heads,
            causal=causal,
            context=context,
            max_period=max_period,
            use_rope=positional_embedding in {"rope", "sin_rope"},
        )
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        self.ffn = [
            nn.Linear(d_model, dim_feedforward, bias=False),
            Identity(),
            nn.Linear(dim_feedforward, d_model, bias=False),
        ]
        self.layer_scale_1 = (
            LayerScale(d_model, layer_scale) if layer_scale is not None else Identity()
        )
        self.layer_scale_2 = (
            LayerScale(d_model, layer_scale) if layer_scale is not None else Identity()
        )

    def __call__(self, x: mx.array, input_lengths: mx.array) -> mx.array:
        residual = x
        x = self.norm1(x)
        x = residual + self.layer_scale_1(self.self_attn(x, input_lengths))
        residual = x
        x = self.norm2(x)
        x = self.ffn[2](_exact_gelu(self.ffn[0](x)))
        return residual + self.layer_scale_2(x)


class Transformer(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int,
        causal: bool,
        context: int | None,
        positional_embedding: str,
        max_period: float,
        positional_scale: float = 1.0,
        layer_scale: float | None = None,
        norm: str = "layer_norm",
        gating: str = "none",
        **kwargs,
    ):
        super().__init__()
        del kwargs
        if gating != "none":
            raise ValueError(f"Unsupported MOSS audio tokenizer gating: {gating}")
        self.positional_embedding = str(positional_embedding)
        self.max_period = float(max_period)
        self.positional_scale = float(positional_scale)
        self.layers = [
            TransformerLayer(
                d_model,
                num_heads,
                dim_feedforward,
                causal=causal,
                context=context,
                positional_embedding=self.positional_embedding,
                max_period=self.max_period,
                layer_scale=layer_scale,
                norm=norm,
            )
            for _ in range(num_layers)
        ]

    def __call__(self, x: mx.array, input_lengths: mx.array) -> mx.array:
        if self.positional_embedding in {"sin", "sin_rope"}:
            positions = mx.arange(x.shape[1], dtype=x.dtype)
            half = x.shape[-1] // 2
            scale = self.max_period ** (
                mx.arange(half, dtype=x.dtype) / max(half - 1, 1)
            )
            phase = positions[:, None] / scale[None, :]
            emb = mx.concatenate([mx.cos(phase), mx.sin(phase)], axis=-1)
            x = x + self.positional_scale * emb[None, :, :]
        for layer in self.layers:
            x = layer(x, input_lengths)
        return x


class ProjectedTransformer(nn.Module):
    def __init__(
        self,
        *,
        input_dimension: int,
        output_dimension: int,
        d_model: int,
        context: int | None,
        conv_layout: bool,
        module_type: str,
        **kwargs,
    ):
        super().__init__()
        del conv_layout, module_type
        self.downsample_ratio = 1
        self.input_proj = nn.Linear(input_dimension, d_model, bias=False)
        self.transformer = Transformer(d_model=d_model, context=context, **kwargs)
        self.output_proj = nn.Linear(d_model, output_dimension, bias=False)

    def __call__(self, x: mx.array, input_lengths: mx.array):
        x = self.input_proj(x.transpose(0, 2, 1))
        x = self.transformer(x, input_lengths=input_lengths)
        return self.output_proj(x).transpose(0, 2, 1), input_lengths


class PatchedPretransform(nn.Module):
    def __init__(self, patch_size: int, *, is_downsample: bool, module_type: str):
        super().__init__()
        del module_type
        self.patch_size = int(patch_size)
        self.downsample_ratio = self.patch_size
        self.is_downsample = bool(is_downsample)

    def encode(self, x: mx.array, input_lengths: mx.array):
        batch, channels, _ = x.shape
        patch = self.patch_size
        x = x.reshape(batch, channels, -1, patch)
        x = x.transpose(0, 1, 3, 2).reshape(batch, channels * patch, -1)
        return x, input_lengths // patch

    def decode(self, x: mx.array, input_lengths: mx.array):
        batch, channels_patch, length = x.shape
        patch = self.patch_size
        channels = channels_patch // patch
        x = x.reshape(batch, channels, patch, length)
        x = x.transpose(0, 1, 3, 2).reshape(batch, channels, length * patch)
        return x, input_lengths * patch

    def __call__(self, x: mx.array, input_lengths: mx.array):
        if self.is_downsample:
            return self.encode(x, input_lengths)
        return self.decode(x, input_lengths)


class LFQ(nn.Module):
    def __init__(self, input_dim: int, codebook_size: int, codebook_dim: int, **kwargs):
        super().__init__()
        del kwargs
        self.input_dim = int(input_dim)
        self.codebook_size = int(codebook_size)
        self.codebook_dim = int(codebook_dim)
        self.in_proj = WNConv1d(self.input_dim, self.codebook_dim, kernel_size=1)
        self.out_proj = WNConv1d(self.codebook_dim, self.input_dim, kernel_size=1)
        self.codebook = nn.Embedding(self.codebook_size, self.codebook_dim)

    def decode_code_wo_out_proj(self, embed_id: mx.array) -> mx.array:
        return self.codebook(embed_id).transpose(0, 2, 1)

    def decode_code(self, embed_id: mx.array) -> mx.array:
        return self.out_proj(self.decode_code_wo_out_proj(embed_id).astype(mx.float32))

    def decode_latents(self, latents: mx.array) -> tuple[mx.array, mx.array]:
        encodings = latents.transpose(0, 2, 1).reshape(-1, latents.shape[1])
        codebook = self.codebook.weight.astype(mx.float32)
        encodings = _l2_normalize(encodings.astype(mx.float32), axis=-1)
        codebook = _l2_normalize(codebook, axis=-1)
        dist = (
            mx.sum(mx.square(encodings), axis=1, keepdims=True)
            - 2.0 * (encodings @ codebook.T)
            + mx.sum(mx.square(codebook), axis=1, keepdims=True).T
        )
        indices = mx.argmax(-dist, axis=1).reshape(latents.shape[0], -1)
        return self.decode_code_wo_out_proj(indices).astype(mx.float32), indices

    def __call__(self, z: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        z_e = self.in_proj(z.astype(mx.float32)).astype(mx.float32)
        z_q, indices = self.decode_latents(z_e)
        z_q = self.out_proj(z_q.astype(mx.float32)).astype(mx.float32)
        return z_q, indices, z_e


class ResidualLFQ(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int = 1024,
        rvq_dim: int | None = None,
        output_dim: int | None = None,
        num_quantizers: int = 32,
        codebook_size: int = 1024,
        codebook_dim: int = 8,
        **kwargs,
    ):
        super().__init__()
        del kwargs
        self.input_dim = int(input_dim)
        self.rvq_dim = int(rvq_dim or input_dim)
        self.output_dim = int(output_dim or input_dim)
        self.num_quantizers = int(num_quantizers)
        self.codebook_size = int(codebook_size)
        self.codebook_dim = int(codebook_dim)
        self.input_proj = WNConv1d(self.input_dim, self.rvq_dim, kernel_size=1)
        self.output_proj = WNConv1d(self.rvq_dim, self.output_dim, kernel_size=1)
        self.quantizers = [
            LFQ(
                input_dim=self.rvq_dim,
                codebook_size=self.codebook_size,
                codebook_dim=self.codebook_dim,
            )
            for _ in range(self.num_quantizers)
        ]

    def __call__(
        self,
        z: mx.array,
        input_length: mx.array,
        n_quantizers: int | None = None,
    ) -> tuple[mx.array, mx.array, mx.array]:
        z = self.input_proj(z.astype(mx.float32)).astype(mx.float32)
        batch, _, max_time = z.shape
        mask = mx.arange(max_time, dtype=mx.int32)[None, :] < input_length[:, None]
        update_mask = mask[:, None, :]
        quantized_out = mx.zeros(z.shape, dtype=mx.float32)
        residual = z
        indices = []
        for quantizer in self.quantizers[: int(n_quantizers or self.num_quantizers)]:
            z_q_i, indices_i, _ = quantizer(residual * update_mask)
            quantized_out = quantized_out + z_q_i * update_mask
            residual = residual - z_q_i * update_mask
            indices.append(indices_i)
        all_indices = (
            mx.stack(indices, axis=0)
            if indices
            else mx.zeros((0, batch, max_time), dtype=mx.int32)
        )
        quantized_out = self.output_proj(quantized_out).astype(mx.float32)
        return quantized_out, all_indices.astype(mx.int32), input_length

    def decode_codes(self, codes: mx.array) -> mx.array:
        nq, batch, time = codes.shape
        emb = mx.zeros((batch, self.rvq_dim, time), dtype=mx.float32)
        for index, quantizer in enumerate(self.quantizers[: int(nq)]):
            emb = emb + quantizer.decode_code(codes[index]).astype(mx.float32)
        return self.output_proj(emb).astype(mx.float32)


class MLXMossAudioTokenizer(nn.Module):
    def __init__(self, config: AudioTokenizerConfig):
        super().__init__()
        self.config = config
        self.sample_rate = int(config.sample_rate)
        self.sampling_rate = int(config.sampling_rate)
        self.downsample_rate = int(config.downsample_rate)
        self.channels = int(config.number_channels)
        self.number_channels = self.channels
        self.enable_channel_interleave = bool(config.enable_channel_interleave)

        channel_factor = (
            self.channels if self.enable_channel_interleave and self.channels > 1 else 1
        )
        current_frame_rate = float(self.sampling_rate * channel_factor)
        self.encoder = []
        for module_kwargs in config.encoder_kwargs or []:
            kwargs = dict(module_kwargs)
            module_type = kwargs.get("module_type")
            if module_type == "PatchedPretransform":
                self.encoder.append(PatchedPretransform(**kwargs, is_downsample=True))
            elif module_type == "Transformer":
                context_duration = float(
                    kwargs.pop(
                        "context_duration",
                        config.causal_transformer_context_duration,
                    )
                )
                kwargs.pop("attention_implementation", None)
                self.encoder.append(
                    ProjectedTransformer(
                        **kwargs,
                        context=int(round(current_frame_rate * context_duration)),
                    )
                )
            else:
                raise ValueError(f"Unsupported encoder module_type={module_type!r}")
            current_frame_rate /= self.encoder[-1].downsample_ratio

        quantizer_kwargs = dict(config.quantizer_kwargs or {})
        quantizer_type = quantizer_kwargs.get("quantizer_type", config.quantizer_type)
        if quantizer_type not in {"rlfq", "random_prefix_rlfq"}:
            raise ValueError(f"Unsupported MOSS quantizer_type={quantizer_type!r}")
        self.quantizer = ResidualLFQ(**quantizer_kwargs)
        self.num_quantizers = self.quantizer.num_quantizers

        self.decoder = []
        for module_kwargs in config.decoder_kwargs or []:
            kwargs = dict(module_kwargs)
            module_type = kwargs.get("module_type")
            if module_type == "PatchedPretransform":
                self.decoder.append(PatchedPretransform(**kwargs, is_downsample=False))
            elif module_type == "Transformer":
                context_duration = float(
                    kwargs.pop(
                        "context_duration",
                        config.causal_transformer_context_duration,
                    )
                )
                kwargs.pop("attention_implementation", None)
                self.decoder.append(
                    ProjectedTransformer(
                        **kwargs,
                        context=int(round(current_frame_rate * context_duration)),
                    )
                )
            else:
                raise ValueError(f"Unsupported decoder module_type={module_type!r}")
            current_frame_rate *= self.decoder[-1].downsample_ratio

    @classmethod
    def from_pretrained(cls, source: str | Path) -> "MLXMossAudioTokenizer":
        model_dir = _resolve_audio_tokenizer_dir(source)
        config = AudioTokenizerConfig.from_file(model_dir / "config.json")
        model = cls(config)
        weights = _load_weights_from_dir(model_dir)
        model.load_weights(list(weights.items()), strict=True)
        model.eval()
        return model

    @classmethod
    def from_model_dir(
        cls,
        model_dir: str | Path | None,
        *,
        fallback_source: str | None = None,
        **kwargs,
    ) -> "MLXMossAudioTokenizer":
        del kwargs
        if model_dir is not None:
            root = Path(model_dir).expanduser()
            for candidate in (root / "audio_tokenizer", root):
                if _is_audio_tokenizer_dir(candidate):
                    return cls.from_pretrained(candidate)
        return cls.from_pretrained(fallback_source or DEFAULT_AUDIO_TOKENIZER_REPO)

    def _load_audio_array(
        self,
        audio: Any,
        sample_rate: int | None = None,
    ) -> np.ndarray:
        if isinstance(audio, (str, Path)):
            from mlx_audio.audio_io import read as audio_read

            array, input_sample_rate = audio_read(str(audio), always_2d=True)
            source_sample_rate = int(input_sample_rate)
        elif isinstance(audio, mx.array):
            array = np.asarray(audio)
            source_sample_rate = int(sample_rate or self.sample_rate)
        else:
            array = np.asarray(audio)
            source_sample_rate = int(sample_rate or self.sample_rate)

        array = np.asarray(array, dtype=np.float32)
        if array.ndim == 1:
            array = array[:, None]
        elif (
            array.ndim == 2 and array.shape[0] <= 8 and array.shape[0] < array.shape[1]
        ):
            array = array.T
        elif array.ndim != 2:
            raise ValueError(f"Unsupported audio shape: {array.shape}")

        if source_sample_rate != self.sample_rate:
            array = np.asarray(
                resample_audio(array, source_sample_rate, self.sample_rate, axis=0),
                dtype=np.float32,
            )

        current_channels = int(array.shape[1])
        if current_channels == self.channels:
            return array
        if current_channels == 1 and self.channels > 1:
            return np.repeat(array, self.channels, axis=1)
        if current_channels > 1 and self.channels == 1:
            return array.mean(axis=1, keepdims=True)
        raise ValueError(
            "Unsupported reference audio channel conversion: "
            f"{current_channels} -> {self.channels}"
        )

    def _prepare_waveform_batch(
        self,
        waveforms: list[mx.array],
    ) -> tuple[mx.array, mx.array]:
        if not waveforms:
            raise ValueError("waveforms must not be empty")
        lengths = [int(waveform.shape[-1]) for waveform in waveforms]
        max_length = max(lengths)
        padded = []
        for waveform in waveforms:
            if waveform.ndim == 1:
                waveform = waveform[None, :]
            if waveform.shape[0] != self.channels:
                raise ValueError(
                    f"Expected waveform shape [{self.channels}, samples], "
                    f"got {waveform.shape}"
                )
            pad = max_length - int(waveform.shape[-1])
            if pad > 0:
                waveform = mx.pad(waveform, [(0, 0), (0, pad)])
            padded.append(waveform)
        return mx.stack(padded, axis=0), mx.array(lengths, dtype=mx.int32)

    def _prepare_codes_batch(
        self,
        codes_list: list[mx.array],
        num_quantizers: int | None = None,
    ) -> tuple[mx.array, mx.array, int]:
        if not codes_list:
            raise ValueError("codes_list must not be empty")
        nqs = [int(codes.shape[0]) for codes in codes_list]
        effective_nq = int(num_quantizers or nqs[0])
        if min(nqs) < effective_nq:
            raise ValueError(
                f"num_quantizers={effective_nq} exceeds available quantizers"
            )
        lengths = [int(codes.shape[-1]) for codes in codes_list]
        max_length = max(lengths)
        padded = mx.zeros(
            (effective_nq, len(codes_list), max_length),
            dtype=mx.int32,
        )
        for index, codes in enumerate(codes_list):
            padded[:, index, : codes.shape[-1]] = codes[:effective_nq].astype(mx.int32)
        return padded, mx.array(lengths, dtype=mx.int32), effective_nq

    def _flatten_channels_for_codec(
        self,
        input_values: mx.array,
        input_lengths: mx.array,
    ) -> tuple[mx.array, mx.array]:
        if input_values.shape[-1] % self.downsample_rate != 0:
            pad_length = self.downsample_rate - (
                input_values.shape[-1] % self.downsample_rate
            )
            input_values = mx.pad(input_values, [(0, 0), (0, 0), (0, pad_length)])
        if self.channels > 1 and self.enable_channel_interleave:
            input_values = input_values.transpose(0, 2, 1).reshape(
                input_values.shape[0], 1, -1
            )
            input_lengths = input_lengths * self.channels
        return input_values, input_lengths

    def _restore_channels_from_codec(
        self,
        output_values: mx.array,
        output_lengths: mx.array,
    ) -> tuple[mx.array, mx.array]:
        if self.channels == 1 or not self.enable_channel_interleave:
            return output_values.astype(mx.float32), output_lengths
        batch = output_values.shape[0]
        output_values = output_values[:, 0, :].reshape(batch, -1, self.channels)
        output_values = output_values.transpose(0, 2, 1).astype(mx.float32)
        return output_values, output_lengths // self.channels

    def _encode_frame(
        self,
        input_values: mx.array,
        input_lengths: mx.array | None = None,
        n_quantizers: int | None = None,
    ) -> tuple[mx.array, mx.array, mx.array]:
        if input_values.ndim == 1:
            input_values = input_values[None, None, :]
        elif input_values.ndim == 2:
            if self.channels == 1:
                input_values = input_values[:, None, :]
            else:
                input_values = input_values[None, :, :]
        if input_lengths is None:
            input_lengths = mx.full(
                (input_values.shape[0],),
                input_values.shape[-1],
                dtype=mx.int32,
            )
        hidden, hidden_lengths = self._flatten_channels_for_codec(
            input_values,
            input_lengths,
        )
        for module in self.encoder:
            hidden, hidden_lengths = module(hidden, hidden_lengths)
        _, audio_codes, audio_code_lengths = self.quantizer(
            hidden.astype(mx.float32),
            hidden_lengths,
            n_quantizers,
        )
        return audio_codes, audio_code_lengths, hidden.astype(mx.float32)

    def _decode_frame(
        self,
        codes: mx.array,
        codes_lengths: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        if codes.ndim != 3:
            raise ValueError(
                f"Expected codes shape [nq, batch, time], got {codes.shape}"
            )
        if codes_lengths is None:
            codes_lengths = mx.full((codes.shape[1],), codes.shape[-1], dtype=mx.int32)
        hidden = self.quantizer.decode_codes(codes.astype(mx.int32))
        audio, audio_lengths = hidden, codes_lengths
        for module in self.decoder:
            audio, audio_lengths = module(audio, audio_lengths)
        return self._restore_channels_from_codec(audio, audio_lengths)

    def encode_audio(
        self,
        audio: Any,
        *,
        sample_rate: int | None = None,
        num_quantizers: int | None = None,
    ) -> mx.array:
        audio_np = self._load_audio_array(audio, sample_rate=sample_rate)
        waveform = mx.array(audio_np.T.copy(), dtype=mx.float32)
        input_values, input_lengths = self._prepare_waveform_batch([waveform])
        codes, code_lengths, _ = self._encode_frame(
            input_values,
            input_lengths,
            n_quantizers=num_quantizers or self.num_quantizers,
        )
        mx.eval(codes, code_lengths)
        code_length = int(code_lengths.tolist()[0])
        return codes[:, 0, :code_length].T.astype(mx.int32)

    def decode_audio_codes(
        self,
        audio_codes: mx.array | np.ndarray,
        *,
        num_quantizers: int | None = None,
    ) -> mx.array:
        codes = (
            audio_codes if isinstance(audio_codes, mx.array) else mx.array(audio_codes)
        )
        codes = codes.astype(mx.int32)
        if codes.ndim == 3:
            if codes.shape[0] != 1:
                raise NotImplementedError(
                    "Batched MOSS audio-tokenizer decode is not implemented."
                )
            codes = codes[0]
        if codes.ndim != 2:
            raise ValueError(f"Expected codes shape [frames, nq], got {codes.shape}")
        if codes.shape[0] == 0:
            return mx.zeros((0, self.channels), dtype=mx.float32)

        effective_nq = int(num_quantizers or codes.shape[1])
        batched_codes, code_lengths, _ = self._prepare_codes_batch(
            [codes[:, :effective_nq].T],
            num_quantizers=effective_nq,
        )
        audio, audio_lengths = self._decode_frame(batched_codes, code_lengths)
        mx.eval(audio, audio_lengths)
        audio_length = int(audio_lengths.tolist()[0])
        return audio[0, :, :audio_length].T.astype(mx.float32)
