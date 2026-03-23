# pyright: reportMissingImports=false

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from safetensors import safe_open

from .config import MossAudioTokenizerConfig


def _create_additive_causal_mask(
    N: int, offset: int = 0, context: Optional[int] = None
) -> mx.array:
    rinds = mx.arange(offset + N)
    linds = mx.arange(offset, offset + N) if offset else rinds
    delta = linds[:, None] - rinds[None]
    allow = delta >= 0
    if context is not None:
        allow = mx.logical_and(allow, delta < int(context))
    deny = mx.logical_not(allow)
    return deny.astype(mx.float32) * -1e9


class _Attention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        causal: bool,
        rope: bool,
        max_period: float,
        context: Optional[int] = None,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim**-0.5
        self.causal = causal
        self.context = int(context) if context is not None else None

        self.in_projs = [nn.Linear(d_model, 3 * d_model, bias=False)]
        self.out_projs = [nn.Linear(d_model, d_model, bias=False)]

        self.rope = (
            nn.RoPE(self.head_dim, traditional=True, base=max_period) if rope else None
        )

    def __call__(self, x: mx.array) -> mx.array:
        B, T, _ = x.shape

        qkv = self.in_projs[0](x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q = qkv[:, :, 0].transpose(0, 2, 1, 3)
        k = qkv[:, :, 1].transpose(0, 2, 1, 3)
        v = qkv[:, :, 2].transpose(0, 2, 1, 3)

        offset = 0
        state = getattr(self, "_streaming_state", None)
        if state is not None:
            offset = int(getattr(state, "offset", 0))

        if self.rope is not None:
            q = self.rope(q, offset=offset)
            k = self.rope(k, offset=offset)

        mask = (
            _create_additive_causal_mask(
                T,
                offset=offset,
                context=self.context,
            ).astype(x.dtype)
            if self.causal
            else None
        )

        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        if mask is not None:
            scores = scores + mask

        attn = mx.softmax(scores, axis=-1)
        out = attn @ v
        out = out.transpose(0, 2, 1, 3).reshape(B, T, self.d_model)
        out = self.out_projs[0](out)

        if state is not None:
            state.offset = int(offset + int(T))
        return out


class _LayerScale(nn.Module):
    def __init__(self, dim: int, init: float):
        super().__init__()
        self.scale = mx.full((int(dim),), float(init), dtype=mx.float32)

    def __call__(self, xs: mx.array) -> mx.array:
        return xs * self.scale


class _TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        causal: bool = True,
        rope: bool = True,
        max_period: float = 10000.0,
        layer_scale: float = 1.0,
        context: Optional[int] = None,
    ):
        super().__init__()
        ls = float(layer_scale)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.self_attn = _Attention(
            d_model,
            num_heads,
            causal,
            rope,
            max_period,
            context=context,
        )
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        self.linear1 = nn.Linear(d_model, d_ff, bias=False)
        self.linear2 = nn.Linear(d_ff, d_model, bias=False)
        self.layer_scale_1 = _LayerScale(d_model, init=ls)
        self.layer_scale_2 = _LayerScale(d_model, init=ls)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.layer_scale_1(self.self_attn(self.norm1(x)))
        x = x + self.layer_scale_2(self.linear2(nn.gelu(self.linear1(self.norm2(x)))))
        return x


class _Transformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        causal: bool,
        rope: bool,
        max_period: float,
        layer_scale: float,
        context: Optional[int],
    ):
        super().__init__()
        self.layers = [
            _TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                causal=causal,
                rope=rope,
                max_period=max_period,
                layer_scale=layer_scale,
                context=context,
            )
            for _ in range(int(num_layers))
        ]

    def __call__(self, xs: mx.array) -> mx.array:
        for layer in self.layers:
            xs = layer(xs)
        return xs


class _ProjectedTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        causal: bool = True,
        rope: bool = True,
        max_period: float = 10000.0,
        layer_scale: float = 1.0,
        output_dim: Optional[int] = None,
        context: Optional[int] = None,
    ):
        super().__init__()
        self.input_proj = (
            nn.Linear(input_dim, d_model, bias=False)
            if int(input_dim) != int(d_model)
            else None
        )
        self.transformer = _Transformer(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_layers,
            causal=causal,
            rope=rope,
            max_period=max_period,
            layer_scale=layer_scale,
            context=context,
        )
        self.output_proj = (
            nn.Linear(d_model, int(output_dim), bias=False)
            if output_dim is not None and int(output_dim) != int(d_model)
            else None
        )

    def __call__(self, x: mx.array) -> mx.array:
        if self.input_proj is not None:
            x = self.input_proj(x)
        x = self.transformer(x)
        if self.output_proj is not None:
            x = self.output_proj(x)
        return x


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "1", "yes", "y", "on"}:
            return True
        if v in {"false", "0", "no", "n", "off"}:
            return False
    return bool(value)


def _ensure_projection_cfg(cfg: dict[str, Any], *, block_name: str) -> dict[str, Any]:
    missing = [
        name
        for name in ("input_dimension", "output_dimension", "d_model")
        if name not in cfg
    ]
    if missing:
        raise ValueError(f"{block_name} missing required keys: {', '.join(missing)}")
    return cfg


def _build_module_from_cfg(
    cfg: dict[str, Any],
    *,
    is_encoder: bool,
    context: Optional[int] = None,
) -> nn.Module:
    module_type = str(cfg.get("module_type", "")).strip()
    if module_type == "PatchedPretransform":
        patch_size = int(cfg.get("patch_size", 1))
        if is_encoder:
            return _PatchDownsample(patch_size)
        return _PatchUpsample(patch_size)

    if module_type != "Transformer":
        raise ValueError(f"Unsupported module_type in config: {module_type}")

    tcfg = _ensure_projection_cfg(cfg, block_name="Transformer")
    input_dim = int(tcfg["input_dimension"])
    output_dim = int(tcfg["output_dimension"])
    d_model = int(tcfg["d_model"])
    num_heads = int(tcfg.get("num_heads", 12))
    d_ff = int(tcfg.get("dim_feedforward", 4 * d_model))
    num_layers = int(tcfg.get("num_layers", 1))
    causal = _as_bool(tcfg.get("causal", True), default=True)
    rope = str(tcfg.get("positional_embedding", "rope")).lower() == "rope"
    max_period = float(tcfg.get("max_period", 10000.0))
    layer_scale = float(tcfg.get("layer_scale", 1.0))

    return _ProjectedTransformer(
        input_dim=input_dim,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        causal=causal,
        rope=rope,
        max_period=max_period,
        layer_scale=layer_scale,
        output_dim=output_dim,
        context=context,
    )


def _default_encoder_kwargs() -> list[dict[str, Any]]:
    return [
        {
            "module_type": "PatchedPretransform",
            "patch_size": 240,
        },
        {
            "module_type": "Transformer",
            "input_dimension": 240,
            "output_dimension": 384,
            "d_model": 768,
            "num_heads": 12,
            "num_layers": 12,
            "dim_feedforward": 3072,
            "causal": True,
            "positional_embedding": "rope",
            "max_period": 10000,
            "layer_scale": 0.01,
        },
        {
            "module_type": "PatchedPretransform",
            "patch_size": 2,
        },
        {
            "module_type": "Transformer",
            "input_dimension": 768,
            "output_dimension": 384,
            "d_model": 768,
            "num_heads": 12,
            "num_layers": 12,
            "dim_feedforward": 3072,
            "causal": True,
            "positional_embedding": "rope",
            "max_period": 10000,
            "layer_scale": 0.01,
        },
        {
            "module_type": "PatchedPretransform",
            "patch_size": 2,
        },
        {
            "module_type": "Transformer",
            "input_dimension": 768,
            "output_dimension": 640,
            "d_model": 768,
            "num_heads": 12,
            "num_layers": 12,
            "dim_feedforward": 3072,
            "causal": True,
            "positional_embedding": "rope",
            "max_period": 10000,
            "layer_scale": 0.01,
        },
        {
            "module_type": "PatchedPretransform",
            "patch_size": 2,
        },
        {
            "module_type": "Transformer",
            "input_dimension": 1280,
            "output_dimension": 768,
            "d_model": 1280,
            "num_heads": 20,
            "num_layers": 32,
            "dim_feedforward": 5120,
            "causal": True,
            "positional_embedding": "rope",
            "max_period": 10000,
            "layer_scale": 0.01,
        },
    ]


def _default_decoder_kwargs() -> list[dict[str, Any]]:
    return [
        {
            "module_type": "Transformer",
            "input_dimension": 768,
            "output_dimension": 1280,
            "d_model": 1280,
            "num_heads": 20,
            "num_layers": 32,
            "dim_feedforward": 5120,
            "causal": True,
            "positional_embedding": "rope",
            "max_period": 10000,
            "layer_scale": 0.01,
        },
        {
            "module_type": "PatchedPretransform",
            "patch_size": 2,
        },
        {
            "module_type": "Transformer",
            "input_dimension": 640,
            "output_dimension": 768,
            "d_model": 768,
            "num_heads": 12,
            "num_layers": 12,
            "dim_feedforward": 3072,
            "causal": True,
            "positional_embedding": "rope",
            "max_period": 10000,
            "layer_scale": 0.01,
        },
        {
            "module_type": "PatchedPretransform",
            "patch_size": 2,
        },
        {
            "module_type": "Transformer",
            "input_dimension": 384,
            "output_dimension": 768,
            "d_model": 768,
            "num_heads": 12,
            "num_layers": 12,
            "dim_feedforward": 3072,
            "causal": True,
            "positional_embedding": "rope",
            "max_period": 10000,
            "layer_scale": 0.01,
        },
        {
            "module_type": "PatchedPretransform",
            "patch_size": 2,
        },
        {
            "module_type": "Transformer",
            "input_dimension": 384,
            "output_dimension": 240,
            "d_model": 768,
            "num_heads": 12,
            "num_layers": 12,
            "dim_feedforward": 3072,
            "causal": True,
            "positional_embedding": "rope",
            "max_period": 10000,
            "layer_scale": 0.01,
        },
        {
            "module_type": "PatchedPretransform",
            "patch_size": 240,
        },
    ]


class _PatchUpsample(nn.Module):
    def __init__(self, patch_size: int, input_dim: Optional[int] = None):
        super().__init__()
        self.patch_size = int(patch_size)
        self.input_dim = int(input_dim) if input_dim is not None else None

    def __call__(self, x: mx.array) -> mx.array:
        b, t, c = x.shape
        if self.input_dim is not None and int(c) != int(self.input_dim):
            raise ValueError("input_dim mismatch")
        p = int(self.patch_size)
        if p <= 0 or int(c) % p != 0:
            raise ValueError("invalid patch_size")
        x = x.reshape(b, t, c // p, p)  # (b, t, out_channels, patch_idx)
        x = x.transpose(0, 1, 3, 2)  # (b, t, patch_idx, out_channels)
        x = x.reshape(b, t * p, c // p)  # interleaved time
        return x


class _WeightParam(nn.Module):
    def __init__(
        self, original0_shape: Tuple[int, ...], original1_shape: Tuple[int, ...]
    ):
        super().__init__()
        self.original0 = mx.zeros(original0_shape, dtype=mx.float32)
        self.original1 = mx.zeros(original1_shape, dtype=mx.float32)


class _WeightNorm1x1(nn.Module):
    def __init__(
        self,
        out_dim: int,
        in_dim: int,
        *,
        original1_layout: str,
    ):
        super().__init__()
        if original1_layout not in {"out_1_in", "out_in_1"}:
            raise ValueError("invalid original1 layout")
        if original1_layout == "out_1_in":
            o1 = (int(out_dim), 1, int(in_dim))
        else:
            o1 = (int(out_dim), int(in_dim), 1)
        self.parametrizations = {"weight": _WeightParam((int(out_dim), 1, 1), o1)}
        self.bias = mx.zeros((int(out_dim),), dtype=mx.float32)
        self._original1_layout = original1_layout

    def __call__(self, x: mx.array) -> mx.array:
        w0 = self.parametrizations["weight"].original0
        w1 = self.parametrizations["weight"].original1

        if self._original1_layout == "out_1_in":
            if int(w1.shape[1]) != 1 and int(w1.shape[2]) == 1:
                w1 = mx.transpose(w1, (0, 2, 1))
                self.parametrizations["weight"].original1 = w1
            if int(w1.shape[1]) != 1:
                raise ValueError(
                    f"Unexpected WeightNorm original1 shape {tuple(w1.shape)} for layout out_1_in"
                )
        else:
            if int(w1.shape[1]) == 1 and int(w1.shape[2]) != 1:
                w1 = mx.transpose(w1, (0, 2, 1))
                self.parametrizations["weight"].original1 = w1
            if int(w1.shape[2]) != 1:
                raise ValueError(
                    f"Unexpected WeightNorm original1 shape {tuple(w1.shape)} for layout out_in_1"
                )

        norm = mx.sqrt(mx.sum(w1.astype(mx.float32) ** 2, axis=(1, 2), keepdims=True))
        norm = mx.maximum(norm, mx.array(1e-12, dtype=norm.dtype))
        w = w0 * (w1 / norm.astype(w1.dtype))
        if self._original1_layout == "out_1_in":
            w2 = w[:, 0, :]
        else:
            w2 = w[:, :, 0]
        y = x @ w2.T
        return y + self.bias[None, None, :]

    def load_weights(self, file_or_weights, strict: bool = True):
        super().load_weights(file_or_weights, strict=strict)

        w1 = self.parametrizations["weight"].original1
        if not isinstance(w1, mx.array) or w1.ndim != 3:
            return self

        if self._original1_layout == "out_1_in":
            expected = (int(w1.shape[0]), 1, int(w1.shape[2]))
            if tuple(w1.shape) == expected:
                return self
            if int(w1.shape[1]) == int(expected[2]) and int(w1.shape[2]) == 1:
                self.parametrizations["weight"].original1 = mx.transpose(w1, (0, 2, 1))
                return self
        else:
            expected = (int(w1.shape[0]), int(w1.shape[1]), 1)
            if tuple(w1.shape) == expected:
                return self
            if int(w1.shape[1]) == 1 and int(w1.shape[2]) == int(expected[1]):
                self.parametrizations["weight"].original1 = mx.transpose(w1, (0, 2, 1))
                return self

        if strict:
            raise ValueError(
                f"Unexpected WeightNorm original1 shape {tuple(w1.shape)} for layout {self._original1_layout}"
            )

        return self


class _PatchDownsample(nn.Module):
    def __init__(self, patch_size: int):
        super().__init__()
        self.patch_size = int(patch_size)

    def __call__(self, x: mx.array) -> mx.array:
        b, c, t = x.shape
        p = int(self.patch_size)
        if p <= 0:
            raise ValueError("invalid patch_size")
        if int(t) % p != 0:
            raise ValueError("time length must be divisible by patch_size")
        x = x.reshape(int(b), int(c), int(t) // p, p)
        x = x.transpose(0, 1, 3, 2)
        x = x.reshape(int(b), int(c) * p, int(t) // p)
        return x


class _ResidualLFQ(nn.Module):
    def __init__(self, num_quantizers: int, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        self.input_proj = _WeightNorm1x1(512, 768, original1_layout="out_1_in")
        self.output_proj = _WeightNorm1x1(768, 512, original1_layout="out_1_in")

        self.quantizers = [
            _LFQQuantizer(codebook_size, codebook_dim) for _ in range(num_quantizers)
        ]

    def decode_codes(self, codes: mx.array) -> mx.array:
        """Decode quantizer codes to embeddings.

        Args:
            codes: Shape (num_quantizers, batch, time)

        Returns:
            Decoded embeddings of shape (batch, time, 768)
        """
        nq, b, t = codes.shape
        if int(nq) != int(self.num_quantizers):
            raise ValueError("num_quantizers mismatch")

        x = mx.zeros((int(b), int(t), 512), dtype=mx.float32)
        for i in range(self.num_quantizers):
            x = x + self.quantizers[i].decode_codes(codes[i])
        x = self.output_proj(x)
        return x

    def encode(
        self, z: mx.array, input_length: mx.array, n_quantizers: Optional[int] = None
    ) -> tuple[mx.array, mx.array, mx.array]:
        z = self.input_proj(z).astype(mx.float32)
        b, max_time, _ = z.shape

        if n_quantizers is None:
            n_quantizers = self.num_quantizers
        nq = int(n_quantizers)
        if nq < 1 or nq > self.num_quantizers:
            raise ValueError("invalid n_quantizers")

        t_idx = mx.arange(int(max_time), dtype=mx.int32)[None, :]
        valid = t_idx < input_length[:, None].astype(mx.int32)
        valid_f = valid[:, :, None].astype(mx.float32)

        quantized_out = mx.zeros(z.shape, dtype=mx.float32)
        residual = z.astype(mx.float32)
        all_indices: List[mx.array] = []

        for i in range(nq):
            masked = residual * valid_f
            quantizer = self.quantizers[i]

            z_e = quantizer.in_proj(masked).astype(mx.float32)
            enc = z_e.reshape(-1, int(z_e.shape[-1]))

            codebook = quantizer.codebook.weight.astype(mx.float32)
            enc_norm = enc / mx.maximum(
                mx.sqrt(mx.sum(enc**2, axis=1, keepdims=True)),
                mx.array(1e-12, dtype=mx.float32),
            )
            codebook_norm = codebook / mx.maximum(
                mx.sqrt(mx.sum(codebook**2, axis=1, keepdims=True)),
                mx.array(1e-12, dtype=mx.float32),
            )

            dist = (
                mx.sum(enc_norm**2, axis=1, keepdims=True)
                - 2.0 * (enc_norm @ mx.transpose(codebook_norm, (1, 0)))
                + mx.transpose(mx.sum(codebook_norm**2, axis=1, keepdims=True), (1, 0))
            )
            flat_idx = mx.argmax(-dist, axis=1).astype(mx.int32)
            idx = flat_idx.reshape(int(b), int(max_time))

            z_q = quantizer.decode_codes(idx)

            quantized_out = quantized_out + z_q * valid_f
            residual = residual - z_q * valid_f
            all_indices.append(idx)

        stacked = mx.stack(all_indices, axis=0).astype(mx.int32)
        quantized_out = self.output_proj(quantized_out).astype(mx.float32)
        return quantized_out, stacked, input_length


class _LFQQuantizer(nn.Module):
    def __init__(self, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.codebook = nn.Embedding(int(codebook_size), int(codebook_dim))
        self.in_proj = _WeightNorm1x1(
            int(codebook_dim), 512, original1_layout="out_1_in"
        )
        self.out_proj = _WeightNorm1x1(
            512, int(codebook_dim), original1_layout="out_1_in"
        )

    def decode_codes(self, codes: mx.array) -> mx.array:
        emb = self.codebook(codes.astype(mx.int32)).astype(mx.float32)
        return self.out_proj(emb)


class MossAudioTokenizer(nn.Module):
    def __init__(self, config: MossAudioTokenizerConfig):
        super().__init__()
        self.config = config
        current_frame_rate = float(config.sampling_rate)
        context_duration = float(
            getattr(config, "causal_transformer_context_duration", 10.0)
        )

        def _transformer_context() -> int:
            return max(1, int(current_frame_rate * context_duration))

        encoder_cfgs = config.encoder_kwargs
        if encoder_cfgs is None:
            encoder_cfgs = _default_encoder_kwargs()
        self.encoder = []
        for module_cfg in encoder_cfgs:
            cfg_i = dict(module_cfg)
            module = _build_module_from_cfg(
                cfg_i,
                is_encoder=True,
                context=_transformer_context(),
            )
            self.encoder.append(module)
            if str(cfg_i.get("module_type", "")) == "PatchedPretransform":
                current_frame_rate /= max(1, int(cfg_i.get("patch_size", 1)))

        decoder_cfgs = config.decoder_kwargs
        if decoder_cfgs is None:
            decoder_cfgs = _default_decoder_kwargs()
        self.decoder = []
        for module_cfg in decoder_cfgs:
            cfg_i = dict(module_cfg)
            module = _build_module_from_cfg(
                cfg_i,
                is_encoder=False,
                context=_transformer_context(),
            )
            self.decoder.append(module)
            if str(cfg_i.get("module_type", "")) == "PatchedPretransform":
                current_frame_rate *= max(1, int(cfg_i.get("patch_size", 1)))

        quantizer_type = str(getattr(config, "quantizer_type", "rlfq")).lower()
        if quantizer_type not in {"rlfq", "random_prefix_rlfq"}:
            raise ValueError(
                f"Unsupported quantizer_type for MLX MossAudioTokenizer: {quantizer_type}"
            )

        self.quantizer = _ResidualLFQ(
            int(config.num_quantizers),
            int(config.codebook_size),
            int(config.codebook_dim),
        )

        self.has_encoder = True

    def decode(self, audio_codes: mx.array, **kwargs: Any) -> dict[str, mx.array]:
        if audio_codes.ndim != 3:
            raise ValueError(
                "audio_codes must have shape (num_quantizers, batch, time)"
            )

        if int(audio_codes.shape[0]) != int(self.config.num_quantizers):
            raise ValueError(
                "audio_codes must have shape (num_quantizers, batch, time)"
            )

        x = self.quantizer.decode_codes(audio_codes)
        for module in self.decoder:
            x = module(x)
        audio = x.transpose(0, 2, 1).astype(mx.float32)
        return {"audio": audio}

    def encode(
        self, input_values: mx.array, num_quantizers: Optional[int] = None
    ) -> dict[str, mx.array]:
        if not bool(self.has_encoder):
            raise ValueError(
                "Codec encoder weights are not loaded; provide a codec directory with encoder weights"
            )

        if input_values.ndim == 1:
            input_values = input_values[None, None, :]
        elif input_values.ndim == 2:
            input_values = input_values[:, None, :]
        elif input_values.ndim != 3:
            raise ValueError(
                "input_values must have shape (time), (batch, time), or (batch, channels, time)"
            )

        if int(input_values.shape[1]) != 1:
            raise ValueError("input_values must have a single channel")

        b, _, t = input_values.shape
        ds = int(self.config.downsample_rate)
        if int(t) % ds != 0:
            pad = ds - (int(t) % ds)
            input_values = mx.pad(input_values, [(0, 0), (0, 0), (0, pad)])
            t = int(input_values.shape[2])

        input_lengths = mx.full((int(b),), int(t), dtype=mx.int32)
        e = input_values.astype(mx.float32)
        e_lengths = input_lengths

        for m in self.encoder:
            if isinstance(m, _PatchDownsample):
                e = m(e)
                e_lengths = e_lengths // int(m.patch_size)
            else:
                e_t = e.transpose(0, 2, 1)
                e_t = m(e_t)
                e = e_t.transpose(0, 2, 1)

        e_t = e.transpose(0, 2, 1)
        _, audio_codes, audio_codes_lengths = self.quantizer.encode(
            e_t, e_lengths, num_quantizers
        )
        return {
            "audio_codes": audio_codes,
            "audio_codes_lengths": audio_codes_lengths,
            "encoder_hidden_states": e_t,
        }

    @staticmethod
    def sanitize(weights: dict[str, mx.array]) -> dict[str, mx.array]:
        _KEEP_PREFIXES = ("decoder.", "quantizer.")

        def _is_mlx_conv1d_weight(arr: mx.array) -> bool:
            shape = arr.shape
            if len(shape) != 3:
                return False
            _, dim2, dim3 = shape
            if dim2 == 1:
                return dim3 > 64
            if dim3 == 1:
                return dim2 == 1
            return dim2 < dim3

        sanitized: dict[str, mx.array] = {}

        for k, v in weights.items():
            if not any(k.startswith(p) for p in _KEEP_PREFIXES):
                continue

            if k.endswith(".out_proj.parametrizations.weight.original1"):
                sanitized[k] = v
                continue

            if isinstance(v, mx.array) and v.ndim == 3 and not _is_mlx_conv1d_weight(v):
                sanitized[k] = mx.transpose(v, (0, 2, 1))
            else:
                sanitized[k] = v
        return sanitized

    @classmethod
    def from_pretrained(cls, path_or_repo: str) -> "MossAudioTokenizer":
        """Load a pretrained MOSS Audio Tokenizer model.

        Args:
            path_or_repo: Path to local directory containing config.json and model weights

        Returns:
            MossAudioTokenizer: Initialized model with loaded weights
        """
        path = Path(path_or_repo)

        config_path = path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found at {config_path}")

        with open(config_path) as f:
            config_dict = json.load(f)
        config = MossAudioTokenizerConfig.from_dict(config_dict)

        model = cls(config)

        weights: Dict[str, mx.array] = {}
        single_path = path / "model.safetensors"
        if single_path.exists():
            with safe_open(str(single_path), framework="numpy") as f:
                for key in f.keys():
                    weights[key] = mx.array(f.get_tensor(key))
        else:
            shards = sorted(path.glob("model-*.safetensors"))
            if not shards:
                raise FileNotFoundError(
                    f"No safetensors found in {path} "
                    "(expected model.safetensors or model-*.safetensors)"
                )
            for shard in shards:
                with safe_open(str(shard), framework="numpy") as f:
                    for key in f.keys():
                        weights[key] = mx.array(f.get_tensor(key))

        has_encoder_weights = any(k.startswith("encoder.") for k in weights)
        sanitized = cls.sanitize(weights)
        if has_encoder_weights:
            for k, v in weights.items():
                if k.startswith("encoder."):
                    sanitized[k] = v

        try:
            model.load_weights(list(sanitized.items()), strict=False)
        except Exception as exc:
            params = tree_flatten(model.parameters(), destination={})
            if not isinstance(params, dict):
                params = dict(params)
            w_keys = set(sanitized.keys())
            p_keys = set(params.keys())
            extra = sorted(w_keys - p_keys)
            missing = sorted(p_keys - w_keys)
            raise RuntimeError(
                f"Failed to load MossAudioTokenizer weights. "
                f"extras={len(extra)} missing={len(missing)}. "
                f"extra[:20]={', '.join(extra[:20])}. "
                f"missing[:20]={', '.join(missing[:20])}"
            ) from exc

        model.has_encoder = has_encoder_weights
        return model
