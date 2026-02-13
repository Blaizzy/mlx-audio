"""Configuration helpers for the MOSS audio tokenizer MLX port."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


LEGACY_MODEL_TYPES = {"speech_tokenizer", "moss-audio-tokenizer"}
CANONICAL_MODEL_TYPE = "moss_audio_tokenizer"


@dataclass(frozen=True)
class MossAudioTokenizerModuleConfig:
    module_type: str
    patch_size: Optional[int] = None
    input_dimension: Optional[int] = None
    output_dimension: Optional[int] = None
    d_model: Optional[int] = None
    num_heads: Optional[int] = None
    num_layers: Optional[int] = None
    dim_feedforward: Optional[int] = None
    causal: Optional[bool] = None
    norm: Optional[str] = None
    positional_embedding: Optional[str] = None
    max_period: Optional[float] = None
    gating: Optional[str] = None
    layer_scale: Optional[float] = None
    conv_layout: Optional[bool] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MossAudioTokenizerModuleConfig":
        return cls(
            module_type=str(data["module_type"]),
            patch_size=data.get("patch_size"),
            input_dimension=data.get("input_dimension"),
            output_dimension=data.get("output_dimension"),
            d_model=data.get("d_model"),
            num_heads=data.get("num_heads"),
            num_layers=data.get("num_layers"),
            dim_feedforward=data.get("dim_feedforward"),
            causal=data.get("causal"),
            norm=data.get("norm"),
            positional_embedding=data.get("positional_embedding"),
            max_period=data.get("max_period"),
            gating=data.get("gating"),
            layer_scale=data.get("layer_scale"),
            conv_layout=data.get("conv_layout"),
        )

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"module_type": self.module_type}
        for field_name in [
            "patch_size",
            "input_dimension",
            "output_dimension",
            "d_model",
            "num_heads",
            "num_layers",
            "dim_feedforward",
            "causal",
            "norm",
            "positional_embedding",
            "max_period",
            "gating",
            "layer_scale",
            "conv_layout",
        ]:
            value = getattr(self, field_name)
            if value is not None:
                payload[field_name] = value
        return payload


@dataclass(frozen=True)
class MossAudioTokenizerQuantizerConfig:
    input_dim: int
    rvq_dim: int
    output_dim: int
    num_quantizers: int
    codebook_size: int
    codebook_dim: int
    quantizer_type: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MossAudioTokenizerQuantizerConfig":
        return cls(
            input_dim=int(data["input_dim"]),
            rvq_dim=int(data["rvq_dim"]),
            output_dim=int(data["output_dim"]),
            num_quantizers=int(data["num_quantizers"]),
            codebook_size=int(data["codebook_size"]),
            codebook_dim=int(data["codebook_dim"]),
            quantizer_type=str(data["quantizer_type"]),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "rvq_dim": self.rvq_dim,
            "output_dim": self.output_dim,
            "num_quantizers": self.num_quantizers,
            "codebook_size": self.codebook_size,
            "codebook_dim": self.codebook_dim,
            "quantizer_type": self.quantizer_type,
        }


@dataclass(frozen=True)
class MossAudioTokenizerConfig:
    model_type: str
    sampling_rate: int
    downsample_rate: int
    causal_transformer_context_duration: float
    encoder_modules: List[MossAudioTokenizerModuleConfig]
    decoder_modules: List[MossAudioTokenizerModuleConfig]
    quantizer: MossAudioTokenizerQuantizerConfig

    @property
    def frame_rate(self) -> float:
        return self.sampling_rate / self.downsample_rate

    @property
    def encoder_patch_product(self) -> int:
        return _patch_product(self.encoder_modules)

    @property
    def decoder_patch_product(self) -> int:
        return _patch_product(self.decoder_modules)

    def patch_alignment_is_valid(self) -> bool:
        # Encoder and decoder patch products should both match configured downsample.
        return (
            self.encoder_patch_product == self.downsample_rate
            and self.decoder_patch_product == self.downsample_rate
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MossAudioTokenizerConfig":
        model_type = str(data.get("model_type", CANONICAL_MODEL_TYPE))
        if model_type in LEGACY_MODEL_TYPES:
            model_type = CANONICAL_MODEL_TYPE

        encoder_payload = data.get("encoder_kwargs", data.get("encoder_modules", []))
        decoder_payload = data.get("decoder_kwargs", data.get("decoder_modules", []))
        encoder_modules = [
            MossAudioTokenizerModuleConfig.from_dict(item) for item in encoder_payload
        ]
        decoder_modules = [
            MossAudioTokenizerModuleConfig.from_dict(item) for item in decoder_payload
        ]
        quantizer_payload = data.get("quantizer_kwargs", data.get("quantizer"))
        if quantizer_payload is None:
            raise ValueError("Missing quantizer_kwargs/quantizer in config payload")
        quantizer = MossAudioTokenizerQuantizerConfig.from_dict(quantizer_payload)

        return cls(
            model_type=model_type,
            sampling_rate=int(data["sampling_rate"]),
            downsample_rate=int(data["downsample_rate"]),
            causal_transformer_context_duration=float(
                data["causal_transformer_context_duration"]
            ),
            encoder_modules=encoder_modules,
            decoder_modules=decoder_modules,
            quantizer=quantizer,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_type": self.model_type,
            "sampling_rate": self.sampling_rate,
            "downsample_rate": self.downsample_rate,
            "causal_transformer_context_duration": self.causal_transformer_context_duration,
            "encoder_kwargs": [module.to_dict() for module in self.encoder_modules],
            "decoder_kwargs": [module.to_dict() for module in self.decoder_modules],
            "quantizer_kwargs": self.quantizer.to_dict(),
            "quantizer_type": self.quantizer.quantizer_type,
        }


def _patch_product(modules: List[MossAudioTokenizerModuleConfig]) -> int:
    product = 1
    for module in modules:
        if module.module_type == "PatchedPretransform":
            if module.patch_size is None:
                raise ValueError("PatchedPretransform module is missing patch_size")
            product *= int(module.patch_size)
    return product


def load_moss_audio_tokenizer_config(path: str | Path) -> MossAudioTokenizerConfig:
    config_path = Path(path)
    data = json.loads(config_path.read_text(encoding="utf-8"))
    return MossAudioTokenizerConfig.from_dict(data)


__all__ = [
    "CANONICAL_MODEL_TYPE",
    "LEGACY_MODEL_TYPES",
    "MossAudioTokenizerConfig",
    "MossAudioTokenizerModuleConfig",
    "MossAudioTokenizerQuantizerConfig",
    "load_moss_audio_tokenizer_config",
]
