from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

from mlx_audio.tts.models.base import BaseModelArgs


def _filter_fields(cls, data: dict[str, Any]) -> dict[str, Any]:
    fields = cls.__dataclass_fields__.keys()
    return {k: v for k, v in data.items() if k in fields}


@dataclass
class FlowConfig(BaseModelArgs):
    dim: int
    depth: int

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "FlowConfig":
        return cls(**_filter_fields(cls, config))


@dataclass
class FlowLMTransformerConfig(BaseModelArgs):
    hidden_scale: int
    max_period: int
    d_model: int
    num_heads: int
    num_layers: int

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "FlowLMTransformerConfig":
        return cls(**_filter_fields(cls, config))


@dataclass
class LookupTable(BaseModelArgs):
    dim: int
    n_bins: int
    tokenizer: str
    tokenizer_path: str

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "LookupTable":
        return cls(**_filter_fields(cls, config))


@dataclass
class FlowLMConfig(BaseModelArgs):
    dtype: str
    flow: FlowConfig
    transformer: FlowLMTransformerConfig
    lookup_table: LookupTable
    weights_path: Optional[str] = None

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "FlowLMConfig":
        return cls(
            dtype=config.get("dtype"),
            flow=FlowConfig.from_dict(config.get("flow", {})),
            transformer=FlowLMTransformerConfig.from_dict(
                config.get("transformer", {})
            ),
            lookup_table=LookupTable.from_dict(config.get("lookup_table", {})),
            weights_path=config.get("weights_path"),
        )


@dataclass
class SEANetConfig(BaseModelArgs):
    dimension: int
    channels: int
    n_filters: int
    n_residual_layers: int
    ratios: list[int]
    kernel_size: int
    residual_kernel_size: int
    last_kernel_size: int
    dilation_base: int
    pad_mode: str
    compress: int

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "SEANetConfig":
        return cls(**_filter_fields(cls, config))


@dataclass
class MimiTransformerConfig(BaseModelArgs):
    d_model: int
    input_dimension: int
    output_dimensions: Tuple[int, ...]
    num_heads: int
    num_layers: int
    layer_scale: float
    context: int
    dim_feedforward: int
    max_period: float = 10000.0

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "MimiTransformerConfig":
        values = dict(_filter_fields(cls, config))
        output_dims = values.get("output_dimensions", ())
        if isinstance(output_dims, list):
            output_dims = tuple(output_dims)
        values["output_dimensions"] = output_dims
        return cls(**values)


@dataclass
class QuantizerConfig(BaseModelArgs):
    dimension: int
    output_dimension: int

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "QuantizerConfig":
        return cls(**_filter_fields(cls, config))


@dataclass
class MimiConfig(BaseModelArgs):
    dtype: str
    sample_rate: int
    channels: int
    frame_rate: float
    seanet: SEANetConfig
    transformer: MimiTransformerConfig
    quantizer: QuantizerConfig
    weights_path: Optional[str] = None

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "MimiConfig":
        return cls(
            dtype=config.get("dtype"),
            sample_rate=config.get("sample_rate"),
            channels=config.get("channels"),
            frame_rate=config.get("frame_rate"),
            seanet=SEANetConfig.from_dict(config.get("seanet", {})),
            transformer=MimiTransformerConfig.from_dict(config.get("transformer", {})),
            quantizer=QuantizerConfig.from_dict(config.get("quantizer", {})),
            weights_path=config.get("weights_path"),
        )


@dataclass
class ModelConfig(BaseModelArgs):
    model_type: str = "pocket_tts"
    flow_lm: Optional[FlowLMConfig] = None
    mimi: Optional[MimiConfig] = None
    weights_path: Optional[str] = None
    weights_path_without_voice_cloning: Optional[str] = None
    model_path: Optional[str] = None

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "ModelConfig":
        return cls(
            model_type=config.get("model_type", "pocket_tts"),
            flow_lm=FlowLMConfig.from_dict(config.get("flow_lm", {})),
            mimi=MimiConfig.from_dict(config.get("mimi", {})),
            weights_path=config.get("weights_path"),
            weights_path_without_voice_cloning=config.get(
                "weights_path_without_voice_cloning"
            ),
            model_path=config.get("model_path"),
        )


def load_yaml_config(path: str | Path) -> ModelConfig:
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return ModelConfig.from_dict(data)
