from __future__ import annotations

from dataclasses import dataclass, field
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
    hidden_scale: float
    max_period: int
    d_model: int
    num_heads: int
    num_layers: int
    gelu: str = "erf"

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
    ldim: Optional[int] = None
    time_rms_norm: str = "rms_var"

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "FlowLMConfig":
        if "d_model" in config:
            config = {
                **config,
                "flow": {"dim": config["flow_dim"], "depth": config["flow_depth"]},
                "transformer": {
                    **config,
                    "hidden_scale": config["dim_feedforward"] / config["d_model"],
                },
                "lookup_table": {
                    "dim": config["lut_dim"],
                    "n_bins": config["n_bins"],
                    "tokenizer": "huggingface",
                    "tokenizer_path": "tokenizer.json",
                },
            }
        return cls(
            dtype=config.get("dtype"),
            flow=FlowConfig.from_dict(config.get("flow", {})),
            transformer=FlowLMTransformerConfig.from_dict(
                config.get("transformer", {})
            ),
            lookup_table=LookupTable.from_dict(config.get("lookup_table", {})),
            weights_path=config.get("weights_path"),
            ldim=config.get("ldim"),
            time_rms_norm=config.get("time_rms_norm", "rms_var"),
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
    encoder_enabled: bool = True
    gelu: str = "tanh"

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "MimiConfig":
        if "dimension" in config:
            config = {
                **config,
                "seanet": {**config, "pad_mode": config.get("pad_mode", "constant")},
                "transformer": {
                    **{
                        k.removeprefix("transformer_"): v
                        for k, v in config.items()
                        if k.startswith("transformer_")
                    },
                    "input_dimension": config["dimension"],
                    "output_dimensions": [config["dimension"]],
                },
                "quantizer": {
                    "dimension": config["quantizer_dimension"],
                    "output_dimension": config["quantizer_output_dimension"],
                },
            }
        return cls(
            dtype=config.get("dtype"),
            sample_rate=config.get("sample_rate"),
            channels=config.get("channels"),
            frame_rate=config.get("frame_rate"),
            seanet=SEANetConfig.from_dict(config.get("seanet", {})),
            transformer=MimiTransformerConfig.from_dict(config.get("transformer", {})),
            quantizer=QuantizerConfig.from_dict(config.get("quantizer", {})),
            weights_path=config.get("weights_path"),
            encoder_enabled=config.get("encoder_enabled", True),
            gelu=config.get("gelu", "tanh"),
        )


@dataclass
class ModelConfig(BaseModelArgs):
    model_type: str = "pocket_tts"
    flow_lm: Optional[FlowLMConfig] = None
    mimi: Optional[MimiConfig] = None
    weights_path: Optional[str] = None
    weights_path_without_voice_cloning: Optional[str] = None
    model_path: Optional[str] = None
    temp: float = 0.7
    lsd_decode_steps: int = 1
    eos_threshold: float = -4.0
    noise_clamp: Optional[float] = None
    audio_prompt_min_duration: float = 0.0
    audio_prompt_max_duration: Optional[float] = None
    conditioners: list[dict[str, Any]] = field(default_factory=list)
    fuser: dict[str, Any] = field(default_factory=dict)
    weights_name: Optional[str] = None
    tokenizer_name: Optional[str] = None
    quantization: Optional[dict[str, Any]] = None

    def __post_init__(self):
        if self.audio_prompt_min_duration < 0:
            raise ValueError("audio_prompt_min_duration must be >= 0.")
        if (
            self.audio_prompt_max_duration is not None
            and self.audio_prompt_max_duration < self.audio_prompt_min_duration
        ):
            raise ValueError(
                "audio_prompt_max_duration must be >= audio_prompt_min_duration."
            )

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "ModelConfig":
        values = _filter_fields(cls, config)
        if "quantization_config" in config and "quantization" not in config:
            values["quantization"] = config["quantization_config"]
        values["flow_lm"] = FlowLMConfig.from_dict(config.get("flow_lm", {}))
        values["mimi"] = MimiConfig.from_dict(config.get("mimi", {}))
        model = cls(**values)
        # Prefer a bundled tokenizer over the configured download path.
        if model.model_path:
            root = Path(model.model_path)
            lut = model.flow_lm.lookup_table
            candidates = [
                root / name
                for name in (model.tokenizer_name, "tokenizer.model", "tokenizer.json")
                if name
            ]
            if "://" not in lut.tokenizer_path:
                candidates.insert(0, root / lut.tokenizer_path)
            for path in candidates:
                if path.is_file():
                    lut.tokenizer_path = str(path)
                    lut.tokenizer = (
                        "huggingface" if path.suffix == ".json" else "sentencepiece"
                    )
                    break
        return model


def load_yaml_config(path: str | Path) -> ModelConfig:
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return ModelConfig.from_dict(data)
