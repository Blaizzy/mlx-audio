from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from mlx_audio.stt.models.qwen3_asr.config import TextConfig


def _filter_kwargs(cls, params: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in params.items() if k in inspect.signature(cls).parameters}


@dataclass
class FrontendConfig:
    fs: int = 16000
    window: str = "hamming"
    n_mels: int = 80
    frame_length: int = 25
    frame_shift: int = 10
    lfr_m: int = 7
    lfr_n: int = 6
    cmvn_file: Optional[str] = None

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> "FrontendConfig":
        return cls(**_filter_kwargs(cls, params))


@dataclass
class SenseVoiceEncoderConfig:
    output_size: int = 512
    attention_heads: int = 4
    linear_units: int = 2048
    num_blocks: int = 50
    tp_blocks: int = 20
    dropout_rate: float = 0.0
    positional_dropout_rate: float = 0.0
    attention_dropout_rate: float = 0.0
    normalize_before: bool = True
    kernel_size: int = 11
    sanm_shift: int = 0
    feat_permute: bool = False

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> "SenseVoiceEncoderConfig":
        params = params.copy()
        if "sanm_shfit" in params and "sanm_shift" not in params:
            params["sanm_shift"] = params["sanm_shfit"]
        return cls(**_filter_kwargs(cls, params))


@dataclass
class AdaptorConfig:
    downsample_rate: int = 1
    use_low_frame_rate: bool = True
    ffn_dim: int = 2048
    llm_dim: int = 1024
    encoder_dim: int = 512
    n_layer: int = 2
    attention_heads: int = 8
    dropout_rate: float = 0.0
    attention_dropout_rate: float = 0.0

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> "AdaptorConfig":
        return cls(**_filter_kwargs(cls, params))


@dataclass
class FunASRNanoConfig:
    model_type: str = "fun_asr_nano"
    model_repo: Optional[str] = "FunAudioLLM/Fun-ASR-Nano-2512"
    model_path: Optional[str] = None

    input_size: int = 560
    qwen_tokenizer_path: str = "Qwen3-0.6B"

    frontend_conf: FrontendConfig = field(default_factory=FrontendConfig)
    audio_encoder_conf: SenseVoiceEncoderConfig = field(
        default_factory=SenseVoiceEncoderConfig
    )
    audio_adaptor_conf: AdaptorConfig = field(default_factory=AdaptorConfig)
    text_config: TextConfig = field(default_factory=TextConfig)

    default_max_tokens: int = 512

    def __post_init__(self):
        if isinstance(self.frontend_conf, dict):
            self.frontend_conf = FrontendConfig.from_dict(self.frontend_conf)
        if isinstance(self.audio_encoder_conf, dict):
            self.audio_encoder_conf = SenseVoiceEncoderConfig.from_dict(
                self.audio_encoder_conf
            )
        if isinstance(self.audio_adaptor_conf, dict):
            self.audio_adaptor_conf = AdaptorConfig.from_dict(self.audio_adaptor_conf)
        if isinstance(self.text_config, dict):
            self.text_config = TextConfig.from_dict(self.text_config)

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> "FunASRNanoConfig":
        params = params.copy()

        if "llm_config" in params and "text_config" not in params:
            params["text_config"] = params.pop("llm_config")

        # Allow converted configs to keep the upstream YAML names.
        for key, target in (
            ("frontend_conf", FrontendConfig),
            ("audio_encoder_conf", SenseVoiceEncoderConfig),
            ("audio_adaptor_conf", AdaptorConfig),
        ):
            if isinstance(params.get(key), dict):
                params[key] = target.from_dict(params[key])

        if isinstance(params.get("text_config"), dict):
            params["text_config"] = TextConfig.from_dict(params["text_config"])

        valid = _filter_kwargs(cls, params)
        return cls(**valid)
