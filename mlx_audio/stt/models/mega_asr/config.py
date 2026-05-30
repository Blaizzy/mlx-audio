from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from mlx_audio.stt.models.qwen3_asr.config import AudioEncoderConfig
from mlx_audio.stt.models.qwen3_asr.config import ModelConfig as Qwen3ModelConfig
from mlx_audio.stt.models.qwen3_asr.config import TextConfig


@dataclass
class MegaASRConfig:
    model_type: str = "mega_asr"
    model_repo: Optional[str] = None

    audio_config: Optional[AudioEncoderConfig] = None
    text_config: Optional[TextConfig] = None

    router_config: Optional[Dict[str, Any]] = field(default_factory=dict)
    lora_config: Optional[Dict[str, Any]] = field(default_factory=dict)

    router_weights: str = "extras/router.safetensors"
    lora_weights: str = "extras/lora.safetensors"

    audio_token_id: int = 151676
    audio_start_token_id: int = 151669
    audio_end_token_id: int = 151670

    support_languages: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.audio_config is None:
            self.audio_config = AudioEncoderConfig()
        elif isinstance(self.audio_config, dict):
            self.audio_config = AudioEncoderConfig.from_dict(self.audio_config)

        if self.text_config is None:
            self.text_config = TextConfig()
        elif isinstance(self.text_config, dict):
            self.text_config = TextConfig.from_dict(self.text_config)

    def to_qwen3_config(self) -> Qwen3ModelConfig:
        assert self.audio_config is not None
        assert self.text_config is not None
        return Qwen3ModelConfig(
            audio_config=self.audio_config,
            text_config=self.text_config,
            model_type="qwen3_asr",
            model_repo=self.model_repo or "",
            audio_token_id=self.audio_token_id,
            audio_start_token_id=self.audio_start_token_id,
            audio_end_token_id=self.audio_end_token_id,
            support_languages=self.support_languages,
        )

    def to_qwen3_dict(self) -> Dict[str, Any]:
        return {
            "model_type": "qwen3_asr",
            "model_repo": self.model_repo,
            "audio_token_id": self.audio_token_id,
            "audio_start_token_id": self.audio_start_token_id,
            "audio_end_token_id": self.audio_end_token_id,
            "support_languages": self.support_languages,
            "audio_config": _config_to_dict(self.audio_config),
            "text_config": _config_to_dict(self.text_config),
        }

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> "MegaASRConfig":
        params = params.copy()

        if "thinker_config" in params:
            thinker = params.pop("thinker_config")
            for key in (
                "audio_config",
                "text_config",
                "audio_token_id",
                "audio_start_token_id",
                "audio_end_token_id",
            ):
                if key in thinker:
                    params[key] = thinker[key]

        if "audio_config" in params and isinstance(params["audio_config"], dict):
            params["audio_config"] = AudioEncoderConfig.from_dict(
                params["audio_config"]
            )
        elif "audio_config" not in params:
            params["audio_config"] = AudioEncoderConfig()

        if "text_config" in params and isinstance(params["text_config"], dict):
            params["text_config"] = TextConfig.from_dict(params["text_config"])
        elif "text_config" not in params:
            params["text_config"] = TextConfig()

        router_cfg: Dict[str, Any] = {}
        for key in (
            "d_model",
            "num_layers",
            "nhead",
            "dim_feedforward",
            "frontend_hidden_dim",
            "classifier_hidden_dim",
            "max_len",
        ):
            if key in params:
                router_cfg[key] = params.pop(key)
        if "router_config" in params and isinstance(params["router_config"], dict):
            router_cfg.update(params.pop("router_config"))
        params["router_config"] = router_cfg or None

        lora_cfg: Dict[str, Any] = {}
        for key in ("r", "lora_alpha", "rank_pattern", "alpha_pattern"):
            if key in params:
                lora_cfg[key] = params.pop(key)
        if "lora_config" in params and isinstance(params["lora_config"], dict):
            lora_cfg.update(params.pop("lora_config"))
        params["lora_config"] = lora_cfg or None

        known = inspect.signature(cls).parameters
        known_params = {k: v for k, v in params.items() if k in known}
        extra_params = {k: v for k, v in params.items() if k not in known}
        instance = cls(**known_params)
        for k, v in extra_params.items():
            setattr(instance, k, v)
        return instance


def _config_to_dict(cfg: Any) -> Dict[str, Any]:
    if cfg is None:
        return {}
    if hasattr(cfg, "__dict__"):
        return {k: v for k, v in cfg.__dict__.items() if not k.startswith("_")}
    return dict(cfg) if isinstance(cfg, dict) else {}
