from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Union

from mlx_audio.tts.models.bailingmm.bailingmm import Model as BailingMMModel
from mlx_audio.tts.models.bailingmm.bailingmm import ModelConfig as BailingMMModelConfig


@dataclass
class ModelConfig(BailingMMModelConfig):
    model_type: str = "dense"

    @classmethod
    def from_dict(cls, config: dict) -> "ModelConfig":
        return cls(
            model_type="dense",
            text_config=config.get("llm_config", config.get("text_config")),
            audio_tokenizer_config=config.get("audio_tokenizer_config"),
            ditar_config=config.get("ditar_config"),
            aggregator_config=config.get("aggregator_config"),
            model_path=config.get("model_path"),
        )


class Model(BailingMMModel):
    def __init__(self, config: Union[ModelConfig, Dict[str, Any]]):
        if isinstance(config, dict):
            config = ModelConfig.from_dict(config)
        else:
            config.model_type = "dense"
        super().__init__(config)
        self.model_type = "dense"

    @staticmethod
    def _is_moe_llm_config(_llm_cfg: Dict[str, Any]) -> bool:
        # Dense variants should always build the Qwen2 backbone path.
        return False
