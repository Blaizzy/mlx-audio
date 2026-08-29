from __future__ import annotations

from typing import Any, Dict

from mlx_audio.stt.models.qwen3_asr.config import ModelConfig as Qwen3ASRConfig


class ModelConfig(Qwen3ASRConfig):
    """Qwen3-ASR configuration with Phonon artifact metadata attached."""

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> "ModelConfig":
        config = super().from_dict(params)
        config.model_path = params.get("model_path")
        config.model_repo = params.get("model_repo")
        # The released Phonon-1 checkpoints are English-only even though their
        # embedded teacher config retains Qwen3-ASR's multilingual list.
        config.support_languages = ["English"]
        return config
