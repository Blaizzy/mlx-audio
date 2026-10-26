from dataclasses import dataclass

from mlx_audio.stt.models.ming_v3_asr.config import ModelConfig as ASRConfig


@dataclass
class ModelConfig(ASRConfig):
    model_type: str = "ming_v3_audio"
    talker_path: str | None = None
