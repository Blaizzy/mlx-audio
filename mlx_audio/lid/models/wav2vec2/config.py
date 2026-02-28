from dataclasses import dataclass
from typing import Dict, Optional

from mlx_audio.stt.models.wav2vec.wav2vec import ModelConfig as Wav2Vec2ModelConfig


@dataclass
class ModelConfig(Wav2Vec2ModelConfig):
    """Wav2Vec2 config extended with classification head fields for LID."""

    classifier_proj_size: int = 256
    num_labels: int = 2
    id2label: Optional[Dict[str, str]] = None

    def __post_init__(self):
        if self.id2label is not None:
            self.num_labels = len(self.id2label)
