from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ModelConfig:
    """Configuration for ECAPA-TDNN language identification model.

    Matches the SpeechBrain ``speechbrain/lang-id-voxlingua107-ecapa`` format
    with defaults for VoxLingua107 (107 languages).
    """

    n_mels: int = 60
    channels: int = 1024
    kernel_sizes: List[int] = field(default_factory=lambda: [5, 3, 3, 3, 1])
    dilations: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 1])
    attention_channels: int = 128
    res2net_scale: int = 8
    se_channels: int = 128
    embedding_dim: int = 256
    classifier_hidden_dim: int = 512
    num_classes: int = 107
    id2label: Optional[Dict[str, str]] = None

    def __post_init__(self):
        if self.id2label is not None:
            self.num_classes = len(self.id2label)
