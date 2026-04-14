from dataclasses import dataclass, field
from typing import Dict, List, Optional

from mlx_audio.tts.models.base import BaseModelArgs


@dataclass
class OmniVoiceConfig(BaseModelArgs):
    model_type: str = "omnivoice"
    audio_vocab_size: int = 1025  # 1024 real tokens + 1 mask token
    audio_mask_id: int = 1024
    num_audio_codebook: int = 8
    audio_codebook_weights: List[int] = field(
        default_factory=lambda: [8, 8, 6, 6, 4, 4, 2, 2]
    )
    sample_rate: int = 24000
    llm_config: Optional[Dict] = None  # raw Qwen3 config dict — parsed lazily
