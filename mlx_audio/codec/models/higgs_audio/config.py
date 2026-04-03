from dataclasses import dataclass, field
from typing import List, Optional

from mlx_audio.tts.models.base import BaseModelArgs


@dataclass
class HiggsAudioConfig(BaseModelArgs):
    model_type: str = "higgs_audio_v2_tokenizer"
    sample_rate: int = 24000
    codebook_size: int = 1024
    codebook_dim: int = 64
    downsample_factor: int = 960  # audio frames per token at sample_rate (8*5*4*2*3)
    # DAC acoustic sub-model
    dac_sample_rate: int = 24000
    dac_num_codebooks: int = 8
    dac_encoder_ratios: List[int] = field(default_factory=lambda: [8, 5, 4, 2, 3])
    dac_encoder_hidden: int = 64
    dac_decoder_hidden: int = 1024

    @property
    def tokens_per_second(self) -> float:
        return self.sample_rate / self.downsample_factor
