import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from mlx_audio.tts.models.base import BaseModelArgs


@dataclass
class HiggsAudioConfig(BaseModelArgs):
    model_type: str = "higgs_audio_v2_tokenizer"
    sample_rate: int = 24000
    codebook_size: int = 1024
    codebook_dim: int = 64
    # config.json stores HuBERT conv downsample factor here (320).
    # Acoustic hop is computed from dac_encoder_ratios instead.
    downsample_factor: int = 320
    # DAC acoustic sub-model
    dac_sample_rate: int = 24000
    dac_num_codebooks: int = 8
    dac_encoder_ratios: List[int] = field(default_factory=lambda: [8, 5, 4, 2, 3])
    dac_encoder_hidden: int = 64
    dac_decoder_hidden: int = 1024
    # Semantic / encode path
    semantic_sample_rate: int = 16000
    semantic_model_config: Optional[Dict[str, Any]] = None
    strides: List[int] = field(default_factory=lambda: [1, 1])
    block_dilations: List[int] = field(default_factory=lambda: [1, 1])
    channel_ratios: List[int] = field(default_factory=lambda: [1, 1])
    kernel_size: int = 3
    unit_kernel_size: int = 3

    @property
    def acoustic_hop(self) -> int:
        """Product of DAC encoder strides: 8*5*4*2*3 = 960."""
        return math.prod(self.dac_encoder_ratios)

    @property
    def tokens_per_second(self) -> float:
        return self.sample_rate / self.acoustic_hop

    @property
    def semantic_downsample_factor(self) -> int:
        """Factor to stride-slice HuBERT output to match acoustic frame rate.

        HuBERT at 16kHz/320 = 50fps. Acoustic at 24kHz/960 = 25fps.
        Factor = 50/25 = 2.
        """
        hubert_fps = self.semantic_sample_rate / self.downsample_factor
        acoustic_fps = self.sample_rate / self.acoustic_hop
        return max(1, round(hubert_fps / acoustic_fps))
