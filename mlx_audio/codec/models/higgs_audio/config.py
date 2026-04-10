from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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
    # Semantic / encode path
    semantic_sample_rate: int = 16000
    semantic_model_config: Optional[Dict[str, Any]] = None
    strides: List[int] = field(default_factory=lambda: [1, 1])
    block_dilations: List[int] = field(default_factory=lambda: [1, 1])
    channel_ratios: List[int] = field(default_factory=lambda: [1, 1])
    kernel_size: int = 3
    unit_kernel_size: int = 3
    # HuBERT conv downsample factor (product of HuBERT conv strides)
    hubert_downsample_factor: int = 320

    @property
    def tokens_per_second(self) -> float:
        return self.sample_rate / self.downsample_factor

    @property
    def semantic_downsample_factor(self) -> int:
        """Factor to stride-slice HuBERT output to match acoustic frame rate.

        HuBERT produces ~50fps at 16kHz/320. Acoustic produces ~25fps at 24kHz/960.
        This factor = 2 downsamples semantic from 50fps to 25fps.
        """
        sr_ratio = self.sample_rate / self.semantic_sample_rate
        return int(self.downsample_factor / sr_ratio / self.hubert_downsample_factor)
