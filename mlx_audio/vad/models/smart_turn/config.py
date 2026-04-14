import inspect
from dataclasses import dataclass
from typing import Optional

from mlx_audio.base import BaseModelArgs


@dataclass
class EncoderConfig(BaseModelArgs):
    """Whisper encoder configuration for Smart Turn."""

    model_type: str = "smart_turn_encoder"
    num_mel_bins: int = 80
    max_source_positions: int = 400
    d_model: int = 384
    encoder_attention_heads: int = 6
    encoder_layers: int = 4
    encoder_ffn_dim: int = 1536
    k_proj_bias: bool = False


@dataclass
class ProcessorConfig(BaseModelArgs):
    """Audio preprocessing configuration for Smart Turn."""

    sampling_rate: int = 16000
    max_audio_seconds: int = 8
    n_fft: int = 400
    hop_length: int = 160
    n_mels: int = 80
    normalize_audio: bool = True
    threshold: float = 0.5


@dataclass
class ModelConfig(BaseModelArgs):
    """Smart Turn endpoint detector configuration."""

    model_type: str = "smart_turn"
    architecture: str = "smart_turn"
    dtype: str = "float32"
    encoder_config: Optional[EncoderConfig] = None
    processor_config: Optional[ProcessorConfig] = None

    # Compatibility with conversion script stubs
    sample_rate: int = 16000
    max_audio_seconds: int = 8
    threshold: float = 0.5

    def __post_init__(self):
        if isinstance(self.encoder_config, dict):
            self.encoder_config = EncoderConfig.from_dict(self.encoder_config)
        if self.encoder_config is None:
            self.encoder_config = EncoderConfig()

        if isinstance(self.processor_config, dict):
            self.processor_config = ProcessorConfig.from_dict(self.processor_config)
        if self.processor_config is None:
            self.processor_config = ProcessorConfig(
                sampling_rate=self.sample_rate,
                max_audio_seconds=self.max_audio_seconds,
                threshold=self.threshold,
                n_mels=self.encoder_config.num_mel_bins,
            )

    @classmethod
    def from_dict(cls, params):
        params = params.copy()
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )
