from dataclasses import dataclass
from typing import Any, Dict, Optional

from mlx_audio.base import BaseModelArgs


@dataclass
class AudioConfig(BaseModelArgs):
    """Whisper encoder configuration used by MOSS-Transcribe-Diarize."""

    model_type: str = "whisper"
    num_mel_bins: int = 80
    d_model: int = 1024
    encoder_layers: int = 24
    encoder_attention_heads: int = 16
    encoder_ffn_dim: int = 4096
    max_source_positions: int = 1500
    dropout: float = 0.0
    attention_dropout: float = 0.0
    activation_dropout: float = 0.0
    activation_function: str = "gelu"
    encoder_layerdrop: float = 0.0
    scale_embedding: bool = False
    initializer_range: float = 0.02
    rope_traditional: bool = True


@dataclass
class TextConfig(BaseModelArgs):
    """Qwen3 text decoder configuration used by MOSS-Transcribe-Diarize."""

    model_type: str = "qwen3"
    vocab_size: int = 151936
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 40960
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    tie_word_embeddings: bool = True
    rope_theta: float = 1000000.0
    rope_scaling: Optional[Dict[str, Any]] = None
    attention_bias: bool = False
    attention_dropout: float = 0.0


@dataclass
class ModelConfig(BaseModelArgs):
    """Configuration for MOSS-Transcribe-Diarize."""

    model_type: str = "moss_transcribe_diarize"
    text_config: TextConfig = None
    audio_config: AudioConfig = None
    audio_token_id: int = 151671
    audio_merge_size: int = 4
    adaptor_input_dim: Optional[int] = None
    tie_word_embeddings: bool = True
    sample_rate: int = 16000

    def __post_init__(self):
        if self.audio_config is None:
            self.audio_config = AudioConfig()
        elif isinstance(self.audio_config, dict):
            self.audio_config = AudioConfig.from_dict(self.audio_config)

        if self.text_config is None:
            self.text_config = TextConfig()
        elif isinstance(self.text_config, dict):
            self.text_config = TextConfig.from_dict(self.text_config)

        self.text_config.tie_word_embeddings = self.tie_word_embeddings
        if self.adaptor_input_dim is None:
            self.adaptor_input_dim = self.audio_config.d_model * self.audio_merge_size
