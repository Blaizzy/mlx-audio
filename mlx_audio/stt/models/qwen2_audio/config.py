import inspect
from dataclasses import dataclass
from typing import Optional


@dataclass
class EncoderConfig:
    d_model: int = 1280
    encoder_layers: int = 32
    encoder_attention_heads: int = 20
    encoder_ffn_dim: int = 5120
    num_mel_bins: int = 128
    max_source_positions: int = 1500
    activation_function: str = "gelu"
    scale_embedding: bool = False
    dropout: float = 0.0
    attention_dropout: float = 0.0
    model_type: str = "qwen2_audio_encoder"

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


@dataclass
class TextConfig:
    model_type: str = "qwen2"
    vocab_size: int = 156032
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    hidden_act: str = "silu"
    max_position_embeddings: int = 8192
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    attention_bias: bool = True
    tie_word_embeddings: bool = False

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


@dataclass
class ModelConfig:
    model_type: str = "qwen2_audio"
    audio_config: EncoderConfig = None
    text_config: TextConfig = None
    audio_token_id: int = 151646

    def __post_init__(self):
        if isinstance(self.audio_config, dict):
            self.audio_config = EncoderConfig.from_dict(self.audio_config)
        elif self.audio_config is None:
            self.audio_config = EncoderConfig()

        if isinstance(self.text_config, dict):
            self.text_config = TextConfig.from_dict(self.text_config)
        elif self.text_config is None:
            self.text_config = TextConfig()

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )
