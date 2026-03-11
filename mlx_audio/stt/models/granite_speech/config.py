import inspect
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class EncoderConfig:
    input_dim: int = 160
    num_layers: int = 10
    hidden_dim: int = 1024
    feedforward_mult: int = 4
    num_heads: int = 8
    dim_head: int = 128
    output_dim: int = 42
    context_size: int = 200
    max_pos_emb: int = 512
    dropout: float = 0.1
    conv_kernel_size: int = 15
    conv_expansion_factor: int = 2
    model_type: str = "granite_speech_encoder"

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
class ProjectorConfig:
    hidden_size: int = 1024
    num_hidden_layers: int = 2
    num_attention_heads: int = 16
    intermediate_size: int = 4096
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-12
    encoder_hidden_size: int = 1024
    cross_attention_frequency: int = 1
    model_type: str = "blip_2_qformer"

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
    model_type: str = "granite"
    vocab_size: int = 100353
    hidden_size: int = 2048
    intermediate_size: int = 4096
    num_hidden_layers: int = 40
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    hidden_act: str = "silu"
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict] = None
    attention_bias: bool = False
    mlp_bias: bool = False
    attention_multiplier: float = 0.0078125
    embedding_multiplier: float = 12.0
    residual_multiplier: float = 0.22
    logits_scaling: float = 8.0
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
    model_type: str = "granite_speech"
    encoder_config: EncoderConfig = None
    projector_config: ProjectorConfig = None
    text_config: TextConfig = None
    audio_token_index: int = 100352
    downsample_rate: int = 5
    window_size: int = 15
    has_lora_adapter: bool = False

    def __post_init__(self):
        if isinstance(self.encoder_config, dict):
            self.encoder_config = EncoderConfig.from_dict(self.encoder_config)
        elif self.encoder_config is None:
            self.encoder_config = EncoderConfig()

        if isinstance(self.projector_config, dict):
            self.projector_config = ProjectorConfig.from_dict(self.projector_config)
        elif self.projector_config is None:
            self.projector_config = ProjectorConfig()

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
