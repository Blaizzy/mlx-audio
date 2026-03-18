import inspect
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union


@dataclass
class EncoderConfig:
    model_type: str = "granite_speech_encoder"
    num_layers: int = 16
    hidden_dim: int = 1024
    input_dim: int = 160
    output_dim: int = 348
    num_heads: int = 8
    dim_head: int = 128
    feedforward_mult: int = 4
    conv_kernel_size: int = 15
    conv_expansion_factor: int = 2
    context_size: int = 200
    max_pos_emb: int = 512
    dropout: float = 0.1

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> "EncoderConfig":
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


@dataclass
class ProjectorConfig:
    model_type: str = "blip_2_qformer"
    num_hidden_layers: int = 2
    hidden_size: int = 1024
    num_attention_heads: int = 16
    intermediate_size: int = 4096
    hidden_act: str = "gelu"
    encoder_hidden_size: int = 1024
    cross_attention_frequency: int = 1
    layer_norm_eps: float = 1e-12
    max_position_embeddings: int = 2048
    vocab_size: int = 30522

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> "ProjectorConfig":
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
    hidden_size: int = 2048
    num_hidden_layers: int = 40
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    intermediate_size: int = 4096
    attention_multiplier: float = 0.0078125
    embedding_multiplier: float = 12.0
    logits_scaling: float = 8.0
    residual_multiplier: float = 0.22
    vocab_size: int = 100353
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    rms_norm_eps: float = 1e-5
    attention_bias: bool = False
    mlp_bias: bool = False
    max_position_embeddings: int = 4096
    bos_token_id: int = 100257
    eos_token_id: int = 100257
    pad_token_id: int = 100256
    tie_word_embeddings: bool = False

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> "TextConfig":
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
    audio_token_index: int = 100352
    downsample_rate: int = 5
    window_size: int = 15
    encoder_config: EncoderConfig = None
    projector_config: ProjectorConfig = None
    text_config: TextConfig = None

    def __post_init__(self):
        if self.encoder_config is None:
            self.encoder_config = EncoderConfig()
        elif isinstance(self.encoder_config, dict):
            self.encoder_config = EncoderConfig.from_dict(self.encoder_config)

        if self.projector_config is None:
            self.projector_config = ProjectorConfig()
        elif isinstance(self.projector_config, dict):
            self.projector_config = ProjectorConfig.from_dict(self.projector_config)

        if self.text_config is None:
            self.text_config = TextConfig()
        elif isinstance(self.text_config, dict):
            self.text_config = TextConfig.from_dict(self.text_config)

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> "ModelConfig":
        params = params.copy()

        encoder_config = params.pop("encoder_config", None)
        projector_config = params.pop("projector_config", None)
        text_config = params.pop("text_config", None)

        if encoder_config is not None and isinstance(encoder_config, dict):
            encoder_config = EncoderConfig.from_dict(encoder_config)
        if projector_config is not None and isinstance(projector_config, dict):
            projector_config = ProjectorConfig.from_dict(projector_config)
        if text_config is not None and isinstance(text_config, dict):
            text_config = TextConfig.from_dict(text_config)

        filtered = {
            k: v for k, v in params.items() if k in inspect.signature(cls).parameters
        }

        return cls(
            encoder_config=encoder_config,
            projector_config=projector_config,
            text_config=text_config,
            **filtered,
        )
