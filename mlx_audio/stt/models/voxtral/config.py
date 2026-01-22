import inspect
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class AudioConfig:
    hidden_size: int = 1280
    num_hidden_layers: int = 32
    intermediate_size: int = 5120
    num_attention_heads: int = 20
    num_key_value_heads: int = 20
    rms_norm_eps: float = 1e-5
    head_dim: int = 64
    rope_theta: float = 1000000.0
    vocab_size: int = 51866
    num_mel_bins: int = 128
    encoder_layers: int = 32
    encoder_attention_heads: int = 20
    encoder_ffn_dim: int = 5120
    encoder_layerdrop: float = 0.0
    d_model: int = 1280
    dropout: float = 0.0
    attention_dropout: float = 0.0
    activation_function: str = "gelu"
    activation_dropout: float = 0.0
    scale_embedding: bool = False
    initializer_range: float = 0.02
    max_source_positions: int = 1500

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
    model_type: str = "llama"
    vocab_size: int = 131072
    max_position_embeddings: int = 131072
    hidden_size: int = 3072
    intermediate_size: int = 8192
    num_hidden_layers: int = 30
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    hidden_act: str = "silu"
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    rope_scaling: Optional[Dict[str, Any]] = None
    attention_bias: bool = False
    attention_dropout: float = 0.0
    mlp_bias: bool = False
    head_dim: int = 128
    tie_word_embeddings: bool = False
    bos_token_id: int = 1
    eos_token_id: int = 2
    sliding_window: Optional[int] = None
    rope_traditional: bool = False
    rope_scaling: Optional[Dict[str, Any]] = None
    rope_theta: float = 100000000.0
    layer_types: List[str] = None

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )

    def __post_init__(self):
        if self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers


@dataclass
class ModelConfig:
    audio_config: AudioConfig
    text_config: TextConfig
    model_repo: str = None
    model_type: str = "voxtral"
    audio_token_id: int = 24
    projector_hidden_act: str = "gelu"
    vocab_size: int = 131072
    hidden_size: int = 3072

    def __post_init__(self):
        if isinstance(self.audio_config, dict):
            self.audio_config = AudioConfig.from_dict(self.audio_config)
        if isinstance(self.text_config, dict):
            self.text_config = TextConfig.from_dict(self.text_config)

        # Update vocab_size and hidden_size from text_config
        self.vocab_size = self.text_config.vocab_size
        self.hidden_size = self.text_config.hidden_size

    @classmethod
    def from_dict(cls, params):
        params = params.copy()
        # Handle nested configs
        if "audio_config" in params and isinstance(params["audio_config"], dict):
            params["audio_config"] = AudioConfig.from_dict(params["audio_config"])
        elif "audio_config" not in params:
            params["audio_config"] = AudioConfig()

        if "text_config" in params and isinstance(params["text_config"], dict):
            params["text_config"] = TextConfig.from_dict(params["text_config"])
        elif "text_config" not in params:
            params["text_config"] = TextConfig()

        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )
