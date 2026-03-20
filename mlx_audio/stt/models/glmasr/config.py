import inspect
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class WhisperConfig:
    """Configuration for the Whisper audio encoder."""

    model_type: str = "whisper"
    activation_function: str = "gelu"
    d_model: int = 1280
    encoder_attention_heads: int = 20
    encoder_ffn_dim: int = 5120
    encoder_layers: int = 32
    encoder_layerdrop: float = 0.0
    num_mel_bins: int = 128
    max_source_positions: int = 1500
    dropout: float = 0.0
    attention_dropout: float = 0.0
    activation_dropout: float = 0.0
    init_std: float = 0.02
    scale_embedding: bool = False
    rope_traditional: bool = True

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> "WhisperConfig":
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


@dataclass
class LlamaConfig:
    """Configuration for the LLaMA language model."""

    model_type: str = "llama"
    vocab_size: int = 59264
    hidden_size: int = 2048
    intermediate_size: int = 6144
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    hidden_act: str = "silu"
    head_dim: int = None
    max_position_embeddings: int = 8192
    layer_types: List[str] = None
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    sliding_window: Optional[int] = None
    rope_traditional: bool = False
    rope_scaling: Optional[Dict[str, Any]] = None
    rope_theta: float = 10000.0
    rope_dim: int = 128
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    mlp_bias: bool = False
    pad_token_id: int = 59260
    eos_token_id: List[int] = field(default_factory=lambda: [59246, 59253, 59255])

    def __post_init__(self):
        if self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> "LlamaConfig":
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


@dataclass
class ModelConfig:
    """Configuration for the GLM-ASR model."""

    model_type: str = "glmasr"
    whisper_config: WhisperConfig = None
    lm_config: LlamaConfig = None

    # Adapter configuration
    adapter_type: str = "mlp"
    merge_factor: int = 4
    mlp_adapter_act: str = "gelu"

    # Audio processing
    use_rope: bool = True
    max_whisper_length: int = 1500
    max_length: int = 65536

    sample_rate: int = 16000

    def __post_init__(self):
        if self.whisper_config is None:
            self.whisper_config = WhisperConfig()
        elif isinstance(self.whisper_config, dict):
            self.whisper_config = WhisperConfig.from_dict(self.whisper_config)

        if self.lm_config is None:
            self.lm_config = LlamaConfig()
        elif isinstance(self.lm_config, dict):
            self.lm_config = LlamaConfig.from_dict(self.lm_config)

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> "ModelConfig":
        # Handle nested configs
        whisper_config = params.pop("whisper_config", None)
        lm_config = params.pop("lm_config", None)

        if whisper_config is not None:
            if isinstance(whisper_config, dict):
                whisper_config = WhisperConfig.from_dict(whisper_config)
        else:
            whisper_config = WhisperConfig()

        if lm_config is not None:
            if isinstance(lm_config, dict):
                lm_config = LlamaConfig.from_dict(lm_config)
        else:
            lm_config = LlamaConfig()

        filtered_params = {
            k: v for k, v in params.items() if k in inspect.signature(cls).parameters
        }

        return cls(
            whisper_config=whisper_config,
            lm_config=lm_config,
            **filtered_params,
        )
