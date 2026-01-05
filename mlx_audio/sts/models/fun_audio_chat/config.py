# Copyright (c) 2025 Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Qwen3Config:
    """Configuration for Qwen3 language model."""

    architectures: List[str] = field(default_factory=lambda: ["Qwen3ForCausalLM"])
    attention_bias: bool = False
    attention_dropout: float = 0.0
    bos_token_id: int = 151643
    eos_token_id: int = 151643
    head_dim: int = 128
    hidden_act: str = "silu"
    hidden_size: int = 4096
    initializer_range: float = 0.02
    intermediate_size: int = 12288
    max_position_embeddings: int = 262144
    max_window_layers: int = 28
    model_type: str = "qwen3"
    num_attention_heads: int = 32
    num_hidden_layers: int = 36
    num_key_value_heads: int = 8
    rms_norm_eps: float = 1e-06
    rope_scaling: Optional[Dict[str, Any]] = None
    rope_theta: float = 5000000
    sliding_window: Optional[int] = None
    tie_word_embeddings: bool = False
    use_cache: bool = False
    use_sliding_window: bool = False
    vocab_size: int = 151936
    # Audio-specific tokens
    audio_bos_index: int = 151670
    audio_eos_index: int = 151671
    sil_index: int = 151673

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Qwen3Config":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})


@dataclass
class CRQTransformerConfig:
    """Configuration for the CRQ (Continuous Residual Quantization) transformer.

    This is a smaller Qwen3 model used for speech decoding/inversion.
    """

    architectures: List[str] = field(default_factory=lambda: ["Qwen3ForCausalLM"])
    attention_bias: bool = False
    attention_dropout: float = 0.0
    bos_token_id: int = 151643
    eos_token_id: int = 151643
    head_dim: int = 128
    hidden_act: str = "silu"
    hidden_size: int = 1024
    initializer_range: float = 0.02
    intermediate_size: int = 3072
    max_position_embeddings: int = 32768
    max_window_layers: int = 28
    model_type: str = "qwen3"
    num_attention_heads: int = 16
    num_hidden_layers: int = 28
    num_key_value_heads: int = 8
    rms_norm_eps: float = 1e-06
    rope_scaling: Optional[Dict[str, Any]] = None
    rope_theta: float = 1000000
    sliding_window: Optional[int] = None
    tie_word_embeddings: bool = True
    use_cache: bool = True
    use_sliding_window: bool = False
    vocab_size: int = 151936

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "CRQTransformerConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})


@dataclass
class FunAudioChatAudioEncoderConfig:
    """Configuration for the FunAudioChat audio encoder.

    This encoder processes mel-spectrogram features using a transformer architecture.
    """

    activation_dropout: float = 0.0
    activation_function: str = "gelu"
    attention_dropout: float = 0.0
    bos_token_id: int = 6561
    codebook_size: int = 6565
    continuous_features_mode: str = "replace"
    d_model: int = 1280
    dropout: float = 0.0
    enable_audio_invert_tower: bool = True
    encoder_attention_heads: int = 20
    encoder_ffn_dim: int = 5120
    encoder_layers: int = 32
    eos_token_id: int = 6562
    group_size: int = 5
    initializer_range: float = 0.02
    max_source_positions: int = 1500
    model_type: str = "funaudiochat_audio_encoder"
    n_window: int = 100
    num_hidden_layers: int = 32
    num_mel_bins: int = 128
    output_dim: int = 4096
    pad_token_id: int = 6563
    scale_embedding: bool = False
    crq_transformer_config: Optional[CRQTransformerConfig] = None

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "FunAudioChatAudioEncoderConfig":
        """Create config from dictionary."""
        crq_config = None
        if "crq_transformer_config" in config_dict and config_dict["crq_transformer_config"]:
            crq_config = CRQTransformerConfig.from_dict(config_dict["crq_transformer_config"])

        filtered = {k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__ and k != "crq_transformer_config"}
        return cls(crq_transformer_config=crq_config, **filtered)


@dataclass
class FunAudioChatConfig:
    """Main configuration for FunAudioChat model.

    Combines audio encoder and text (language model) configurations.
    """

    architectures: List[str] = field(default_factory=lambda: ["FunAudioChatForConditionalGeneration"])
    audio_config: FunAudioChatAudioEncoderConfig = field(default_factory=FunAudioChatAudioEncoderConfig)
    text_config: Qwen3Config = field(default_factory=Qwen3Config)
    audio_token_index: int = 151669
    ignore_index: int = -100
    model_type: str = "funaudiochat"
    use_cache: bool = False

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "FunAudioChatConfig":
        """Create config from dictionary."""
        audio_config = FunAudioChatAudioEncoderConfig.from_dict(
            config_dict.get("audio_config", {})
        )
        text_config = Qwen3Config.from_dict(config_dict.get("text_config", {}))

        return cls(
            architectures=config_dict.get("architectures", ["FunAudioChatForConditionalGeneration"]),
            audio_config=audio_config,
            text_config=text_config,
            audio_token_index=config_dict.get("audio_token_index", 151669),
            ignore_index=config_dict.get("ignore_index", -100),
            model_type=config_dict.get("model_type", "funaudiochat"),
            use_cache=config_dict.get("use_cache", False),
        )
