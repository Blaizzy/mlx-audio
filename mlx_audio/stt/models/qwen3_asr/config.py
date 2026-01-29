import inspect
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AudioEncoderConfig:
    """Configuration for Qwen3-ASR audio encoder."""

    num_mel_bins: int = 128
    encoder_layers: int = 24
    encoder_attention_heads: int = 16
    encoder_ffn_dim: int = 4096
    d_model: int = 1024
    dropout: float = 0.0
    attention_dropout: float = 0.0
    activation_function: str = "gelu"
    activation_dropout: float = 0.0
    scale_embedding: bool = False
    initializer_range: float = 0.02
    max_source_positions: int = 1500
    n_window: int = 50
    output_dim: int = 2048
    n_window_infer: int = 800
    conv_chunksize: int = 500
    downsample_hidden_size: int = 480

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> "AudioEncoderConfig":
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


@dataclass
class TextConfig:
    """Configuration for Qwen3-ASR text decoder (Qwen3-based)."""

    model_type: str = "qwen3"
    vocab_size: int = 151936
    hidden_size: int = 2048
    intermediate_size: int = 6144
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 65536
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    tie_word_embeddings: bool = True
    rope_theta: float = 1000000.0
    rope_scaling: Optional[Dict[str, Any]] = None
    attention_bias: bool = False
    attention_dropout: float = 0.0

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
    """Configuration for Qwen3-ASR model."""

    audio_config: AudioEncoderConfig = None
    text_config: TextConfig = None
    model_type: str = "qwen3_asr"
    model_repo: str = None
    audio_token_id: int = 151676
    audio_start_token_id: int = 151669
    audio_end_token_id: int = 151670
    support_languages: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.audio_config is None:
            self.audio_config = AudioEncoderConfig()
        elif isinstance(self.audio_config, dict):
            self.audio_config = AudioEncoderConfig.from_dict(self.audio_config)

        if self.text_config is None:
            self.text_config = TextConfig()
        elif isinstance(self.text_config, dict):
            self.text_config = TextConfig.from_dict(self.text_config)

    @classmethod
    def from_dict(cls, params: Dict[str, Any]):
        """Create config from dict, returning ForcedAlignerConfig if appropriate."""
        # Check if this is a forced aligner config
        if "thinker_config" in params:
            thinker = params.get("thinker_config", {})
            if thinker.get("model_type") == "qwen3_forced_aligner":
                from .qwen3_forced_aligner import ForcedAlignerConfig

                return ForcedAlignerConfig.from_dict(params)

        params = params.copy()

        # Handle nested thinker_config (from HF config)
        if "thinker_config" in params:
            thinker = params.pop("thinker_config")
            if "audio_config" in thinker:
                params["audio_config"] = thinker["audio_config"]
            if "text_config" in thinker:
                params["text_config"] = thinker["text_config"]
            if "audio_token_id" in thinker:
                params["audio_token_id"] = thinker["audio_token_id"]
            if "audio_start_token_id" in thinker:
                params["audio_start_token_id"] = thinker["audio_start_token_id"]
            if "audio_end_token_id" in thinker:
                params["audio_end_token_id"] = thinker["audio_end_token_id"]

        # Handle nested configs
        if "audio_config" in params and isinstance(params["audio_config"], dict):
            params["audio_config"] = AudioEncoderConfig.from_dict(
                params["audio_config"]
            )
        elif "audio_config" not in params:
            params["audio_config"] = AudioEncoderConfig()

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
