from dataclasses import dataclass
from typing import Any, Dict

from ..base import BaseModelArgs

LLAMA_520M_CONFIG = {
    "model_type": "llama",
    "vocab_size": 8,  # Unused due to custom input layers
    "hidden_size": 1024,
    "num_hidden_layers": 30,
    "intermediate_size": 4096,
    "num_attention_heads": 16,
    "num_key_value_heads": 16,
    "head_dim": 64,
    "max_position_embeddings": 131072,
    "rms_norm_eps": 1e-05,
    "rope_theta": 500000.0,
    "rope_scaling": {
        "factor": 8.0,
        "high_freq_factor": 4.0,
        "low_freq_factor": 1.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3",
    },
    "attention_bias": False,
    "mlp_bias": False,
    "tie_word_embeddings": False,
}

LLAMA_CONFIGS = {
    "Llama_520M": LLAMA_520M_CONFIG,
}


@dataclass
class T3Config:
    """Configuration for T3 (Token-to-Token) model."""

    # Text token configuration
    text_tokens_dict_size: int = 704  # English: 704, Multilingual: 2454
    start_text_token: int = 255
    stop_text_token: int = 0
    max_text_tokens: int = 2048

    # Speech token configuration
    speech_tokens_dict_size: int = 8194
    start_speech_token: int = 6561
    stop_speech_token: int = 6562
    max_speech_tokens: int = 4096

    # Model architecture
    llama_config_name: str = "Llama_520M"
    input_pos_emb: str = "learned"  # "learned" or "rope"
    speech_cond_prompt_len: int = 150

    # Conditioning
    encoder_type: str = "voice_encoder"
    speaker_embed_size: int = 256
    use_perceiver_resampler: bool = True
    emotion_adv: bool = True

    @property
    def n_channels(self) -> int:
        """Get hidden size from LLaMA config."""
        return LLAMA_CONFIGS[self.llama_config_name]["hidden_size"]

    @property
    def is_multilingual(self) -> bool:
        """Check if model is multilingual."""
        return self.text_tokens_dict_size == 2454

    @classmethod
    def english_only(cls) -> "T3Config":
        """Create configuration for English-only TTS model."""
        return cls(text_tokens_dict_size=704)

    @classmethod
    def multilingual(cls) -> "T3Config":
        """Create configuration for multilingual TTS model."""
        return cls(text_tokens_dict_size=2454)


@dataclass
class ModelConfig(BaseModelArgs):
    """Main configuration for Chatterbox TTS model."""

    # Model type for auto-detection
    model_type: str = "chatterbox"

    # Model components
    t3_config: T3Config = None

    # Sample rates
    s3_sr: int = 16000  # S3 tokenizer sample rate
    s3gen_sr: int = 24000  # S3Gen output sample rate
    sample_rate: int = 24000  # Output sample rate (alias for s3gen_sr)

    # Conditioning lengths
    enc_cond_len: int = 6 * 16000  # 6 seconds at 16kHz
    dec_cond_len: int = 10 * 24000  # 10 seconds at 24kHz

    # Model path (set by load_model for tokenizer initialization)
    model_path: str = None

    def __post_init__(self):
        if self.t3_config is None:
            self.t3_config = T3Config.english_only()

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "ModelConfig":
        """Create config from dictionary."""
        t3_config = None
        if "t3_config" in config:
            t3_config = T3Config(**config["t3_config"])

        return cls(
            model_type=config.get("model_type", "chatterbox"),
            t3_config=t3_config,
            s3_sr=config.get("s3_sr", 16000),
            s3gen_sr=config.get("s3gen_sr", 24000),
            sample_rate=config.get("sample_rate", config.get("s3gen_sr", 24000)),
            enc_cond_len=config.get("enc_cond_len", 6 * 16000),
            dec_cond_len=config.get("dec_cond_len", 10 * 24000),
            model_path=config.get("model_path"),
        )
