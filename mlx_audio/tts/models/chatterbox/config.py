from dataclasses import dataclass
from typing import Any, Dict

from ..base import BaseModelArgs

LLAMA_520M_CONFIG = {
    "model_type": "llama",
    "vocab_size": 4000,
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
    text_tokens_dict_size: int = 704
    start_text_token: int = 255
    stop_text_token: int = 0
    max_text_tokens: int = 2048
    speech_tokens_dict_size: int = 8194
    start_speech_token: int = 6561
    stop_speech_token: int = 6562
    max_speech_tokens: int = 4096
    llama_config_name: str = "Llama_520M"
    input_pos_emb: str = "learned"
    speech_cond_prompt_len: int = 150
    encoder_type: str = "voice_encoder"
    speaker_embed_size: int = 256
    use_perceiver_resampler: bool = True
    emotion_adv: bool = True

    @property
    def n_channels(self) -> int:
        return LLAMA_CONFIGS[self.llama_config_name]["hidden_size"]

    @property
    def is_multilingual(self) -> bool:
        return self.text_tokens_dict_size == 2454

    @classmethod
    def english_only(cls) -> "T3Config":
        return cls(text_tokens_dict_size=704)

    @classmethod
    def multilingual(cls) -> "T3Config":
        return cls(text_tokens_dict_size=2454)


@dataclass
class ModelConfig(BaseModelArgs):
    model_type: str = "chatterbox"
    t3_config: T3Config = None
    s3_sr: int = 16000  # S3 tokenizer sample rate
    s3gen_sr: int = 24000  # S3Gen output sample rate
    sample_rate: int = 24000  # Output sample rate (alias for s3gen_sr)

    enc_cond_len: int = 6 * 16000  # 6 seconds at 16kHz
    dec_cond_len: int = 10 * 24000  # 10 seconds at 24kHz

    model_path: str = None

    def __post_init__(self):
        if self.t3_config is None:
            self.t3_config = T3Config.english_only()

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "ModelConfig":
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
