# Copyright (c) 2025 Resemble AI
# MIT License
# MLX port of T3Config

from dataclasses import dataclass
from typing import Optional

# GPT2 Medium configuration for Turbo model
GPT2_MEDIUM_CONFIG = {
    "activation_function": "gelu_new",
    "n_ctx": 8196,
    "n_embd": 1024,
    "hidden_size": 1024,
    "n_head": 16,
    "n_layer": 24,
    "n_positions": 8196,
    "vocab_size": 50276,
    "layer_norm_epsilon": 1e-05,
    "attn_pdrop": 0.1,
    "embd_pdrop": 0.1,
    "resid_pdrop": 0.1,
}


@dataclass
class T3Config:
    """Configuration for T3 TTS model."""

    # Text tokens
    start_text_token: int = 255
    stop_text_token: int = 0
    text_tokens_dict_size: int = 50276
    max_text_tokens: int = 2048

    # Speech tokens
    start_speech_token: int = 6561
    stop_speech_token: int = 6562
    speech_tokens_dict_size: int = 6563
    max_speech_tokens: int = 4096

    # Model architecture
    llama_config_name: str = "GPT2_medium"
    input_pos_emb: Optional[str] = None
    speech_cond_prompt_len: int = 375

    # Conditioning
    encoder_type: str = "voice_encoder"
    speaker_embed_size: int = 256
    use_perceiver_resampler: bool = False
    emotion_adv: bool = False

    @property
    def n_channels(self) -> int:
        """Get hidden size from config."""
        return GPT2_MEDIUM_CONFIG["hidden_size"]

    @property
    def is_multilingual(self) -> bool:
        """Check if this is a multilingual model."""
        return self.text_tokens_dict_size == 2454

    @classmethod
    def turbo(cls) -> "T3Config":
        """Create configuration for Turbo TTS model."""
        return cls(
            text_tokens_dict_size=50276,
            speech_tokens_dict_size=6563,
            llama_config_name="GPT2_medium",
            input_pos_emb=None,
            speech_cond_prompt_len=375,
            use_perceiver_resampler=False,
            emotion_adv=False,
        )
