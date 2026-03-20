from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from mlx_audio.tts.models.base import BaseModelArgs


@dataclass
class FishTextConfig(BaseModelArgs):
    model_type: str = "fish_qwen3"
    vocab_size: int = 155776
    n_layer: int = 36
    n_head: int = 32
    dim: int = 2560
    intermediate_size: int = 9728
    n_local_heads: int = 8
    head_dim: int = 128
    rope_base: float = 1000000.0
    norm_eps: float = 1e-6
    max_seq_len: int = 32768
    dropout: float = 0.0
    tie_word_embeddings: bool = True
    attention_qkv_bias: bool = False
    attention_o_bias: bool = False
    attention_qk_norm: bool = True
    use_gradient_checkpointing: bool = False
    initializer_range: float = 0.01976423537605237


@dataclass
class FishAudioDecoderConfig(BaseModelArgs):
    model_type: str = "fish_qwen3_audio_decoder"
    vocab_size: int = 4096
    n_layer: int = 4
    n_head: int = 32
    dim: int = 2560
    intermediate_size: int = 9728
    n_local_heads: int = 8
    head_dim: int = 128
    rope_base: float = 1000000.0
    norm_eps: float = 1e-6
    max_seq_len: int = 11
    dropout: float = 0.0
    tie_word_embeddings: bool = False
    attention_qkv_bias: bool = False
    attention_o_bias: bool = False
    attention_qk_norm: bool = False
    use_gradient_checkpointing: bool = False
    initializer_range: float = 0.01976423537605237
    text_dim: int = 2560
    num_codebooks: int = 10


@dataclass
class ModelConfig(BaseModelArgs):
    model_type: str = "fish_speech"
    model_path: Optional[str] = None
    dtype: str = "bfloat16"
    pad_token_id: int = 151669
    eos_token_id: int = 151645
    audio_pad_token_id: int = 151677
    semantic_start_token_id: int = 151678
    semantic_end_token_id: int = 155773
    sample_rate: int = 44100
    text_config: FishTextConfig = field(default_factory=FishTextConfig)
    audio_decoder_config: FishAudioDecoderConfig = field(
        default_factory=FishAudioDecoderConfig
    )

    @classmethod
    def from_dict(cls, params: dict) -> "ModelConfig":
        text_config = params.get("text_config", {})
        audio_decoder_config = params.get("audio_decoder_config", {})
        return cls(
            model_type=params.get("model_type", "fish_speech"),
            model_path=params.get("model_path"),
            dtype=params.get("dtype", "bfloat16"),
            pad_token_id=params.get("pad_token_id", 151669),
            eos_token_id=params.get("eos_token_id", 151645),
            audio_pad_token_id=params.get("audio_pad_token_id", 151677),
            semantic_start_token_id=params.get("semantic_start_token_id", 151678),
            semantic_end_token_id=params.get("semantic_end_token_id", 155773),
            sample_rate=params.get("sample_rate", 44100),
            text_config=(
                text_config
                if isinstance(text_config, FishTextConfig)
                else FishTextConfig.from_dict(text_config)
            ),
            audio_decoder_config=(
                audio_decoder_config
                if isinstance(audio_decoder_config, FishAudioDecoderConfig)
                else FishAudioDecoderConfig.from_dict(audio_decoder_config)
            ),
        )
