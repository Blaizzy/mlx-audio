"""Checkpoint contract and architecture sizes for MiniMax Music 3.

Adapted and modified from mikolaj92/minimax-music3-mlx under Apache-2.0.
See LICENSE and NOTICE.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from mlx_audio.tts.models.base import BaseModelArgs

OFFICIAL_AUDIO_END_TOKEN_ID = 151670
OFFICIAL_AUDIO_CFG_TOKEN_ID = 151654
OFFICIAL_AUDIO_CODE_OFFSET = 151675
OFFICIAL_SEMANTIC_VOCAB_SIZE = 16384

AR_CFG_SCALE = 1.5
AR_CFG_TOP_K = 50
AR_SAMPLING_TOP_K = 50
DIT_CFG_SCALE = 1.7

FRAME_RATE = 25.0
MAX_AUDIO_FRAMES = 9_000
MAX_PROMPT_TOKENS = 5_000
CHUNK_FRAMES = 200
CHUNK_HOP = 100
LATENT_HOP_LENGTH = 512
OVERLAP_LATENT_LENGTH = 172
CROP_LEFT_LATENT = 86
CROP_RIGHT_LATENT = 344 - CROP_LEFT_LATENT
SAMPLING_RATE = 44_100


@dataclass
class ModelConfig(BaseModelArgs):
    model_type: str = "minimax_music3"
    model_path: str = ""

    hidden_size: int = 4096
    num_codebooks: int = 8
    audio_vocab_size: int = 1024
    semantic_vocab_size: int = OFFICIAL_SEMANTIC_VOCAB_SIZE
    vocab_size: int = 200_000
    audio_code_offset: int = OFFICIAL_AUDIO_CODE_OFFSET
    audio_end_token_id: int = OFFICIAL_AUDIO_END_TOKEN_ID
    audio_cfg_token_id: int = OFFICIAL_AUDIO_CFG_TOKEN_ID

    num_hidden_layers: int = 36
    intermediate_size: int = 12_288
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 10_240
    rope_theta: float = 1_000_000.0
    tie_word_embeddings: bool = False

    depth_num_layers: int = 4
    depth_num_heads: int = 16
    depth_intermediate_size: int = 6144
    depth_max_position_embeddings: int = 16

    condition_out_dim: int = 2048
    num_condition_layers: int = 8
    input_sampling_rate: int = 24_000
    input_hop_length: int = 960
    output_sampling_rate: int = SAMPLING_RATE
    output_hop_length: int = LATENT_HOP_LENGTH
    dit_in_channels: int = 128
    dit_num_layers: int = 36
    dit_num_heads: int = 32
    dit_head_dim: int = 64
    dit_ff_inner_dim: int = 8192
    dit_rotary_dim: int = 32
    dit_fourier_dim: int = 256

    vocoder_input_dim: int = 1024
    vocoder_hidden_dim: int = 1536
    vocoder_upsampling_ratios: tuple[int, ...] = (8, 8, 4, 2)
    sample_rate: int = SAMPLING_RATE
    frame_rate: float = FRAME_RATE
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["vocoder_upsampling_ratios"] = list(self.vocoder_upsampling_ratios)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        known = {
            key: value for key, value in data.items() if key in cls.__dataclass_fields__
        }
        if "vocoder_upsampling_ratios" in known:
            known["vocoder_upsampling_ratios"] = tuple(
                known["vocoder_upsampling_ratios"]
            )
        return cls(**known)

    @classmethod
    def tiny(cls) -> "ModelConfig":
        return cls(
            hidden_size=64,
            num_codebooks=8,
            audio_vocab_size=32,
            semantic_vocab_size=64,
            vocab_size=512,
            audio_code_offset=200,
            audio_end_token_id=199,
            audio_cfg_token_id=198,
            num_hidden_layers=2,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=256,
            rope_theta=10_000.0,
            depth_num_layers=2,
            depth_num_heads=4,
            depth_intermediate_size=128,
            condition_out_dim=32,
            num_condition_layers=8,
            dit_in_channels=16,
            dit_num_layers=2,
            dit_num_heads=4,
            dit_head_dim=16,
            dit_ff_inner_dim=64,
            dit_rotary_dim=8,
            dit_fourier_dim=16,
            vocoder_input_dim=16,
            vocoder_hidden_dim=32,
            vocoder_upsampling_ratios=(4, 2),
        )

    @property
    def residual_codebooks(self) -> int:
        return self.num_codebooks - 1
