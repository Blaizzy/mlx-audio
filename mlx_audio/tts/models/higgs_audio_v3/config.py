from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from mlx_lm.models.qwen3 import ModelArgs as Qwen3ModelArgs


@dataclass
class HiggsAudioV3TextConfig:
    model_type: str = "qwen3"
    hidden_size: int = 2560
    num_hidden_layers: int = 36
    intermediate_size: int = 9728
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    max_position_embeddings: int = 32768
    rope_theta: float = 1_000_000.0
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    vocab_size: int = 151936
    tie_word_embeddings: bool = True
    rope_scaling: Optional[dict[str, Any]] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "HiggsAudioV3TextConfig":
        data = dict(data or {})
        rope_parameters = data.get("rope_parameters") or {}
        return cls(
            model_type=data.get("model_type", "qwen3"),
            hidden_size=int(data.get("hidden_size", 2560)),
            num_hidden_layers=int(data.get("num_hidden_layers", 36)),
            intermediate_size=int(data.get("intermediate_size", 9728)),
            num_attention_heads=int(data.get("num_attention_heads", 32)),
            num_key_value_heads=int(data.get("num_key_value_heads", 8)),
            max_position_embeddings=int(data.get("max_position_embeddings", 32768)),
            rope_theta=float(
                data.get("rope_theta", rope_parameters.get("rope_theta", 1_000_000))
            ),
            head_dim=int(data.get("head_dim", 128)),
            rms_norm_eps=float(data.get("rms_norm_eps", 1e-6)),
            vocab_size=int(data.get("vocab_size", 151936)),
            tie_word_embeddings=bool(data.get("tie_word_embeddings", True)),
            rope_scaling=data.get("rope_scaling"),
        )

    def to_qwen3_args(self) -> Qwen3ModelArgs:
        return Qwen3ModelArgs(
            model_type=self.model_type,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            intermediate_size=self.intermediate_size,
            num_attention_heads=self.num_attention_heads,
            rms_norm_eps=self.rms_norm_eps,
            vocab_size=self.vocab_size,
            num_key_value_heads=self.num_key_value_heads,
            max_position_embeddings=self.max_position_embeddings,
            rope_theta=self.rope_theta,
            head_dim=self.head_dim,
            tie_word_embeddings=self.tie_word_embeddings,
            rope_scaling=self.rope_scaling,
        )


@dataclass
class HiggsAudioV3Config:
    model_type: str = "higgs_audio_v3"
    text_config: HiggsAudioV3TextConfig = field(default_factory=HiggsAudioV3TextConfig)
    audio_token_id: int = -100
    audio_num_codebooks: int = 8
    audio_codebook_size: int = 1026
    audio_boc_token_id: int = 1024
    audio_eoc_token_id: int = 1025
    use_delay_pattern: bool = True
    sample_rate: int = 24000

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HiggsAudioV3Config":
        text_config = HiggsAudioV3TextConfig.from_dict(data.get("text_config"))
        enc = dict(data.get("audio_encoder_config") or {})
        codebook_size = int(
            data.get("audio_codebook_size", enc.get("vocab_size", 1026))
        )
        return cls(
            model_type=data.get("model_type", "higgs_audio_v3"),
            text_config=text_config,
            audio_token_id=int(data.get("audio_token_id", -100)),
            audio_num_codebooks=int(
                data.get("audio_num_codebooks", enc.get("num_codebooks", 8))
            ),
            audio_codebook_size=codebook_size,
            audio_boc_token_id=int(data.get("audio_boc_token_id", codebook_size - 2)),
            audio_eoc_token_id=int(data.get("audio_eoc_token_id", codebook_size - 1)),
            use_delay_pattern=bool(
                data.get("use_delay_pattern", enc.get("use_delay_pattern", True))
            ),
            sample_rate=int(data.get("sample_rate", 24000)),
        )
