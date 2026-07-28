from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


def _normalize_special_topk_layers(value: Any) -> dict[int, int]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError("special_topk_layers must be a mapping")
    return {int(k): int(v) for k, v in value.items()}


@dataclass
class Zonos2Config:
    model_type: str = "zonos2"
    dtype: str = "bfloat16"

    n_layers: int = 28
    dim: int = 2048
    head_dim: int = 128
    n_heads: Optional[int] = None
    n_kv_heads: Optional[int] = 4
    ffn_dim_multiplier: float = 1.5
    multiple_of: int = 256
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    max_seqlen: int = 6144

    n_codebooks: int = 9
    codebook_size: int = 1024
    eoa_id: int = 1024
    audio_pad_id: int = 1025
    text_vocab: Optional[int] = 519
    loss_softcap: float = 15.0
    sample_rate: int = 44100
    dac_model_id: str = "mlx-community/descript-audio-codec-44khz"

    speaker_enabled: bool = True
    speaker_embedding_dim: int = 2048
    speaker_lda_dim: Optional[int] = 1024
    speaker_encoder_model_id: str = "marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B"
    speaker_encoder_path: Optional[str] = "speaker_encoder"
    speaker_encoder_sample_rate: int = 24000
    speaker_background_token_enabled: bool = True
    accurate_mode_token_enabled: bool = True

    speaking_rate_num_buckets: int = 8
    speaking_rate_buckets: tuple[str, ...] = (
        "0-8",
        "8-11",
        "11-14",
        "14-17",
        "17-21",
        "21-28",
        "28-40",
        "40+",
    )
    quality_num_buckets: int = 60
    quality_features: tuple[str, ...] = (
        "lufs",
        "estimated_snr",
        "max_pause",
        "estimated_bandlimit_hz",
        "leading_silence_s",
        "trailing_silence_s",
    )
    quality_buckets: dict[str, tuple[str, ...]] = field(default_factory=dict)
    quality_dropout: dict[str, float] = field(default_factory=dict)

    moe_impl: str = "sonic"
    moe_n_experts: int = 16
    moe_router_topk: int = 1
    special_topk_layers: dict[int, int] = field(default_factory=lambda: {26: 2})
    moe_router_dim: int = 128
    moe_start_from_layer: int = 3
    moe_end_from_layer: int = 1
    norm_topk_prob: bool = False
    moe_balancing_strategy: str = "legacy"

    model_path: Optional[str] = None

    def __post_init__(self) -> None:
        self.model_type = str(self.model_type)
        self.n_layers = int(self.n_layers)
        self.dim = int(self.dim)
        self.head_dim = int(self.head_dim)
        self.n_heads = None if self.n_heads is None else int(self.n_heads)
        self.n_kv_heads = None if self.n_kv_heads is None else int(self.n_kv_heads)
        self.multiple_of = int(self.multiple_of)
        self.n_codebooks = int(self.n_codebooks)
        self.codebook_size = int(self.codebook_size)
        self.eoa_id = int(self.eoa_id)
        self.audio_pad_id = int(self.audio_pad_id)
        self.text_vocab = None if self.text_vocab is None else int(self.text_vocab)
        self.sample_rate = int(self.sample_rate)
        self.speaker_enabled = bool(self.speaker_enabled)
        self.speaker_embedding_dim = int(self.speaker_embedding_dim)
        self.speaker_lda_dim = (
            None if self.speaker_lda_dim is None else int(self.speaker_lda_dim)
        )
        self.speaker_encoder_model_id = str(self.speaker_encoder_model_id)
        self.speaker_encoder_path = (
            None
            if self.speaker_encoder_path is None
            else str(self.speaker_encoder_path)
        )
        self.speaker_encoder_sample_rate = int(self.speaker_encoder_sample_rate)
        self.speaker_background_token_enabled = bool(
            self.speaker_background_token_enabled
        )
        self.accurate_mode_token_enabled = bool(self.accurate_mode_token_enabled)
        self.speaking_rate_num_buckets = int(self.speaking_rate_num_buckets)
        self.speaking_rate_buckets = tuple(str(x) for x in self.speaking_rate_buckets)
        self.quality_num_buckets = int(self.quality_num_buckets or 0)
        self.quality_features = tuple(str(x) for x in self.quality_features)
        self.quality_buckets = {
            str(k): tuple(str(x) for x in (v or ()))
            for k, v in (self.quality_buckets or {}).items()
        }
        if not self.quality_buckets:
            self.quality_buckets = _default_quality_buckets()
        if not self.quality_features and self.quality_buckets:
            self.quality_features = tuple(self.quality_buckets.keys())
        if self.quality_num_buckets <= 0:
            self.quality_num_buckets = sum(
                len(self.quality_buckets.get(feature, ()))
                for feature in self.quality_features
            )
        self.quality_dropout = {
            str(k): float(v) for k, v in (self.quality_dropout or {}).items()
        }
        self.moe_n_experts = int(self.moe_n_experts)
        self.moe_router_topk = int(self.moe_router_topk)
        self.special_topk_layers = _normalize_special_topk_layers(
            self.special_topk_layers
        )
        self.moe_router_dim = int(self.moe_router_dim)
        self.moe_start_from_layer = int(self.moe_start_from_layer)
        self.moe_end_from_layer = int(self.moe_end_from_layer)
        self.norm_topk_prob = bool(self.norm_topk_prob)
        self.moe_balancing_strategy = (
            str(self.moe_balancing_strategy).strip().lower().replace("-", "_")
        )

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> "Zonos2Config":
        data = dict(params or {})
        if "model" in data and isinstance(data["model"], dict):
            data = {**data, **data["model"]}
        data.pop("model", None)
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in data.items() if k in allowed})

    @property
    def num_heads(self) -> int:
        return self.n_heads if self.n_heads is not None else self.dim // self.head_dim

    @property
    def num_kv_heads(self) -> int:
        return self.n_kv_heads if self.n_kv_heads is not None else self.num_heads

    @property
    def intermediate_size(self) -> int:
        raw = int(self.ffn_dim_multiplier * self.dim)
        return self.multiple_of * ((raw + self.multiple_of - 1) // self.multiple_of)

    @property
    def audio_vocab_size(self) -> int:
        return self.codebook_size + 2

    @property
    def frame_width(self) -> int:
        return self.n_codebooks + 1

    @property
    def quality_bucket_counts(self) -> tuple[int, ...]:
        return tuple(
            len(self.quality_buckets.get(feature, ()))
            for feature in self.quality_features
        )

    @property
    def speaker_background_num_buckets(self) -> int:
        return 2 if self.speaker_background_token_enabled else 0

    @property
    def accurate_mode_num_buckets(self) -> int:
        return (
            1
            if self.accurate_mode_token_enabled
            and self.speaker_background_num_buckets > 0
            else 0
        )

    def is_moe_layer(self, layer_idx: int) -> bool:
        if self.moe_n_experts <= 1:
            return False
        if layer_idx < self.moe_start_from_layer:
            return False
        if (self.n_layers - layer_idx) <= self.moe_end_from_layer:
            return False
        return True

    def num_experts_per_tok(self, layer_idx: int) -> int:
        return int(self.special_topk_layers.get(layer_idx, self.moe_router_topk))


def _default_quality_buckets() -> dict[str, tuple[str, ...]]:
    return {
        "lufs": (
            "-1000--50",
            "-50--45.5",
            "-45.5--41",
            "-41--36.5",
            "-36.5--32",
            "-32--27.5",
            "-27.5--23",
            "-23--18.5",
            "-18.5--14",
            "-14--9.5",
            "-9.5--5",
            "-5+",
        ),
        "estimated_snr": (
            "-1000-0",
            "0-6",
            "6-12",
            "12-18",
            "18-24",
            "24-30",
            "30-36",
            "36-42",
            "42-48",
            "48-54",
            "54-60",
            "60+",
        ),
        "max_pause": (
            "0-0.5",
            "0.5-1",
            "1-1.5",
            "1.5-2",
            "2-2.5",
            "2.5-3",
            "3-3.5",
            "3.5-4",
            "4-4.5",
            "4.5-5",
            "5-5.5",
            "5.5-6",
        ),
        "estimated_bandlimit_hz": (
            "495.3-3433",
            "3433-6371",
            "6371-9310",
            "9310-12248",
            "12248-15186",
            "15186-18124",
            "18124-21062",
            "21062-24000",
        ),
        "leading_silence_s": (
            "0-0.05",
            "0.05-0.1",
            "0.1-0.25",
            "0.25-0.5",
            "0.5-1",
            "1-2",
            "2-4",
            "4+",
        ),
        "trailing_silence_s": (
            "0-0.05",
            "0.05-0.1",
            "0.1-0.25",
            "0.25-0.5",
            "0.5-1",
            "1-2",
            "2-4",
            "4+",
        ),
    }


ModelConfig = Zonos2Config
