from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Dict

from mlx_audio.tts.models.base import BaseModelArgs


def _filter_dict_for_dataclass(cls: type, data: Dict[str, Any]) -> Dict[str, Any]:
    valid_fields = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in valid_fields}


def _default_language_config() -> Dict[str, Any]:
    return {
        "hidden_size": 4096,
        "num_hidden_layers": 36,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "intermediate_size": 12288,
        "head_dim": 128,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000,
        "vocab_size": 155648,
    }


def _coerce_language_config(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(k): v for k, v in value.items()}
    try:
        as_dict = dict(value)
    except Exception as exc:
        raise ValueError("language_config must be dict-like") from exc
    return {str(k): v for k, v in as_dict.items()}


@dataclass
class ModelConfig(BaseModelArgs):
    model_type: str = "moss_tts_delay"

    language_config: Dict[str, Any] = field(default_factory=_default_language_config)

    n_vq: int = 32
    audio_vocab_size: int = 1024
    audio_pad_code: int = 1024
    sampling_rate: int = 24000
    sample_rate: int = 24000

    audio_start_token_id: int = 151652
    audio_end_token_id: int = 151653
    audio_user_slot_token_id: int = 151654
    audio_assistant_gen_slot_token_id: int = 151656
    audio_assistant_delay_slot_token_id: int = 151662
    pad_token_id: int = 151643
    im_start_token_id: int = 151644
    im_end_token_id: int = 151645

    def __post_init__(self):
        merged = _default_language_config()
        merged.update(_coerce_language_config(self.language_config))
        self.language_config = merged

        if (
            getattr(self, "sampling_rate", None) is None
            and getattr(self, "sample_rate", None) is not None
        ):
            self.sampling_rate = self.sample_rate
        if (
            getattr(self, "sample_rate", None) is None
            and getattr(self, "sampling_rate", None) is not None
        ):
            self.sample_rate = self.sampling_rate

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> "ModelConfig":
        params = dict(params)

        language_config = params.get("language_config")
        to_dict = getattr(language_config, "to_dict", None)
        if callable(to_dict):
            language_config = to_dict()

        if language_config is None:
            language_config = _default_language_config()
        else:
            merged = _default_language_config()
            merged.update(_coerce_language_config(language_config))
            language_config = merged

        params["language_config"] = language_config
        filtered = _filter_dict_for_dataclass(cls, params)
        return cls(**filtered)
