from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from mlx_lm.models.qwen3 import ModelArgs as Qwen3ModelConfig

from mlx_audio.tts.models.base import BaseModelArgs

DEFAULT_AUDIO_TOKENIZER_REPO = "OpenMOSS-Team/MOSS-Audio-Tokenizer"


@dataclass
class ModelConfig(BaseModelArgs):
    model_type: str = "moss_tts_delay"
    model_path: str | None = None
    language_config: Qwen3ModelConfig | None = None
    initializer_range: float = 0.02
    n_vq: int = 32
    audio_vocab_size: int = 1024
    audio_user_slot_token_id: int = 151654
    audio_assistant_gen_slot_token_id: int = 151656
    audio_assistant_delay_slot_token_id: int = 151662
    audio_start_token_id: int = 151652
    audio_end_token_id: int = 151653
    audio_pad_code: int = 1024
    pad_token_id: int = 151643
    im_start_token_id: int = 151644
    im_end_token_id: int = 151645
    sampling_rate: int = 24000
    audio_tokenizer_pretrained_name_or_path: str | None = None
    additional_mlp_ffn_hidden_size: int | None = None
    local_ffn_hidden_size: int | None = None
    local_hidden_size: int | None = None
    local_num_layers: int | None = None

    @property
    def hidden_size(self) -> int:
        if self.language_config is None:
            raise ValueError("language_config is not initialized")
        return int(self.language_config.hidden_size)

    @property
    def vocab_size(self) -> int:
        if self.language_config is None:
            raise ValueError("language_config is not initialized")
        return int(self.language_config.vocab_size)

    @property
    def is_local_transformer(self) -> bool:
        return (
            self.additional_mlp_ffn_hidden_size is not None
            and self.local_ffn_hidden_size is not None
            and self.local_hidden_size is not None
            and self.local_num_layers is not None
        )

    def local_transformer_config(self) -> Qwen3ModelConfig:
        if self.language_config is None:
            raise ValueError("language_config is not initialized")
        if not self.is_local_transformer:
            raise ValueError("local transformer configuration is not initialized")
        return replace(
            self.language_config,
            hidden_size=int(self.local_hidden_size),
            intermediate_size=int(self.local_ffn_hidden_size),
            num_hidden_layers=int(self.local_num_layers),
        )

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> "ModelConfig":
        params = dict(params or {})
        language_params = dict(params.get("language_config") or {})
        language_params.setdefault("model_type", "qwen3")
        # The upstream checkpoint has an explicit text lm head at lm_heads.0.
        language_params.setdefault("tie_word_embeddings", False)
        if "rope_theta" not in language_params and isinstance(
            language_params.get("rope_parameters"), dict
        ):
            rope_parameters = language_params["rope_parameters"]
            if "rope_theta" in rope_parameters:
                language_params["rope_theta"] = rope_parameters["rope_theta"]

        language_config = Qwen3ModelConfig.from_dict(language_params)
        return cls(
            model_type=str(params.get("model_type", "moss_tts_delay")),
            model_path=params.get("model_path"),
            language_config=language_config,
            initializer_range=float(params.get("initializer_range", 0.02)),
            n_vq=int(params.get("n_vq", 32)),
            audio_vocab_size=int(params.get("audio_vocab_size", 1024)),
            audio_user_slot_token_id=int(
                params.get("audio_user_slot_token_id", 151654)
            ),
            audio_assistant_gen_slot_token_id=int(
                params.get("audio_assistant_gen_slot_token_id", 151656)
            ),
            audio_assistant_delay_slot_token_id=int(
                params.get("audio_assistant_delay_slot_token_id", 151662)
            ),
            audio_start_token_id=int(params.get("audio_start_token_id", 151652)),
            audio_end_token_id=int(params.get("audio_end_token_id", 151653)),
            audio_pad_code=int(params.get("audio_pad_code", 1024)),
            pad_token_id=int(params.get("pad_token_id", 151643)),
            im_start_token_id=int(params.get("im_start_token_id", 151644)),
            im_end_token_id=int(params.get("im_end_token_id", 151645)),
            sampling_rate=int(
                params.get("sampling_rate", params.get("sample_rate", 24000))
            ),
            audio_tokenizer_pretrained_name_or_path=params.get(
                "audio_tokenizer_pretrained_name_or_path",
                DEFAULT_AUDIO_TOKENIZER_REPO,
            ),
            additional_mlp_ffn_hidden_size=(
                int(params["additional_mlp_ffn_hidden_size"])
                if "additional_mlp_ffn_hidden_size" in params
                else None
            ),
            local_ffn_hidden_size=(
                int(params["local_ffn_hidden_size"])
                if "local_ffn_hidden_size" in params
                else None
            ),
            local_hidden_size=(
                int(params["local_hidden_size"])
                if "local_hidden_size" in params
                else None
            ),
            local_num_layers=(
                int(params["local_num_layers"])
                if "local_num_layers" in params
                else None
            ),
        )
