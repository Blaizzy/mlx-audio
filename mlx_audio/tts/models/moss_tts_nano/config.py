from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mlx_audio.tts.models.base import BaseModelArgs

DEFAULT_AUDIO_TOKENIZER_REPO = "mlx-community/MOSS-Audio-Tokenizer-Nano"


@dataclass
class GPT2Config(BaseModelArgs):
    model_type: str = "gpt2"
    vocab_size: int = 16384
    n_positions: int = 32768
    n_ctx: int = 32768
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    n_inner: int = 3072
    activation_function: str = "gelu_new"
    resid_pdrop: float = 0.0
    embd_pdrop: float = 0.0
    attn_pdrop: float = 0.0
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    scale_attn_weights: bool = True
    scale_attn_by_inverse_layer_idx: bool = False
    position_embedding_type: str = "rope"
    rope_base: float = 10000.0
    pad_token_id: int = 3
    bos_token_id: int = 1
    eos_token_id: int = 2
    tie_word_embeddings: bool = True
    use_cache: bool = True

    @property
    def hidden_size(self) -> int:
        return self.n_embd

    @property
    def num_attention_heads(self) -> int:
        return self.n_head

    @classmethod
    def from_dict(cls, params: dict[str, Any] | None) -> "GPT2Config":
        params = dict(params or {})
        if "hidden_size" in params and "n_embd" not in params:
            params["n_embd"] = params["hidden_size"]
        if "num_hidden_layers" in params and "n_layer" not in params:
            params["n_layer"] = params["num_hidden_layers"]
        if "num_attention_heads" in params and "n_head" not in params:
            params["n_head"] = params["num_attention_heads"]
        if "intermediate_size" in params and "n_inner" not in params:
            params["n_inner"] = params["intermediate_size"]
        return super().from_dict(params)


@dataclass
class ModelConfig(BaseModelArgs):
    model_type: str = "moss_tts_nano"
    model_path: str | None = None
    gpt2_config: GPT2Config = field(default_factory=GPT2Config)
    n_vq: int = 16
    audio_vocab_size: int = 1024
    audio_codebook_sizes: list[int] = field(default_factory=lambda: [1024] * 16)
    audio_pad_token_id: int = 1024
    pad_token_id: int = 3
    im_start_token_id: int = 4
    im_end_token_id: int = 5
    audio_start_token_id: int = 6
    audio_end_token_id: int = 7
    audio_user_slot_token_id: int = 8
    audio_assistant_slot_token_id: int = 9
    audio_tokenizer_type: str = "moss-audio-tokenizer-nano"
    audio_tokenizer_pretrained_name_or_path: str | None = None
    audio_tokenizer_sample_rate: int = 48000
    tokenizer_use_fast: bool = False
    attn_implementation: str = "sdpa"
    local_transformer_layers: int = 1
    local_transformer_attn_implementation: str | None = None
    initializer_range: float = 0.02
    max_position_embeddings: int = 32768
    hidden_size: int = 768
    vocab_size: int = 16384

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> "ModelConfig":
        gpt2_config = GPT2Config.from_dict(params.get("gpt2_config", {}))
        n_vq = int(params.get("n_vq", 16))
        audio_codebook_sizes = params.get("audio_codebook_sizes")
        audio_vocab_size = int(params.get("audio_vocab_size", 1024))
        if audio_codebook_sizes is None:
            audio_codebook_sizes = [audio_vocab_size] * n_vq
        else:
            audio_codebook_sizes = [int(size) for size in audio_codebook_sizes]
        if len(audio_codebook_sizes) != n_vq:
            raise ValueError(
                "audio_codebook_sizes must have one entry per VQ channel "
                f"(expected {n_vq}, got {len(audio_codebook_sizes)})"
            )

        local_attn = params.get("local_transformer_attn_implementation")
        if local_attn is None:
            local_attn = params.get("attn_implementation", "sdpa")

        audio_tokenizer_repo = params.get(
            "audio_tokenizer_pretrained_name_or_path",
            None,
        )

        return cls(
            model_type="moss_tts_nano",
            model_path=params.get("model_path"),
            gpt2_config=gpt2_config,
            n_vq=n_vq,
            audio_vocab_size=audio_vocab_size,
            audio_codebook_sizes=audio_codebook_sizes,
            audio_pad_token_id=int(params.get("audio_pad_token_id", 1024)),
            pad_token_id=int(params.get("pad_token_id", 3)),
            im_start_token_id=int(params.get("im_start_token_id", 4)),
            im_end_token_id=int(params.get("im_end_token_id", 5)),
            audio_start_token_id=int(params.get("audio_start_token_id", 6)),
            audio_end_token_id=int(params.get("audio_end_token_id", 7)),
            audio_user_slot_token_id=int(params.get("audio_user_slot_token_id", 8)),
            audio_assistant_slot_token_id=int(
                params.get("audio_assistant_slot_token_id", 9)
            ),
            audio_tokenizer_type=params.get(
                "audio_tokenizer_type", "moss-audio-tokenizer-nano"
            ),
            audio_tokenizer_pretrained_name_or_path=audio_tokenizer_repo,
            audio_tokenizer_sample_rate=int(
                params.get("audio_tokenizer_sample_rate", 48000)
            ),
            tokenizer_use_fast=bool(params.get("tokenizer_use_fast", False)),
            attn_implementation=params.get("attn_implementation", "sdpa"),
            local_transformer_layers=int(params.get("local_transformer_layers", 1)),
            local_transformer_attn_implementation=str(local_attn),
            initializer_range=float(params.get("initializer_range", 0.02)),
            max_position_embeddings=int(
                params.get("max_position_embeddings", gpt2_config.n_positions)
            ),
            hidden_size=int(params.get("hidden_size", gpt2_config.n_embd)),
            vocab_size=int(params.get("vocab_size", gpt2_config.vocab_size)),
        )

    def local_gpt2_config(self) -> GPT2Config:
        return GPT2Config(
            model_type=self.gpt2_config.model_type,
            vocab_size=self.gpt2_config.vocab_size,
            n_positions=self.n_vq + 1,
            n_ctx=self.n_vq + 1,
            n_embd=self.gpt2_config.n_embd,
            n_layer=self.local_transformer_layers,
            n_head=self.gpt2_config.n_head,
            n_inner=self.gpt2_config.n_inner,
            activation_function=self.gpt2_config.activation_function,
            resid_pdrop=self.gpt2_config.resid_pdrop,
            embd_pdrop=self.gpt2_config.embd_pdrop,
            attn_pdrop=self.gpt2_config.attn_pdrop,
            layer_norm_epsilon=self.gpt2_config.layer_norm_epsilon,
            initializer_range=self.gpt2_config.initializer_range,
            scale_attn_weights=self.gpt2_config.scale_attn_weights,
            scale_attn_by_inverse_layer_idx=(
                self.gpt2_config.scale_attn_by_inverse_layer_idx
            ),
            position_embedding_type=self.gpt2_config.position_embedding_type,
            rope_base=self.gpt2_config.rope_base,
            pad_token_id=self.gpt2_config.pad_token_id,
            bos_token_id=self.gpt2_config.bos_token_id,
            eos_token_id=self.gpt2_config.eos_token_id,
            tie_word_embeddings=self.gpt2_config.tie_word_embeddings,
            use_cache=self.gpt2_config.use_cache,
        )
