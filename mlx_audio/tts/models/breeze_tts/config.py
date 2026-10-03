"""Configuration helpers for Breeze TTS 2 checkpoints."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from mlx_audio.tts.models.base import BaseModelArgs


@dataclass
class ModelConfig(BaseModelArgs):
    """The subset of the upstream Breeze config required for MLX inference."""

    model_type: str = "breeze"
    hidden_size: int = 2048
    intermediate_size: int = 6144
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-5
    attention_bias: bool = False
    attention_dropout: float = 0.0
    mlp_bias: bool = False
    initializer_range: float = 0.02
    use_cache: bool = True
    pad_token_id: int = 0
    bos_token_id: int = 2
    eos_token_id: int | None = 1
    tie_word_embeddings: bool = False
    rope_theta: float = 500000.0
    rope_scaling: dict | None = None
    max_position_embeddings: int = 2048
    num_codebooks: int = 16
    vocab_size: int = 2051
    # Newer Breeze checkpoints retain the canonical ``num_codebooks`` and
    # ``vocab_size`` fields, while some exports also expose these as
    # ``audio_*`` aliases.  Keep both so loading does not accidentally use the
    # nested Qwen vocabulary for the audio wrapper.
    audio_num_codebooks: int | None = None
    audio_vocab_size: int | None = None
    audio_embed_size: int | None = None
    text_vocab_size: int = 262158
    audio_token_id: int = 262144
    audio_eos_token_id: int = 262145
    codebook_pad_token_id: int = 2050
    codebook_eos_token_id: int = 0
    sample_rate: int = 24000
    backbone_config: dict | None = None
    codec_config: dict | None = None
    depth_decoder_config: dict | None = None
    text_encoder_config: dict | None = None
    tie_codebooks_embeddings: bool = True

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> "ModelConfig":
        """Build a wrapper config while retaining nested model configs.

        ``BreezeConfig`` is a multimodal wrapper.  Its nested Qwen config has
        a large text vocabulary (151936 for Breeze-TTS-2), whereas the wrapper
        vocabulary is the 2051-entry codec vocabulary.  The checkpoint also
        contains ``audio_vocab_size``/``audio_num_codebooks`` aliases in some
        revisions.  Normalize those aliases here so all callers, including
        the generation path, continue to read the wrapper-level fields.
        """
        params = dict(params)
        if "num_codebooks" not in params and "audio_num_codebooks" in params:
            params["num_codebooks"] = params["audio_num_codebooks"]
        if "vocab_size" not in params and "audio_vocab_size" in params:
            params["vocab_size"] = params["audio_vocab_size"]

        codec = cls._as_mapping(params.get("codec_config"))
        audio_num_codebooks = params.get("audio_num_codebooks")
        audio_vocab_size = params.get("audio_vocab_size")
        # Prefer explicit audio aliases when both names are present.  They are
        # the unambiguous wrapper fields in checkpoints produced by newer
        # Transformers versions; the nested backbone's ``vocab_size`` is
        # never used for this purpose.
        if audio_num_codebooks is not None:
            params["num_codebooks"] = audio_num_codebooks
        if audio_vocab_size is not None:
            params["vocab_size"] = audio_vocab_size

        values = {
            key: value
            for key, value in params.items()
            if key in cls.__dataclass_fields__ and key != "sample_rate"
        }
        return cls(
            **values,
            sample_rate=codec.get(
                "sampling_rate",
                codec.get("output_sample_rate", params.get("sample_rate", 24000)),
            ),
        )

    @staticmethod
    def _as_mapping(value: Any) -> Mapping[str, Any]:
        """Return a nested config as a mapping when possible.

        ``base_load_model`` supplies dictionaries, but accepting a
        ``PretrainedConfig``-like object here keeps direct callers and tests
        compatible with the upstream configuration classes.
        """
        if value is None:
            return {}
        if isinstance(value, Mapping):
            return value
        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            converted = to_dict()
            if isinstance(converted, Mapping):
                return converted
        return {
            name: getattr(value, name)
            for name in dir(value)
            if not name.startswith("_") and not callable(getattr(value, name, None))
        }

    def backbone_value(self, name: str, default: Any = None) -> Any:
        """Return a Qwen backbone value, preferring the checkpoint sub-config.

        Breeze's top-level fields describe its multimodal wrapper and retain
        values inherited from another architecture.  The nested
        ``backbone_config`` is the authoritative Qwen3 configuration.
        """
        backbone = self._as_mapping(self.backbone_config)
        if name in backbone:
            return backbone[name]
        return getattr(self, name, default)

    @property
    def wrapper_num_codebooks(self) -> int:
        """Number of codebooks used by the audio wrapper.

        This is deliberately separate from ``backbone_config.num_codebooks``
        (which is not a Qwen field today, but may appear in future exports).
        """
        value = (
            self.audio_num_codebooks
            if self.audio_num_codebooks is not None
            else self.num_codebooks
        )
        return int(value)

    @property
    def wrapper_vocab_size(self) -> int:
        """Size of each audio codebook, excluding the backbone EOS class."""
        value = (
            self.audio_vocab_size
            if self.audio_vocab_size is not None
            else self.vocab_size
        )
        return int(value)

    @property
    def wrapper_audio_embed_size(self) -> int:
        """Embedding width consumed by the audio-token wrapper."""
        if self.audio_embed_size is not None:
            return int(self.audio_embed_size)
        return self.backbone_hidden_size

    @property
    def backbone_hidden_size(self) -> int:
        return int(self.backbone_value("hidden_size"))

    @property
    def codec_vocab_size(self) -> int:
        """Number of actual codec entries before Breeze's reserved ids."""
        codec = self._as_mapping(self.codec_config)
        size = int(codec.get("codebook_size", self.wrapper_vocab_size - 3))
        if not 0 < size <= self.wrapper_vocab_size:
            raise ValueError(
                "codec codebook_size must be between 1 and the wrapper "
                f"vocab_size ({self.wrapper_vocab_size}), got {size}"
            )
        return size
