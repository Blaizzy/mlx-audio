from dataclasses import dataclass

from mlx_audio.base import BaseModelArgs
from mlx_audio.lm.models.qwen2 import ModelArgs as Qwen2Args


@dataclass
class ModelConfig(BaseModelArgs):
    model_type: str = "mimo_audio"
    hidden_size: int = 4096
    num_hidden_layers: int = 36
    intermediate_size: int = 11008
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    rms_norm_eps: float = 1e-6
    vocab_size: int = 151680
    max_position_embeddings: int = 8192
    rope_theta: float = 640000
    tie_word_embeddings: bool = False
    group_size: int = 4
    audio_channels: int = 8
    local_dim: int = 1024
    local_layers: int = 16
    local_attn_heads: int = 64
    local_ffn_dim: int = 4096
    input_local_dim: int = 1024
    input_local_layers: int = 6
    input_full_attention: bool = True
    speech_vocab_size: str | list[int] = "1025-1025-129-129-129-129-129-129"
    speech_zeroemb_idx: str | list[int] = "1024-1024-128-128-128-128-128-128"
    delay_pattern: str | list[int] = "0-1-2-3-4-5-6-7"
    empty_token_id: int = 151667
    audio_tokenizer_path: str = "XiaomiMiMo/MiMo-Audio-Tokenizer"
    audio_tokenizer_revision: str | None = "5df9914f72d3acda1320d7fecde7d91622edb0c1"

    def __post_init__(self):
        self.model_type = "mimo_audio"
        if self.group_size < 1 or self.audio_channels < 1:
            raise ValueError("MiMo group size and audio channels must be positive")
        for sizes in (self.speech_vocab_sizes, self.speech_empty_ids, self.delays):
            if len(sizes) != self.audio_channels:
                raise ValueError(
                    "Speech vocabularies, padding and delays must match audio_channels"
                )
        if any(d < 0 for d in self.delays):
            raise ValueError("Audio delays cannot be negative")
        if any(
            p != v - 1 for p, v in zip(self.speech_empty_ids, self.speech_vocab_sizes)
        ):
            raise ValueError("MiMo speech padding must be the final vocabulary entry")

    @staticmethod
    def _list(value):
        return (
            [int(x) for x in value.split("-")]
            if isinstance(value, str)
            else list(value)
        )

    @property
    def speech_vocab_sizes(self):
        return self._list(self.speech_vocab_size)

    @property
    def speech_empty_ids(self):
        return self._list(self.speech_zeroemb_idx)

    @property
    def delays(self):
        return self._list(self.delay_pattern)

    def qwen_args(self, local=None):
        args = Qwen2Args(
            model_type="qwen2",
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            intermediate_size=self.intermediate_size,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            rms_norm_eps=self.rms_norm_eps,
            vocab_size=self.vocab_size,
            max_position_embeddings=self.max_position_embeddings,
            rope_theta=self.rope_theta,
            tie_word_embeddings=False,
        )
        if local is not None:
            args.hidden_size = (
                self.input_local_dim if local == "input" else self.local_dim
            )
            args.num_hidden_layers = (
                self.input_local_layers if local == "input" else self.local_layers
            )
            args.intermediate_size = (
                args.hidden_size * 4 if local == "input" else self.local_ffn_dim
            )
            args.num_attention_heads = args.num_key_value_heads = self.local_attn_heads
        return args


def prepare_config(config, model_path=None):
    return {
        "audio_tokenizer_path": ModelConfig.audio_tokenizer_path,
        "audio_tokenizer_revision": ModelConfig.audio_tokenizer_revision,
        **config,
        "model_type": "mimo_audio",
    }
