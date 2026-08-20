"""
MiMo-V2.5-ASR configuration for MLX.

Matches the HuggingFace MiMoAudioConfig, which extends Qwen2Config
with speech-specific parameters.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class MiMoAudioConfig:
    """Configuration for MiMo-V2.5-ASR (Qwen2-based ASR model)."""

    # ── Qwen2 LLM backbone ──
    vocab_size: int = 151680
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 640000.0
    rope_scaling: Optional[dict] = None
    max_position_embeddings: int = 8192
    use_sliding_window: bool = False
    sliding_window: Optional[int] = None
    attention_dropout: float = 0.0

    # ── Speech ──
    audio_channels: int = 8
    group_size: int = 4
    delay_pattern: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6, 7])
    speech_vocab_size: str = "1025-1025-129-129-129-129-129-129"
    speech_zeroemb_idx: str = "1024-1024-128-128-128-128-128-128"
    n_rvq: int = 20  # full RVQ channels (only first 8 used)

    # ── Input Local Transformer ──
    input_local_layers: int = 6
    input_local_dim: int = 1024
    input_local_attn_heads: int = 64
    input_local_intermediate_size: int = 4096
    input_local_head_dim: int = 16
    input_local_hidden_dropout: float = 0.1
    input_full_attention: bool = True

    # ── Local Transformer ──
    local_layers: int = 16
    local_dim: int = 1024
    local_attn_heads: int = 64
    local_ffn_dim: int = 4096
    local_attn_dropout: float = 0.1
    local_hidden_dropout: float = 0.1
    local_rotary_base: float = 640000.0

    # ── Projections ──
    projection_layers: int = 1
    add_post_norm: bool = True
    out_hidden_size: int = 4096

    # ── Special tokens (defaults from tokenizer_config.json) ──
    eot_idx: int = 151672
    sosp_idx: int = 151665
    eosp_idx: int = 151666
    eostm_idx: int = 151671
    sostm_idx: int = 151670
    speechlm_idx: int = 151669
    empty_idx: int = 151667

    @classmethod
    def from_dict(cls, d: dict) -> "MiMoAudioConfig":
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid})

    def parsed_speech_vocab_sizes(self) -> List[int]:
        if "-" in str(self.speech_vocab_size):
            return [int(s) for s in str(self.speech_vocab_size).split("-")]
        return [int(self.speech_vocab_size)] * self.audio_channels

    def parsed_speech_empty_ids(self) -> List[int]:
        if "-" in str(self.speech_zeroemb_idx):
            return [int(s) for s in str(self.speech_zeroemb_idx).split("-")]
        return [int(self.speech_zeroemb_idx)] * self.audio_channels

    def parsed_delay_pattern(self) -> List[int]:
        if isinstance(self.delay_pattern, str):
            return [int(x) for x in self.delay_pattern.split("-")]
        return list(self.delay_pattern)
