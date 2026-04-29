"""Higgs Audio v2 (Boson AI) — MLX port config.

Maps the PyTorch HiggsAudioConfig (transformers-style) into dataclasses
usable by this MLX port.

Upstream reference:
  bosonai/higgs-audio-v2-generation-3B-base/config.json
  boson_multimodal/model/higgs_audio/configuration_higgs_audio.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class HiggsTextConfig:
    """Llama-3.2-3B backbone configuration."""

    hidden_size: int = 3072
    num_hidden_layers: int = 28
    num_attention_heads: int = 24
    num_key_value_heads: int = 8
    intermediate_size: int = 8192
    vocab_size: int = 128256
    rope_theta: float = 500000.0
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = True
    rope_scaling: Optional[dict] = field(
        default_factory=lambda: {
            "factor": 32.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3",
        }
    )
    model_type: str = "llama"


@dataclass
class HiggsAudioConfig:
    """Top-level Higgs v2 config: Llama backbone + Higgs-specific audio extensions."""

    text_config: HiggsTextConfig = field(default_factory=HiggsTextConfig)

    # Audio codebook params (match bosonai/higgs-audio-v2-tokenizer)
    audio_num_codebooks: int = 8
    audio_codebook_size: int = 1024
    audio_stream_bos_id: int = 1024
    audio_stream_eos_id: int = 1025

    # Dual-FFN layer indices — which backbone layers run the audio MLP path.
    # For v2 3B this is [0..27], i.e. all layers.
    audio_dual_ffn_layers: List[int] = field(default_factory=lambda: list(range(28)))

    # If True, insert a separate audio-attention layer per dual-FFN block.
    # For v2 3B this is 0 — attention is fully shared between text and audio.
    use_audio_out_self_attention: bool = False

    # If > 0, add an extra transformer stack inside HiggsAudioDecoderProjector
    # before the audio head. For v2 3B this is 0 — the decoder projector is
    # literally just two nn.Linear heads (text_lm_head + audio_lm_head).
    audio_decoder_proj_num_layers: int = 0

    # If True, generation uses the delay-pattern trick (codebook i is emitted
    # with i-frame delay so one forward pass predicts K codebooks at once).
    # For v2 3B: True. Requires revert_delay_pattern at decode time.
    use_delay_pattern: bool = True

    # Special-token ids in the text vocab that mark audio-in and audio-out
    # positions. Used to build the audio_out_mask during generation.
    audio_in_token_idx: Optional[int] = None
    audio_out_token_idx: Optional[int] = None
    audio_out_bos_token_id: Optional[int] = None
    audio_eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None

    @classmethod
    def from_dict(cls, d: dict) -> "HiggsAudioConfig":
        """Construct from the HF config.json dict (permissive — ignores extras)."""
        tc = d.get("text_config", {})
        text_config = HiggsTextConfig(
            hidden_size=tc.get("hidden_size", 3072),
            num_hidden_layers=tc.get("num_hidden_layers", 28),
            num_attention_heads=tc.get("num_attention_heads", 24),
            num_key_value_heads=tc.get("num_key_value_heads", 8),
            intermediate_size=tc.get("intermediate_size", 8192),
            vocab_size=tc.get("vocab_size", 128256),
            rope_theta=tc.get("rope_theta", 500000.0),
            rms_norm_eps=tc.get("rms_norm_eps", 1e-5),
            tie_word_embeddings=tc.get("tie_word_embeddings", True),
            rope_scaling=tc.get("rope_scaling"),
        )
        return cls(
            text_config=text_config,
            audio_num_codebooks=d.get("audio_num_codebooks", 8),
            audio_codebook_size=d.get("audio_codebook_size", 1024),
            audio_stream_bos_id=d.get("audio_stream_bos_id", 1024),
            audio_stream_eos_id=d.get("audio_stream_eos_id", 1025),
            audio_dual_ffn_layers=d.get("audio_dual_ffn_layers", list(range(28))),
            use_audio_out_self_attention=bool(
                d.get("use_audio_out_self_attention", False)
            ),
            audio_decoder_proj_num_layers=d.get("audio_decoder_proj_num_layers", 0),
            use_delay_pattern=d.get("use_delay_pattern", True),
            audio_in_token_idx=d.get("audio_in_token_idx"),
            audio_out_token_idx=d.get("audio_out_token_idx"),
            audio_out_bos_token_id=d.get("audio_out_bos_token_id"),
            audio_eos_token_id=d.get("audio_eos_token_id"),
            pad_token_id=d.get("pad_token_id"),
        )
