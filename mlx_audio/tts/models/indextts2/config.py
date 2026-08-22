from dataclasses import dataclass
from typing import Any, Optional

from mlx_audio.codec.models.bigvgan.bigvgan import BigVGANConfig
from mlx_audio.tts.models.base import BaseModelArgs
from mlx_audio.tts.models.indextts2.semantic_codec import RepCodecConfig


@dataclass
class ModelConfig(BaseModelArgs):
    """MLX-native IndexTTS2 config.

    This module is a scaffold for a full MLX port of IndexTTS2.
    Weight conversion + component implementations will be added incrementally.
    """

    model_type: str = "indextts2"

    # Audio
    sample_rate: int = 22050

    # Optional LLM used to map `emo_text` -> `emo_vector`.
    # This is separate from the core TTS weights.
    qwen_emotion_model: str = "Qwen/Qwen2.5-0.5B-Instruct-4bit"

    # Vocoder (BigVGAN v2 22khz 80-band for official IndexTTS2)
    vocoder: Optional[BigVGANConfig] = None

    # Style encoder (CAMPPlus) config dict (matches CAMPPlus __init__ args)
    campplus: Optional[dict[str, Any]] = None

    # Semantic codec (MaskGCT / RepCodec)
    semantic_codec: Optional[RepCodecConfig] = None

    # W2V-BERT semantic encoder (facebook/w2v-bert-2.0)
    w2vbert: Optional[dict[str, Any]] = None

    # UnifiedVoice (semantic token generator)
    unifiedvoice: Optional[dict[str, Any]] = None

    # s2mel flow-matching model
    s2mel: Optional[dict[str, Any]] = None

    # Paths within the MLX model folder (relative to model_path) for submodules.
    # These will be populated once converters exist.
    bigvgan_weights: Optional[str] = None
    campplus_weights: Optional[str] = None
    maskgct_weights: Optional[str] = None
    w2vbert_weights: Optional[str] = None
    unifiedvoice_weights: Optional[str] = None
    s2mel_diffusion_weights: Optional[str] = None

    # Set by loader
    model_path: Optional[str] = None

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "ModelConfig":
        vocoder_cfg = config.get("vocoder", None)
        vocoder = (
            BigVGANConfig(**vocoder_cfg)
            if isinstance(vocoder_cfg, dict)
            else None
        )

        semantic_codec_cfg = config.get("semantic_codec", None)
        semantic_codec = (
            RepCodecConfig(**semantic_codec_cfg)
            if isinstance(semantic_codec_cfg, dict)
            else None
        )
        return cls(
            model_type=config.get("model_type", "indextts2"),
            sample_rate=int(config.get("sample_rate", 22050)),
            qwen_emotion_model=config.get(
                "qwen_emotion_model", "Qwen/Qwen2.5-0.5B-Instruct-4bit"
            ),
            vocoder=vocoder,
            campplus=config.get("campplus"),
            semantic_codec=semantic_codec,
            w2vbert=config.get("w2vbert"),
            unifiedvoice=config.get("unifiedvoice"),
            s2mel=config.get("s2mel"),
            bigvgan_weights=config.get("bigvgan_weights"),
            campplus_weights=config.get("campplus_weights"),
            maskgct_weights=config.get("maskgct_weights"),
            w2vbert_weights=config.get("w2vbert_weights"),
            unifiedvoice_weights=config.get("unifiedvoice_weights"),
            s2mel_diffusion_weights=config.get("s2mel_diffusion_weights"),
            model_path=config.get("model_path"),
        )
