from dataclasses import dataclass, fields
from typing import Any, Optional

from mlx_audio.codec.models.bigvgan.bigvgan import BigVGANConfig
from mlx_audio.tts.models.base import BaseModelArgs
from mlx_audio.tts.models.indextts2.semantic_codec import EnhancedCodecConfig


@dataclass
class ModelConfig(BaseModelArgs):
    """MLX-native config shared by IndexTTS2 and IndexTTS-2.5.

    `version` selects generation-specific behavior; both generations keep the
    same model type so converted checkpoints can share one model package.
    """

    model_type: str = "indextts2"
    version: str = "2.0"

    # Audio
    sample_rate: int = 22050

    # Optional LLM used to map `emo_text` -> `emo_vector`.
    # This is separate from the core TTS weights.
    qwen_emotion_model: str = "Qwen/Qwen2.5-0.5B-Instruct-4bit"

    # Emotion matrix bank used by explicit 8-D vectors and Qwen text emotion.
    emo_num: tuple[int, ...] = (3, 17, 2, 8, 4, 5, 10, 24)
    emotion_matrices: Optional[str] = None

    # Vocoder (BigVGAN v2 22khz 80-band for official IndexTTS2)
    vocoder: Optional[BigVGANConfig] = None

    # Style encoder (CAMPPlus) config dict (matches CAMPPlus __init__ args)
    campplus: Optional[dict[str, Any]] = None

    # Semantic codec (EnhancedCodec)
    semantic_codec: Optional[EnhancedCodecConfig] = None

    # W2V-BERT semantic encoder (facebook/w2v-bert-2.0)
    w2vbert: Optional[dict[str, Any]] = None

    # UnifiedVoice (semantic token generator)
    unifiedvoice: Optional[dict[str, Any]] = None

    # s2mel flow-matching model
    s2mel: Optional[dict[str, Any]] = None

    # Generation-specific switches, inferred from `version` when omitted.
    spk_cond_mode: Optional[str] = None
    use_gpt_latent: Optional[bool] = None
    tokenizer: Optional[str] = None

    # Paths within the MLX model folder (relative to model_path) for submodules.
    bigvgan_weights: Optional[str] = None
    campplus_weights: Optional[str] = None
    semantic_codec_weights: Optional[str] = None
    w2vbert_weights: Optional[str] = None
    unifiedvoice_weights: Optional[str] = None
    s2mel_diffusion_weights: Optional[str] = None

    # Set by loader
    model_path: Optional[str] = None

    @property
    def is_2_5(self) -> bool:
        return self.version.replace("_", ".").startswith("2.5")

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "ModelConfig":
        vocoder_cfg = config.get("vocoder", None)
        if isinstance(vocoder_cfg, dict):
            allowed = {field.name for field in fields(BigVGANConfig)}
            vocoder_cfg = {
                key: value for key, value in vocoder_cfg.items() if key in allowed
            }
        vocoder = (
            BigVGANConfig(**vocoder_cfg) if isinstance(vocoder_cfg, dict) else None
        )

        semantic_codec_cfg = config.get("semantic_codec", None)
        model_type = config.get("model_type", "indextts2")
        version = str(
            config.get("version", "2.5" if model_type == "indextts2_5" else "2.0")
        )
        is_2_5 = version.replace("_", ".").startswith("2.5")
        if isinstance(semantic_codec_cfg, dict):
            allowed = {field.name for field in fields(EnhancedCodecConfig)}
            semantic_codec_cfg = {
                key: value
                for key, value in semantic_codec_cfg.items()
                if key in allowed
            }
            # IndexTTS2's MaskGCT codec has no temporal downsampling; 2.5's
            # EnhancedCodec doubles the frame rate internally.
            semantic_codec_cfg.setdefault("downsample_scale", 2 if is_2_5 else 1)
        semantic_codec = (
            EnhancedCodecConfig(**semantic_codec_cfg)
            if isinstance(semantic_codec_cfg, dict)
            else None
        )

        spk_cond_mode = config.get("spk_cond_mode")
        if spk_cond_mode is None:
            spk_cond_mode = "campplus" if is_2_5 else "conformer"
        use_gpt_latent = config.get("use_gpt_latent")
        if use_gpt_latent is None:
            use_gpt_latent = not is_2_5
        tokenizer = config.get("tokenizer")
        if tokenizer is None:
            tokenizer = "tiktoken" if is_2_5 else "sentencepiece"

        return cls(
            model_type=model_type,
            version=version,
            sample_rate=int(config.get("sample_rate", 22050)),
            qwen_emotion_model=config.get(
                "qwen_emotion_model", "Qwen/Qwen2.5-0.5B-Instruct-4bit"
            ),
            emo_num=tuple(config.get("emo_num", (3, 17, 2, 8, 4, 5, 10, 24))),
            emotion_matrices=config.get("emotion_matrices"),
            vocoder=vocoder,
            campplus=config.get("campplus"),
            semantic_codec=semantic_codec,
            w2vbert=config.get("w2vbert"),
            unifiedvoice=config.get("unifiedvoice"),
            s2mel=config.get("s2mel"),
            spk_cond_mode=spk_cond_mode,
            use_gpt_latent=bool(use_gpt_latent),
            tokenizer=tokenizer,
            bigvgan_weights=config.get("bigvgan_weights"),
            campplus_weights=config.get("campplus_weights"),
            semantic_codec_weights=config.get("semantic_codec_weights"),
            w2vbert_weights=config.get("w2vbert_weights"),
            unifiedvoice_weights=config.get("unifiedvoice_weights"),
            s2mel_diffusion_weights=config.get("s2mel_diffusion_weights"),
            model_path=config.get("model_path"),
        )
