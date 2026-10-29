import json
from dataclasses import dataclass, field
from pathlib import Path

from mlx_audio.lm.models.qwen3 import ModelArgs as Qwen3Config
from mlx_audio.tts.models.base import BaseModelArgs

LANGUAGES = ("as", "bn", "en", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te")


def default_backbone():
    return {
        "model_type": "qwen3",
        "hidden_size": 2048,
        "num_hidden_layers": 28,
        "intermediate_size": 6144,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "rms_norm_eps": 1e-6,
        "vocab_size": 194548,
        "max_position_embeddings": 32768,
        "rope_theta": 1000000.0,
        "tie_word_embeddings": True,
    }


@dataclass
class DepthConfig(BaseModelArgs):
    hidden_size: int = 1024
    num_attention_heads: int = 16
    intermediate_size: int = 3072
    num_hidden_layers: int = 4
    embedding_rank: int = 128


@dataclass
class ModelConfig(BaseModelArgs):
    model_type: str = "maya2"
    model_path: str | None = None
    backbone_config: dict = field(default_factory=default_backbone)
    depth_config: DepthConfig = field(default_factory=DepthConfig)
    num_codebooks: int = 32
    codebook_size: int = 2048
    sample_rate: int = 24000
    frame_rate: float = 12.5
    speaker_dim: int = 192
    speaker_anchor_hidden: int = 512
    speaker_anchor_enabled: bool = True
    codec_repo: str = "kyutai/moshiko-pytorch-bf16"
    codec_filename: str = "tokenizer-e351c8d8-checkpoint125.safetensors"
    speaker_encoder_repo: str = "aufklarer/SpeechBrain-ECAPA-VoxCeleb-20M-MLX"

    @classmethod
    def from_dict(cls, params):
        params = dict(params)
        path = params.get("model_path")
        if path and "backbone_config" not in params:
            path = Path(path)
            backbone = path / "backbone" / "config.json"
            if backbone.is_file():
                params["backbone_config"] = json.loads(backbone.read_text())
            sidecar = path / "maya3_config.json"
            if sidecar.is_file():
                source = json.loads(sidecar.read_text())
                anchor = source.get("model", {}).get("speaker_anchor", {})
                params["speaker_anchor_enabled"] = anchor.get("enabled", False)
                params["speaker_anchor_hidden"] = anchor.get("hidden", 512)
                params["speaker_dim"] = anchor.get("input_dim", 192)
        cfg = super().from_dict(params)
        if isinstance(cfg.depth_config, dict):
            cfg.depth_config = DepthConfig.from_dict(cfg.depth_config)
        cfg.backbone_config = {**default_backbone(), **cfg.backbone_config}
        rope = cfg.backbone_config.get("rope_parameters", {})
        cfg.backbone_config["rope_theta"] = rope.get(
            "rope_theta", cfg.backbone_config["rope_theta"]
        )
        if not 2 <= cfg.num_codebooks <= 32:
            raise ValueError("num_codebooks must be between 2 and 32")
        return cfg

    def qwen_config(self):
        return Qwen3Config.from_dict(self.backbone_config)
