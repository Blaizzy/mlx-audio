from .config import HiggsAudioV3Config, HiggsAudioV3TextConfig
from .generation import (
    HiggsSamplerState,
    apply_delay_pattern,
    reverse_delay_pattern,
    sample_independent,
    step,
)
from .model import Model, ModelConfig
from .prompt import HiggsAudioV3PromptBuilder, ReferenceCodes

DETECTION_HINTS = {
    "model_type_aliases": {"higgs_multimodal_qwen3"},
    "path_patterns": {"higgs_audio_v3", "higgsaudiov3", "higgs-audio-v3"},
}

__all__ = [
    "Model",
    "ModelConfig",
    "DETECTION_HINTS",
    "HiggsAudioV3Config",
    "HiggsAudioV3TextConfig",
    "HiggsAudioV3PromptBuilder",
    "ReferenceCodes",
    "HiggsSamplerState",
    "apply_delay_pattern",
    "reverse_delay_pattern",
    "sample_independent",
    "step",
]
