from .config import BranchConfig, ModelConfig
from .silero_vad import Model, SileroVADState, VADOutput

DETECTION_HINTS = {
    "architectures": ["silero_vad", "SileroVAD"],
    "config_keys": ["branch_16k", "branch_8k", "min_speech_duration_ms"],
    "path_patterns": ["silero", "silero-vad", "silero_vad"],
}

__all__ = [
    "BranchConfig",
    "ModelConfig",
    "Model",
    "SileroVADState",
    "VADOutput",
    "DETECTION_HINTS",
]
