from .asr import MiMoASR
from .config import MiMoAudioConfig as ModelConfig

Model = MiMoASR

__all__ = ["MiMoASR", "Model", "ModelConfig"]
