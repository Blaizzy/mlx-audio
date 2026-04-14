from .config import OmniVoiceConfig
from .omnivoice import Model
from .utils import create_voice_clone_prompt

ModelConfig = OmniVoiceConfig  # alias expected by mlx_audio.utils.base_load_model

__all__ = ["Model", "ModelConfig", "OmniVoiceConfig", "create_voice_clone_prompt"]
