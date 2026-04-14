"""DeepFilterNet speech enhancement model for MLX."""

from .config import DeepFilterNet2Config, DeepFilterNet3Config, DeepFilterNetConfig
from .model import DeepFilterNetModel
from .streaming import DeepFilterNetStreamer, DeepFilterNetStreamingConfig

Model = DeepFilterNetModel
ModelConfig = DeepFilterNetConfig

__all__ = [
    "DeepFilterNetModel",
    "DeepFilterNetConfig",
    "DeepFilterNet2Config",
    "DeepFilterNet3Config",
    "DeepFilterNetStreamer",
    "DeepFilterNetStreamingConfig",
    "Model",
    "ModelConfig",
]
