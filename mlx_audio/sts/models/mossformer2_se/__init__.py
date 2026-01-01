# Original: ClearerVoice-Studio (github.com/modelscope/ClearerVoice-Studio)
# Copyright (c) Speech Lab, Alibaba Group
# Licensed under Apache License 2.0
# MLX port by Dmitry Starkov

"""
MossFormer2 SE speech enhancement model for MLX.

This module provides a speech enhancement model based on MossFormer2 architecture,
optimized for 48kHz audio on Apple Silicon.
"""

# Reuse audio utilities from sam_audio
from ..sam_audio.processor import load_audio, save_audio
from .config import MossFormer2SEConfig
from .mossformer2_se_wrapper import MossFormer2SE
from .processor import MossFormer2SEProcessor

__all__ = [
    "MossFormer2SEConfig",
    "MossFormer2SE",
    "MossFormer2SEProcessor",
    "load_audio",
    "save_audio",
]
