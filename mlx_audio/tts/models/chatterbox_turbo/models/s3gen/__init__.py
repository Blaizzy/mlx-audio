# Copyright (c) 2025 Resemble AI
# MIT License
# S3Gen module for MLX

from .decoder import ConditionalDecoder
from .encoder import UpsampleConformerEncoder
from .flow_matching import CausalConditionalCFM
from .hifigan import HiFTGenerator
from .mel import mel_spectrogram
from .s3gen import (
    S3GEN_SIL,
    S3GEN_SR,
    SPEECH_VOCAB_SIZE,
    S3Gen,
    S3Token2Mel,
    S3Token2Wav,
)

__all__ = [
    "S3Gen",
    "S3Token2Mel",
    "S3Token2Wav",
    "S3GEN_SR",
    "S3GEN_SIL",
    "SPEECH_VOCAB_SIZE",
    "HiFTGenerator",
    "UpsampleConformerEncoder",
    "ConditionalDecoder",
    "CausalConditionalCFM",
    "mel_spectrogram",
]
