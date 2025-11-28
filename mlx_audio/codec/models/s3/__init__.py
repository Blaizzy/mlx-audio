from .model_v2 import ModelConfig, S3TokenizerV2
from .utils import (
    log_mel_spectrogram,
    make_non_pad_mask,
    mask_to_bias,
    merge_tokenized_segments,
    padding,
)

# S3Tokenizer constants
S3_SR = 16_000  # Sample rate for S3Tokenizer
S3_HOP = 160  # 100 frames/sec
S3_TOKEN_HOP = 640  # 25 tokens/sec
S3_TOKEN_RATE = 25
SPEECH_VOCAB_SIZE = 6561  # 3^8

__all__ = [
    "S3TokenizerV2",
    "ModelConfig",
    "log_mel_spectrogram",
    "make_non_pad_mask",
    "mask_to_bias",
    "padding",
    "merge_tokenized_segments",
    "S3_SR",
    "S3_HOP",
    "S3_TOKEN_HOP",
    "S3_TOKEN_RATE",
    "SPEECH_VOCAB_SIZE",
]
