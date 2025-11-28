# Re-export from codec S3 implementation
# Chatterbox uses the same S3TokenizerV2 as CosyVoice (from s3tokenizer package)

from mlx_audio.codec.models.s3 import (
    S3_HOP,
    S3_SR,
    S3_TOKEN_HOP,
    S3_TOKEN_RATE,
    SPEECH_VOCAB_SIZE,
    ModelConfig,
    S3TokenizerV2,
    make_non_pad_mask,
    mask_to_bias,
    merge_tokenized_segments,
    padding,
)

# Chatterbox uses a specific log_mel_spectrogram that matches PyTorch's torch.stft behavior
# (drops the last frame). We keep this local implementation for compatibility.
from .utils import log_mel_spectrogram

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
