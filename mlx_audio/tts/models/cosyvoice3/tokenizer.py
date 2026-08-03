"""Speech tokenizer for CosyVoice3 (waveform -> discrete speech tokens)."""

from typing import Optional, Tuple

import mlx.core as mx


class SpeechTokenizer:
    """Lazy wrapper around the configured speech-tokenizer implementation."""

    def __init__(self, name: str = "speech_tokenizer_v2_25hz", use_onnx: bool = False):
        self.name = name
        self.use_onnx = use_onnx
        self._impl = None  # lazy

    def _ensure(self):
        if self._impl is not None:
            return
        if self.use_onnx:
            raise NotImplementedError("onnxruntime speech tokenizer is not implemented")
        from mlx_audio.codec.models.s3 import S3TokenizerV2

        self._impl = S3TokenizerV2.from_pretrained(self.name)
        mx.eval(self._impl.parameters())

    def quantize(self, audio_16k: mx.array) -> Tuple[mx.array, mx.array]:
        """audio (16k mono) -> (token_ids [1, N], token_lens [1])."""
        self._ensure()
        from mlx_audio.codec.models.s3 import log_mel_spectrogram

        mels = log_mel_spectrogram(audio_16k)
        mels = mx.expand_dims(mels, 0)
        mel_lens = mx.array([mels.shape[-1]], dtype=mx.int32)
        return self._impl.quantize(mels, mel_lens)
