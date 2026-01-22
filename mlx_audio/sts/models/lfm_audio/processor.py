# Copyright (c) 2025 Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)
# LFM2.5-Audio: Processor for audio and text

import json
import math
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download

from mlx_audio.codec.models.mimi import Mimi, MimiStreamingDecoder
from mlx_audio.codec.models.mimi.mimi import MimiConfig, mimi_202407
from mlx_audio.dsp import STR_TO_WINDOW_FN, mel_filters, stft

from .config import LFM2AudioConfig, PreprocessorConfig
from .detokenizer import LFM2AudioDetokenizer


class LFMModality(IntEnum):
    """Modality types for LFM2 Audio.

    Note: Values 1, 2, 3 match PyTorch implementation (0 is reserved/unused).
    """

    TEXT = 1
    AUDIO_IN = 2
    AUDIO_OUT = 3


class AudioPreprocessor(nn.Module):
    """Preprocessor for converting audio to mel spectrogram features."""

    def __init__(self, config: PreprocessorConfig):
        super().__init__()
        self.config = config

        # Precompute mel filterbank
        # Use slaney mel scale to match PyTorch/NeMo (not HTK)
        self._mel_filters = mel_filters(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            n_mels=config.features,
            f_min=0.0,
            f_max=config.sample_rate // 2,
            norm="slaney",
            mel_scale="slaney",
        )

    @property
    def hop_length(self) -> int:
        return int(self.config.sample_rate * self.config.window_stride)

    @property
    def win_length(self) -> int:
        return int(self.config.sample_rate * self.config.window_size)

    def __call__(self, audio: mx.array) -> mx.array:
        """
        Convert audio waveform to mel spectrogram features.

        Args:
            audio: Audio waveform (B, T) or (T,)

        Returns:
            Mel spectrogram features (B, T', features) or (T', features)
        """
        single_input = audio.ndim == 1
        if single_input:
            audio = audio[None, :]

        B = audio.shape[0]
        features_list = []

        for i in range(B):
            # Add dithering
            waveform = audio[i]
            if self.config.dither > 0:
                waveform = waveform + self.config.dither * mx.random.normal(
                    waveform.shape
                )

            # Pre-emphasis high-pass filter: y[n] = x[n] - preemph * x[n-1]
            if self.config.preemph > 0:
                waveform = mx.concatenate(
                    [waveform[:1], waveform[1:] - self.config.preemph * waveform[:-1]]
                )

            # STFT (use constant padding to match PyTorch/NeMo)
            spec = stft(
                waveform,
                n_fft=self.config.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.config.window,
                center=True,
                pad_mode="constant",
            )

            # Power spectrum
            power_spec = mx.abs(spec) ** 2

            # Apply mel filterbank - transpose to match dimensions
            # power_spec: (T, n_fft//2+1), mel_filters: (n_mels, n_fft//2+1)
            # We need (T, n_mels) so use mel_filters.T
            mel_spec = power_spec @ self._mel_filters.T

            # Log mel (use add guard like PyTorch, not max guard)
            if self.config.log:
                log_zero_guard = 5.96e-8  # Same as PyTorch (2^-24)
                mel_spec = mx.log(mel_spec + log_zero_guard)

            # Normalize with Bessel's correction (ddof=1) to match PyTorch
            # Note: PyTorch computes seq_len differently and excludes last frame from normalization
            if self.config.normalize == "per_feature":
                # Compute valid sequence length (matches PyTorch's get_seq_len)
                # seq_len = floor((audio_len + n_fft - n_fft) / hop_length) = audio_len / hop_length
                valid_frames = len(waveform) // self.hop_length
                n = min(valid_frames, mel_spec.shape[0])

                # Compute mean/std only over valid frames
                valid_mel = mel_spec[:n]
                mean = mx.mean(valid_mel, axis=0, keepdims=True)
                # Bessel's correction: divide by (n-1) instead of n
                variance = mx.sum((valid_mel - mean) ** 2, axis=0, keepdims=True) / (
                    n - 1
                )
                std = mx.sqrt(variance) + 1e-5
                # Apply normalization to ALL frames (including last one)
                mel_spec = (mel_spec - mean) / std

            features_list.append(mel_spec)

        features = mx.stack(features_list, axis=0)

        if single_input:
            return features[0]

        return features


class LFM2AudioProcessor:
    """
    Processor for LFM2.5-Audio model.

    Handles:
    - Text tokenization
    - Audio preprocessing (mel spectrogram)
    - Audio tokenization (Mimi codec)
    - Audio detokenization
    """

    def __init__(
        self,
        config: LFM2AudioConfig,
        tokenizer: Optional[Any] = None,
        mimi: Optional[Mimi] = None,
        detokenizer: Optional[LFM2AudioDetokenizer] = None,
    ):
        self.config = config

        # Text tokenizer (lazy loaded)
        self._tokenizer = tokenizer

        # Audio preprocessor for mel features
        self.audio_preprocessor = AudioPreprocessor(config.preprocessor)

        # Mimi codec for audio tokenization (lazy loaded)
        self._mimi = mimi

        # Detokenizer for audio output (lazy loaded)
        self._detokenizer = detokenizer

        self.model_path = None

    @property
    def tokenizer(self):
        """Lazy load tokenizer."""
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                )
            except ImportError:
                raise ImportError(
                    "transformers is required for text tokenization. "
                    "Install with: pip install transformers"
                )
        return self._tokenizer

    @property
    def mimi(self) -> Mimi:
        """Lazy load Mimi codec."""
        if self._mimi is None:
            # The checkpoint has 32 codebooks (Kyutai's full Mimi)
            # LFM2.5-Audio uses only the first 8 codebooks
            cfg = mimi_202407(num_codebooks=32)
            self._mimi = Mimi(cfg)
            # Load pretrained weights
            model_file = (
                self.model_path / "tokenizer-e351c8d8-checkpoint125.safetensors"
            )
            # Use strict=False to skip training-only params (cluster_usage, embedding_sum, initialized)
            self._mimi.load_pytorch_weights(str(model_file), strict=False)
        return self._mimi

    @property
    def detokenizer(self) -> LFM2AudioDetokenizer:
        """Lazy load detokenizer with pretrained weights."""
        if self._detokenizer is None:
            self._detokenizer = LFM2AudioDetokenizer.from_pretrained(self.model_path)
        return self._detokenizer

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
    ) -> "LFM2AudioProcessor":
        """Load processor from pretrained model."""
        # Download or get local path
        if Path(model_name_or_path).exists():
            model_path = Path(model_name_or_path)
        else:
            model_path = Path(
                snapshot_download(
                    model_name_or_path,
                    allow_patterns=["*.json", "*.safetensors", "tokenizer*"],
                )
            )

        # Load config
        config_path = model_path / "config.json"
        with open(config_path) as f:
            config_dict = json.load(f)
        config = LFM2AudioConfig.from_dict(config_dict)
        processor = cls(config)
        processor.model_path = model_path

        return processor

    def preprocess_audio(
        self,
        audio: mx.array,
        sample_rate: int = 16000,
    ) -> mx.array:
        """
        Preprocess audio to mel spectrogram features.

        Args:
            audio: Audio waveform (B, T) or (T,)
            sample_rate: Input sample rate

        Returns:
            Mel features (B, T', features) or (T', features)
        """
        # Resample if needed
        if sample_rate != self.config.preprocessor.sample_rate:
            audio = self._resample(
                audio, sample_rate, self.config.preprocessor.sample_rate
            )

        return self.audio_preprocessor(audio)

    def tokenize_audio(self, audio: mx.array, sample_rate: int = 24000) -> mx.array:
        """
        Tokenize audio waveform using Mimi codec.

        Args:
            audio: Audio waveform (B, 1, T) or (1, T) or (T,)
            sample_rate: Input sample rate

        Returns:
            Audio codes (B, num_codebooks, T')
        """
        # Ensure correct shape: (B, 1, T)
        if audio.ndim == 1:
            audio = audio[None, None, :]
        elif audio.ndim == 2:
            audio = audio[None, :]

        # Resample if needed
        if sample_rate != int(self.mimi.sample_rate):
            audio = self._resample(audio, sample_rate, int(self.mimi.sample_rate))

        # Encode with Mimi
        codes = self.mimi.encode(audio)

        return codes

    def decode_audio(
        self, codes: mx.array, codec: Optional[str] = "detokenizer"
    ) -> mx.array:
        """
        Decode audio codes to waveform using LFM2 detokenizer or Mimi codec.

        Args:
            codes: Audio codes (B, num_codebooks, T) or (num_codebooks, T)
                   LFM2.5-Audio uses 8 codebooks
            codec: Decoder to use, either "detokenizer" or "mimi"
        Returns:
            Audio waveform (B, 1, T_audio) or (1, T_audio)
        """
        if codec == "detokenizer":
            return self.detokenizer(codes)
        elif codec == "mimi":
            return self.mimi.decode(codes)
        else:
            raise ValueError(f"Invalid codec: {codec}")

    def tokenize_text(self, text: str) -> mx.array:
        """
        Tokenize text.

        Args:
            text: Input text

        Returns:
            Token IDs as mx.array
        """
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        return mx.array(tokens)

    def format_chat(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        """
        Format chat messages using the tokenizer's chat template.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
                      Roles: 'system', 'user', 'assistant'
            add_generation_prompt: Whether to add assistant prompt for generation

        Returns:
            Formatted chat string
        """
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    def tokenize_chat(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> mx.array:
        """
        Format and tokenize chat messages.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            add_generation_prompt: Whether to add assistant prompt for generation

        Returns:
            Token IDs as mx.array
        """
        formatted = self.format_chat(messages, add_generation_prompt)
        tokens = self.tokenizer.encode(formatted, add_special_tokens=False)
        return mx.array(tokens)

    def decode_text(self, tokens: Union[mx.array, List[int]]) -> str:
        """
        Decode text tokens.

        Args:
            tokens: Token IDs (B, T) or (T,)

        Returns:
            Decoded text string
        """

        if hasattr(tokens, "ndim"):
            tokens_ = tokens.squeeze().tolist() if tokens.ndim > 1 else tokens.tolist()
        elif isinstance(tokens, list):
            tokens_ = tokens
        else:
            raise ValueError(f"Invalid tokens type: {type(tokens)}")
        return self.tokenizer.decode(tokens_)

    def _resample(
        self,
        audio: mx.array,
        orig_sr: int,
        target_sr: int,
    ) -> mx.array:
        """Resample audio using scipy's high-quality polyphase resampling."""
        if orig_sr == target_sr:
            return audio

        import numpy as np
        from scipy import signal

        # Convert to numpy for scipy processing
        audio_np = np.array(audio)
        orig_dtype = audio_np.dtype

        # Calculate target length
        if audio_np.ndim == 1:
            orig_len = audio_np.shape[0]
            new_len = int(orig_len * target_sr / orig_sr)
            resampled_np = signal.resample_poly(
                audio_np.astype(np.float64),
                target_sr,
                orig_sr,
            ).astype(orig_dtype)
        elif audio_np.ndim == 2:
            # Shape: (channels, samples) or (batch, samples)
            orig_len = audio_np.shape[-1]
            new_len = int(orig_len * target_sr / orig_sr)
            resampled_np = signal.resample_poly(
                audio_np.astype(np.float64),
                target_sr,
                orig_sr,
                axis=-1,
            ).astype(orig_dtype)
        else:  # 3D: (batch, channels, samples)
            orig_len = audio_np.shape[-1]
            new_len = int(orig_len * target_sr / orig_sr)
            resampled_np = signal.resample_poly(
                audio_np.astype(np.float64),
                target_sr,
                orig_sr,
                axis=-1,
            ).astype(orig_dtype)

        return mx.array(resampled_np)


@dataclass
class ChatState:
    """
    State container for multi-turn conversations.

    Maintains parallel tensors for text tokens, audio input, audio output codes,
    and modality flags.
    """

    processor: LFM2AudioProcessor
    text_tokens: List[int]
    audio_features: Optional[mx.array]
    audio_out_codes: List[mx.array]
    modalities: List[LFMModality]
    current_turn: Optional[str]

    def __init__(self, processor: LFM2AudioProcessor, add_bos: bool = True):
        self.processor = processor
        self.text_tokens = []
        self.audio_features = None
        self.audio_out_codes = []
        self.modalities = []
        self.current_turn = None

        # Add BOS token at the start (token ID 1)
        if add_bos:
            bos_token_id = getattr(processor.tokenizer, "bos_token_id", 1)
            if bos_token_id is not None:
                self.text_tokens.append(bos_token_id)
                self.modalities.append(LFMModality.TEXT)

    def new_turn(self, role: str):
        """Start a new conversation turn."""
        self.current_turn = role

        # Add role tokens: <|im_start|>role\n
        # Note: tokenizer uses <|im_start|> (id=6) and <|im_end|> (id=7)
        turn_prefix = f"<|im_start|>{role}\n"
        tokens = self.processor.tokenizer.encode(turn_prefix, add_special_tokens=False)
        self.text_tokens.extend(tokens)

        # Add TEXT modality for each token (not based on difference, which breaks after audio)
        for _ in range(len(tokens)):
            self.modalities.append(LFMModality.TEXT)

    def end_turn(self):
        """End the current turn."""
        # Add <|im_end|>\n
        tokens = self.processor.tokenizer.encode(
            "<|im_end|>\n", add_special_tokens=False
        )
        self.text_tokens.extend(tokens)
        # Add TEXT modality for each token (not based on difference, which breaks after audio)
        for _ in range(len(tokens)):
            self.modalities.append(LFMModality.TEXT)
        self.current_turn = None

    def add_text(self, text: str):
        """Add text to the current turn."""
        tokens = self.processor.tokenizer.encode(text, add_special_tokens=False)
        self.text_tokens.extend(tokens)
        for _ in range(len(tokens)):
            self.modalities.append(LFMModality.TEXT)

    def add_audio(self, audio: mx.array, sample_rate: int = 16000):
        """Add audio to the current turn."""
        # Preprocess to mel features
        features = self.processor.preprocess_audio(audio, sample_rate)
        if self.audio_features is None:
            self.audio_features = features
        else:
            self.audio_features = mx.concatenate(
                [self.audio_features, features], axis=0
            )

        # Calculate the actual encoder output length after subsampling
        # Subsampling uses 3 stride-2 convolutions with kernel=3, padding=1
        # Formula: output = floor((input + 2*padding - kernel) / stride) + 1
        def calc_conv_output(input_len, kernel=3, stride=2, padding=1):
            return (input_len + 2 * padding - kernel) // stride + 1

        mel_frames = features.shape[0]
        t = calc_conv_output(mel_frames)  # After first stride-2 conv
        t = calc_conv_output(t)  # After second stride-2 conv
        t = calc_conv_output(t)  # After third stride-2 conv
        num_frames = t

        for _ in range(num_frames):
            self.modalities.append(LFMModality.AUDIO_IN)

    def append(self, token: mx.array, modality: LFMModality):
        """Append a generated token to the state."""
        if modality == LFMModality.TEXT:
            self.text_tokens.append(int(token.item()))
        elif modality == LFMModality.AUDIO_OUT:
            self.audio_out_codes.append(token)
        self.modalities.append(modality)

    def get_text_tokens(self) -> mx.array:
        """Get text tokens as tensor."""
        return mx.array(self.text_tokens)[None, :]

    def get_audio_features(self) -> Optional[mx.array]:
        """Get audio features as tensor."""
        if self.audio_features is None:
            return None
        if self.audio_features.ndim == 2:
            return self.audio_features[None, :]
        return self.audio_features

    def get_modalities(self) -> mx.array:
        """Get modality flags as tensor."""
        return mx.array([int(m) for m in self.modalities])[None, :]

    def __iter__(self):
        """Allow unpacking for model input."""
        return iter(
            [
                ("text_tokens", self.get_text_tokens()),
                ("audio_features", self.get_audio_features()),
                ("modalities", self.get_modalities()),
            ]
        )

    def items(self):
        """Dict-like items for model input."""
        return [
            ("text_tokens", self.get_text_tokens()),
            ("audio_features", self.get_audio_features()),
            ("modalities", self.get_modalities()),
        ]
