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
from huggingface_hub import hf_hub_download, snapshot_download

from mlx_audio.codec.models.mimi import Mimi, MimiStreamingDecoder
from mlx_audio.codec.models.mimi.mimi import MimiConfig, mimi_202407
from mlx_audio.dsp import mel_filters, stft, STR_TO_WINDOW_FN

from .config import LFM2AudioConfig, PreprocessorConfig


class LFMModality(IntEnum):
    """Modality types for LFM2 Audio."""
    TEXT = 0
    AUDIO_IN = 1
    AUDIO_OUT = 2


class AudioPreprocessor(nn.Module):
    """Preprocessor for converting audio to mel spectrogram features."""

    def __init__(self, config: PreprocessorConfig):
        super().__init__()
        self.config = config

        # Precompute mel filterbank
        self._mel_filters = mel_filters(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            n_mels=config.features,
            f_min=0.0,
            f_max=config.sample_rate // 2,
            norm="slaney",
            mel_scale="htk",
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
                waveform = waveform + self.config.dither * mx.random.normal(waveform.shape)

            # STFT
            spec = stft(
                waveform,
                n_fft=self.config.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.config.window,
                center=True,
            )

            # Power spectrum
            power_spec = mx.abs(spec) ** 2

            # Apply mel filterbank - transpose to match dimensions
            # power_spec: (T, n_fft//2+1), mel_filters: (n_mels, n_fft//2+1)
            # We need (T, n_mels) so use mel_filters.T
            mel_spec = power_spec @ self._mel_filters.T

            # Log mel
            if self.config.log:
                mel_spec = mx.log(mx.maximum(mel_spec, 1e-10))

            # Normalize
            if self.config.normalize == "per_feature":
                mean = mx.mean(mel_spec, axis=0, keepdims=True)
                std = mx.std(mel_spec, axis=0, keepdims=True) + 1e-5
                mel_spec = (mel_spec - mean) / std

            features_list.append(mel_spec)

        features = mx.stack(features_list, axis=0)

        if single_input:
            return features[0]

        return features


class LFM2AudioDetokenizer(nn.Module):
    """
    Audio detokenizer that converts audio codes to waveforms.

    Uses the LFM detokenizer architecture: embeddings -> upsample -> transformer -> ISTFT
    """

    def __init__(
        self,
        dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        num_codebooks: int = 8,
        vocab_size: int = 2048,
        n_fft: int = 1280,
        hop_length: int = 320,
        win_length: int = 1280,
        sliding_window: int = 30,
    ):
        super().__init__()
        self.dim = dim
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sliding_window = sliding_window
        self.num_codebooks = num_codebooks

        # Fused embedding for codebooks (sum of individual embeddings)
        self.embeddings = [nn.Embedding(vocab_size, dim) for _ in range(num_codebooks)]

        # Simple transformer layers
        from .transformer import TransformerBlock

        self.layers = [
            TransformerBlock(
                dim=dim,
                num_heads=num_heads,
                num_kv_heads=num_heads,
                ff_dim=dim * 4,
                max_seq_len=4096,
                rope_theta=10000.0,
            )
            for _ in range(num_layers)
        ]

        # Output projection to STFT bins
        self.out_proj = nn.Linear(dim, n_fft + 2)

    def _embed_codes(self, codes: mx.array) -> mx.array:
        """Embed audio codes by summing embeddings from all codebooks."""
        # codes: (B, num_codebooks, T)
        B, K, T = codes.shape
        embedded = mx.zeros((B, T, self.dim))

        for i, emb in enumerate(self.embeddings):
            if i < K:
                embedded = embedded + emb(codes[:, i, :])

        return embedded

    def __call__(self, codes: mx.array) -> mx.array:
        """
        Convert audio codes to waveform.

        Args:
            codes: (B, num_codebooks, T) with values in [0, vocab_size)

        Returns:
            Waveform (B, T_audio)
        """
        # Embed codes
        x = self._embed_codes(codes)  # (B, T, dim)

        # Upsample by 6x using nearest neighbor
        B, T, D = x.shape
        upsample_size = 6 * T
        # Transpose: (B, T, D) -> (B, D, T)
        x = x.transpose(0, 2, 1)
        # Nearest neighbor upsample
        indices = mx.arange(upsample_size) // 6
        x = x[:, :, indices]
        # Transpose back: (B, D, T') -> (B, T', D)
        x = x.transpose(0, 2, 1)

        # Create sliding window attention mask
        T_new = x.shape[1]
        idx = mx.arange(T_new)
        d_idx = idx[None, :] - idx[:, None]
        mask = mx.logical_and(d_idx <= 0, d_idx > -self.sliding_window)
        mask = mx.where(mask, 0.0, float("-inf"))
        mask = mx.expand_dims(mask, axis=(0, 1))

        # Apply transformer layers
        for layer in self.layers:
            x, _ = layer(x, mask=mask)

        # Project to STFT space
        x = self.out_proj(x)  # (B, T', n_fft + 2)

        # Split into magnitude and phase
        n_bins = self.n_fft // 2 + 1
        log_mag = x[:, :, :n_bins]
        angle = x[:, :, n_bins:]

        # Reconstruct magnitude
        mag = mx.exp(log_mag)

        # ISTFT reconstruction using DSP utilities
        waveform = self._istft(mag, angle)

        return waveform

    def _istft(self, mag: mx.array, angle: mx.array) -> mx.array:
        """Inverse STFT to reconstruct waveform using overlap-add."""
        from mlx_audio.dsp import hanning

        B, T, F = mag.shape

        # Create complex STFT
        real = mag * mx.cos(angle)
        imag = mag * mx.sin(angle)

        # Get window
        window = hanning(self.win_length, periodic=True)
        if window.shape[0] < self.n_fft:
            pad = self.n_fft - window.shape[0]
            window = mx.concatenate([window, mx.zeros((pad,))])

        # Overlap-add reconstruction
        output_length = (T - 1) * self.hop_length + self.win_length
        output = mx.zeros((B, output_length))
        window_sum = mx.zeros((output_length,))

        # Prepare for IRFFT
        stft_complex = real + 1j * imag

        # IRFFT each frame
        time_frames = mx.fft.irfft(stft_complex, n=self.n_fft, axis=-1)

        # Overlap-add
        for t in range(T):
            start = t * self.hop_length
            end = start + self.n_fft

            # Add windowed frame
            frame = time_frames[:, t, :] * window
            output = output.at[:, start:end].add(frame)
            window_sum = window_sum.at[start:end].add(window**2)

        # Normalize
        window_sum = mx.maximum(window_sum, 1e-8)
        output = output / window_sum[None, :]

        # Trim center padding
        trim = self.n_fft // 2
        output = output[:, trim:-trim] if trim > 0 else output

        return output


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

    @property
    def tokenizer(self):
        """Lazy load tokenizer."""
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(
                    "LiquidAI/LFM2.5-Audio-1.5B",
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
            model_file = hf_hub_download(
                "LiquidAI/LFM2.5-Audio-1.5B",
                "tokenizer-e351c8d8-checkpoint125.safetensors"
            )
            # Use strict=False to skip training-only params (cluster_usage, embedding_sum, initialized)
            self._mimi.load_pytorch_weights(model_file, strict=False)
        return self._mimi

    @property
    def detokenizer(self) -> LFM2AudioDetokenizer:
        """Lazy load detokenizer."""
        if self._detokenizer is None:
            self._detokenizer = LFM2AudioDetokenizer(
                dim=512,
                num_layers=4,
                num_heads=8,
                num_codebooks=self.config.codebooks,
                vocab_size=self.config.audio_vocab_size - 1,  # Exclude padding token
            )
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

        return cls(config)

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
            audio = self._resample(audio, sample_rate, self.config.preprocessor.sample_rate)

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

    def decode_audio(self, codes: mx.array) -> mx.array:
        """
        Decode audio codes to waveform using Mimi codec.

        Args:
            codes: Audio codes (B, num_codebooks, T) or (num_codebooks, T)
                   LFM2.5-Audio uses 8 codebooks

        Returns:
            Audio waveform (B, 1, T_audio) or (1, T_audio)
        """
        single_input = codes.ndim == 2
        if single_input:
            codes = codes[None, :]

        # LFM2.5-Audio uses 8 codebooks, Mimi has 32
        # Pad with zeros for unused codebooks
        B, K, T = codes.shape
        if K < 32:
            # Pad to 32 codebooks with zeros
            padding = mx.zeros((B, 32 - K, T), dtype=codes.dtype)
            codes = mx.concatenate([codes, padding], axis=1)

        # Decode with Mimi
        audio = self.mimi.decode(codes)

        if single_input:
            return audio[0]

        return audio

    def decode_with_detokenizer(self, codes: mx.array) -> mx.array:
        """
        Decode audio codes using the LFM detokenizer (ISTFT-based).

        Args:
            codes: Audio codes (B, num_codebooks, T)

        Returns:
            Audio waveform (B, T_audio)
        """
        return self.detokenizer(codes)

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

    def decode_text(self, tokens: mx.array) -> str:
        """
        Decode text tokens.

        Args:
            tokens: Token IDs (B, T) or (T,)

        Returns:
            Decoded text string
        """
        if tokens.ndim == 2:
            tokens = tokens[0]

        return self.tokenizer.decode(tokens.tolist())

    def _resample(
        self,
        audio: mx.array,
        orig_sr: int,
        target_sr: int,
    ) -> mx.array:
        """Simple resampling using linear interpolation."""
        if orig_sr == target_sr:
            return audio

        # Get original length
        if audio.ndim == 3:
            orig_len = audio.shape[-1]
        elif audio.ndim == 2:
            orig_len = audio.shape[-1]
        else:
            orig_len = audio.shape[0]

        # Calculate new length
        new_len = int(orig_len * target_sr / orig_sr)

        # Create interpolation indices
        indices = mx.arange(new_len) * (orig_len - 1) / (new_len - 1)
        idx_low = mx.floor(indices).astype(mx.int32)
        idx_high = mx.minimum(idx_low + 1, orig_len - 1)
        weights = indices - idx_low.astype(mx.float32)

        # Interpolate
        if audio.ndim == 1:
            resampled = audio[idx_low] * (1 - weights) + audio[idx_high] * weights
        elif audio.ndim == 2:
            resampled = audio[:, idx_low] * (1 - weights) + audio[:, idx_high] * weights
        else:
            resampled = audio[:, :, idx_low] * (1 - weights) + audio[:, :, idx_high] * weights

        return resampled


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

    def __init__(self, processor: LFM2AudioProcessor):
        self.processor = processor
        self.text_tokens = []
        self.audio_features = None
        self.audio_out_codes = []
        self.modalities = []
        self.current_turn = None

    def new_turn(self, role: str):
        """Start a new conversation turn."""
        self.current_turn = role

        # Add role tokens: <|im_start|>role\n
        # Note: tokenizer uses <|im_start|> (id=6) and <|im_end|> (id=7)
        turn_prefix = f"<|im_start|>{role}\n"
        self.text_tokens.extend(
            self.processor.tokenizer.encode(turn_prefix, add_special_tokens=False)
        )

        for _ in range(len(self.text_tokens) - len(self.modalities)):
            self.modalities.append(LFMModality.TEXT)

    def end_turn(self):
        """End the current turn."""
        # Add <|im_end|>\n
        self.text_tokens.extend(
            self.processor.tokenizer.encode("<|im_end|>\n", add_special_tokens=False)
        )
        for _ in range(len(self.text_tokens) - len(self.modalities)):
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
            self.audio_features = mx.concatenate([self.audio_features, features], axis=0)

        # Add audio in modality markers
        num_frames = features.shape[0] // self.processor.config.encoder.subsampling_factor
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
        return iter([
            ("text_tokens", self.get_text_tokens()),
            ("audio_features", self.get_audio_features()),
            ("modalities", self.get_modalities()),
        ])

    def items(self):
        """Dict-like items for model input."""
        return [
            ("text_tokens", self.get_text_tokens()),
            ("audio_features", self.get_audio_features()),
            ("modalities", self.get_modalities()),
        ]
