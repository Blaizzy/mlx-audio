# Copyright (c) 2025 Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

from typing import List, Optional, Union

import mlx.core as mx
import numpy as np


class FunAudioChatProcessor:
    """Audio processor for FunAudioChat model.

    Handles audio preprocessing including:
    - Loading audio files
    - Resampling
    - Computing mel spectrograms
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
        n_mels: int = 128,
        chunk_length: int = 30,
    ):
        """
        Initialize processor.

        Args:
            sample_rate: Target sample rate (default 16kHz)
            n_fft: FFT window size
            hop_length: Hop length for STFT
            n_mels: Number of mel frequency bins
            chunk_length: Maximum chunk length in seconds
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.chunk_length = chunk_length
        self.n_samples = chunk_length * sample_rate

    def load_audio(self, audio_path: str) -> mx.array:
        """Load audio file and resample to target sample rate.

        Args:
            audio_path: Path to audio file

        Returns:
            Audio waveform as MLX array
        """
        from mlx_audio.stt.utils import load_audio

        return load_audio(audio_path, sr=self.sample_rate)

    def compute_mel_spectrogram(
        self,
        audio: Union[mx.array, np.ndarray],
        padding: str = "longest",
    ) -> mx.array:
        """Compute log-mel spectrogram from audio waveform.

        Args:
            audio: Audio waveform
            padding: Padding strategy ("longest" or "max_length")

        Returns:
            Log-mel spectrogram (batch, seq_len, n_mels)
        """
        from mlx_audio.utils import hanning, mel_filters, stft

        if isinstance(audio, np.ndarray):
            audio = mx.array(audio)

        # Ensure 1D
        if audio.ndim == 2:
            audio = audio.mean(axis=-1)

        # Pad or truncate
        if padding == "max_length":
            if audio.shape[0] > self.n_samples:
                audio = audio[: self.n_samples]
            elif audio.shape[0] < self.n_samples:
                pad_length = self.n_samples - audio.shape[0]
                audio = mx.pad(audio, [(0, pad_length)])

        # Compute STFT
        window = hanning(self.n_fft)
        freqs = stft(audio, window=window, n_fft=self.n_fft, hop_length=self.hop_length)

        # Compute magnitude spectrogram
        magnitudes = freqs[:-1, :].abs().square()

        # Apply mel filterbank
        filters = mel_filters(
            self.sample_rate, self.n_fft, self.n_mels, norm="slaney", mel_scale=None
        )
        mel_spec = magnitudes @ filters.T

        # Convert to log scale
        log_spec = mx.maximum(mel_spec, 1e-10).log10()
        log_spec = mx.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0

        return log_spec[None]  # Add batch dimension

    def __call__(
        self,
        audio: Optional[Union[str, mx.array, np.ndarray, List]] = None,
        text: Optional[Union[str, List[str]]] = None,
        padding: str = "longest",
        return_tensors: str = "mlx",
    ) -> dict:
        """Process audio and/or text inputs.

        Args:
            audio: Audio input (path, waveform, or list)
            text: Text input
            padding: Padding strategy
            return_tensors: Return format ("mlx" or "np")

        Returns:
            Dictionary with processed inputs
        """
        result = {}

        if audio is not None:
            if isinstance(audio, str):
                audio = self.load_audio(audio)
            elif isinstance(audio, list):
                audio = [self.load_audio(a) if isinstance(a, str) else mx.array(a) for a in audio]
                # Stack into batch
                max_len = max(a.shape[0] for a in audio)
                padded = []
                for a in audio:
                    if a.shape[0] < max_len:
                        a = mx.pad(a, [(0, max_len - a.shape[0])])
                    padded.append(a)
                audio = mx.stack(padded)

            if isinstance(audio, mx.array) and audio.ndim == 1:
                mel = self.compute_mel_spectrogram(audio, padding=padding)
            elif isinstance(audio, mx.array) and audio.ndim == 2:
                # Batch processing
                mels = []
                for i in range(audio.shape[0]):
                    mel = self.compute_mel_spectrogram(audio[i], padding=padding)
                    mels.append(mel)
                mel = mx.concatenate(mels, axis=0)
            else:
                mel = audio

            result["input_features"] = mel

        if text is not None:
            result["text"] = text

        return result

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> "FunAudioChatProcessor":
        """Load processor from pretrained model.

        Args:
            model_path: Path to model directory

        Returns:
            Processor instance
        """
        import json
        from pathlib import Path
        from huggingface_hub import hf_hub_download

        # Try to load processor config
        try:
            if Path(model_path).exists():
                config_path = Path(model_path) / "preprocessor_config.json"
            else:
                config_path = hf_hub_download(
                    repo_id=model_path, filename="preprocessor_config.json"
                )

            with open(config_path, "r") as f:
                config = json.load(f)

            return cls(
                sample_rate=config.get("sampling_rate", 16000),
                n_fft=config.get("n_fft", 400),
                hop_length=config.get("hop_length", 160),
                n_mels=config.get("feature_size", 128),
                chunk_length=config.get("chunk_length", 30),
            )
        except Exception:
            # Return default processor
            return cls()
