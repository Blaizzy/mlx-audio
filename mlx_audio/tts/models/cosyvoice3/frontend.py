"""
CosyVoice3 Frontend for text processing and speaker embedding.

Based on: https://github.com/FunAudioLLM/CosyVoice
"""

from pathlib import Path
from typing import Optional, Tuple

import mlx.core as mx
import numpy as np

try:
    import onnxruntime as ort

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    from transformers import AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import librosa

    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


class CosyVoice3Frontend:
    """
    Frontend for CosyVoice3 text-to-speech.

    Handles:
    - Text tokenization using the CosyVoice tokenizer
    - Speaker embedding extraction using CAMPPlus
    - Audio feature extraction for prompts
    """

    def __init__(
        self,
        tokenizer_path: Optional[str] = None,
        campplus_path: Optional[str] = None,
        speech_tokenizer_path: Optional[str] = None,
        sample_rate: int = 24000,
    ):
        """
        Initialize the frontend.

        Args:
            tokenizer_path: Path to tokenizer (HuggingFace format)
            campplus_path: Path to CAMPPlus ONNX model
            speech_tokenizer_path: Path to speech tokenizer ONNX model
            sample_rate: Audio sample rate
        """
        self.sample_rate = sample_rate
        self.tokenizer = None
        self.campplus_session = None
        self.speech_tokenizer_session = None

        # Load tokenizer
        if tokenizer_path and TRANSFORMERS_AVAILABLE:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # Load CAMPPlus for speaker embedding
        if campplus_path and ONNX_AVAILABLE:
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            self.campplus_session = ort.InferenceSession(
                campplus_path, sess_options, providers=["CPUExecutionProvider"]
            )

        # Load speech tokenizer
        if speech_tokenizer_path and ONNX_AVAILABLE:
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            self.speech_tokenizer_session = ort.InferenceSession(
                speech_tokenizer_path, sess_options, providers=["CPUExecutionProvider"]
            )

    def tokenize(self, text: str) -> mx.array:
        """
        Tokenize text using the CosyVoice tokenizer.

        Args:
            text: Input text

        Returns:
            Token IDs as mx.array (1, T)
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Please provide tokenizer_path.")

        # CosyVoice uses special tokens
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        return mx.array([tokens], dtype=mx.int32)

    def extract_speaker_embedding(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> mx.array:
        """
        Extract speaker embedding from audio using CAMPPlus.

        Args:
            audio: Audio waveform (numpy array)
            sample_rate: Audio sample rate (should be 16000 for CAMPPlus)

        Returns:
            Speaker embedding as mx.array (1, 192)
        """
        if self.campplus_session is None:
            raise RuntimeError("CAMPPlus not loaded. Please provide campplus_path.")

        if not LIBROSA_AVAILABLE:
            raise RuntimeError(
                "librosa required for audio processing. Install with: pip install librosa"
            )

        # Resample to 16kHz if needed
        if sample_rate != 16000:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)

        # Compute mel filterbank features (80 bins, like Kaldi)
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=16000,
            n_fft=512,
            hop_length=160,
            n_mels=80,
            fmin=20,
            fmax=7600,
        )

        # Log mel
        log_mel = np.log(np.maximum(mel_spec, 1e-6))

        # Normalize (remove mean)
        log_mel = log_mel - np.mean(log_mel, axis=0, keepdims=True)

        # Transpose to (T, 80) and add batch dimension
        features = log_mel.T[np.newaxis, :, :].astype(np.float32)

        # Run CAMPPlus
        input_name = self.campplus_session.get_inputs()[0].name
        outputs = self.campplus_session.run(None, {input_name: features})
        embedding = outputs[0].flatten()

        return mx.array(embedding[np.newaxis, :])

    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio from file.

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (audio_waveform, sample_rate)
        """
        if not LIBROSA_AVAILABLE:
            raise RuntimeError("librosa required. Install with: pip install librosa")

        audio, sr = librosa.load(audio_path, sr=None)
        return audio, sr

    def extract_mel_features(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> mx.array:
        """
        Extract mel spectrogram features for prompt audio.

        Uses hop_length=480 to match CosyVoice3 training config.
        50 fps (24000/480) = 25 tokens/s * token_mel_ratio (2)

        Args:
            audio: Audio waveform
            sample_rate: Audio sample rate

        Returns:
            Mel features as mx.array (1, 80, T)
        """
        if not LIBROSA_AVAILABLE:
            raise RuntimeError("librosa required. Install with: pip install librosa")

        # Resample to model sample rate if needed
        if sample_rate != self.sample_rate:
            audio = librosa.resample(
                audio, orig_sr=sample_rate, target_sr=self.sample_rate
            )

        # Compute mel spectrogram matching CosyVoice3 training config
        # n_fft=1920, hop_size=480, win_size=1920 from cosyvoice3.yaml
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=1920,
            hop_length=480,
            win_length=1920,
            n_mels=80,
            fmin=0,
            fmax=self.sample_rate // 2,
        )

        # Log mel
        log_mel = np.log(np.maximum(mel_spec, 1e-6))

        # Add batch dimension: (80, T) -> (1, 80, T)
        return mx.array(log_mel[np.newaxis, :, :].astype(np.float32))

    def extract_speech_tokens(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> mx.array:
        """
        Extract speech tokens from audio using the speech tokenizer.

        Uses whisper-style 128-bin log mel spectrogram.

        Args:
            audio: Audio waveform (numpy array)
            sample_rate: Audio sample rate

        Returns:
            Speech tokens as mx.array (1, T)
        """
        if self.speech_tokenizer_session is None:
            raise RuntimeError(
                "Speech tokenizer not loaded. Please provide speech_tokenizer_path."
            )

        if not LIBROSA_AVAILABLE:
            raise RuntimeError("librosa required. Install with: pip install librosa")

        # Resample to 16kHz (whisper uses 16kHz)
        if sample_rate != 16000:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)

        # Limit to 30 seconds
        max_samples = 30 * 16000
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        # Compute 128-bin log mel spectrogram (whisper-style)
        # whisper uses: n_fft=400, hop_length=160, n_mels=128
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=16000,
            n_fft=400,
            hop_length=160,
            n_mels=128,
            fmin=0,
            fmax=8000,
        )

        # Log mel (whisper uses log10 with floor)
        log_mel = np.log10(np.maximum(mel_spec, 1e-10))

        # Normalize (whisper normalizes to [-1, 1] range approximately)
        log_mel = (log_mel + 4.0) / 4.0  # Approximate whisper normalization

        # Shape: (1, 128, T) for ONNX
        feats = log_mel[np.newaxis, :, :].astype(np.float32)
        feats_length = np.array([feats.shape[2]], dtype=np.int32)

        # Run speech tokenizer
        outputs = self.speech_tokenizer_session.run(
            None, {"feats": feats, "feats_length": feats_length}
        )
        tokens = outputs[0].flatten()

        return mx.array(tokens[np.newaxis, :], dtype=mx.int32)

    def frontend_zero_shot(
        self,
        text: str,
        prompt_text: str,
        prompt_audio_path: str,
    ) -> dict:
        """
        Prepare inputs for zero-shot inference.

        Args:
            text: Text to synthesize
            prompt_text: Transcript of the prompt audio
            prompt_audio_path: Path to the prompt audio file

        Returns:
            Dictionary with model inputs
        """
        # Tokenize text
        text_tokens = self.tokenize(text)
        prompt_text_tokens = self.tokenize(prompt_text)

        # Load and process prompt audio
        prompt_audio, prompt_sr = self.load_audio(prompt_audio_path)

        # Extract speaker embedding
        speaker_embedding = self.extract_speaker_embedding(prompt_audio, prompt_sr)

        # Extract mel features for prompt (for Flow model conditioning)
        prompt_mel = self.extract_mel_features(prompt_audio, prompt_sr)

        # Extract speech tokens from prompt audio (for LLM context)
        prompt_speech_tokens = None
        if self.speech_tokenizer_session is not None:
            prompt_speech_tokens = self.extract_speech_tokens(prompt_audio, prompt_sr)

        return {
            "text_tokens": text_tokens,
            "prompt_text_tokens": prompt_text_tokens,
            "speaker_embedding": speaker_embedding,
            "prompt_mel": prompt_mel,
            "prompt_speech_tokens": prompt_speech_tokens,
        }
