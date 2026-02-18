import math
from dataclasses import dataclass
from typing import Dict, Optional, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_audio.audio_io import read as audio_read
from mlx_audio.stt.models.whisper.audio import log_mel_spectrogram

from .config import EncoderConfig, ModelConfig


@dataclass
class EndpointOutput:
    prediction: int
    probability: float


class WhisperAttention(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.num_heads = config.encoder_attention_heads
        self.head_dim = config.d_model // config.encoder_attention_heads
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=True)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=config.k_proj_bias)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=True)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        bsz, seq_len, _ = x.shape

        q = self.q_proj(x).reshape(bsz, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(bsz, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(bsz, seq_len, self.num_heads, self.head_dim)

        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 3, 1))
        v = mx.transpose(v, (0, 2, 1, 3))

        attn = (q @ k) / math.sqrt(self.head_dim)
        attn = mx.softmax(attn, axis=-1, precise=True)

        out = attn @ v
        out = mx.transpose(out, (0, 2, 1, 3)).reshape(bsz, seq_len, -1)
        return self.out_proj(out)


class WhisperEncoderLayer(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.self_attn = WhisperAttention(config)
        self.fc1 = nn.Linear(config.d_model, config.encoder_ffn_dim, bias=True)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, config.d_model, bias=True)
        self.final_layer_norm = nn.LayerNorm(config.d_model)

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(x)
        x = x + residual

        residual = x
        x = self.final_layer_norm(x)
        x = self.fc2(nn.gelu(self.fc1(x)))
        x = x + residual
        return x


class WhisperEncoder(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        self.conv1 = nn.Conv1d(
            config.num_mel_bins, config.d_model, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv1d(
            config.d_model, config.d_model, kernel_size=3, stride=2, padding=1
        )
        self.embed_positions = nn.Embedding(config.max_source_positions, config.d_model)
        self.layers = [
            WhisperEncoderLayer(config) for _ in range(config.encoder_layers)
        ]
        self.layer_norm = nn.LayerNorm(config.d_model)

    def __call__(self, input_features: mx.array) -> mx.array:
        # HF Whisper expects (batch, n_mels, n_frames). MLX Conv1d uses
        # (batch, n_frames, channels), so we transpose before conv.
        x = mx.transpose(input_features, (0, 2, 1))
        x = nn.gelu(self.conv1(x))
        x = nn.gelu(self.conv2(x))

        positions = mx.arange(x.shape[1], dtype=mx.int32)[None, :]
        x = x + self.embed_positions(positions)

        for layer in self.layers:
            x = layer(x)

        return self.layer_norm(x)


class Model(nn.Module):
    """
    Smart Turn endpoint detector (Whisper encoder + attention pooling + MLP).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.dtype = mx.float16 if config.dtype == "float16" else mx.float32

        d_model = config.encoder_config.d_model
        self.encoder = WhisperEncoder(config.encoder_config)

        # Named with suffixes to keep MLX paths valid and predictable.
        self.pool_attention_0 = nn.Linear(d_model, 256)
        self.pool_attention_2 = nn.Linear(256, 1)

        self.classifier_0 = nn.Linear(d_model, 256)
        self.classifier_1 = nn.LayerNorm(256)
        self.classifier_4 = nn.Linear(256, 64)
        self.classifier_6 = nn.Linear(64, 1)

    def __call__(
        self, input_features: mx.array, return_logits: bool = False
    ) -> mx.array:
        if input_features.ndim == 2:
            input_features = input_features[None, ...]

        hidden = self.encoder(input_features.astype(self.dtype))
        attn = self.pool_attention_2(mx.tanh(self.pool_attention_0(hidden)))
        attn = mx.softmax(attn, axis=1, precise=True)
        pooled = mx.sum(hidden * attn, axis=1)

        x = self.classifier_0(pooled)
        x = self.classifier_1(x)
        x = nn.gelu(x)
        x = self.classifier_4(x)
        x = nn.gelu(x)
        logits = self.classifier_6(x)

        if return_logits:
            return logits
        return mx.sigmoid(logits)

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        if orig_sr == target_sr:
            return audio
        if audio.size == 0:
            return audio
        new_len = max(1, int(round(audio.shape[0] * target_sr / orig_sr)))
        x_old = np.linspace(0.0, 1.0, audio.shape[0], endpoint=False)
        x_new = np.linspace(0.0, 1.0, new_len, endpoint=False)
        return np.interp(x_new, x_old, audio).astype(np.float32)

    def _prepare_audio_array(
        self,
        audio: Union[str, np.ndarray, mx.array],
        sample_rate: Optional[int] = None,
    ) -> np.ndarray:
        sr = (
            self.config.processor_config.sampling_rate
            if sample_rate is None
            else sample_rate
        )

        if isinstance(audio, str):
            waveform, file_sr = audio_read(audio)
            if waveform.ndim > 1:
                waveform = np.mean(waveform, axis=1)
            audio_np = np.asarray(waveform, dtype=np.float32)
            sr = int(file_sr)
        elif isinstance(audio, mx.array):
            audio_np = np.asarray(audio, dtype=np.float32)
        else:
            audio_np = np.asarray(audio, dtype=np.float32)

        if audio_np.ndim != 1:
            raise ValueError(f"Expected mono audio (1-D), got shape {audio_np.shape}")

        target_sr = self.config.processor_config.sampling_rate
        audio_np = self._resample(audio_np, sr, target_sr)

        max_samples = (
            self.config.processor_config.max_audio_seconds
            * self.config.processor_config.sampling_rate
        )
        if audio_np.shape[0] > max_samples:
            audio_np = audio_np[-max_samples:]
        elif audio_np.shape[0] < max_samples:
            pad = max_samples - audio_np.shape[0]
            audio_np = np.pad(audio_np, (pad, 0), mode="constant")

        if self.config.processor_config.normalize_audio and audio_np.size > 0:
            mean = float(audio_np.mean())
            std = float(audio_np.std())
            audio_np = (audio_np - mean) / max(std, 1e-7)

        return audio_np.astype(np.float32, copy=False)

    def prepare_input_features(
        self,
        audio: Union[str, np.ndarray, mx.array],
        sample_rate: Optional[int] = None,
    ) -> mx.array:
        audio_np = self._prepare_audio_array(audio, sample_rate=sample_rate)
        mel = log_mel_spectrogram(audio_np, n_mels=self.config.processor_config.n_mels)

        # Convert to [frames, n_mels] if needed.
        if mel.ndim != 2:
            raise ValueError(f"Expected 2-D mel features, got shape {mel.shape}")
        if mel.shape[-1] != self.config.processor_config.n_mels:
            mel = mx.transpose(mel, (1, 0))

        frames = mel.shape[0]
        target_frames = (
            self.config.processor_config.max_audio_seconds
            * self.config.processor_config.sampling_rate
            // self.config.processor_config.hop_length
        )
        if frames > target_frames:
            mel = mel[-target_frames:, :]
        elif frames < target_frames:
            mel = mx.pad(mel, [(target_frames - frames, 0), (0, 0)])

        # Convert to HF-style shape [n_mels, n_frames].
        return mx.transpose(mel, (1, 0)).astype(self.dtype)

    def predict_endpoint(
        self,
        audio: Union[str, np.ndarray, mx.array],
        sample_rate: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> EndpointOutput:
        features = self.prepare_input_features(audio, sample_rate=sample_rate)
        probability = float(self(features)[0, 0].item())
        thr = (
            self.config.processor_config.threshold
            if threshold is None
            else float(threshold)
        )
        prediction = 1 if probability > thr else 0
        return EndpointOutput(prediction=prediction, probability=probability)

    @staticmethod
    def _remap_key(key: str) -> str:
        key = key.replace("inner.", "", 1) if key.startswith("inner.") else key
        key = key.replace("pool_attention.0.", "pool_attention_0.")
        key = key.replace("pool_attention.2.", "pool_attention_2.")
        key = key.replace("classifier.0.", "classifier_0.")
        key = key.replace("classifier.1.", "classifier_1.")
        key = key.replace("classifier.4.", "classifier_4.")
        key = key.replace("classifier.6.", "classifier_6.")
        return key

    @classmethod
    def sanitize(cls, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        sanitized = {}

        for key, value in weights.items():
            if key.startswith("val_"):
                # ONNX helper constants are not model parameters. We expect
                # conversion to remap trainable val_* tensors to semantic names.
                continue

            target_key = key
            target_key = cls._remap_key(target_key)

            # Conv1d: PyTorch/HF stores (out, in, k), MLX expects (out, k, in).
            if (
                target_key in {"encoder.conv1.weight", "encoder.conv2.weight"}
                and value.ndim == 3
            ):
                value = mx.transpose(value, (0, 2, 1))

            # Handle pre-remapped semantic keys that still carry ONNX MatMul layout.
            if (
                target_key.endswith("fc1.weight")
                and value.ndim == 2
                and value.shape[0] < value.shape[1]
            ):
                value = mx.transpose(value, (1, 0))
            if (
                target_key.endswith("fc2.weight")
                and value.ndim == 2
                and value.shape[0] > value.shape[1]
            ):
                value = mx.transpose(value, (1, 0))
            if (
                target_key == "pool_attention_0.weight"
                and value.ndim == 2
                and value.shape[0] != 256
            ):
                value = mx.transpose(value, (1, 0))
            if (
                target_key == "pool_attention_2.weight"
                and value.ndim == 2
                and value.shape[0] != 1
            ):
                value = mx.transpose(value, (1, 0))

            sanitized[target_key] = value

        return sanitized
