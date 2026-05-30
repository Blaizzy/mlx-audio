from __future__ import annotations

import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.dsp import hanning, mel_filters, stft
from mlx_audio.stt.utils import load_audio


class LogMel80(nn.Module):
    def __init__(self):
        super().__init__()
        self.sample_rate = 16000
        self.n_fft = 400
        self.hop_length = 160
        self.win_length = 400
        self.window = hanning(self.win_length, periodic=True)
        self.filters = mel_filters(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=80,
            f_min=0.0,
            f_max=8000.0,
            norm="slaney",
            mel_scale="slaney",
        )

    def __call__(self, waveform: mx.array) -> mx.array:
        if waveform.ndim != 1:
            waveform = mx.squeeze(waveform)

        spec = stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            pad_mode="reflect",
        )
        power = mx.abs(spec) ** 2.0
        mel = power @ self.filters.T
        log_mel = mx.log10(mx.maximum(mel, 1e-10))
        return (log_mel + 4.0) / 4.0


class BatchNorm1d(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((num_features,))
        self.bias = mx.zeros((num_features,))
        self.running_mean = mx.zeros((num_features,))
        self.running_var = mx.ones((num_features,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return (x - self.running_mean) / mx.sqrt(
            self.running_var + self.eps
        ) * self.weight + self.bias


class ConvFrontend(nn.Module):
    def __init__(self, n_mels: int = 80, hidden_dim: int = 128, d_model: int = 256):
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, hidden_dim, kernel_size=3, stride=2, padding=1)
        self.bn1 = BatchNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, d_model, kernel_size=3, stride=2, padding=1)
        self.bn2 = BatchNorm1d(d_model)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.gelu(self.bn1(self.conv1(x)))
        x = nn.gelu(self.bn2(self.conv2(x)))
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 850):
        super().__init__()
        position = mx.arange(max_len, dtype=mx.float32)[:, None]
        div_term = mx.exp(
            mx.arange(0, d_model, 2, dtype=mx.float32) * (-math.log(10000.0) / d_model)
        )
        pe = mx.zeros((1, max_len, d_model), dtype=mx.float32)
        pe[:, :, 0::2] = mx.sin(position * div_term)
        pe[:, :, 1::2] = mx.cos(position * div_term)
        self.pe = pe

    def __call__(self, x: mx.array) -> mx.array:
        return x + self.pe[:, : x.shape[1]]


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int = 256, nhead: int = 4):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def __call__(self, x: mx.array) -> mx.array:
        batch_size, steps, _ = x.shape
        q = self.q_proj(x).reshape(batch_size, steps, self.nhead, self.head_dim)
        k = self.k_proj(x).reshape(batch_size, steps, self.nhead, self.head_dim)
        v = self.v_proj(x).reshape(batch_size, steps, self.nhead, self.head_dim)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 3, 1)
        v = v.transpose(0, 2, 1, 3)

        attn = mx.softmax((q * self.scale) @ k, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(batch_size, steps, self.d_model)
        return self.out_proj(out)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int = 256, nhead: int = 4, dim_feedforward: int = 1024):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadSelfAttention(d_model=d_model, nhead=nhead)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.self_attn(self.norm1(x))
        x = x + self.linear2(nn.gelu(self.linear1(self.norm2(x))))
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 4,
        dim_feedforward: int = 1024,
        num_layers: int = 1,
    ):
        super().__init__()
        self.layers = [
            TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
            )
            for _ in range(num_layers)
        ]
        self.norm = nn.LayerNorm(d_model)

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class AttentionPooling(nn.Module):
    def __init__(self, d_model: int = 256):
        super().__init__()
        self.query = nn.Linear(d_model, 1)

    def __call__(self, x: mx.array) -> mx.array:
        weights = mx.softmax(mx.squeeze(self.query(x), axis=-1), axis=-1)
        return mx.sum(weights[..., None] * x, axis=1)


class ClassifierHead(nn.Module):
    def __init__(self, d_model: int = 256, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc1(x)))


class AudioQualityRouter(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 4,
        dim_feedforward: int = 1024,
        num_layers: int = 1,
        n_mels: int = 80,
        frontend_hidden_dim: int = 128,
        classifier_hidden_dim: int = 128,
        max_len: int = 850,
    ):
        super().__init__()
        self.logmel = LogMel80()
        self.frontend = ConvFrontend(
            n_mels=n_mels,
            hidden_dim=frontend_hidden_dim,
            d_model=d_model,
        )
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=max_len)
        self.transformer = TransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_layers=num_layers,
        )
        self.pooling = AttentionPooling(d_model=d_model)
        self.classifier = ClassifierHead(
            d_model=d_model,
            hidden_dim=classifier_hidden_dim,
        )

    @classmethod
    def from_converted(cls, weights: dict[str, mx.array]) -> "AudioQualityRouter":
        d_model = int(weights["transformer.norm.weight"].shape[0])
        dim_feedforward = int(weights["transformer.layers.0.linear1.weight"].shape[0])
        num_layers = len(
            {
                key.split(".")[2]
                for key in weights
                if key.startswith("transformer.layers.")
            }
        )
        n_mels = int(weights["frontend.conv.0.weight"].shape[2])
        frontend_hidden_dim = int(weights["frontend.conv.0.weight"].shape[0])
        classifier_hidden_dim = int(weights["classifier.0.weight"].shape[0])
        max_len = int(weights["pos_encoder.pe"].shape[1])

        model = cls(
            d_model=d_model,
            nhead=4,
            dim_feedforward=dim_feedforward,
            num_layers=num_layers,
            n_mels=n_mels,
            frontend_hidden_dim=frontend_hidden_dim,
            classifier_hidden_dim=classifier_hidden_dim,
            max_len=max_len,
        )
        model._load_weights(weights)
        return model

    def _load_weights(self, weights: dict[str, mx.array]) -> None:
        self.frontend.conv1.weight = weights["frontend.conv.0.weight"]
        self.frontend.conv1.bias = weights["frontend.conv.0.bias"]
        self.frontend.bn1.weight = weights["frontend.conv.1.weight"]
        self.frontend.bn1.bias = weights["frontend.conv.1.bias"]
        self.frontend.bn1.running_mean = weights["frontend.conv.1.running_mean"]
        self.frontend.bn1.running_var = weights["frontend.conv.1.running_var"]

        self.frontend.conv2.weight = weights["frontend.conv.4.weight"]
        self.frontend.conv2.bias = weights["frontend.conv.4.bias"]
        self.frontend.bn2.weight = weights["frontend.conv.5.weight"]
        self.frontend.bn2.bias = weights["frontend.conv.5.bias"]
        self.frontend.bn2.running_mean = weights["frontend.conv.5.running_mean"]
        self.frontend.bn2.running_var = weights["frontend.conv.5.running_var"]

        self.pos_encoder.pe = weights["pos_encoder.pe"]

        for index, layer in enumerate(self.transformer.layers):
            prefix = f"transformer.layers.{index}"
            in_proj_weight = weights[f"{prefix}.self_attn.in_proj_weight"]
            in_proj_bias = weights[f"{prefix}.self_attn.in_proj_bias"]
            q_weight, k_weight, v_weight = mx.split(in_proj_weight, 3, axis=0)
            q_bias, k_bias, v_bias = mx.split(in_proj_bias, 3, axis=0)

            layer.self_attn.q_proj.weight = q_weight
            layer.self_attn.q_proj.bias = q_bias
            layer.self_attn.k_proj.weight = k_weight
            layer.self_attn.k_proj.bias = k_bias
            layer.self_attn.v_proj.weight = v_weight
            layer.self_attn.v_proj.bias = v_bias
            layer.self_attn.out_proj.weight = weights[
                f"{prefix}.self_attn.out_proj.weight"
            ]
            layer.self_attn.out_proj.bias = weights[f"{prefix}.self_attn.out_proj.bias"]

            layer.linear1.weight = weights[f"{prefix}.linear1.weight"]
            layer.linear1.bias = weights[f"{prefix}.linear1.bias"]
            layer.linear2.weight = weights[f"{prefix}.linear2.weight"]
            layer.linear2.bias = weights[f"{prefix}.linear2.bias"]
            layer.norm1.weight = weights[f"{prefix}.norm1.weight"]
            layer.norm1.bias = weights[f"{prefix}.norm1.bias"]
            layer.norm2.weight = weights[f"{prefix}.norm2.weight"]
            layer.norm2.bias = weights[f"{prefix}.norm2.bias"]

        self.transformer.norm.weight = weights["transformer.norm.weight"]
        self.transformer.norm.bias = weights["transformer.norm.bias"]
        self.pooling.query.weight = weights["pooling.query.weight"]
        self.pooling.query.bias = weights["pooling.query.bias"]
        self.classifier.fc1.weight = weights["classifier.0.weight"]
        self.classifier.fc1.bias = weights["classifier.0.bias"]
        self.classifier.fc2.weight = weights["classifier.3.weight"]
        self.classifier.fc2.bias = weights["classifier.3.bias"]

    def _prepare_waveform(self, wav: str | Path | mx.array) -> mx.array:
        if isinstance(wav, (str, Path)):
            return load_audio(str(wav))

        waveform = wav if isinstance(wav, mx.array) else mx.array(wav, mx.float32)
        if waveform.ndim > 1:
            waveform = mx.mean(waveform, axis=-1)
        return waveform.astype(mx.float32)

    def logits(self, wav: str | Path | mx.array) -> mx.array:
        waveform = self._prepare_waveform(wav)
        hidden_states = mx.expand_dims(self.logmel(waveform), axis=0)
        hidden_states = self.frontend(hidden_states)
        hidden_states = self.pos_encoder(hidden_states)
        hidden_states = self.transformer(hidden_states)
        pooled = self.pooling(hidden_states)
        return mx.squeeze(self.classifier(pooled), axis=0)

    def degraded_prob(self, wav: str | Path | mx.array) -> float:
        return float(mx.softmax(self.logits(wav), axis=-1)[1])

    def route(self, wav: str | Path | mx.array) -> dict[str, float | bool]:
        degraded_prob = self.degraded_prob(wav)
        return {
            "degraded_prob": degraded_prob,
            "use_lora": degraded_prob >= 0.5,
        }
