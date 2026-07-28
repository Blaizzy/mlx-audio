from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any, Optional

import mlx.core as mx

from mlx_audio.dsp import hanning, mel_filters, stft
from mlx_audio.tts.models.qwen3_tts.config import Qwen3TTSSpeakerEncoderConfig
from mlx_audio.tts.models.qwen3_tts.speaker_encoder import Qwen3TTSSpeakerEncoder
from mlx_audio.utils import load_audio, resample_audio

DEFAULT_SPEAKER_ENCODER_REPO = "marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B"


def _filter_config(data: dict[str, Any]) -> dict[str, Any]:
    valid = {field.name for field in fields(Qwen3TTSSpeakerEncoderConfig)}
    return {key: value for key, value in data.items() if key in valid}


def speaker_encoder_config_from_dict(
    data: dict[str, Any] | None,
) -> Qwen3TTSSpeakerEncoderConfig:
    return Qwen3TTSSpeakerEncoderConfig(**_filter_config(data or {}))


def _is_mlx_conv_weight(value: mx.array) -> bool:
    if len(value.shape) != 3:
        return False
    _, dim2, dim3 = value.shape
    if dim2 == 1:
        return dim3 > 64
    if dim3 == 1:
        return not (dim2 > 64)
    return dim2 < dim3


def sanitize_speaker_encoder_weights(
    weights: dict[str, mx.array],
    *,
    dtype: str | None = None,
) -> dict[str, mx.array]:
    sanitized: dict[str, mx.array] = {}
    for key, value in weights.items():
        new_key = key.removeprefix("speaker_encoder.")
        if new_key.endswith(".weight") and len(value.shape) == 3:
            value = (
                value if _is_mlx_conv_weight(value) else mx.transpose(value, (0, 2, 1))
            )
        if dtype == "bfloat16" and value.dtype in (
            mx.float16,
            mx.float32,
            mx.bfloat16,
        ):
            value = value.astype(mx.bfloat16)
        elif dtype == "float16" and value.dtype in (
            mx.float16,
            mx.float32,
            mx.bfloat16,
        ):
            value = value.astype(mx.float16)
        elif dtype == "float32" and value.dtype in (
            mx.float16,
            mx.float32,
            mx.bfloat16,
        ):
            value = value.astype(mx.float32)
        sanitized[new_key] = value
    return sanitized


def resolve_speaker_encoder_path(
    model_path: str | Path | None,
    *,
    speaker_encoder_path: str | Path | None = "speaker_encoder",
    speaker_encoder_model_id: str = DEFAULT_SPEAKER_ENCODER_REPO,
) -> Path:
    if speaker_encoder_path is not None:
        path = Path(speaker_encoder_path).expanduser()
        if not path.is_absolute() and model_path is not None:
            path = Path(model_path).expanduser() / path
        if path.exists():
            return path

    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            speaker_encoder_model_id,
            allow_patterns=[
                "config.json",
                "preprocessor_config.json",
                "model.safetensors",
            ],
        )
    )


def load_speaker_encoder(model_dir: str | Path) -> Qwen3TTSSpeakerEncoder:
    model_dir = Path(model_dir)
    config_path = model_dir / "config.json"
    config_data = json.loads(config_path.read_text()) if config_path.exists() else {}
    config = speaker_encoder_config_from_dict(config_data)
    model = Qwen3TTSSpeakerEncoder(config)

    weights_path = model_dir / "model.safetensors"
    if not weights_path.exists():
        raise FileNotFoundError(f"speaker encoder weights not found: {weights_path}")
    weights = sanitize_speaker_encoder_weights(dict(mx.load(str(weights_path))))
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    model.eval()
    return model


def speaker_log_mel_spectrogram(
    audio: mx.array,
    *,
    sample_rate: int = 24000,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 128,
    f_min: float = 0.0,
    f_max: float = 12000.0,
) -> mx.array:
    if audio.ndim == 2:
        audio = mx.mean(audio, axis=0)
    elif audio.ndim != 1:
        raise ValueError(f"speaker audio must be 1-D or 2-D, got shape {audio.shape}")

    padding = (n_fft - hop_length) // 2
    if audio.shape[0] <= padding:
        raise ValueError(
            f"speaker audio must contain more than {padding} samples at {sample_rate} Hz"
        )

    left = audio[1 : padding + 1][::-1]
    right = audio[-(padding + 1) : -1][::-1]
    padded = mx.concatenate([left, audio, right], axis=0)

    spec = stft(
        padded,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=hanning(n_fft, periodic=True),
        center=False,
        pad_mode="reflect",
    )
    magnitude = mx.abs(spec)
    filters = mel_filters(
        sample_rate,
        n_fft,
        n_mels,
        f_min=f_min,
        f_max=f_max,
        norm="slaney",
        mel_scale="slaney",
    )
    mel = magnitude @ filters.T
    return mx.log(mx.maximum(mel, 1e-5))[None, :, :]


class Zonos2SpeakerEmbeddingExtractor:
    def __init__(
        self,
        model_dir: str | Path,
        *,
        sample_rate: int = 24000,
    ):
        self.sample_rate = int(sample_rate)
        self.model = load_speaker_encoder(model_dir)

    def _prepare_audio(
        self,
        audio: str | Path | mx.array | tuple[Any, int],
        sample_rate: Optional[int] = None,
    ) -> mx.array:
        if isinstance(audio, tuple):
            if len(audio) != 2:
                raise ValueError("speaker audio tuple must be (audio, sample_rate)")
            audio, sample_rate = audio

        if isinstance(audio, (str, Path)):
            return load_audio(str(audio), sample_rate=self.sample_rate)

        waveform = mx.array(audio, dtype=mx.float32)
        if waveform.ndim == 2:
            if waveform.shape[0] <= waveform.shape[1]:
                waveform = mx.mean(waveform, axis=0)
            else:
                waveform = mx.mean(waveform, axis=1)
        elif waveform.ndim != 1:
            raise ValueError(
                f"speaker audio must be 1-D or 2-D, got shape {waveform.shape}"
            )

        source_rate = int(sample_rate or self.sample_rate)
        if source_rate != self.sample_rate:
            waveform = resample_audio(waveform, source_rate, self.sample_rate)
            waveform = mx.array(waveform, dtype=mx.float32)
        return waveform

    def encode(
        self,
        audio: str | Path | mx.array | tuple[Any, int],
        *,
        sample_rate: Optional[int] = None,
    ) -> mx.array:
        waveform = self._prepare_audio(audio, sample_rate)
        mel = speaker_log_mel_spectrogram(waveform, sample_rate=self.sample_rate)
        embedding = self.model(mel).astype(mx.float32)
        mx.eval(embedding)
        return embedding


def convert_speaker_encoder(
    source_repo: str,
    out_dir: Path,
    *,
    dtype: str = "bfloat16",
    revision: str | None = None,
) -> Path:
    from huggingface_hub import snapshot_download

    source_dir = Path(
        snapshot_download(
            source_repo,
            revision=revision,
            allow_patterns=[
                "config.json",
                "preprocessor_config.json",
                "model.safetensors",
            ],
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    weights = sanitize_speaker_encoder_weights(
        dict(mx.load(str(source_dir / "model.safetensors"))),
        dtype=dtype,
    )
    mx.save_safetensors(
        str(out_dir / "model.safetensors"),
        weights,
        metadata={"format": "mlx"},
    )

    for name in ("config.json", "preprocessor_config.json"):
        src = source_dir / name
        if src.exists():
            (out_dir / name).write_text(src.read_text(), encoding="utf-8")
    return out_dir
