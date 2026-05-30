"""DeepFilterNet MLX model runtime.

This module implements the full DeepFilterNet inference pipeline in MLX:
- STFT/ISTFT with Vorbis window
- ERB + DF feature extraction and normalization
- Neural network inference
- Spectral reconstruction and delay compensation
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from huggingface_hub import hf_hub_download

from mlx_audio import audio_io
from mlx_audio.dsp import istft, stft

from .config import DeepFilterNet2Config, DeepFilterNet3Config, DeepFilterNetConfig
from .network import DfNet
from .network_df1 import DfNetV1
from .weight_loader import load_weights as load_df_weights

DEFAULT_REPO = "mlx-community/DeepFilterNet-mlx"
DEFAULT_SUBFOLDER = "v3"

VERSION_SUBFOLDER = {
    1: "v1",
    2: "v2",
    3: "v3",
}

DEFAULT_CONFIGS = {
    "DeepFilterNet": DeepFilterNetConfig,
    "DeepFilterNet2": DeepFilterNet2Config,
    "DeepFilterNet3": DeepFilterNet3Config,
}


class DeepFilterNetModel(nn.Module):
    """Pure-MLX DeepFilterNet inference runtime."""

    def __init__(
        self,
        config: DeepFilterNetConfig,
        model: Optional[Union[DfNet, DfNetV1]] = None,
        model_dir: Optional[Path] = None,
        model_version: Optional[str] = None,
    ):
        super().__init__()
        if model_version is None:
            model_version = getattr(config, "model_version", "DeepFilterNet3")
        if model is None:
            model = (
                DfNetV1(config) if model_version == "DeepFilterNet" else DfNet(config)
            )

        self.model = model
        self.config = config
        self.model_dir = Path(model_dir) if model_dir is not None else None
        self.model_version = model_version

        # From libDF: wnorm = 1 / (fft_size^2 / (2 * hop_size))
        self.wnorm = 1.0 / (config.fft_size * config.fft_size / (2.0 * config.hop_size))

        self._vorbis = self._vorbis_window(config.fft_size)

        # Filterbanks loaded from model weights via weight_loader fallback
        self.erb_fb = self.model.erb_fb
        self.erb_inv_fb = self.model.mask.erb_inv_fb
        self.erb_widths = config.erb_widths
        self._has_erb_fb = (
            hasattr(self.erb_fb, "shape")
            and len(self.erb_fb.shape) == 2
            and self.erb_fb.shape[0] == (self.config.fft_size // 2 + 1)
            and self.erb_fb.shape[1] == self.config.nb_erb
        )

    def load_weights(self, weights, strict: bool = False):
        weight_dict = dict(weights)
        loaded = load_df_weights(self.model, weight_dict)
        if strict and loaded != len(weight_dict):
            raise ValueError(
                f"Loaded {loaded} of {len(weight_dict)} DeepFilterNet tensors"
            )
        return self

    def post_load_hook(self, model_path: Path) -> "DeepFilterNetModel":
        self.model_dir = Path(model_path)
        return self

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = DEFAULT_REPO,
        subfolder: Optional[str] = DEFAULT_SUBFOLDER,
        version: Optional[int] = None,
    ) -> "DeepFilterNetModel":
        """Load a pretrained DeepFilterNet model.

        Args:
            model_name_or_path: HuggingFace repo id or local directory path.
            subfolder: Subfolder within the repo (e.g. "v1", "v2", "v3").
            version: Convenience alias — 1, 2, or 3 maps to subfolder "v1"/"v2"/"v3".
                     Overrides *subfolder* when provided.
        """
        if version is not None:
            subfolder = VERSION_SUBFOLDER.get(version)
            if subfolder is None:
                raise ValueError(
                    f"Unsupported version={version}. Choose from 1, 2, or 3."
                )

        local = Path(model_name_or_path).expanduser().resolve()
        if local.exists():
            if not local.is_dir():
                raise ValueError(
                    f"Local model path must be a directory containing "
                    f"config.json and model.safetensors: {local}"
                )
            model_dir = local / subfolder if subfolder else local
            config_path = model_dir / "config.json"
            weights_path = model_dir / "model.safetensors"
            if not config_path.exists():
                raise FileNotFoundError(
                    f"Missing config.json in model directory: {model_dir}"
                )
            if not weights_path.exists():
                raise FileNotFoundError(f"Missing model.safetensors in: {model_dir}")
            return cls._load_from_files(
                config_path=config_path,
                weights_path=weights_path,
                model_dir=model_dir,
            )

        # Treat as HuggingFace repo ID.
        hf_kwargs = {"repo_id": model_name_or_path}
        if subfolder:
            hf_kwargs["subfolder"] = subfolder
        config_path = Path(hf_hub_download(filename="config.json", **hf_kwargs))
        weights_path = Path(hf_hub_download(filename="model.safetensors", **hf_kwargs))
        return cls._load_from_files(
            config_path=config_path,
            weights_path=weights_path,
            model_dir=config_path.parent,
        )

    @classmethod
    def _load_from_files(
        cls,
        config_path: Path,
        weights_path: Path,
        model_dir: Path,
    ) -> "DeepFilterNetModel":
        with open(config_path, encoding="utf-8") as f:
            config_dict = json.load(f)

        model_version = config_dict.get("model_version")
        if not model_version:
            model_version = "DeepFilterNet3"
        config_class = DEFAULT_CONFIGS.get(model_version, DeepFilterNetConfig)
        config = config_class.from_dict(config_dict)

        if model_version == "DeepFilterNet":
            model = DfNetV1(config)
        else:
            model = DfNet(config)

        weights = mx.load(str(weights_path))
        loaded = load_df_weights(model, weights)
        print(f"Loaded DeepFilterNet weights: {loaded} tensors from {weights_path}")

        return cls(
            model=model,
            config=config,
            model_dir=model_dir,
            model_version=model_version,
        )

    def enhance_file(
        self, input_path: Union[str, Path], output_path: Union[str, Path]
    ) -> Path:
        input_path = Path(input_path)
        output_path = Path(output_path)

        audio, sr = audio_io.read(str(input_path), always_2d=False, dtype="float32")
        if sr != self.config.sample_rate:
            raise ValueError(
                f"Expected {self.config.sample_rate} Hz audio, got {sr} Hz: {input_path}"
            )

        if audio.ndim > 1:
            audio = audio[:, 0]

        enhanced = self.enhance_array(audio)
        audio_io.write(str(output_path), enhanced, self.config.sample_rate)
        return output_path

    def create_streamer(
        self,
        *,
        pad_end_frames: int = 3,
        compensate_delay: bool = True,
    ):
        from .streaming import DeepFilterNetStreamer, DeepFilterNetStreamingConfig

        return DeepFilterNetStreamer(
            model=self,
            config=DeepFilterNetStreamingConfig(
                pad_end_frames=pad_end_frames,
                compensate_delay=compensate_delay,
            ),
        )

    def enhance_array_streaming(
        self,
        audio: np.ndarray,
        chunk_samples: Optional[int] = None,
        *,
        pad_end_frames: int = 3,
        compensate_delay: bool = True,
    ) -> np.ndarray:
        x = np.asarray(audio, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return x

        streamer = self.create_streamer(
            pad_end_frames=pad_end_frames,
            compensate_delay=compensate_delay,
        )
        if chunk_samples is None:
            chunk_samples = self.config.hop_size * 8
        chunk_samples = max(int(chunk_samples), self.config.hop_size)

        outs = []
        for start in range(0, x.shape[0], chunk_samples):
            out = streamer.process_chunk(
                x[start : start + chunk_samples], is_last=False
            )
            if out.size > 0:
                outs.append(out)
        tail = streamer.flush()
        if tail.size > 0:
            outs.append(tail)
        if not outs:
            return np.zeros((0,), dtype=np.float32)
        return np.concatenate(outs, axis=0)

    def enhance_file_streaming(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        chunk_samples: Optional[int] = None,
        *,
        pad_end_frames: int = 3,
        compensate_delay: bool = True,
    ) -> Path:
        input_path = Path(input_path)
        output_path = Path(output_path)

        audio, sr = audio_io.read(str(input_path), always_2d=False, dtype="float32")
        if sr != self.config.sample_rate:
            raise ValueError(
                f"Expected {self.config.sample_rate} Hz audio, got {sr} Hz: {input_path}"
            )
        if audio.ndim > 1:
            audio = audio[:, 0]

        enhanced = self.enhance_array_streaming(
            audio=audio,
            chunk_samples=chunk_samples,
            pad_end_frames=pad_end_frames,
            compensate_delay=compensate_delay,
        )
        audio_io.write(str(output_path), enhanced, self.config.sample_rate)
        return output_path

    def enhance_array(self, audio: np.ndarray) -> np.ndarray:
        x = mx.array(audio.astype(np.float32))
        p = self.config

        orig_len = x.shape[0]

        # Match libDF streaming analysis:
        # - implicit left context of one hop (analysis memory initialized with zeros)
        # - right padding by fft_size for delay compensation path
        x = mx.pad(x, [(p.hop_size, p.fft_size)])

        # STFT [T, F], no center padding
        spec = stft(
            x,
            n_fft=p.fft_size,
            hop_length=p.hop_size,
            win_length=p.fft_size,
            window=self._vorbis,
            center=False,
        )

        # Match libDF analysis normalization.
        spec = spec * self.wnorm

        alpha = self._norm_alpha()

        # ERB features
        spec_mag_sq = mx.real(spec) ** 2 + mx.imag(spec) ** 2  # [T, F]
        erb = self._erb(spec_mag_sq)  # [T, E]
        # Match libDF exactly: 10*log10(erb + 1e-10), not clamp-based log.
        erb_db = 10.0 * mx.log10(erb + 1e-10)
        erb_norm = self._band_mean_norm(erb_db, alpha, p.nb_erb)
        feat_erb = erb_norm[None, None, :, :]  # [B, 1, T, E]

        # DF complex features
        df_spec = spec[:, : p.nb_df]  # [T, D]
        df_re, df_im = self._band_unit_norm(df_spec, alpha, p.nb_df)
        feat_df = mx.stack([df_re, df_im], axis=-1)[
            None, None, :, :, :
        ]  # [B, 1, T, D, 2]

        spec_in = mx.stack([mx.real(spec), mx.imag(spec)], axis=-1)[None, None, :, :, :]

        spec_e, _mask, _lsnr, _df_coefs = self.model(spec_in, feat_erb, feat_df)

        enh = spec_e[0, 0]  # [T, F, 2]
        enh_complex = enh[..., 0] + 1j * enh[..., 1]  # [T, F]

        # Undo libDF normalization for ISTFT implementation
        enh_complex = enh_complex / self.wnorm

        # istft expects [F, T]
        enh_complex = mx.transpose(enh_complex, (1, 0))
        audio_out = istft(
            enh_complex,
            hop_length=p.hop_size,
            win_length=p.fft_size,
            window=self._vorbis,
            center=False,
            # Include the explicit left hop padding used before STFT.
            length=orig_len + p.hop_size + p.fft_size,
            # Match libDF overlap-add scaling behavior.
            normalized=True,
        )

        # Match official DeepFilterNet/libDF delay compensation.
        d = p.fft_size - p.hop_size
        audio_out = audio_out[d : orig_len + d]

        y = np.array(audio_out, dtype=np.float32)
        return np.clip(y, -1.0, 1.0)

    def _norm_alpha(self) -> float:
        # Match df.utils.get_norm_alpha() rounding behavior exactly.
        a_raw = math.exp(-self.config.hop_size / self.config.sample_rate)
        precision = 3
        a = 1.0
        while a >= 1.0:
            a = round(a_raw, precision)
            precision += 1
        return a

    def _band_mean_norm(self, x: mx.array, alpha: float, nb_bands: int) -> mx.array:
        # Use numpy for the sequential EMA loop to avoid per-frame MLX kernel launch overhead.
        x_np = np.array(x, dtype=np.float32)
        state = np.linspace(-60.0, -90.0, nb_bands, dtype=np.float32)
        out = np.empty_like(x_np)
        a = np.float32(alpha)
        one_minus_a = np.float32(1.0 - alpha)
        for i in range(x_np.shape[0]):
            state = x_np[i] * one_minus_a + state * a
            out[i] = (x_np[i] - state) / np.float32(40.0)
        return mx.array(out)

    def _band_unit_norm(self, x: mx.array, alpha: float, nb_freqs: int):
        # Same EMA recurrence as libDF, computed in numpy for performance.
        x_np = np.array(x, dtype=np.complex64)
        mag = np.abs(x_np)
        state = np.linspace(0.001, 0.0001, nb_freqs, dtype=np.float32)

        out_r = np.empty((x_np.shape[0], nb_freqs), dtype=np.float32)
        out_i = np.empty((x_np.shape[0], nb_freqs), dtype=np.float32)
        a = np.float32(alpha)
        one_minus_a = np.float32(1.0 - alpha)
        for i in range(x_np.shape[0]):
            state = mag[i] * one_minus_a + state * a
            denom = np.sqrt(state)
            out_r[i] = x_np[i].real / denom
            out_i[i] = x_np[i].imag / denom

        return mx.array(out_r), mx.array(out_i)

    def _erb(self, spec_mag_sq: mx.array) -> mx.array:
        """Compute ERB energies, preferring learned filterbank projection when present."""
        if self._has_erb_fb:
            return spec_mag_sq @ self.erb_fb

        if self.erb_widths is None:
            raise ValueError("Missing both ERB filterbank and ERB band widths.")

        bands = []
        start = 0
        for width in self.erb_widths:
            stop = start + int(width)
            bands.append(mx.mean(spec_mag_sq[:, start:stop], axis=1))
            start = stop
        return mx.stack(bands, axis=1)

    @staticmethod
    def _vorbis_window(size: int) -> mx.array:
        # libDF Vorbis window:
        # sin(0.5*pi*sin(0.5*pi*(n+0.5)/(N/2))^2)
        n = np.arange(size, dtype=np.float32)
        n_half = size // 2
        inner = np.sin(0.5 * np.pi * (n + 0.5) / n_half)
        window = np.sin(0.5 * np.pi * inner * inner)
        return mx.array(window.astype(np.float32))
