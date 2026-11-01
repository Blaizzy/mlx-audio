# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
# Adapted for MLX from NVIDIA-NemotronLabs-RE-USE.
"""Native RE-USE speech enhancement at 8–48 kHz."""

import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from scipy.signal import resample_poly

from mlx_audio import audio_io
from mlx_audio.dsp import istft, stft

from ..base import STSOutput
from .config import ModelConfig
from .mamba import TFMambaBlock
from .network import Decoder, DenseEncoder


class Model(nn.Module):
    output_suffix = "_enhanced"

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.dense_encoder = DenseEncoder(config)
        self.TSMamba = [TFMambaBlock(config) for _ in range(config.num_tfmamba)]
        self.mask_decoder = Decoder(config)
        self.phase_decoder = Decoder(config, phase=True)

    def __call__(self, magnitude, phase):
        """Map compressed magnitude and phase [batch, frequency, time]."""
        freq, time = magnitude.shape[-2:]
        x = mx.stack([magnitude, phase], axis=-1).transpose(0, 2, 1, 3)
        x = mx.pad(x, ((0, 0), (0, 2), (0, 2), (0, 0)))
        x = self.dense_encoder(x)
        for block in self.TSMamba:
            x = block(x)
        magnitude = self.mask_decoder(x)[:, :time, :freq, 0].transpose(0, 2, 1)
        phase = self.phase_decoder(x)[:, :time, :freq, 0].transpose(0, 2, 1)
        return magnitude, phase

    def sanitize(self, weights):
        key = "dense_encoder.dense_conv_1.0.weight"
        if (
            key not in weights
            or weights[key].shape == self.dense_encoder.dense_conv_1[0].weight.shape
        ):
            return weights
        return {
            key: (
                value.transpose(0, 2, 3, 1)
                if value.ndim == 4
                else (
                    value.transpose(0, 2, 1) if key.endswith("conv1d.weight") else value
                )
            )
            for key, value in weights.items()
        }

    @property
    def model_quant_predicate(self):
        # This small enhancement model is kept at source precision.
        return lambda path, module: False

    def _enhance_chunk(self, audio, sample_rate):
        length = audio.shape[0]
        n_fft, hop, win = self.config.stft_sizes(sample_rate)
        # Reflection padding needs more than half an FFT of input.
        if length <= n_fft // 2:
            audio = mx.pad(audio, ((0, n_fft // 2 + 1 - length),))
        window = 0.5 - 0.5 * mx.cos(2 * mx.pi * mx.arange(win) / win)
        spectrum = stft(audio, n_fft, hop, win, window=window).T
        magnitude = mx.abs(spectrum)
        compression = self.config.compress_factor
        if compression in ("log1p", "relu_log1p", "signed_log1p"):
            magnitude = mx.log1p(magnitude)
        else:
            magnitude = magnitude**compression
        # Real FFT endpoints can contain -0j in MLX. PyTorch returns +0j;
        # feeding -pi instead of +pi into the network changes its prediction.
        imaginary = mx.where(spectrum.imag == 0, 0.0, spectrum.imag)
        phase = mx.arctan2(imaginary, spectrum.real)
        magnitude, phase = self(magnitude[None], phase[None])
        magnitude, phase = magnitude[0], phase[0]
        # Match the released inference script's sweep-artifact suppression.
        silent = mx.mean((magnitude <= 0).astype(mx.float32), axis=0) > 0.5
        magnitude = mx.where(silent[None], 0, magnitude)
        if compression == "relu_log1p":
            magnitude = mx.expm1(nn.relu(magnitude))
        elif compression == "log1p":
            magnitude = mx.expm1(magnitude)
        elif compression == "signed_log1p":
            magnitude = mx.sign(magnitude) * mx.expm1(mx.abs(magnitude))
        else:
            magnitude = nn.relu(magnitude) ** (1 / compression)
        spectrum = magnitude * (mx.cos(phase) + 1j * mx.sin(phase))
        enhanced = istft(spectrum, hop, win, window=window, normalized=True)
        if enhanced.shape[0] < length:
            enhanced = mx.pad(
                enhanced, ((0, length - enhanced.shape[0]),), constant_values=1e-8
            )
        return enhanced[:length]

    def generate(
        self,
        audio: str | Path | np.ndarray | mx.array,
        *,
        sample_rate: int | None = None,
        target_sample_rate: int | None = None,
        chunk_seconds: float | None = 5.0,
        overlap_seconds: float | None = None,
    ) -> STSOutput:
        """Enhance mono or samples-first multichannel audio independently per channel.

        File inputs retain their sample rate. Arrays require ``sample_rate``.
        Set ``target_sample_rate`` for bandwidth extension, or ``chunk_seconds=None``
        to process the complete recording in one pass. This bidirectional model
        performs offline enhancement, not causal streaming.
        """
        if isinstance(audio, (str, Path)):
            audio, rate = audio_io.read(str(Path(audio).expanduser()))
            if sample_rate is not None and sample_rate != rate:
                raise ValueError("sample_rate disagrees with the audio file")
            sample_rate = rate
        elif audio is None:
            raise ValueError("Input audio is required for enhancement")
        if sample_rate is None:
            raise ValueError("sample_rate is required for audio arrays")
        self._validate_rate(sample_rate)
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim not in (1, 2) or audio.size == 0 or not np.isfinite(audio).all():
            raise ValueError(
                "Expected nonempty, finite audio with shape [samples] or [samples, channels]"
            )
        if target_sample_rate is not None:
            self._validate_rate(target_sample_rate)
            divisor = math.gcd(sample_rate, target_sample_rate)
            audio = resample_poly(
                audio, target_sample_rate // divisor, sample_rate // divisor, axis=0
            )
            sample_rate = target_sample_rate
        mono = audio.ndim == 1
        if mono:
            audio = audio[:, None]
        length = audio.shape[0]
        if chunk_seconds is None:
            chunk = length
            overlap = 0
        else:
            if not math.isfinite(chunk_seconds) or chunk_seconds <= 0:
                raise ValueError("chunk_seconds must be positive or None")
            if overlap_seconds is None:
                overlap_seconds = chunk_seconds / 2
            if (
                not math.isfinite(overlap_seconds)
                or not 0 <= overlap_seconds < chunk_seconds
            ):
                raise ValueError(
                    "overlap_seconds must be nonnegative and less than chunk_seconds"
                )
            chunk = max(1, int(chunk_seconds * sample_rate))
            overlap = int(overlap_seconds * sample_rate)
            if overlap >= chunk:
                raise ValueError("Chunk hop must be at least one sample")
        if length <= chunk:
            channels = [
                self._enhance_chunk(mx.array(audio[:, i]), sample_rate)
                for i in range(audio.shape[1])
            ]
            enhanced = mx.stack(channels, axis=-1)
        else:
            hop = chunk - overlap
            starts = list(range(0, length - chunk, hop)) + [length - chunk]
            window = np.maximum(np.hanning(chunk + 1)[:-1], 1e-7).astype(np.float32)
            output = np.zeros_like(audio)
            total = np.zeros((length, 1), dtype=np.float32)
            for start in starts:
                for channel in range(audio.shape[1]):
                    result = self._enhance_chunk(
                        mx.array(audio[start : start + chunk, channel]), sample_rate
                    )
                    output[start : start + chunk, channel] += (
                        np.asarray(result) * window
                    )
                total[start : start + chunk] += window[:, None]
            enhanced = mx.array(output / total)
        mx.eval(enhanced)
        return STSOutput(
            audio=enhanced[:, 0] if mono else enhanced, sample_rate=sample_rate
        )

    @staticmethod
    def _validate_rate(rate):
        if not isinstance(rate, (int, np.integer)) or not 8000 <= rate <= 48000:
            raise ValueError("Sample rate must be an integer between 8000 and 48000 Hz")
