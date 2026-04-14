from pathlib import Path
from typing import Iterable, List, Tuple

import mlx.core as mx
import numpy as np

from mlx_audio.dsp import STR_TO_WINDOW_FN, hanning, mel_filters, stft

from .config import PreprocessorConfig

DITHER_EPS = 1e-5


class CohereAudioFrontend:
    def __init__(self, config: PreprocessorConfig):
        self.config = config
        self.window = self._build_window()
        self.fb = mel_filters(
            config.sample_rate,
            config.n_fft,
            config.features,
            norm="slaney",
            mel_scale="slaney",
        ).astype(mx.float32)

    def _build_window(self) -> mx.array:
        window_fn = STR_TO_WINDOW_FN.get(self.config.window, None)
        window = (
            window_fn(self.config.win_length, periodic=False)
            if window_fn is not None
            else hanning(self.config.win_length)
        )
        return window.astype(mx.float32)

    def _stft_window(self) -> mx.array:
        if self.window.shape[0] == self.config.n_fft:
            return self.window.astype(mx.float32)

        total_pad = self.config.n_fft - self.window.shape[0]
        left_pad = total_pad // 2
        right_pad = total_pad - left_pad
        return mx.concatenate(
            [
                mx.zeros((left_pad,), dtype=mx.float32),
                self.window.astype(mx.float32),
                mx.zeros((right_pad,), dtype=mx.float32),
            ]
        )

    def load_buffers_from_checkpoint(self, model_path: Path) -> None:
        safetensor_path = model_path / "model.safetensors"
        if not safetensor_path.exists():
            return

        weights = mx.load(str(safetensor_path))

        fb_key = "preprocessor.featurizer.fb"
        if fb_key in weights:
            fb = weights[fb_key]
            if fb.ndim == 3:
                fb = fb.squeeze(0)
            self.fb = fb.astype(mx.float32)

        win_key = "preprocessor.featurizer.window"
        if win_key in weights:
            self.window = weights[win_key].astype(mx.float32)

    def _normalize_waveform(self, waveform) -> np.ndarray:
        if isinstance(waveform, mx.array):
            waveform = np.array(waveform)
        arr = np.asarray(waveform, dtype=np.float32)
        if arr.ndim == 2:
            if arr.shape[0] <= 8 and arr.shape[1] > arr.shape[0]:
                arr = arr.mean(axis=0)
            else:
                arr = arr.mean(axis=1)
        if arr.ndim != 1:
            raise ValueError(f"Expected mono waveform, got shape {arr.shape}.")
        return arr.astype(np.float32, copy=False)

    def _apply_dither(self, waveform: np.ndarray) -> np.ndarray:
        if self.config.dither <= 0:
            return waveform
        rng = np.random.default_rng(seed=waveform.shape[0])
        noise = rng.standard_normal(waveform.shape[0], dtype=np.float32)
        return waveform + self.config.dither * noise

    def _sequence_length(self, num_samples: int) -> int:
        return max(num_samples // self.config.hop_length, 0)

    def _extract_single(self, waveform) -> Tuple[mx.array, int]:
        waveform = self._normalize_waveform(waveform)
        waveform = self._apply_dither(waveform)

        x = mx.array(waveform, dtype=mx.float32)
        if self.config.preemph > 0 and x.shape[0] > 1:
            x = mx.concatenate(
                [x[:1], x[1:] - self.config.preemph * x[:-1]],
                axis=0,
            )

        spec = stft(
            x,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.n_fft,
            window=self._stft_window(),
            center=True,
            pad_mode="constant",
        )
        power = mx.square(mx.abs(spec)).astype(mx.float32)
        mel = self.fb.astype(power.dtype) @ power.T

        if self.config.log:
            mel = mx.log(mel + self.config.log_zero_guard_value)

        seq_len = self._sequence_length(len(waveform))
        seq_len = min(seq_len, mel.shape[1])

        if self.config.normalize == "per_feature" and seq_len > 0:
            valid = mel[:, :seq_len]
            mean = mx.mean(valid, axis=1, keepdims=True)
            if seq_len > 1:
                var = mx.sum((valid - mean) ** 2, axis=1, keepdims=True) / (seq_len - 1)
            else:
                var = mx.zeros_like(mean)
            std = mx.sqrt(var)
            mel = (mel - mean) / (std + DITHER_EPS)

        if seq_len < mel.shape[1]:
            valid_mask = mx.arange(mel.shape[1])[None, :] < seq_len
            mel = mx.where(valid_mask, mel, self.config.pad_value)

        mel = mel.T

        if self.config.pad_to > 0 and mel.shape[0] % self.config.pad_to != 0:
            pad_frames = self.config.pad_to - (mel.shape[0] % self.config.pad_to)
            mel = mx.pad(
                mel,
                ((0, pad_frames), (0, 0)),
                constant_values=self.config.pad_value,
            )

        return mel.astype(mx.float32), seq_len

    def __call__(self, waveforms: Iterable) -> Tuple[mx.array, mx.array]:
        features: List[mx.array] = []
        lengths: List[int] = []

        for waveform in waveforms:
            mel, length = self._extract_single(waveform)
            features.append(mel)
            lengths.append(length)

        if not features:
            raise ValueError("At least one waveform is required.")

        max_frames = max(feature.shape[0] for feature in features)
        padded = []
        for feature in features:
            if feature.shape[0] < max_frames:
                feature = mx.pad(
                    feature,
                    ((0, max_frames - feature.shape[0]), (0, 0)),
                    constant_values=self.config.pad_value,
                )
            padded.append(feature)

        return mx.stack(padded, axis=0), mx.array(lengths, dtype=mx.int32)
