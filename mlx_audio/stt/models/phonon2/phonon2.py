"""Phonon-2 on the standard Parakeet graph, with exact packed encoder layers.

Packed layer integration adapted from fermion-research 0.2.11, Copyright
2026 Fermion Research, Apache-2.0.
"""

from os import PathLike

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.stt.models.parakeet.audio import log_mel_spectrogram
from mlx_audio.stt.models.parakeet.parakeet import ParakeetTDT

from .config import ModelConfig
from .packed import ENCODER_SUFFIXES, PackedFiveValueLinear


class Model(ParakeetTDT):
    def __init__(self, config: ModelConfig):
        decoding = config.decoding
        durations = decoding.durations
        greedy = decoding.greedy or {}
        maximum = greedy.get("max_symbols")
        if (
            decoding.model_type != "tdt"
            or not isinstance(durations, list)
            or not durations
            or any(type(d) is not int or d < 0 for d in durations)
            or len(durations) != config.joint.num_extra_outputs
            or type(maximum) is not int
            or maximum <= 0
        ):
            raise ValueError(
                "Phonon-2 requires valid durations and a positive max_symbols"
            )
        super().__init__(config)
        self.config = config
        expected = {
            f"encoder.layers.{i}.{suffix}"
            for i in range(self.encoder_config.n_layers)
            for suffix in ENCODER_SUFFIXES
        }
        modules = config.phonon2_modules
        names = [r["name"] for r in modules]
        if len(names) != len(set(names)) or set(names) != expected:
            raise ValueError(
                "Phonon-2 must replace every encoder projection exactly once"
            )
        self._packed_modules = []
        for row in modules:
            parts = row["name"].split(".")
            parent = self
            for part in parts[:-1]:
                parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
            old = getattr(parent, parts[-1])
            outputs, inputs = row["shape"]
            shape = (
                (outputs, 1, inputs)
                if isinstance(old, nn.Conv1d)
                else (outputs, inputs)
            )
            if tuple(old.weight.shape) != shape:
                raise ValueError(f"Phonon-2 encoder shape mismatch: {row['name']}")
            layer = PackedFiveValueLinear(inputs, outputs, bias="bias" in old)
            setattr(parent, parts[-1], layer)
            self._packed_modules.append(layer)

    def _log_mel_spectrogram(self, audio_data: mx.array) -> mx.array:
        if audio_data.ndim != 1 or audio_data.size == 0:
            raise ValueError("Expected a nonempty mono audio waveform")
        dtype = audio_data.dtype
        minimum = 2 * self.preprocessor_config.hop_length
        if audio_data.size < minimum:
            audio_data = mx.pad(audio_data, (0, minimum - audio_data.size))
        # Match Fermion's frontend: compute features in fp32, then cast for
        # inference. Keep this policy local to Phonon-2, including chunk tails.
        return log_mel_spectrogram(
            audio_data.astype(mx.float32), self.preprocessor_config
        ).astype(dtype)

    def load_weights(self, weights, strict=False):
        # Partial loads invalidate the source model's numerical contract.
        if isinstance(weights, (str, PathLike)):
            weights = list(mx.load(str(weights)).items())
        else:
            weights = list(weights)
        for name, value in weights:
            if name.endswith((".base_q", ".residual_q")) and value.dtype != mx.uint32:
                raise ValueError("Phonon-2 packed codes must remain uint32")
            if name.endswith((".row_lo", ".row_hi")) and value.dtype != mx.float16:
                raise ValueError("Phonon-2 row palettes must remain float16")
        result = super().load_weights(weights, strict=True)
        for module in self._packed_modules:
            module.materialize()
        return result
