import math
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

SUPPORTED_VERSIONS = ["1.0.0"]


@dataclass
class DACFile:
    codes: mx.array

    # Metadata
    chunk_length: int
    original_length: int
    input_db: float
    channels: int
    sample_rate: int
    padding: bool
    dac_version: str

    def save(self, path):
        artifacts = {
            "codes": np.array(self.codes).astype(np.uint16),
            "metadata": {
                "input_db": self.input_db,
                "original_length": self.original_length,
                "sample_rate": self.sample_rate,
                "chunk_length": self.chunk_length,
                "channels": self.channels,
                "padding": self.padding,
                "dac_version": SUPPORTED_VERSIONS[-1],
            },
        }
        path = Path(path).with_suffix(".dac")
        with open(path, "wb") as f:
            np.save(f, artifacts)
        return path

    @classmethod
    def load(cls, path):
        artifacts = np.load(path, allow_pickle=True)[()]
        codes = mx.array(artifacts["codes"], dtype=mx.int32)
        if artifacts["metadata"].get("dac_version", None) not in SUPPORTED_VERSIONS:
            raise RuntimeError(
                f"Given file {path} can't be loaded with this version of descript-audio-codec."
            )
        return cls(codes=codes, **artifacts["metadata"])


class CodecMixin:
    @property
    def padding(self):
        if not hasattr(self, "_padding"):
            self._padding = True
        return self._padding

    @padding.setter
    def padding(self, value):
        assert isinstance(value, bool)

        layers = [
            layer
            for layer in self.modules()
            if isinstance(layer, (nn.Conv1d, nn.ConvTranspose1d))
        ]

        for layer in layers:
            if value:
                if hasattr(layer, "original_padding"):
                    layer.padding = layer.original_padding
            else:
                layer.original_padding = layer.padding
                layer.padding = tuple(0 for _ in range(len(layer.padding)))

        self._padding = value

    def get_delay(self):
        l_out = self.get_output_length(0)
        L = l_out

        layers = []
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.ConvTranspose1d)):
                layers.append(layer)

        for layer in reversed(layers):
            d = layer.dilation
            k = layer.weight.shape[1]
            s = layer.stride

            if isinstance(layer, nn.ConvTranspose1d):
                L = ((L - d * (k - 1) - 1) / s) + 1
            elif isinstance(layer, nn.Conv1d):
                L = (L - 1) * s + d * (k - 1) + 1

            L = math.ceil(L)

        l_in = L

        return (l_in - l_out) // 2

    def get_output_length(self, input_length):
        L = input_length
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.ConvTranspose1d)):
                d = layer.dilation
                k = layer.weight.shape[1]
                s = layer.stride

                if isinstance(layer, nn.Conv1d):
                    L = ((L - d * (k - 1) - 1) / s) + 1
                elif isinstance(layer, nn.ConvTranspose1d):
                    L = (L - 1) * s + d * (k - 1) + 1

                L = math.floor(L)
        return L

    def compress(
        self,
        audio_path: Union[str, Path],
        win_duration: float = 1.0,
        normalize_db: float = -16,
        n_quantizers: int = None,
    ) -> DACFile:
        from mlx_audio.audio_io import read as audio_read

        audio_signal, original_sr = audio_read(audio_path)
        signal_duration = audio_signal.shape[-1] / original_sr

        original_padding = self.padding
        if original_sr != self.sample_rate:
            raise ValueError(
                f"Sample rate of the audio signal ({original_sr}) does not match the sample rate of the model ({self.sample_rate})."
            )

        audio_data = mx.array(audio_signal)

        rms = mx.sqrt(mx.mean(mx.power(audio_data, 2), axis=-1) + 1e-12)
        input_db = 20 * mx.log10(rms / 1.0 + 1e-12)

        if normalize_db is not None:
            audio_data = audio_data * mx.power(10, (normalize_db - input_db) / 20)

        # rearrange "n -> 1 1 n"
        audio_data = audio_data[None, None, :]
        nb, nac, nt = audio_data.shape
        # rearrange "nb nac nt -> (nb nac) 1 nt"
        audio_data = audio_data.reshape(nb * nac, 1, nt)

        win_duration = signal_duration if win_duration is None else win_duration

        if signal_duration <= win_duration:
            self.padding = True
            n_samples = nt
            hop = nt
        else:
            self.padding = False
            audio_data = mx.pad(audio_data, [(0, 0), (0, 0), (self.delay, self.delay)])

            n_samples = int(win_duration * self.sample_rate)
            n_samples = int(math.ceil(n_samples / self.hop_length) * self.hop_length)
            hop = self.get_output_length(n_samples)

        codes = []
        for i in range(0, nt, hop):
            x = audio_data[..., i : i + n_samples]
            x = mx.pad(x, [(0, 0), (0, 0), (0, max(0, n_samples - x.shape[-1]))])

            x = self.preprocess(x, self.sample_rate)
            _, c, _, _, _ = self.encode(x, n_quantizers)
            codes.append(c)
            chunk_length = c.shape[-1]

        codes = mx.concatenate(codes, axis=-1)

        dac_file = DACFile(
            codes=codes,
            chunk_length=chunk_length,
            original_length=signal_duration,
            input_db=input_db,
            channels=nac,
            sample_rate=original_sr,
            padding=self.padding,
            dac_version=SUPPORTED_VERSIONS[-1],
        )

        if n_quantizers is not None:
            codes = codes[:, :n_quantizers, :]

        self.padding = original_padding
        return dac_file

    def decompress(self, obj: Union[str, Path, DACFile]) -> mx.array:
        if isinstance(obj, (str, Path)):
            obj = DACFile.load(obj)

        if self.sample_rate != obj.sample_rate:
            raise ValueError(
                f"Sample rate of the audio signal ({obj.sample_rate}) does not match the sample rate of the model ({self.sample_rate})."
            )

        original_padding = self.padding
        self.padding = obj.padding

        codes = obj.codes
        chunk_length = obj.chunk_length
        recons = []

        for i in range(0, codes.shape[-1], chunk_length):
            c = codes[..., i : i + chunk_length]
            z = self.quantizer.from_codes(c)[0]
            r = self.decode(z)
            recons.append(r)

        recons = mx.concatenate(recons, axis=1)
        # rearrange "1 n 1 -> 1 n" (squeeze last dim)
        recons = recons.squeeze(-1)

        target_db = obj.input_db
        normalize_db = -16

        if normalize_db is not None:
            recons = recons * mx.power(10, (target_db - normalize_db) / 20)

        self.padding = original_padding
        return recons
