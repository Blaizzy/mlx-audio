import json
import math
from pathlib import Path
from typing import List, Literal, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from huggingface_hub import snapshot_download

from .base import CodecMixin
from .nn.layers import Snake1d, WNConv1d, WNConvTranspose1d
from .nn.quantize import ResidualVectorQuantize


class ResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2

        self.block = nn.Sequential(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def __call__(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


class EncoderBlock(nn.Module):
    def __init__(self, dim: int = 16, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1),
            ResidualUnit(dim // 2, dilation=3),
            ResidualUnit(dim // 2, dilation=9),
            Snake1d(dim // 2),
            WNConv1d(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def __call__(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        strides: list = [2, 4, 8, 8],
        d_latent: int = 64,
    ):
        super().__init__()
        self.block = [WNConv1d(1, d_model, kernel_size=7, padding=3)]

        for stride in strides:
            d_model *= 2
            self.block += [EncoderBlock(d_model, stride=stride)]

        self.block += [
            Snake1d(d_model),
            WNConv1d(d_model, d_latent, kernel_size=3, padding=1),
        ]

        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model

    def __call__(self, x):
        return self.block(x).moveaxis(1, 2)


class DecoderBlock(nn.Module):
    def __init__(self, input_dim: int = 16, output_dim: int = 8, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(input_dim),
            WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
            ResidualUnit(output_dim, dilation=1),
            ResidualUnit(output_dim, dilation=3),
            ResidualUnit(output_dim, dilation=9),
        )

    def __call__(self, x):
        return self.block(x)


class Decoder(nn.Module):
    def __init__(
        self,
        input_channel,
        channels,
        rates,
        d_out: int = 1,
    ):
        super().__init__()
        layers = [WNConv1d(input_channel, channels, kernel_size=7, padding=3)]

        for i, stride in enumerate(rates):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            layers += [DecoderBlock(input_dim, output_dim, stride)]

        layers += [
            Snake1d(output_dim),
            WNConv1d(output_dim, d_out, kernel_size=7, padding=3),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

    def __call__(self, x):
        return self.model(x)


class DAC(nn.Module, CodecMixin):
    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: List[int] = [2, 4, 5, 8],
        latent_dim: int = None,
        decoder_dim: int = 1536,
        decoder_rates: List[int] = [8, 5, 4, 2],
        n_codebooks: int = 32,
        codebook_size: int = 1024,
        codebook_dim: Union[int, list] = 8,
        sample_rate: int = 44100,
        **kwargs,
    ):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        self.sample_rate = sample_rate

        if latent_dim is None:
            latent_dim = encoder_dim * (2 ** len(encoder_rates))

        self.latent_dim = latent_dim

        self.hop_length = np.prod(encoder_rates)
        self.encoder = Encoder(encoder_dim, encoder_rates, latent_dim)

        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.quantizer = ResidualVectorQuantize(
            input_dim=latent_dim,
            n_codebooks=n_codebooks,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
        )

        self.decoder = Decoder(
            latent_dim,
            decoder_dim,
            decoder_rates,
        )

        self.sample_rate = sample_rate

        self.delay = self.get_delay()

    def preprocess(self, audio_data, sample_rate):
        if sample_rate is None:
            sample_rate = self.sample_rate
        assert sample_rate == self.sample_rate

        length = audio_data.shape[-1]
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        audio_data = mx.pad(audio_data, [(0, 0), (0, 0), (0, right_pad)])

        return audio_data

    def encode(
        self,
        audio_data: mx.array,
        n_quantizers: int = None,
    ):
        z = self.encoder(audio_data.moveaxis(1, 2))
        z, codes, latents, commitment_loss, codebook_loss = self.quantizer(
            z, n_quantizers
        )
        return z, codes, latents, commitment_loss, codebook_loss

    def decode(self, z: mx.array):
        return self.decoder(z.moveaxis(1, 2))

    def _extra_repr(self):
        return (
            f"encoder_dim={self.encoder_dim}, "
            f"encoder_rates={self.encoder_rates}, "
            f"latent_dim={self.latent_dim}, "
            f"decoder_dim={self.decoder_dim}, "
            f"decoder_rates={self.decoder_rates}, "
            f"n_codebooks={self.n_codebooks}, "
            f"codebook_size={self.codebook_size}, "
            f"codebook_dim={self.codebook_dim}"
        )

    def __call__(
        self,
        audio_data: mx.array,
        sample_rate: int = None,
        n_quantizers: int = None,
        use_rvq: bool = True,
        return_loss: bool = False,
    ):
        length = audio_data.shape[-1]
        audio_data = self.preprocess(audio_data, sample_rate)

        if use_rvq:
            z, codes, latents, commitment_loss, codebook_loss = self.encode(
                audio_data, n_quantizers
            )
        else:
            z = self.encoder(audio_data.moveaxis(1, 2))

        x = self.decode(z)

        if return_loss:
            return mx.losses.mse(x, audio_data)

        return {
            "audio": x[..., :length],
            "z": z,
            "codes": codes,
            "latents": latents,
            "vq/commitment_loss": commitment_loss,
            "vq/codebook_loss": codebook_loss,
        }

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
    ) -> "DAC":
        path = fetch_from_hub(repo_id)
        if path is None:
            raise ValueError(f"Could not find model {path}")

        model_path = path / "model.safetensors"
        config_path = path / "config.json"

        with open(config_path) as f:
            config = json.load(f)

        dac = DAC(**config)

        weights = mx.load(model_path.as_posix(), format="safetensors")
        dac.load_weights(list(weights.items()))
        mx.eval(dac.parameters())

        return dac


# fetch model from hub


def fetch_from_hub(hf_repo: str) -> Path:
    model_path = Path(
        snapshot_download(
            repo_id=hf_repo,
            allow_patterns=["*.safetensors", "*.json"],
        )
    )
    return model_path
