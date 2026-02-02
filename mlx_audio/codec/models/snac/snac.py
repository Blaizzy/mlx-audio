import json
import math
from pathlib import Path
from typing import List, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from huggingface_hub import snapshot_download

from .layers import Decoder, Encoder
from .vq import ResidualVectorQuantize


class SNAC(nn.Module):
    def __init__(
        self,
        sampling_rate=44100,
        encoder_dim=64,
        encoder_rates=[3, 3, 7, 7],
        latent_dim=None,
        decoder_dim=1536,
        decoder_rates=[7, 7, 3, 3],
        attn_window_size=32,
        codebook_size=4096,
        codebook_dim=8,
        vq_strides=[8, 4, 2, 1],
        noise=True,
        depthwise=True,
    ):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        if latent_dim is None:
            latent_dim = encoder_dim * (2 ** len(encoder_rates))
        self.latent_dim = latent_dim
        self.hop_length = np.prod(encoder_rates)
        self.encoder = Encoder(
            encoder_dim,
            encoder_rates,
            depthwise=depthwise,
            attn_window_size=attn_window_size,
        )
        self.n_codebooks = len(vq_strides)
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.vq_strides = vq_strides
        self.attn_window_size = attn_window_size
        self.quantizer = ResidualVectorQuantize(
            input_dim=latent_dim,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            vq_strides=vq_strides,
        )
        self.decoder = Decoder(
            latent_dim,
            decoder_dim,
            decoder_rates,
            noise,
            depthwise=depthwise,
            attn_window_size=attn_window_size,
        )

    def preprocess(self, audio_data):
        length = audio_data.shape[-1]

        def lcm(a, b):
            return abs(a * b) // math.gcd(a, b)

        lcm_value = self.vq_strides[0]
        for i in range(1, len(self.vq_strides)):
            lcm_value = lcm(lcm_value, self.vq_strides[i])

        if self.attn_window_size:
            lcm_value = lcm(lcm_value, self.attn_window_size)

        pad_to = self.hop_length * lcm_value
        right_pad = math.ceil(length / pad_to) * pad_to - length

        # Pad the audio data
        audio_data = mx.pad(audio_data, [(0, 0), (0, 0), (0, right_pad)])
        return audio_data

    def __call__(self, audio_data: mx.array) -> Tuple[mx.array, List[mx.array]]:
        length = audio_data.shape[-1]
        audio_data = self.preprocess(audio_data)
        z = self.encoder(audio_data.moveaxis(1, 2))
        z_q, codes = self.quantizer(z)
        audio_hat = self.decoder(z_q)
        return audio_hat[..., :length], codes

    def encode(self, audio_data: mx.array) -> List[mx.array]:
        audio_data = self.preprocess(audio_data)
        z = self.encoder(audio_data.moveaxis(1, 2))
        _, codes = self.quantizer(z)
        return codes

    def decode(self, codes: List[mx.array]) -> mx.array:
        z_q = self.quantizer.from_codes(codes)
        audio_hat = self.decoder(z_q.moveaxis(1, 2))
        return audio_hat

    def decode_stream(
        self,
        codes: List[mx.array],
        prev_codes: List[mx.array] = None,
        context_frames: int = 8,
    ) -> Tuple[mx.array, List[mx.array]]:
        """Streaming decode that maintains context for consistent output.

        Args:
            codes: New code frames to decode [layer1, layer2, layer3]
            prev_codes: Previous context codes (None for first call)
            context_frames: Number of previous frames to use as context

        Returns:
            Tuple of (new_audio_samples, updated_context_codes)
        """
        if prev_codes is None:
            # First call - decode normally and save context
            audio = self.decode(codes)
            # Save last context_frames as context for next call
            new_context = [
                c[:, -context_frames:] if c.shape[1] > context_frames else c
                for c in codes
            ]
            return audio, new_context

        # Concatenate context with new codes for each layer
        # Need to handle different strides: layer0 has 1x, layer1 has 2x, layer2 has 4x the frames
        combined_codes = []
        for i, (prev, new) in enumerate(zip(prev_codes, codes)):
            # Adjust context size based on VQ stride
            stride = self.vq_strides[i]
            layer_context = max(1, context_frames // stride)
            if prev.shape[1] > layer_context:
                prev = prev[:, -layer_context:]
            combined = mx.concatenate([prev, new], axis=1)
            combined_codes.append(combined)

        # Decode the combined sequence
        full_audio = self.decode(combined_codes)

        # Calculate how many samples came from context
        # Each code frame produces hop_length samples after all upsampling
        context_samples = context_frames * self.hop_length

        # Return only new samples (after context region)
        if full_audio.shape[-1] > context_samples:
            new_audio = full_audio[..., context_samples:]
        else:
            new_audio = full_audio

        # Update context with end of current codes
        new_context = [
            c[:, -context_frames:] if c.shape[1] > context_frames else c for c in codes
        ]

        return new_audio, new_context

    def _extra_repr(self):
        return (
            f"sampling_rate={self.sampling_rate}, "
            f"encoder_dim={self.encoder_dim}, "
            f"encoder_rates={self.encoder_rates}, "
            f"latent_dim={self.latent_dim}, "
            f"decoder_dim={self.decoder_dim}, "
            f"decoder_rates={self.decoder_rates}, "
            f"codebook_size={self.codebook_size}, "
            f"codebook_dim={self.codebook_dim}, "
            f"vq_strides={self.vq_strides}"
        )

    @classmethod
    def from_config(cls, config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        model = cls(**config)
        return model

    @classmethod
    def from_pretrained(cls, repo_id, **kwargs):
        path = fetch_from_hub(repo_id)

        if path is None:
            raise ValueError(f"Could not find model {path}")

        model_path = path / "model.safetensors"
        config_path = path / "config.json"
        snac = cls.from_config(config_path)

        weights = mx.load(model_path.as_posix(), format="safetensors")
        snac.load_weights(list(weights.items()))
        mx.eval(snac.parameters())

        return snac


# fetch model from hub


def fetch_from_hub(hf_repo: str) -> Path:
    model_path = Path(
        snapshot_download(
            repo_id=hf_repo,
            allow_patterns=["*.safetensors", "*.json"],
        )
    )
    return model_path
