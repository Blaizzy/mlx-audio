"""Causal LFM2 speech encoder and parallel LFM3 speech depth decoder."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_audio.lm.models.lfm3 import Lfm3Model


def _fp32(activation, x):
    return activation(x.astype(mx.float32)).astype(x.dtype)


class Downsampler(nn.Module):
    def __init__(self, config):
        super().__init__()
        c = config.conv_channels
        self.conv_in = nn.Conv2d(1, c, 3, stride=2, padding=1)
        self.conv_mid = nn.Conv2d(c, c, 3, stride=2, padding=1, groups=c)
        self.lin_mid = nn.Conv2d(c, c, 1)
        self.conv_end = nn.Conv2d(c, c, 3, stride=2, padding=1, groups=c)
        self.lin_end = nn.Conv2d(c, c, 1)
        self.lin_out = nn.Linear(config.num_mel_bins * c // 8, config.hidden_size)

    def __call__(self, features, lengths):
        def mask(x, n):
            return x * (mx.arange(x.shape[2]) < n[:, None])[:, None, :, None]

        x = mask(features[..., None], lengths)  # B, mel, time, channels
        x = _fp32(nn.silu, self.conv_in(x))
        lengths = (lengths + 1) // 2
        x = mask(x, lengths)
        x = _fp32(nn.silu, self.lin_mid(self.conv_mid(x)))
        lengths = (lengths + 1) // 2
        x = mask(x, lengths)
        x = _fp32(nn.silu, self.lin_end(self.conv_end(x)))
        b, f, t, c = x.shape
        # PyTorch flattens channel then mel, rather than mel then channel.
        x = x.transpose(0, 2, 3, 1).reshape(b, t, c * f)
        return self.lin_out(x)


class AudioEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.downsample = Downsampler(config)
        self.lfm = Lfm3Model(config.text_config, embed_tokens=False)

    def __call__(self, features, lengths=None):
        if features.ndim != 3 or features.shape[1] != self.config.num_mel_bins:
            raise ValueError(
                f"Expected features [clips, {self.config.num_mel_bins}, frames]"
            )
        if lengths is None:
            lengths = mx.full((features.shape[0],), features.shape[-1])
        x = self.downsample(
            features.astype(self.downsample.conv_in.weight.dtype), lengths
        )
        h = self.lfm(input_embeddings=x, window_size=self.config.sliding_window)
        counts = ((np.asarray(lengths) + 7) // 8).tolist()
        return mx.concatenate([h[i, :n] for i, n in enumerate(counts)])


class AudioProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config.audio_encoder_config.hidden_size
        out = config.text_config.hidden_size
        self.layer_norm = nn.LayerNorm(dim)
        self.linear_1 = nn.Linear(dim, out)
        self.linear_2 = nn.Linear(out, out)

    def __call__(self, x):
        return self.linear_2(_fp32(nn.gelu, self.linear_1(self.layer_norm(x))))
