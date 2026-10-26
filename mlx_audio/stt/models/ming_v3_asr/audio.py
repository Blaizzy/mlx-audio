"""Variable-length Whisper encoder and Ming's log-mel frontend.

The encoder follows OpenAI Whisper (MIT) and inclusionAI's Ming V3 ASR
implementation (Apache-2.0). It processes each chunk independently.
"""

import math

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.stt.models.whisper.audio import log_mel_spectrogram


class AudioAttention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width)
        self.out = nn.Linear(width, width)

    def __call__(self, x):
        b, length, width = x.shape
        q, k, v = [
            projection(x).reshape(b, length, self.heads, -1).transpose(0, 2, 1, 3)
            for projection in (self.query, self.key, self.value)
        ]
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=(width // self.heads) ** -0.5
        )
        return self.out(out.transpose(0, 2, 1, 3).reshape(b, length, width))


class AudioBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.attn = AudioAttention(width, heads)
        self.attn_ln = nn.LayerNorm(width)
        self.mlp_ln = nn.LayerNorm(width)
        self.mlp = [nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width)]

    def __call__(self, x):
        x = x + self.attn(self.attn_ln(x))
        y = self.mlp_ln(x)
        for layer in self.mlp:
            y = layer(y)
        return x + y


class AudioEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.n_audio_state
        self.conv1 = nn.Conv1d(config.n_mels, width, 3, padding=1)
        self.conv2 = nn.Conv1d(width, width, 3, stride=2, padding=1)
        inverse = mx.exp(-math.log(10000) * mx.arange(width // 2) / (width // 2 - 1))
        positions = mx.arange(config.n_audio_ctx)[:, None] * inverse
        self.positional_embedding = mx.concatenate(
            [mx.sin(positions), mx.cos(positions)], axis=-1
        )
        self.blocks = [
            AudioBlock(width, config.n_audio_head) for _ in range(config.n_audio_layer)
        ]
        self.ln_post = nn.LayerNorm(width)

    def __call__(self, features):
        x = nn.gelu(self.conv1(features))
        x = nn.gelu(self.conv2(x))
        if x.shape[1] > self.positional_embedding.shape[0]:
            raise ValueError("Audio chunk exceeds the encoder positional context")
        x = (x + self.positional_embedding[: x.shape[1]]).astype(x.dtype)
        for block in self.blocks:
            x = block(x)
        return self.ln_post(x)


def extract_features(waveform, n_mels=128):
    if waveform.ndim != 1 or waveform.size <= 200:
        raise ValueError("Audio must be mono with more than 200 samples at 16 kHz")
    if not bool(mx.all(mx.isfinite(waveform))):
        raise ValueError("Audio must contain finite samples")
    return log_mel_spectrogram(waveform, n_mels=n_mels).astype(mx.bfloat16)
