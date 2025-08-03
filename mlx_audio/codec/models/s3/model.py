from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from einops.array_api import rearrange

from .utils import make_non_pad_mask, mask_to_bias


@dataclass
class ModelConfig:
    n_mels: int = 128
    n_audio_ctx: int = 1500
    n_audio_state: int = 1280
    n_audio_head: int = 20
    n_audio_layer: int = 6
    n_codebook_size: int = 4096


def sinusoids(length: int, channels: int, max_timescale: float = 10000) -> mx.array:
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = mx.exp(-log_timescale_increment * mx.arange(channels // 2))
    scaled_time = mx.arange(length)[:, None] * inv_timescales[None, :]
    return mx.concatenate([mx.sin(scaled_time), mx.cos(scaled_time)], axis=1)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = nn.Linear(n_state, n_state)
        self.key = nn.Linear(n_state, n_state, bias=False)
        self.value = nn.Linear(n_state, n_state)
        self.out = nn.Linear(n_state, n_state)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array]:
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

    def qkv_attention(
        self, q: mx.array, k: mx.array, v: mx.array, mask: Optional[mx.array] = None
    ) -> Tuple[mx.array, mx.array | None]:
        B, T, D = q.shape
        scale = (D // self.n_head) ** -0.25

        q = q.reshape(B, T, self.n_head, -1).transpose(0, 2, 1, 3) * scale
        k = k.reshape(B, T, self.n_head, -1).transpose(0, 2, 1, 3) * scale
        v = v.reshape(B, T, self.n_head, -1).transpose(0, 2, 1, 3)

        output = mx.fast.scaled_dot_product_attention(q, k, v, scale=1, mask=mask)
        output = output.transpose(0, 2, 1, 3).reshape(B, T, D)
        return output, None


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = nn.LayerNorm(n_state)

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            nn.Linear(n_state, n_mlp), nn.GELU(), nn.Linear(n_mlp, n_state)
        )
        self.mlp_ln = nn.LayerNorm(n_state)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        x = x + self.attn(self.attn_ln(x), mask=mask)[0]
        x = x + self.mlp(self.mlp_ln(x))
        return x


class AudioEncoder(nn.Module):
    def __init__(
        self,
        n_mels: int,
        n_ctx: int,
        n_state: int,
        n_head: int,
        n_layer: int,
        stride: int,
    ):
        super().__init__()
        self.stride = stride

        self.conv1 = nn.Conv1d(
            in_channels=n_mels,
            out_channels=n_state,
            kernel_size=3,
            stride=stride,
            padding=1,
        )
        self.conv2 = nn.Conv1d(
            in_channels=n_state,
            out_channels=n_state,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.positional_embedding = sinusoids(n_ctx, n_state)

        self.blocks = [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]

    def __call__(self, x: mx.array, x_len: mx.array) -> Tuple[mx.array, mx.array]:
        """
        x : mx.array, shape = (batch_size, n_mels, T)
            the mel spectrogram of the audio
        x_len: mx.array, shape = (batch_size,)
            length of each audio in x
        """
        mask = make_non_pad_mask(x_len)
        mask = mx.expand_dims(mask, axis=1)  # (B, 1, T)

        x = x.transpose(0, 2, 1)  # (B, T, n_mels)
        mask_transposed = mask.transpose(0, 2, 1)  # (B, T, 1)

        x = self.conv1(x * mask_transposed)
        x = nn.gelu(x)
        x_len = (x_len + 2 - 1 * (3 - 1) - 1) // self.stride + 1

        mask = make_non_pad_mask(x_len)
        mask_transposed = mx.expand_dims(mask, axis=-1)  # (B, T, 1)
        x = self.conv2(x * mask_transposed)
        x = nn.gelu(x)

        x_len = (x_len + 2 - 1 * (3 - 1) - 1) // 2 + 1

        mask = make_non_pad_mask(x_len)
        mask = mask_to_bias(mask, x.dtype)
        mask = mx.expand_dims(mask, axis=1)  # (B, 1, T)

        x = x + self.positional_embedding[: x.shape[1], :]

        for block in self.blocks:
            x = block(x, mx.expand_dims(mask, axis=1))

        return x, x_len


class EuclideanCodebook(nn.Module):
    """Codebook with Euclidean distance.
    Args:
        dim (int): Dimension.
        codebook_size (int): Codebook size.
    """

    def __init__(self, dim: int, codebook_size: int):
        super().__init__()
        self.codebook_size = codebook_size
        self.embed = mx.zeros((codebook_size, dim))

    def preprocess(self, x: mx.array) -> mx.array:
        x = rearrange(x, "... d -> (...) d")
        return x

    def quantize(self, x: mx.array) -> mx.array:
        embed = self.embed.T
        dist = -(
            mx.sum(x.astype(mx.float32) ** 2, axis=1, keepdims=True)
            - 2 * x @ embed
            + mx.sum(embed.astype(mx.float32) ** 2, axis=0, keepdims=True)
        )
        embed_ind = mx.argmax(dist, axis=-1)
        return embed_ind

    def postprocess_emb(self, embed_ind: mx.array, shape: tuple) -> mx.array:
        return embed_ind.reshape(*shape[:-1])

    def dequantize(self, embed_ind: mx.array) -> mx.array:
        quantize = self.embed[embed_ind]
        return quantize

    def encode(self, x: mx.array) -> mx.array:
        shape = x.shape
        # pre-process
        x = self.preprocess(x)
        # quantize
        embed_ind = self.quantize(x)
        # post-process
        embed_ind = self.postprocess_emb(embed_ind, shape)
        return embed_ind

    def decode(self, embed_ind: mx.array) -> mx.array:
        quantize = self.dequantize(embed_ind)
        return quantize


class VectorQuantization(nn.Module):
    """Vector quantization implementation
    Args:
        dim (int): Dimension
        codebook_size (int): Codebook size
    """

    def __init__(self, dim: int, codebook_size: int):
        super().__init__()
        self._codebook = EuclideanCodebook(dim=dim, codebook_size=codebook_size)
        self.codebook_size = codebook_size

    @property
    def codebook(self):
        return self._codebook.embed

    def encode(self, x: mx.array) -> mx.array:
        x = x / mx.sqrt(mx.sum(x**2, axis=-1, keepdims=True) + 1e-8)
        embed_in = self._codebook.encode(x)
        return embed_in

    def decode(self, embed_ind: mx.array) -> mx.array:
        quantize = self._codebook.decode(embed_ind)
        quantize = rearrange(quantize, "b n d -> b d n")
        return quantize


class S3Tokenizer(nn.Module):
    """S3 tokenizer implementation
    Args:
        config (ModelConfig): Config
    """

    def __init__(self, name: str, config: ModelConfig = ModelConfig()):
        super().__init__()
        self.config = config
        self.encoder = AudioEncoder(
            self.config.n_mels,
            self.config.n_audio_ctx,
            self.config.n_audio_state,
            self.config.n_audio_head,
            self.config.n_audio_layer,
            2 if name == "speech_tokenizer_v1_25hz" else 1,
        )
        self.quantizer = VectorQuantization(
            self.config.n_audio_state, self.config.n_codebook_size
        )

    def __call__(self, mel: mx.array, mel_len: mx.array) -> Tuple[mx.array, mx.array]:
        return self.quantize(mel, mel_len)

    def quantize(self, mel: mx.array, mel_len: mx.array) -> Tuple[mx.array, mx.array]:
        hidden, code_len = self.encoder(mel, mel_len)
        code = self.quantizer.encode(hidden)
        return code, code_len
