from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from einops.array_api import rearrange
from huggingface_hub import snapshot_download

from .model import MultiHeadAttention
from .utils import make_non_pad_mask, mask_to_bias


@dataclass
class ModelConfig:
    n_mels: int = 128
    n_audio_ctx: int = 1500
    n_audio_state: int = 1280
    n_audio_head: int = 20
    n_audio_layer: int = 6
    n_codebook_size: int = 3**8


def precompute_freqs_cis(
    dim: int, end: int, theta: float = 10000.0, scaling: Optional[float] = None
) -> mx.array:
    """Precompute frequency tensor for rotary embeddings"""
    freqs = 1.0 / (
        theta ** (mx.arange(0, dim, 2)[: (dim // 2)].astype(mx.float32) / dim)
    )
    t = mx.arange(end)
    if scaling is not None:
        t = t * scaling
    freqs = mx.outer(t, freqs).astype(mx.float32)
    cos_freqs = mx.cos(freqs)
    sin_freqs = mx.sin(freqs)
    cos_freqs = mx.concatenate([cos_freqs, cos_freqs], axis=-1)
    sin_freqs = mx.concatenate([sin_freqs, sin_freqs], axis=-1)
    return cos_freqs, sin_freqs


def apply_rotary_emb(
    xq: mx.array,
    xk: mx.array,
    cos: mx.array,
    sin: mx.array,
) -> Tuple[mx.array, mx.array]:
    """Apply rotary embeddings to query and key tensors"""
    # Expand dimensions for broadcasting
    cos = mx.expand_dims(mx.expand_dims(cos, axis=0), axis=2)
    sin = mx.expand_dims(mx.expand_dims(sin, axis=0), axis=2)

    D = xq.shape[-1]
    # Split and rotate
    xq_half_l, xq_half_r = xq[..., : D // 2], xq[..., D // 2 :]
    xq_rotated = mx.concatenate([-xq_half_r, xq_half_l], axis=-1)

    xk_half_l, xk_half_r = xk[..., : D // 2], xk[..., D // 2 :]
    xk_rotated = mx.concatenate([-xk_half_r, xk_half_l], axis=-1)

    # Apply rotation
    xq_out = xq * cos + xq_rotated * sin
    xk_out = xk * cos + xk_rotated * sin

    return xq_out, xk_out


class FSQCodebook(nn.Module):
    """Finite Scalar Quantization Codebook"""

    def __init__(self, dim: int, level: int = 3):
        super().__init__()
        self.project_down = nn.Linear(dim, 8)
        self.level = level
        self.embed = None

    def preprocess(self, x: mx.array) -> mx.array:
        x = rearrange(x, "... d -> (...) d")
        return x

    def encode(self, x: mx.array) -> mx.array:
        x_shape = x.shape
        # pre-process
        x = self.preprocess(x)
        # quantize
        h = self.project_down(x).astype(mx.float32)
        h = mx.tanh(h)
        h = h * 0.9990000128746033
        h = mx.round(h) + 1

        # Create powers for base conversion
        powers = mx.power(self.level, mx.arange(2**self.level, dtype=h.dtype))
        mu = mx.sum(h * mx.expand_dims(powers, axis=0), axis=-1)
        ind = mu.reshape(x_shape[0], x_shape[1]).astype(mx.int32)
        return ind

    def decode(self, embed_ind: mx.array) -> mx.array:
        raise NotImplementedError("There is no official up project component provided")


class FSQVectorQuantization(nn.Module):
    """Finite Scalar Quantization Vector Quantization"""

    def __init__(
        self,
        dim: int,
        codebook_size: int,
    ):
        super().__init__()
        assert 3**8 == codebook_size
        self.fsq_codebook = FSQCodebook(dim=dim, level=3)
        self.codebook_size = codebook_size

    @property
    def codebook(self):
        return self.fsq_codebook.embed

    def encode(self, x: mx.array) -> mx.array:
        return self.fsq_codebook.encode(x)

    def decode(self, embed_ind: mx.array) -> mx.array:
        quantize = self.fsq_codebook.decode(embed_ind)
        quantize = rearrange(quantize, "b n d -> b d n")
        return quantize


class FSMNMultiHeadAttention(MultiHeadAttention):
    """Multi-head attention with FSMN (Feedforward Sequential Memory Network)"""

    def __init__(
        self,
        n_state: int,
        n_head: int,
        kernel_size: int = 31,
    ):
        super().__init__(n_state, n_head)

        self.fsmn_block = nn.Conv1d(
            in_channels=n_state,
            out_channels=n_state,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
            groups=n_state,
            bias=False,
        )
        self.left_padding = (kernel_size - 1) // 2
        self.right_padding = kernel_size - 1 - self.left_padding

    def forward_fsmn(
        self, inputs: mx.array, mask: Optional[mx.array] = None
    ) -> mx.array:
        b, t, n, d = inputs.shape
        inputs = inputs.reshape(b, t, -1)

        if mask is not None and mask.shape[2] > 0:
            inputs = inputs * mask

        pad_left = mx.zeros((b, self.left_padding, inputs.shape[2]), dtype=inputs.dtype)
        pad_right = mx.zeros(
            (b, self.right_padding, inputs.shape[2]), dtype=inputs.dtype
        )
        x_padded = mx.concatenate([pad_left, inputs, pad_right], axis=1)
        x = self.fsmn_block(x_padded)
        x = x + inputs

        if mask is not None:
            x = x * mask

        return x

    def qkv_attention(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        mask: Optional[mx.array] = None,
        mask_pad: Optional[mx.array] = None,
        freqs_cis: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, mx.array | None, mx.array]:
        B, T, D = q.shape
        scale = (D // self.n_head) ** -0.25

        q = q.reshape(B, T, self.n_head, -1)
        k = k.reshape(B, T, self.n_head, -1)
        v = v.reshape(B, T, self.n_head, -1)

        if freqs_cis is not None:
            cos, sin = freqs_cis
            q, k = apply_rotary_emb(q, k, cos[:T], sin[:T])

        fsm_memory = self.forward_fsmn(v, mask_pad)

        q = q.transpose(0, 2, 1, 3) * scale
        k = k.transpose(0, 2, 1, 3) * scale
        v = v.transpose(0, 2, 1, 3)

        output = mx.fast.scaled_dot_product_attention(q, k, v, scale=1, mask=mask)
        output = output.transpose(0, 2, 1, 3).reshape(B, T, D)

        return output, None, fsm_memory

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        mask_pad: Optional[mx.array] = None,
        freqs_cis: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, mx.array | None]:
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        wv, qk, fsm_memory = self.qkv_attention(q, k, v, mask, mask_pad, freqs_cis)
        return self.out(wv) + fsm_memory, qk


class ResidualAttentionBlock(nn.Module):
    """Residual attention block with FSMN"""

    def __init__(
        self,
        n_state: int,
        n_head: int,
        kernel_size: int = 31,
    ):
        super().__init__()

        self.attn = FSMNMultiHeadAttention(n_state, n_head, kernel_size)
        self.attn_ln = nn.LayerNorm(n_state, eps=1e-6)

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            nn.Linear(n_state, n_mlp), nn.GELU(), nn.Linear(n_mlp, n_state)
        )
        self.mlp_ln = nn.LayerNorm(n_state)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        mask_pad: Optional[mx.array] = None,
        freqs_cis: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        x = (
            x
            + self.attn(
                self.attn_ln(x), mask=mask, mask_pad=mask_pad, freqs_cis=freqs_cis
            )[0]
        )

        x = x + self.mlp(self.mlp_ln(x))
        return x


class AudioEncoderV2(nn.Module):
    def __init__(
        self,
        n_mels: int,
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

        self._freqs_cis = precompute_freqs_cis(64, 1024 * 2)

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
        mask_pad = mx.expand_dims(mask, axis=-1)  # (B, T, 1)
        mask = mask_to_bias(mask, x.dtype)
        mask = mx.expand_dims(mask, axis=1)  # (B, 1, T)

        for block in self.blocks:
            x = block(x, mask, mask_pad, self._freqs_cis)

        return x, x_len


class S3TokenizerV2(nn.Module):
    """S3 tokenizer v2 implementation.
    Args:
        config (ModelConfig): Config
    """

    def __init__(self, name: str, config: ModelConfig = ModelConfig()):
        super().__init__()
        if "v1" not in name:
            assert "v2" in name
            config.n_codebook_size = 3**8
        self.config = config
        self.encoder = AudioEncoderV2(
            self.config.n_mels,
            self.config.n_audio_state,
            self.config.n_audio_head,
            self.config.n_audio_layer,
            2,
        )
        self.quantizer = FSQVectorQuantization(
            self.config.n_audio_state,
            self.config.n_codebook_size,
        )

    def __call__(self, mel: mx.array, mel_len: mx.array) -> Tuple[mx.array, mx.array]:
        return self.quantize(mel, mel_len)

    def quantize(self, mel: mx.array, mel_len: mx.array) -> Tuple[mx.array, mx.array]:
        hidden, code_len = self.encoder(mel, mel_len)
        code = self.quantizer.encode(hidden)
        return code, code_len

    @classmethod
    def from_pretrained(
        cls,
        name: str = "speech_tokenizer_v2_25hz",
        repo_id: str = "mlx-community/CosyVoice2-0.5B-S3Tokenizer",
    ) -> "S3TokenizerV2":
        path = fetch_from_hub(repo_id)
        if path is None:
            raise ValueError(f"Could not find model {path}")

        model = S3TokenizerV2(name)
        model_path = path / f"{name}.safetensors"
        weights = mx.load(model_path.as_posix(), format="safetensors")
        model.load_weights(list(weights.items()))
        mx.eval(model.parameters())

        return model


# fetch model from hub


def fetch_from_hub(hf_repo: str) -> Path:
    model_path = Path(
        snapshot_download(
            repo_id=hf_repo,
            allow_patterns=["*.safetensors", "*.json"],
        )
    )
    return model_path
