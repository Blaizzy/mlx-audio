from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_audio.codec.models.bigvgan.conv import WNConv1d
from mlx_audio.codec.models.vocos.vocos import VocosBackbone


def _l2_normalize(x: mx.array, axis: int = -1, eps: float = 1e-12) -> mx.array:
    denom = mx.sqrt(mx.maximum(mx.sum(x * x, axis=axis, keepdims=True), eps))
    return x / denom


class FactorizedVectorQuantize(nn.Module):
    """MLX port of Amphion FactorizedVectorQuantize.

    Expects latents shaped (B, T, D).
    """

    def __init__(
        self,
        input_dim: int,
        codebook_size: int,
        codebook_dim: int,
        use_l2_normlize: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.use_l2_normlize = use_l2_normlize

        if self.input_dim != self.codebook_dim:
            self.in_project = WNConv1d(self.input_dim, self.codebook_dim, kernel_size=1)
            self.out_project = WNConv1d(self.codebook_dim, self.input_dim, kernel_size=1)
        else:
            self.in_project = nn.Identity()
            self.out_project = nn.Identity()

        self.codebook = nn.Embedding(self.codebook_size, self.codebook_dim)

    def decode_latents(self, latents: mx.array) -> Tuple[mx.array, mx.array]:
        # latents: (B, T, D)
        B, T, D = latents.shape
        enc = latents.reshape(B * T, D)

        codebook = self.codebook.weight  # (K, D)
        if self.use_l2_normlize:
            enc = _l2_normalize(enc, axis=-1)
            codebook_n = _l2_normalize(codebook, axis=-1)
        else:
            codebook_n = codebook

        # Squared euclidean distance
        dist = (
            mx.sum(enc * enc, axis=1, keepdims=True)
            - 2.0 * (enc @ codebook_n.T)
            + mx.sum(codebook_n * codebook_n, axis=1)[None, :]
        )  # (B*T, K)
        indices = mx.argmax(-dist, axis=1).reshape(B, T)

        z_q = self.codebook(indices)  # (B, T, D)
        return z_q, indices

    def __call__(self, z: mx.array) -> Tuple[mx.array, mx.array]:
        # z: (B, T, D)
        z_e = self.in_project(z)
        z_q, indices = self.decode_latents(z_e)
        z_q = self.out_project(z_q)
        return z_q, indices

    def vq2emb(self, vq: mx.array, *, out_proj: bool = True) -> mx.array:
        emb = self.codebook(vq)
        if out_proj:
            emb = self.out_project(emb)
        return emb


class ResidualVQ(nn.Module):
    """MLX port of Amphion ResidualVQ (inference-only).

    Expects latents shaped (B, T, D).
    """

    def __init__(
        self,
        input_dim: int,
        num_quantizers: int,
        codebook_size: int,
        codebook_dim: int,
        use_l2_normlize: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        self.quantizers = [
            FactorizedVectorQuantize(
                input_dim=input_dim,
                codebook_size=codebook_size,
                codebook_dim=codebook_dim,
                use_l2_normlize=use_l2_normlize,
            )
            for _ in range(num_quantizers)
        ]

    def __call__(self, z: mx.array, *, n_quantizers: Optional[int] = None):
        if n_quantizers is None:
            n_quantizers = self.num_quantizers

        quantized_out = mx.zeros_like(z)
        residual = z
        all_indices = []

        for i, q in enumerate(self.quantizers):
            if i >= n_quantizers:
                break
            z_q_i, idx_i = q(residual)
            quantized_out = quantized_out + z_q_i
            residual = residual - z_q_i
            all_indices.append(idx_i)

        all_indices = mx.stack(all_indices, axis=0)  # (N, B, T)
        return quantized_out, all_indices

    def vq2emb(self, vq: mx.array, *, n_quantizers: Optional[int] = None) -> mx.array:
        if n_quantizers is None:
            n_quantizers = self.num_quantizers

        out = 0.0
        for i, q in enumerate(self.quantizers):
            if i >= n_quantizers:
                break
            out = out + q.vq2emb(vq[i])
        return out


@dataclass
class RepCodecConfig:
    codebook_size: int = 8192
    hidden_size: int = 1024
    codebook_dim: int = 8
    vocos_dim: int = 384
    vocos_intermediate_dim: int = 2048
    vocos_num_layers: int = 12
    num_quantizers: int = 1
    downsample_scale: int = 1


class RepCodec(nn.Module):
    """MLX port of MaskGCT RepCodec (semantic codec)."""

    def __init__(self, cfg: RepCodecConfig):
        super().__init__()
        self.cfg = cfg

        self.codebook_size = cfg.codebook_size
        self.codebook_dim = cfg.codebook_dim
        self.hidden_size = cfg.hidden_size

        if cfg.downsample_scale and cfg.downsample_scale > 1:
            self.down = nn.Conv1d(
                cfg.hidden_size, cfg.hidden_size, kernel_size=3, stride=2, padding=1
            )
            self.up = nn.Conv1d(
                cfg.hidden_size, cfg.hidden_size, kernel_size=3, stride=1, padding=1
            )
        else:
            self.down = None
            self.up = None

        self.encoder = [
            VocosBackbone(
                input_channels=cfg.hidden_size,
                dim=cfg.vocos_dim,
                intermediate_dim=cfg.vocos_intermediate_dim,
                num_layers=cfg.vocos_num_layers,
                adanorm_num_embeddings=None,
            ),
            nn.Linear(cfg.vocos_dim, cfg.hidden_size),
        ]
        self.decoder = [
            VocosBackbone(
                input_channels=cfg.hidden_size,
                dim=cfg.vocos_dim,
                intermediate_dim=cfg.vocos_intermediate_dim,
                num_layers=cfg.vocos_num_layers,
                adanorm_num_embeddings=None,
            ),
            nn.Linear(cfg.vocos_dim, cfg.hidden_size),
        ]

        self.quantizer = ResidualVQ(
            input_dim=cfg.hidden_size,
            num_quantizers=cfg.num_quantizers,
            codebook_size=cfg.codebook_size,
            codebook_dim=cfg.codebook_dim,
            use_l2_normlize=True,
        )

    def _encode(self, x: mx.array) -> mx.array:
        # x: (B, T, C)
        y = self.encoder[0](x)
        y = self.encoder[1](y)
        return y

    def _decode(self, x: mx.array) -> mx.array:
        y = self.decoder[0](x)
        y = self.decoder[1](y)
        return y

    def quantize(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        # Downsample (optional)
        if self.down is not None:
            x = self.down(x)
            x = nn.gelu(x)

        x = self._encode(x)
        quantized_out, all_indices = self.quantizer(x)
        # Match torch method return: indices squeezed when N==1
        if all_indices.shape[0] == 1:
            return all_indices[0], quantized_out
        return all_indices, quantized_out

    def vq2emb(self, vq: mx.array, *, n_quantizers: Optional[int] = None) -> mx.array:
        return self.quantizer.vq2emb(vq, n_quantizers=n_quantizers)

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Best-effort sanitizer for PyTorch -> MLX weight layout.

        - Conv1d: (O, I, K) -> (O, K, I)
        - Depthwise conv weights inside VocosBackbone already handled by shape check.
        """
        curr = dict(tree_flatten(self.parameters()))
        out = {}
        for k, v in weights.items():
            if k not in curr:
                out[k] = v
                continue

            if v.ndim == 3 and curr[k].ndim == 3 and v.shape != curr[k].shape:
                # Torch conv1d (O,I,K) -> MLX (O,K,I)
                if v.shape[0] == curr[k].shape[0] and v.shape[1] == curr[k].shape[2]:
                    v = v.transpose(0, 2, 1)

            out[k] = v
        return out
