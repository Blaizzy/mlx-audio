from typing import List

import mlx.core as mx
import mlx.nn as nn
from einops.array_api import rearrange

from .layers import WNConv1d


class VectorQuantize(nn.Module):
    def __init__(
        self, input_dim: int, codebook_size: int, codebook_dim: int, stride: int = 1
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.stride = stride

        self.in_proj = WNConv1d(input_dim, codebook_dim, kernel_size=1)
        self.out_proj = WNConv1d(codebook_dim, input_dim, kernel_size=1)
        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    def __call__(self, z):
        z = z.moveaxis(1, 2)

        if self.stride > 1:
            kernel_size = self.stride
            stride = self.stride
            kernel = mx.ones((z.shape[2], kernel_size, 1)) / kernel_size
            z = mx.conv1d(z, kernel, stride=stride, padding=0, groups=z.shape[2])

        # Factorized codes - Project input into low-dimensional space
        z_e = self.in_proj(z).moveaxis(1, 2)  # z_e : (B x D x T)
        z_q, indices = self.decode_latents(z_e)

        # Straight-through estimator: z_e + stop_gradient(z_q - z_e)
        z_q = z_e + (z_q - z_e)

        z_q = self.out_proj(z_q.moveaxis(1, 2)).moveaxis(1, 2)

        if self.stride > 1:
            # Implement repeat_interleave
            shape = list(z_q.shape)
            shape[-1] *= self.stride
            # Create a new tensor with the expanded shape
            expanded = mx.zeros(shape)

            # Fill the expanded tensor with repeated values
            for i in range(self.stride):
                expanded[..., i :: self.stride] = z_q

            z_q = expanded

        return z_q, indices

    def embed_code(self, embed_id):
        return self.codebook.weight[embed_id]

    def decode_code(self, embed_id):
        return self.embed_code(embed_id).moveaxis(1, 2)

    def decode_latents(self, latents):
        encodings = rearrange(latents, "b d t -> (b t) d")
        codebook = self.codebook.weight  # codebook: (N x D)

        encodings = normalize(encodings)
        codebook = normalize(codebook)

        dist = (
            mx.power(encodings, 2).sum(1, keepdims=True)
            - 2 * encodings @ codebook.T
            + mx.power(codebook, 2).sum(1, keepdims=True).T
        )
        min_dist = (-dist).argmax(1)
        indices = rearrange(min_dist, "(b t) -> b t", b=latents.shape[0])
        z_q = self.decode_code(indices)
        return z_q, indices


class ResidualVectorQuantize(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        codebook_size: int = 1024,
        codebook_dim: int = 8,
        vq_strides: List[int] = [1, 1, 1, 1],
    ):
        super().__init__()
        self.n_codebooks = len(vq_strides)
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size
        self.quantizers = [
            VectorQuantize(input_dim, codebook_size, codebook_dim, stride)
            for stride in vq_strides
        ]

    def __call__(self, z):
        z_q = 0
        residual = z
        codes = []
        for i, quantizer in enumerate(self.quantizers):
            z_q_i, indices_i = quantizer(residual)
            z_q = z_q + z_q_i
            residual = residual - z_q_i
            codes.append(indices_i)

        return z_q, codes

    def from_codes(self, codes: List[mx.array]) -> mx.array:
        z_q = 0.0
        for i in range(self.n_codebooks):
            z_p_i = self.quantizers[i].decode_code(codes[i])
            z_q_i = self.quantizers[i].out_proj(z_p_i.moveaxis(1, 2)).moveaxis(1, 2)

            # Handle repeat_interleave for stride > 1
            if self.quantizers[i].stride > 1:
                stride = self.quantizers[i].stride
                shape = list(z_q_i.shape)
                shape[-1] *= stride
                expanded = mx.zeros(shape)

                for j in range(stride):
                    expanded[..., j::stride] = z_q_i

                z_q_i = expanded

            z_q += z_q_i

        return z_q


def normalize(x, p=2.0, dim=1, eps=1e-12):
    """L2 normalization function"""
    norm = mx.power(mx.sum(mx.power(mx.abs(x), p), axis=dim, keepdims=True), 1 / p)
    return x / mx.maximum(norm, eps)
