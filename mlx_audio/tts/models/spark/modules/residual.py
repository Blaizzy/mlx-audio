from typing import Any, Dict, List

import mlx.core as mx
import mlx.nn as nn
from einops.array_api import rearrange

from mlx_audio.codec.models.descript.nn.layers import WNConv1d


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


class FactorizedVectorQuantize(nn.Module):
    def __init__(
        self,
        input_dim: int,
        codebook_size: int,
        codebook_dim: int,
        commitment: float,
        codebook_loss_weight: float = 1.0,
        decay: float = 0.99,
        threshold_ema_dead_code: float = 2,
        momentum: float = 0.99,
        **kwargs,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.codebook_size = codebook_size
        self.commitment = commitment
        self.codebook_dim = codebook_dim
        self.codebook_loss_weight = codebook_loss_weight
        self.decay = decay
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.momentum = momentum

        requires_projection = input_dim != codebook_dim

        self.in_project = (
            WNConv1d(in_channels=input_dim, out_channels=codebook_dim, kernel_size=1)
            if requires_projection
            else nn.Identity()
        )
        self.out_project = (
            WNConv1d(in_channels=codebook_dim, out_channels=input_dim, kernel_size=1)
            if requires_projection
            else nn.Identity()
        )

        self.codebook = nn.Embedding(self.codebook_size, codebook_dim)
        self.cluster_size = mx.zeros((self.codebook_size,))

    def __call__(self, z: mx.array) -> Dict[str, Any]:
        """Quantized the input tensor using a fixed codebook and returns
        the corresponding codebook vectors

        Parameters
        ----------
        z : Tensor[B x D x T]

        Returns
        -------
        Tensor[B x D x T]
            Quantized continuous representation of input
        Tensor[1]
            Commitment loss to train encoder to predict vectors closer to codebook
            entries
        Tensor[1]
            Codebook loss to update the codebook
        Tensor[B x T]
            Codebook indices (quantized discrete representation of input)
        Tensor[B x D x T]
            Projected latents (continuous representation of input before quantization)
        """
        # transpose since we use linear

        # Factorized codes project input into low-dimensional space if self.input_dim != self.codebook_dim
        z_e = self.in_project(z.transpose(0, 2, 1)).transpose(0, 2, 1)
        z_q, indices, dists = self.decode_latents(z_e)

        # statistic the usage of codes
        embed_onehot = mx.zeros(
            (indices.shape[0], indices.shape[1], self.codebook_size), dtype=z_e.dtype
        )
        for i in range(indices.shape[0]):
            for j in range(indices.shape[1]):
                embed_onehot[i, j, indices[i, j]] = 1.0
        avg_probs = mx.mean(embed_onehot.reshape(-1, self.codebook_size), axis=0)
        perplexity = mx.exp(-mx.sum(avg_probs * mx.log(avg_probs + 1e-10)))

        active_num = (embed_onehot.sum(0).sum(0) > 0).sum()

        commit_loss = mx.zeros(0)
        codebook_loss = mx.zeros(0)

        z_q = z_e + (
            z_q - z_e
        )  # noop in forward pass, straight-through gradient estimator in backward pass

        z_q = self.out_project(z_q.transpose(0, 2, 1)).transpose(0, 2, 1)

        vq_loss = (commit_loss + codebook_loss).mean()

        return {
            "z_q": z_q,
            "indices": indices,
            "dists": dists,
            "vq_loss": vq_loss,
            "perplexity": perplexity,
            "active_num": active_num.astype(mx.float32),
        }

    def vq2emb(self, vq, out_proj=True):
        emb = self.embed_code(vq)
        if out_proj:
            emb = self.out_project(emb)
        return emb

    def tokenize(self, z: mx.array) -> mx.array:
        """tokenize the input tensor"""
        z_e = self.in_project(z.transpose(0, 2, 1)).transpose(0, 2, 1)
        _, indices, _ = self.decode_latents(z_e)
        return indices

    def detokenize(self, indices):
        """detokenize the input indices"""
        # Check if indices are empty
        if indices.shape[0] == 0 or indices.shape[1] == 0:
            # Return an appropriate empty or placeholder tensor
            return mx.zeros((1, self.input_dim, 1))

        z_q = self.decode_code(indices).transpose(0, 2, 1)
        z_q = self.out_project(z_q)
        return z_q

    def get_emb(self):
        return self.codebook.weight

    def embed_code(self, embed_id):
        return mx.take(self.codebook.weight, embed_id, axis=0)

    def decode_code(self, embed_id):

        return self.embed_code(embed_id).transpose(0, 2, 1)

    def normalize(self, x):
        """Normalize input tensor along dimension 1."""
        norm = mx.sqrt(mx.sum(mx.power(x, 2), axis=1, keepdims=True))
        return x / mx.maximum(norm, 1e-12)

    def decode_latents(self, latents):
        encodings = rearrange(latents, "b d t -> (b t) d")
        codebook = self.codebook.weight

        # L2 normalize encodings and codebook
        encodings = self.normalize(encodings)
        codebook = self.normalize(codebook)

        # Compute euclidean distance between encodings and codebook,
        # with L2 normalization, the distance is equal to cosine distance
        dist = (
            mx.sum(mx.power(encodings, 2), axis=1, keepdims=True)
            - 2 * encodings @ codebook.T
            + mx.sum(mx.power(codebook, 2), axis=1, keepdims=True).T
        )
        min_encoding_indices = mx.argmax(-dist, axis=1)
        indices = mx.reshape(min_encoding_indices, (latents.shape[0], latents.shape[2]))
        z_q = self.decode_code(indices)

        return z_q, indices, dist

    def get_codes_from_indices(self, indices):
        """Get codebook vectors from indices.

        Args:
            indices: Tensor of shape [B, T]

        Returns:
            Tensor of shape [B, D, T]
        """
        return self.decode_code(indices)

    def get_output_from_indices(self, indices):
        """Get output from indices.

        Args:
            indices: Tensor of shape [B, T]

        Returns:
            Tensor of shape [B, D, T]
        """
        z_q = self.get_codes_from_indices(indices)
        return self.out_project(z_q.transpose(0, 2, 1)).transpose(0, 2, 1)

    def sanitize(self, weights):
        sanitized_weights = {}
        for k, v in weights.items():
            if "weight_v" in k:
                if v.shape[1] > v.shape[-1]:
                    sanitized_weights[k] = v.transpose(0, 2, 1)
                else:
                    sanitized_weights[k] = v
            else:
                sanitized_weights[k] = v
        return sanitized_weights
