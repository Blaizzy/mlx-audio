"""
Residual Vector Quantizer (RVQ) – encode-only path for MLX.

Ported from ``src/mimo_audio_tokenizer/quantization.py``.
Training logic (EMA, kmeans init, commitment loss) is removed; only the
inference-time ``encode`` path is implemented.

Architecture
------------
ResidualVectorQuantizer
  └── ResidualVectorQuantization
        └── VectorQuantization  × n_q
              └── EuclideanCodebook  (nearest-neighbour lookup)
"""

from typing import List, Optional, Union

import mlx.core as mx
import mlx.nn as nn


# ── EuclideanCodebook (encode-only) ─────────────────────────────────

class EuclideanCodebook(nn.Module):
    """
    Nearest-neighbour codebook with Euclidean distance.

    Only the ``quantize`` (nearest lookup) and ``decode`` (embedding lookup)
    methods are needed for inference.
    """

    def __init__(self, dim: int, codebook_size: int):
        super().__init__()
        # (codebook_size, dim)
        self.embed = mx.zeros((codebook_size, dim))

    def quantize(self, x: mx.array) -> mx.array:
        """
        Find nearest codebook entry for each vector.

        Parameters
        ----------
        x : mx.array, shape (N, dim)

        Returns
        -------
        mx.array, shape (N,), dtype int32 – codebook indices
        """
        # Euclidean distance: ||x - e||² = x² - 2x·e + e²
        #  (N, dim) @ (dim, codebook_size) → (N, codebook_size)
        embed_t = self.embed.T  # (dim, codebook_size)

        x_sq = (x * x).sum(axis=-1, keepdims=True)  # (N, 1)
        e_sq = (self.embed * self.embed).sum(axis=-1)  # (codebook_size,)
        dot = x @ embed_t  # (N, codebook_size)

        # negate so that argmax on neg-dist = argmin on dist
        neg_dist = 2.0 * dot - e_sq  # (N, codebook_size)  (omit x², constant)

        return neg_dist.argmax(axis=-1)  # (N,)

    def decode(self, embed_ind: mx.array) -> mx.array:
        """
        Look up embeddings from indices.

        Parameters
        ----------
        embed_ind : mx.array, shape (...)

        Returns
        -------
        mx.array, shape (..., dim)
        """
        return self.embed[embed_ind]

    @property
    def codebook_size(self) -> int:
        return self.embed.shape[0]


# ── VectorQuantization ──────────────────────────────────────────────

class VectorQuantization(nn.Module):
    """
    Single vector quantizer with optional input/output projections.
    """

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        codebook_dim: Optional[int] = None,
    ):
        super().__init__()
        _codebook_dim = codebook_dim or dim
        self.project_in = (
            nn.Linear(dim, _codebook_dim, bias=False)
            if _codebook_dim != dim
            else nn.Identity()
        )
        self.project_out = (
            nn.Linear(_codebook_dim, dim, bias=False)
            if _codebook_dim != dim
            else nn.Identity()
        )
        self.codebook = EuclideanCodebook(_codebook_dim, codebook_size)

    def encode(self, x: mx.array) -> mx.array:
        """
        Encode vectors to codebook indices.

        Parameters
        ----------
        x : mx.array, shape (N, dim)

        Returns
        -------
        mx.array, shape (N,) – int32 indices
        """
        x_proj = self.project_in(x)
        return self.codebook.quantize(x_proj)

    def decode(self, embed_ind: mx.array) -> mx.array:
        """Decode indices back to vectors.  (N,*) → (N, dim)."""
        quantized = self.codebook.decode(embed_ind)
        return self.project_out(quantized)


# ── ResidualVectorQuantization ──────────────────────────────────────

class ResidualVectorQuantization(nn.Module):
    """Stack of VectorQuantization layers applied in residual fashion."""

    def __init__(self, num_quantizers: int, codebook_size: Union[int, List[int]], **kwargs):
        super().__init__()
        if isinstance(codebook_size, int):
            codebook_size = [codebook_size] * num_quantizers
        self.layers = [
            VectorQuantization(codebook_size=codebook_size[i], **kwargs)
            for i in range(num_quantizers)
        ]

    def encode(self, x: mx.array, n_q: Optional[int] = None, st: int = 0) -> mx.array:
        """
        Iteratively encode residual vectors.

        Parameters
        ----------
        x : mx.array, shape (N, dim)
        n_q : int, optional – number of quantizers to use (default: all)
        st : int – starting quantizer index

        Returns
        -------
        mx.array, shape (n_q, N) – stacked code indices
        """
        n_q = len(self.layers) if n_q is None else n_q
        residual = x
        all_indices = []
        for layer in self.layers[st:n_q]:
            indices = layer.encode(residual)
            quantized = layer.decode(indices)
            residual = residual - quantized
            all_indices.append(indices)
        return mx.stack(all_indices, axis=0)  # (n_q, N)


# ── ResidualVectorQuantizer (public API) ────────────────────────────

class ResidualVectorQuantizer(nn.Module):
    """
    Top-level RVQ module matching the MiMo-Audio-Tokenizer interface.

    Parameters
    ----------
    dimension : int       – hidden dimension of the encoder output
    n_q : int             – number of residual quantizers
    bins : int | list    – codebook size(s)
    """

    def __init__(
        self,
        dimension: int = 1280,
        n_q: int = 20,
        bins: Union[int, list] = 1024,
    ):
        super().__init__()
        self.n_q = n_q
        self.dimension = dimension
        self.vq = ResidualVectorQuantization(
            dim=dimension,
            codebook_size=bins,
            num_quantizers=n_q,
        )

    def encode(self, x: mx.array, n_q: Optional[int] = None, st: int = 0) -> mx.array:
        """
        Encode vectors.

        Parameters
        ----------
        x : mx.array, shape (N, dimension)
        n_q : int, optional
        st : int

        Returns
        -------
        mx.array, shape (n_q, N), dtype int32
        """
        return self.vq.encode(x, n_q=n_q, st=st)

    def decode(self, codes: mx.array, st: int = 0) -> mx.array:
        """Decode codes back to continuous vectors.  (n_q, N) → (N, dim)."""
        quantized = mx.zeros((codes.shape[1], self.dimension))
        for i in range(codes.shape[0]):
            quantized = quantized + self.vq.layers[st + i].decode(codes[i])
        return quantized
