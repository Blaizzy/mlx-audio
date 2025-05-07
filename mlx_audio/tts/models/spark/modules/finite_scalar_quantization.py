"""
Finite Scalar Quantization: VQ-VAE Made Simple - https://arxiv.org/abs/2309.15505
Code adapted from Jax version in Appendix A.1
"""

from __future__ import annotations

from contextlib import nullcontext
from functools import partial, wraps
from typing import List, Tuple

import mlx.core as mx
import mlx.nn as nn

# helper functions


def exists(v):
    return v is not None


def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None


def maybe(fn):
    @wraps(fn)
    def inner(x, *args, **kwargs):
        if not exists(x):
            return x
        return fn(x, *args, **kwargs)

    return inner


# tensor helpers


def round_ste(z: mx.array) -> mx.array:
    """Round with straight through gradients."""
    zhat = z.round()
    return z + (zhat - z)


# main class


class FSQ(nn.Module):
    def __init__(
        self,
        levels: List[int],
        dim: int | None = None,
        num_codebooks=1,
        keep_num_codebooks_dim: bool | None = None,
        scale: float | None = None,
        allowed_dtypes: Tuple[mx.dtype, ...] = (mx.float32, mx.float64),
        channel_first: bool = False,
        projection_has_bias: bool = True,
        return_indices=True,
        force_quantization_f32=True,
    ):
        super().__init__()
        _levels = mx.array(list(levels), dtype=mx.int32)
        self._levels = _levels

        _basis = mx.cumprod(mx.array([1] + list(levels[:-1])), axis=0)
        self._basis = _basis

        self.scale = scale

        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim

        effective_codebook_dim = codebook_dim * num_codebooks
        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = effective_codebook_dim

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        self.dim = default(dim, len(_levels) * num_codebooks)

        self.channel_first = channel_first

        has_projections = self.dim != effective_codebook_dim
        self.project_in = (
            nn.Linear(self.dim, effective_codebook_dim, bias=projection_has_bias)
            if has_projections
            else nn.Identity()
        )
        self.project_out = (
            nn.Linear(effective_codebook_dim, self.dim, bias=projection_has_bias)
            if has_projections
            else nn.Identity()
        )

        self.has_projections = has_projections

        self.return_indices = return_indices
        if return_indices:
            self.codebook_size = self._levels.prod().item()
            implicit_codebook = self._indices_to_codes(mx.arange(self.codebook_size))
            self._implicit_codebook = implicit_codebook

        self.allowed_dtypes = allowed_dtypes
        self.force_quantization_f32 = force_quantization_f32

    def atanh(self, x):
        return mx.log((1 + x) / (1 - x)) / 2

    def bound(self, z, eps: float = 1e-3):
        """Bound `z`, an array of shape (..., d)."""
        half_l = (self._levels - 1) * (1 + eps) / 2
        offset = mx.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = self.atanh(offset / half_l)  # original atanh
        return mx.tanh(z + shift) * half_l - offset

    def quantize(self, z):
        """Quantizes z, returns quantized zhat, same shape as z."""
        quantized = round_ste(self.bound(z))
        half_width = self._levels // 2  # Renormalize to [-1, 1].
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized):
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat):
        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def _indices_to_codes(self, indices):
        level_indices = self.indices_to_level_indices(indices)
        codes = self._scale_and_shift_inverse(level_indices)
        return codes

    def codes_to_indices(self, zhat):
        """Converts a `code` to an index in the codebook."""
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(axis=-1).astype(mx.int32)

    def indices_to_level_indices(self, indices):
        """Converts indices to indices at each level, perhaps needed for a transformer with factorized embeddings"""
        indices = mx.reshape(indices, (indices.shape[0], -1, 1))
        codes_non_centered = (indices // self._basis) % self._levels
        return codes_non_centered

    def indices_to_codes(self, indices):
        """Inverse of `codes_to_indices`."""
        assert exists(indices)

        is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))

        codes = self._indices_to_codes(indices)

        if self.keep_num_codebooks_dim:
            codes = mx.reshape(codes, (codes.shape[0], -1))

        codes = self.project_out(codes)

        if is_img_or_video or self.channel_first:
            codes = mx.reshape(codes, (codes.shape[0], -1, codes.shape[-1]))

        return codes

    def __call__(self, z):
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension
        c - number of codebook dim
        """

        is_img_or_video = z.ndim >= 4
        need_move_channel_last = is_img_or_video or self.channel_first

        # standardize image or video into (batch, seq, dimension)

        if need_move_channel_last:
            z = mx.reshape(z, (z.shape[0], -1, z.shape[-1]))
            # z = mx.reshape(z, (z.shape[0], -1))

        assert (
            z.shape[-1] == self.dim
        ), f"expected dimension of {self.dim} but found dimension of {z.shape[-1]}"

        z = self.project_in(z)

        z = mx.reshape(z, (z.shape[0], z.shape[1], self.num_codebooks, z.shape[-1]))

        # whether to force quantization step to be full precision or not

        force_f32 = self.force_quantization_f32

        orig_dtype = z.dtype

        if force_f32 and orig_dtype not in self.allowed_dtypes:
            z = z.float()

        codes = self.quantize(z)

        # returning indices could be optional

        indices = None

        if self.return_indices:
            indices = self.codes_to_indices(codes)

        codes = mx.reshape(codes, (codes.shape[0], codes.shape[1], -1))

        codes = codes.astype(orig_dtype)

        # project out

        out = self.project_out(codes)

        # reconstitute image or video dimensions

        if need_move_channel_last:
            out = mx.reshape(out, (out.shape[0], -1, out.shape[-1]))
            out = mx.reshape(
                out, (out.shape[0], out.shape[1], out.shape[2], out.shape[-1])
            )

            indices = mx.reshape(indices, (indices.shape[0], -1, indices.shape[-1]))

        if not self.keep_num_codebooks_dim and self.return_indices:
            indices = mx.reshape(indices, (indices.shape[0], -1, indices.shape[-1]))

        # return quantized output and indices

        return out, indices
