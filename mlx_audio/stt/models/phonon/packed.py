"""Packed five-value linear layers used by the Phonon-1 decoder.
"""

# Portions copyright 2026 Fermion Research, licensed under Apache-2.0.

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx
import mlx.nn as nn

_UNPACK_SOURCE = r"""
    uint output_index = thread_position_in_grid.x;
    uint packed_width = in_features / 16;
    uint output_words = quint5_q_shape[0] * packed_width;
    if (output_index >= output_words) return;
    uint row = output_index / packed_width;
    uint word = output_index - row * packed_width;
    uint bytes_per_row = quint5_q_shape[1];
    uint base_word = 0;
    uint residual_word = 0;
    #pragma unroll
    for (uint j = 0; j < 16; ++j) {
        uint logical_index = word * 16 + j;
        uint group = logical_index / 10;
        uint offset = logical_index - group * 10;
        const device uchar* source = quint5_q + row * bytes_per_row + group * 3;
        uint payload = uint(source[0]) | (uint(source[1]) << 8) |
                       (uint(source[2]) << 16);
        uint divisor;
        switch (offset) {
            case 0: divisor = 1; break;
            case 1: divisor = 5; break;
            case 2: divisor = 25; break;
            case 3: divisor = 125; break;
            case 4: divisor = 625; break;
            case 5: divisor = 3125; break;
            case 6: divisor = 15625; break;
            case 7: divisor = 78125; break;
            case 8: divisor = 390625; break;
            default: divisor = 1953125; break;
        }
        uint symbol = (payload / divisor) % 5;
        uint base_code;
        uint residual_code;
        switch (symbol) {
            case 0: base_code = 0; residual_code = 0; break;
            case 1: base_code = 0; residual_code = 2; break;
            case 2: base_code = 1; residual_code = 1; break;
            case 3: base_code = 2; residual_code = 0; break;
            default: base_code = 2; residual_code = 2; break;
        }
        base_word |= base_code << (2 * j);
        residual_word |= residual_code << (2 * j);
    }
    base_q[output_index] = base_word;
    residual_q[output_index] = residual_word;
"""


@lru_cache(maxsize=1)
def _unpack_kernel():
    return mx.fast.metal_kernel(
        name="mlx_audio_phonon_unpack_base5_v1",
        input_names=("quint5_q", "in_features"),
        output_names=("base_q", "residual_q"),
        source=_UNPACK_SOURCE,
    )


def packed_bytes_per_row(in_features: int) -> int:
    return ((in_features + 9) // 10) * 3


def unpack_base5_codes(
    quint5_q: mx.array, in_features: int
) -> tuple[mx.array, mx.array]:
    """Reconstruct the exact pair of native MLX 2-bit affine planes."""

    if quint5_q.dtype != mx.uint8 or quint5_q.ndim != 2:
        raise ValueError("quint5_q must be a two-dimensional uint8 array")
    if in_features % 16 or quint5_q.shape[1] != packed_bytes_per_row(in_features):
        raise ValueError("invalid ten-base-5-symbols-per-24-bit layout")

    out_features = quint5_q.shape[0]
    shape = (out_features, in_features // 16)
    count = out_features * shape[1]
    return tuple(
        _unpack_kernel()(
            inputs=(quint5_q, mx.array(in_features, dtype=mx.uint32)),
            grid=(count, 1, 1),
            threadgroup=(min(256, count), 1, 1),
            output_shapes=(shape, shape),
            output_dtypes=(mx.uint32, mx.uint32),
        )
    )


class PackedTritLinear(nn.Module):
    """A learned five-value linear evaluated as two native 2-bit QMMs."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        group_size: int = 128,
        slim_metadata: bool = False,
        quint5_codes: bool = True,
    ):
        super().__init__()
        if in_features % group_size:
            raise ValueError(
                f"in_features={in_features} is not divisible by {group_size}"
            )

        self.group_size = group_size
        self.slim_metadata = slim_metadata
        self.quint5_codes = quint5_codes
        self._in_features = in_features
        packed_width = in_features // 16
        groups = in_features // group_size

        if quint5_codes:
            self.quint5_q = mx.zeros(
                (out_features, packed_bytes_per_row(in_features)), dtype=mx.uint8
            )
        else:
            self.base_q = mx.zeros((out_features, packed_width), dtype=mx.uint32)
            self.residual_q = mx.zeros((out_features, packed_width), dtype=mx.uint32)

        if slim_metadata:
            self.base_alpha = mx.zeros((out_features,), dtype=mx.bfloat16)
            self.residual_scale = mx.zeros((1,), dtype=mx.bfloat16)
        else:
            self.base_scales = mx.zeros((out_features, groups), dtype=mx.bfloat16)
            self.base_biases = mx.zeros((out_features, groups), dtype=mx.bfloat16)
            self.residual_scales = mx.zeros((out_features, groups), dtype=mx.bfloat16)
            self.residual_biases = mx.zeros((out_features, groups), dtype=mx.bfloat16)

    def materialize(self) -> None:
        if self.quint5_codes:
            self.base_q, self.residual_q = unpack_base5_codes(
                self.quint5_q, self._in_features
            )
            mx.eval(self.base_q, self.residual_q)
            del self.quint5_q

        if self.slim_metadata:
            shape = (
                self.base_q.shape[0],
                self._in_features // self.group_size,
            )
            self._runtime_base_scales = mx.contiguous(
                mx.broadcast_to(self.base_alpha[:, None], shape)
            )
            self._runtime_base_biases = mx.contiguous(-self._runtime_base_scales)
            self._runtime_residual_scales = mx.contiguous(
                mx.broadcast_to(self.residual_scale.reshape((1, 1)), shape)
            )
            self._runtime_residual_biases = mx.contiguous(
                -self._runtime_residual_scales
            )

    def _metadata(self) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        if self.slim_metadata:
            return (
                self._runtime_base_scales,
                self._runtime_base_biases,
                self._runtime_residual_scales,
                self._runtime_residual_biases,
            )
        return (
            self.base_scales,
            self.base_biases,
            self.residual_scales,
            self.residual_biases,
        )

    def __call__(self, x: mx.array) -> mx.array:
        base_scales, base_biases, residual_scales, residual_biases = self._metadata()
        base = mx.quantized_matmul(
            x,
            self.base_q,
            base_scales,
            base_biases,
            transpose=True,
            group_size=self.group_size,
            bits=2,
            mode="affine",
        )
        residual = mx.quantized_matmul(
            x,
            self.residual_q,
            residual_scales,
            residual_biases,
            transpose=True,
            group_size=self.group_size,
            bits=2,
            mode="affine",
        )
        return base + residual
