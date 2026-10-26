"""Exact five-value encoder weights as two native MLX affine 2-bit planes.

Adapted from fermion-research 0.2.11, Copyright 2026 Fermion Research,
Apache-2.0. Packed planes execute with MLX quantized matrix multiplication.
"""

import mlx.core as mx
import mlx.nn as nn

GROUP_SIZE = 128
ENCODER_SUFFIXES = (
    "feed_forward1.linear1",
    "feed_forward1.linear2",
    "feed_forward2.linear1",
    "feed_forward2.linear2",
    "self_attn.linear_q",
    "self_attn.linear_k",
    "self_attn.linear_v",
    "self_attn.linear_out",
    "self_attn.linear_pos",
    "conv.pointwise_conv1",
    "conv.pointwise_conv2",
)


class PackedFiveValueLinear(nn.Module):
    """A linear or kernel-1 Conv1d, preserving the source row palettes."""

    def __init__(self, inputs, outputs, bias=False):
        super().__init__()
        if inputs <= 0 or outputs <= 0 or inputs % GROUP_SIZE:
            raise ValueError("Phonon-2 encoder width must be divisible by 128")
        self._inputs = inputs
        self.base_q = mx.zeros((outputs, inputs // 16), mx.uint32)
        self.residual_q = mx.zeros((outputs, inputs // 16), mx.uint32)
        self.row_lo = mx.zeros((outputs,), mx.float16)
        self.row_hi = mx.zeros((outputs,), mx.float16)
        if bias:
            self.bias = mx.zeros((outputs,), mx.bfloat16)

    def materialize(self):
        shape = (self.row_lo.shape[0], self._inputs // GROUP_SIZE)
        # Subtract in fp32: rounding hi-lo to fp16 changes the learned weights.
        lo = self.row_lo.astype(mx.float32)
        delta = self.row_hi.astype(mx.float32) - lo
        self._base_scales = mx.contiguous(mx.broadcast_to(lo[:, None], shape))
        self._base_biases = -self._base_scales
        self._residual_scales = mx.contiguous(mx.broadcast_to(delta[:, None], shape))
        self._residual_biases = -self._residual_scales

    def __call__(self, x):
        dtype = x.dtype
        x = x.astype(mx.float32)
        a = mx.quantized_matmul(
            x,
            self.base_q,
            self._base_scales,
            self._base_biases,
            transpose=True,
            group_size=GROUP_SIZE,
            bits=2,
        )
        b = mx.quantized_matmul(
            x,
            self.residual_q,
            self._residual_scales,
            self._residual_biases,
            transpose=True,
            group_size=GROUP_SIZE,
            bits=2,
        )
        y = a + b
        if "bias" in self:
            y = y + self.bias.astype(mx.float32)
        return y.astype(dtype)
