"""
Custom Metal kernel for depthwise 1D convolutions used across MossFormer models.

This module exposes `depthwise_conv1d` which mirrors MLX's `mx.conv1d` but
special-cases the common stride-1, symmetric-padding depthwise configuration.
When the fast path does not apply, we fall back to `mx.conv1d` to preserve
behaviour exactly.
"""

from __future__ import annotations

import mlx.core as mx
from mlx.core.fast import metal_kernel

_depthwise_conv1d_kernel = metal_kernel(
    name="custom_depthwise_conv1d",
    input_names=["inp", "weight", "params"],
    output_names=["out"],
    source="""
        const int chan = int(thread_position_in_grid.x);
        const int time = int(thread_position_in_grid.y);
        const int batch = int(thread_position_in_grid.z);

        const int B = params[0];
        const int L_in = params[1];
        const int C = params[2];
        const int K = params[3];
        const int pad = params[4];
        const int L_out = params[5];

        if (batch >= B || time >= L_out || chan >= C) {
            return;
        }

        const int out_index = ((batch * L_out) + time) * C + chan;
        const int weight_base = chan * K;

        T acc = T(0);
        for (int k = 0; k < K; ++k) {
            const int in_time = time + k - pad;
            if (in_time >= 0 && in_time < L_in) {
                const int in_index = ((batch * L_in) + in_time) * C + chan;
                acc += inp[in_index] * weight[weight_base + k];
            }
        }

        out[out_index] = acc;
    """,
)


def depthwise_conv1d(
    x: mx.array,
    weight: mx.array,
    *,
    stride: int = 1,
    padding: int = 0,
    groups: int,
    stream: mx.core.context.StreamOrDevice | None = None,
) -> mx.array:
    """Depthwise 1-D convolution with an optimized Metal kernel fallback."""
    if (
        stride == 1
        and x.ndim == 3
        and weight.ndim == 3
        and weight.shape[2] == 1
        and groups == x.shape[2]
        and weight.shape[0] == x.shape[2]
        and (x.dtype == mx.float32 or x.dtype == mx.float16)
    ):
        kernel_size = weight.shape[1]
        output_length = x.shape[1] + 2 * padding - kernel_size + 1

        if output_length > 0 and padding * 2 == kernel_size - 1:
            x_contig = mx.contiguous(x, allow_col_major=False, stream=stream)
            weight_contig = mx.contiguous(weight, allow_col_major=False, stream=stream)

            params = mx.array(
                [
                    x_contig.shape[0],
                    x_contig.shape[1],
                    x_contig.shape[2],
                    kernel_size,
                    padding,
                    output_length,
                ],
                dtype=mx.int32,
            )

            outputs = _depthwise_conv1d_kernel(
                inputs=[x_contig, weight_contig, params],
                template=[("T", x_contig.dtype)],
                grid=(x_contig.shape[2], output_length, x_contig.shape[0]),
                threadgroup=(1, 1, 1),
                output_shapes=[[x_contig.shape[0], output_length, x_contig.shape[2]]],
                output_dtypes=[x_contig.dtype],
                stream=stream,
            )

            return outputs[0]

    return mx.conv1d(
        x,
        weight,
        stride=stride,
        padding=padding,
        groups=groups,
        stream=stream,
    )
