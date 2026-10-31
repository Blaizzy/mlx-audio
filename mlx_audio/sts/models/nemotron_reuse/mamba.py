# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
# Adapted for MLX from NVIDIA-NemotronLabs-RE-USE.
"""Mamba-1 inference with a bounded-memory native selective scan."""

import math

import mlx.core as mx
import mlx.nn as nn


def _make_scan_kernel():
    if not mx.metal.is_available():
        return None
    return mx.fast.metal_kernel(
        name="reuse_selective_scan",
        input_names=["x", "dt", "A", "B", "C", "D"],
        output_names=["y"],
        source="""
            uint channel = thread_position_in_grid.x;
            uint batch = thread_position_in_grid.y;
            if (channel >= H) return;
            float state[N];
            float a[N];
            for (int s = 0; s < N; ++s) {
                state[s] = 0.0f;
                a[s] = A[channel * N + s];
            }
            for (int t = 0; t < L; ++t) {
                uint index = (batch * L + t) * H + channel;
                uint bc = (batch * L + t) * N;
                float delta = dt[index];
                float u = x[index];
                float value = D[channel] * u;
                for (int s = 0; s < N; ++s) {
                    state[s] = metal::exp(delta * a[s]) * state[s]
                        + delta * B[bc + s] * u;
                    value += state[s] * C[bc + s];
                }
                y[index] = value;
            }
        """,
    )


_scan_kernel = _make_scan_kernel()


def selective_scan(x, dt, a, b, c, d):
    """Scan [batch, length, channels], accumulating the state in float32."""
    batch, length, channels = x.shape
    state_size = a.shape[-1]
    if _scan_kernel is not None and mx.default_device() == mx.gpu:
        return _scan_kernel(
            inputs=[v.astype(mx.float32) for v in (x, dt, a, b, c, d)],
            template=[("H", channels), ("N", state_size), ("L", length)],
            grid=(channels, batch, 1),
            threadgroup=(min(256, channels), 1, 1),
            output_shapes=[x.shape],
            output_dtypes=[mx.float32],
        )[0].astype(x.dtype)
    state = mx.zeros((batch, channels, state_size), dtype=mx.float32)
    output = []
    for t in range(length):
        delta = dt[:, t, :, None].astype(mx.float32)
        u = x[:, t, :, None].astype(mx.float32)
        state = mx.exp(delta * a) * state + delta * b[:, t, None, :] * u
        output.append(mx.sum(state * c[:, t, None, :], axis=-1) + d * x[:, t])
    return mx.stack(output, axis=1).astype(x.dtype)


class Mamba(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config.hid_feature
        inner = dim * config.expand
        self.d_conv = config.d_conv
        self.d_state = config.d_state
        self.dt_rank = math.ceil(dim / 16)
        self.in_proj = nn.Linear(dim, 2 * inner, bias=False)
        self.conv1d = nn.Conv1d(inner, inner, config.d_conv, groups=inner)
        self.x_proj = nn.Linear(inner, self.dt_rank + 2 * self.d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, inner)
        self.A_log = mx.log(mx.arange(1, self.d_state + 1, dtype=mx.float32))
        self.A_log = mx.broadcast_to(self.A_log, (inner, self.d_state))
        self.D = mx.ones((inner,))
        self.out_proj = nn.Linear(inner, dim, bias=False)

    def __call__(self, x):
        x, z = mx.split(self.in_proj(x), 2, axis=-1)
        x = nn.silu(self.conv1d(mx.pad(x, ((0, 0), (self.d_conv - 1, 0), (0, 0)))))
        dt, b, c = mx.split(
            self.x_proj(x), [self.dt_rank, self.dt_rank + self.d_state], axis=-1
        )
        dt = nn.softplus(self.dt_proj(dt).astype(mx.float32))
        y = selective_scan(x, dt, -mx.exp(self.A_log.astype(mx.float32)), b, c, self.D)
        return self.out_proj(y * nn.silu(z))


class BidirectionalMamba(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config.hid_feature
        self.forward_blocks = Mamba(config)
        self.backward_blocks = Mamba(config)
        self.output_proj = nn.Linear(2 * dim, dim)
        self.norm = nn.LayerNorm(dim)

    def __call__(self, x):
        forward = self.forward_blocks(x) + x
        backward = (self.backward_blocks(x[:, ::-1]) + x[:, ::-1])[:, ::-1]
        return self.norm(self.output_proj(mx.concatenate([forward, backward], axis=-1)))


class TFMambaBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.time_mamba = BidirectionalMamba(config)
        self.freq_mamba = BidirectionalMamba(config)

    def __call__(self, x):
        batch, time, freq, channels = x.shape
        x = x.transpose(0, 2, 1, 3).reshape(batch * freq, time, channels)
        x = self.time_mamba(x) + x
        x = x.reshape(batch, freq, time, channels).transpose(0, 2, 1, 3)
        x = x.reshape(batch * time, freq, channels)
        x = self.freq_mamba(x) + x
        return x.reshape(batch, time, freq, channels)
