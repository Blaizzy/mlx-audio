from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn

from .s2mel_utils import sequence_mask


@dataclass
class InterpolateRegulatorConfig:
    channels: int = 512
    sampling_ratios: Tuple[int, ...] = (1, 1, 1, 1)
    in_channels: int = 1024
    out_channels: int = 512
    groups: int = 1


def _nearest_interpolate_1d(x: mx.array, out_len: int) -> mx.array:
    # x: (T, C)
    in_len = int(x.shape[0])
    if in_len == out_len:
        return x
    if in_len <= 1:
        return mx.broadcast_to(x[:1, :], (out_len, x.shape[1]))

    idx = (mx.floor(mx.arange(out_len, dtype=mx.float32) * (in_len / out_len))).astype(mx.int32)
    idx = mx.clip(idx, 0, in_len - 1)
    return x[idx, :]


class InterpolateRegulator(nn.Module):
    def __init__(self, cfg: InterpolateRegulatorConfig):
        super().__init__()
        self.cfg = cfg
        self.interpolate = len(cfg.sampling_ratios) > 0

        self.model = []
        if self.interpolate:
            for _ in cfg.sampling_ratios:
                self.model.append(nn.Conv1d(cfg.channels, cfg.channels, 3, 1, 1))
                self.model.append(nn.GroupNorm(cfg.groups, cfg.channels))
                self.model.append(nn.Mish())

        self.model.append(nn.Conv1d(cfg.channels, cfg.out_channels, 1, 1))

        # Unused in IndexTTS2 continuous mode but present in checkpoints.
        self.embedding = nn.Embedding(2048, cfg.channels)

        self.content_in_proj = nn.Linear(cfg.in_channels, cfg.channels)

        self.mask_token = mx.zeros((1, cfg.channels), dtype=mx.float32)

    def __call__(
        self,
        x: mx.array,
        *,
        ylens: mx.array,
        f0: Optional[mx.array] = None,
        n_quantizers: Optional[int] = None,
    ):
        del f0, n_quantizers
        # x: (B, T, in_channels)
        if x.ndim != 3:
            raise ValueError(f"Expected (B,T,C), got {x.shape}")

        B, T, _ = x.shape
        x = self.content_in_proj(x)  # (B, T, channels)

        out_len = int(mx.max(ylens).item())
        if self.interpolate:
            xs = []
            for i in range(B):
                xs.append(_nearest_interpolate_1d(x[i], out_len))
            x = mx.stack(xs, axis=0)  # (B, out_len, C)
        else:
            x = x[:, :out_len, :]

        h = x
        for layer in self.model:
            h = layer(h)
        out = h  # (B, out_len, out_channels)
        mask = sequence_mask(ylens, max_length=out_len).astype(out.dtype)[:, :, None]
        return out * mask, ylens, None, None, None
