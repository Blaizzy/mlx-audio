from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.tts.models.indextts.ecapa_tdnn.tdnn import TDNN


class AttentiveStatisticsPooling(nn.Module):
    def __init__(
        self, channels: int, attention_channels: int, global_context: bool = True
    ):
        super().__init__()

        self.eps = 1e-12
        self.global_context = global_context

        self.tdnn = TDNN(
            channels * 3 if global_context else channels, attention_channels, 1
        )
        self.tanh = nn.Tanh()
        self.conv = nn.Conv1d(attention_channels, channels, 1)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None):  # NLC
        N, L, C = x.shape

        if mask is not None:
            mask = mask[:, :, None]
        else:
            mask = mx.ones((N, L, 1))

        if self.global_context:
            global_mean = (x * mask).sum(1, keepdims=True) / (
                mask.sum(1, keepdims=True) + self.eps
            )
            global_std = mx.sqrt(
                ((x - global_mean) ** 2 * mask).sum(1, keepdims=True)
                / (mask.sum(1, keepdims=True) + self.eps)
                + self.eps
            )
            attn = mx.concat(
                [
                    x,
                    mx.repeat(global_mean, L, axis=1),
                    mx.repeat(global_std, L, axis=1),
                ],
                axis=2,
            )
        else:
            attn = x

        attn = self.conv(self.tanh(self.tdnn(attn)))

        attn = mx.softmax(mx.where(mask == 0, -mx.inf, attn), axis=1)

        mean = (x * attn).sum(1, keepdims=True)
        std = mx.sqrt(((x - mean) ** 2 * attn).sum(1, keepdims=True) + self.eps)

        return mx.concat([mean, std], axis=2)
