from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.tts.models.indextts.ecapa_tdnn.tdnn import TDNN


class Res2Net(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        scale: int,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        # just make sure it's dividable
        assert in_channels % scale == out_channels % scale == 0

        self.scale = scale

        self.blocks = [
            TDNN(
                in_channels // scale,
                out_channels // scale,
                kernel_size,
                dilation,
                groups,
                bias,
            )
            for _ in range(scale - 1)
        ]

    def __call__(self, x: mx.array) -> mx.array:  # NLC
        segments = mx.split(x, self.scale, axis=-1)

        y = [segments[0]]

        for i in range(1, len(segments)):
            prev = y[-1] if i > 1 else 0
            y.append(self.blocks[i - 1](segments[i] + prev))

        return mx.concat(y, axis=-1)


class SE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        se_channels: int,
        out_channels: int,
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, se_channels, 1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(se_channels, out_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def __call__(
        self, x: mx.array, mask: Optional[mx.array] = None
    ) -> mx.array:  # NLC, NL
        if mask is not None:
            mask = mask[:, :, None]  # NL1
            masked_x = x * mask
            s = masked_x.sum(1, keepdims=True) / mask.sum(1, keepdims=True)
        else:
            s = x.mean(1, keepdims=True)

        s = self.sigmoid(self.conv2(self.relu(self.conv1(s))))

        return s * x


class SeRes2Net(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale: int,
        attention_channels: int,
        kernel_size: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()

        self.out_channels = out_channels

        self.tdnn1 = TDNN(
            in_channels,
            out_channels,
            kernel_size=1,
            dilation=1,
            groups=groups,
        )
        self.res2net_block = Res2Net(
            out_channels,
            out_channels,
            kernel_size,
            scale,
            dilation=dilation,
        )
        self.tdnn2 = TDNN(
            out_channels,
            out_channels,
            kernel_size=1,
            dilation=1,
            groups=groups,
        )
        self.se_block = SE(out_channels, attention_channels, out_channels)
        self.shortcut = (
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
            )
            if in_channels != out_channels
            else nn.Identity()
        )

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        x = self.shortcut(x)

        x += self.se_block(self.tdnn2(self.res2net_block(self.tdnn1(x))), mask)

        return x
