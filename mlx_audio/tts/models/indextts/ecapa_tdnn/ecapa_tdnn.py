from dataclasses import dataclass, field
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.tts.models.indextts.ecapa_tdnn.asp import AttentiveStatisticsPooling
from mlx_audio.tts.models.indextts.ecapa_tdnn.se_res2net import SeRes2Net
from mlx_audio.tts.models.indextts.ecapa_tdnn.tdnn import TDNN


@dataclass
class ECPATDNNArgs:
    input_size: int
    lin_neurons: int = 192
    channels: list[int] = field(default_factory=lambda: [512, 512, 512, 512, 1536])
    kernel_sizes: list[int] = field(default_factory=lambda: [5, 3, 3, 3, 1])
    dilations: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 1])
    attention_channels: int = 128
    res2net_scale: int = 8
    se_channels: int = 128
    global_context: bool = True
    groups: list[int] = field(default_factory=lambda: [1, 1, 1, 1, 1])


class ECPATDNN(nn.Module):
    def __init__(self, args: ECPATDNNArgs):
        super().__init__()
        assert len(args.channels) == len(args.kernel_sizes) and len(
            args.channels
        ) == len(args.dilations)

        self.args = args

        self.blocks = [
            TDNN(
                args.input_size,
                args.channels[0],
                args.kernel_sizes[0],
                dilation=args.dilations[0],
                groups=args.groups[0],
            )
        ] + [
            SeRes2Net(
                args.channels[i - 1],
                args.channels[i],
                scale=args.res2net_scale,
                attention_channels=args.se_channels,
                kernel_size=args.kernel_sizes[i],
                dilation=args.dilations[i],
                groups=args.groups[i],
            )
            for i in range(1, len(args.channels) - 1)
        ]
        self.mfa = TDNN(
            args.channels[-2] * (len(args.channels) - 2),
            args.channels[-1],
            args.kernel_sizes[-1],
            dilation=args.dilations[-1],
            groups=args.groups[-1],
        )
        self.asp = AttentiveStatisticsPooling(
            args.channels[-1],
            attention_channels=args.attention_channels,
            global_context=args.global_context,
        )
        self.asp_bn = nn.BatchNorm(args.channels[-1] * 2)
        self.fc = nn.Conv1d(
            in_channels=args.channels[-1] * 2,
            out_channels=args.lin_neurons,
            kernel_size=1,
        )

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None):  #
        xl = []
        for layer in self.blocks:
            if isinstance(layer, SeRes2Net):
                x = layer(x, mask=mask)
                xl.append(mx.array(x))
            else:
                x = layer(x)

        x = mx.concat(xl, axis=2)
        x = self.mfa(x)

        x = self.asp(x, mask=mask)
        x = self.asp_bn(x)

        x = self.fc(x)

        return x
