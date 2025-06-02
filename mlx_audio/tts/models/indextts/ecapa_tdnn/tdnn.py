import mlx.core as mx
import mlx.nn as nn


# essentially just conv with relu & norm
class TDNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.padding = ((kernel_size - 1) * dilation) // 2

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            1,
            0,
            dilation,
            groups,
            bias,
        )
        self.activation = nn.ReLU()
        self.norm = nn.BatchNorm(out_channels)

    def __call__(self, x: mx.array) -> mx.array:  # NLC
        # reflect padding
        top_pad = x[:, 1 : self.padding + 1, :][:, ::-1, :]
        bottom_pad = x[:, -(self.padding + 1) : -1, :][:, ::-1, :]
        x = mx.concat([top_pad, x, bottom_pad], axis=1)

        res = self.norm(self.activation(self.conv(x)))

        return res
