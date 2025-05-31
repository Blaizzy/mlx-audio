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

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            1,
            ((kernel_size - 1) * dilation) // 2,
            dilation,
            groups,
            bias,
        )
        self.activation = nn.ReLU()
        self.norm = nn.BatchNorm(out_channels)

    def __call__(self, x: mx.array) -> mx.array:  # NLC
        return self.norm(self.activation(self.conv(x)))
