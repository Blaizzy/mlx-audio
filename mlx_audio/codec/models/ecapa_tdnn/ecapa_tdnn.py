import mlx.core as mx
import mlx.nn as nn


class TDNNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            bias=True,
        )
        self.norm = nn.BatchNorm(out_channels)

    def __call__(self, x: mx.array) -> mx.array:
        return self.norm(nn.relu(self.conv(x)))


class Res2NetBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        scale: int = 8,
    ):
        super().__init__()
        assert (
            channels % scale == 0
        ), f"channels ({channels}) must be divisible by scale ({scale})"
        self.scale = scale
        hidden = channels // scale
        self.blocks = [
            TDNNBlock(hidden, hidden, kernel_size, dilation) for _ in range(scale - 1)
        ]

    def __call__(self, x: mx.array) -> mx.array:
        chunks = mx.split(x, self.scale, axis=-1)
        y = [chunks[0]]
        for i, block in enumerate(self.blocks):
            inp = chunks[i + 1] + y[-1] if i > 0 else chunks[i + 1]
            y.append(block(inp))
        return mx.concatenate(y, axis=-1)


class SEBlock(nn.Module):
    def __init__(self, in_dim: int, bottleneck: int = 128):
        super().__init__()
        self.conv1 = nn.Conv1d(in_dim, bottleneck, 1)
        self.conv2 = nn.Conv1d(bottleneck, in_dim, 1)

    def __call__(self, x: mx.array) -> mx.array:
        s = mx.mean(x, axis=1, keepdims=True)
        s = nn.relu(self.conv1(s))
        s = mx.sigmoid(self.conv2(s))
        return x * s


class SERes2NetBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        res2net_scale: int = 8,
        se_channels: int = 128,
    ):
        super().__init__()
        self.tdnn1 = TDNNBlock(channels, channels, 1)
        self.res2net_block = Res2NetBlock(
            channels, kernel_size, dilation, res2net_scale
        )
        self.tdnn2 = TDNNBlock(channels, channels, 1)
        self.se_block = SEBlock(channels, se_channels)

    def __call__(self, x: mx.array) -> mx.array:
        out = self.tdnn1(x)
        out = self.res2net_block(out)
        out = self.tdnn2(out)
        out = self.se_block(out)
        return out + x
