from collections.abc import Callable

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.tts.models.chatterbox.s3gen.xvector import (
    CAMLayer,
    DenseLayer,
    StatsPool,
    kaldi_fbank,
)


class FusedBasicResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=(stride, 1), padding=1
        )
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.shortcut = []
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = [
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=(stride, 1),
                )
            ]

    def __call__(self, x: mx.array) -> mx.array:
        out = nn.relu(self.conv1(x))
        out = self.conv2(out)
        shortcut = x
        for layer in self.shortcut:
            shortcut = layer(shortcut)
        return nn.relu(out + shortcut)


class FusedFCM(nn.Module):
    def __init__(
        self,
        block=FusedBasicResBlock,
        num_blocks: tuple[int, int] = (2, 2),
        m_channels: int = 32,
        feat_dim: int = 80,
    ):
        super().__init__()
        self.in_planes = m_channels
        self.conv1 = nn.Conv2d(1, m_channels, kernel_size=3, stride=1, padding=1)
        self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, m_channels, num_blocks[1], stride=2)
        self.conv2 = nn.Conv2d(
            m_channels, m_channels, kernel_size=3, stride=(2, 1), padding=1
        )
        self.out_channels = m_channels * (feat_dim // 8)

    def _make_layer(self, block, planes: int, num_blocks: int, stride: int):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return layers

    def __call__(self, x: mx.array) -> mx.array:
        x = mx.expand_dims(x, -1)
        out = nn.relu(self.conv1(x))
        for layer in self.layer1:
            out = layer(out)
        for layer in self.layer2:
            out = layer(out)
        out = nn.relu(self.conv2(out))
        b, h, w, c = out.shape
        out = mx.transpose(out, (0, 3, 1, 2))
        return mx.reshape(out, (b, c * h, w))


def _batchnorm_relu(channels: int) -> list[nn.Module | Callable[[mx.array], mx.array]]:
    return [nn.BatchNorm(channels), lambda x: nn.relu(x)]


def _apply_layers(
    x: mx.array, layers: list[nn.Module | Callable[[mx.array], mx.array]]
) -> mx.array:
    for layer in layers:
        x = layer(x)
    return x


class FusedTDNNLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
    ):
        super().__init__()
        if padding < 0:
            if kernel_size % 2 != 1:
                raise ValueError(f"Expected odd kernel size, got {kernel_size}")
            padding = (kernel_size - 1) // 2 * dilation
        self.linear = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = mx.swapaxes(x, 1, 2)
        x = nn.relu(self.linear(x))
        return mx.swapaxes(x, 1, 2)


class FusedCAMDenseTDNNLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bn_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
    ):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError(f"Expected odd kernel size, got {kernel_size}")
        padding = (kernel_size - 1) // 2 * dilation
        self.nonlinear1 = _batchnorm_relu(in_channels)
        self.linear1 = nn.Conv1d(in_channels, bn_channels, 1)
        self.cam_layer = CAMLayer(
            bn_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = mx.swapaxes(x, 1, 2)
        x = _apply_layers(x, self.nonlinear1)
        x = nn.relu(self.linear1(x))
        x = mx.swapaxes(x, 1, 2)
        return self.cam_layer(x)


class FusedCAMDenseTDNNBlock(nn.Module):
    def __init__(
        self,
        num_layers: int,
        in_channels: int,
        out_channels: int,
        bn_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
    ):
        super().__init__()
        self.layers = []
        for i in range(num_layers):
            self.layers.append(
                FusedCAMDenseTDNNLayer(
                    in_channels=in_channels + i * out_channels,
                    out_channels=out_channels,
                    bn_channels=bn_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                )
            )

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = mx.concatenate([x, layer(x)], axis=1)
        return x


class FusedTransitLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = False,
    ):
        super().__init__()
        self.nonlinear = _batchnorm_relu(in_channels)
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        x = mx.swapaxes(x, 1, 2)
        x = _apply_layers(x, self.nonlinear)
        x = self.linear(x)
        return mx.swapaxes(x, 1, 2)


class StepAudio2CAMPPlus(nn.Module):
    """CAMPPlus speaker encoder matching the folded StepAudio2 ONNX export."""

    def __init__(
        self,
        feat_dim: int = 80,
        embedding_size: int = 192,
        growth_rate: int = 32,
        bn_size: int = 4,
        init_channels: int = 128,
        output_level: str = "segment",
    ):
        super().__init__()
        self.head = FusedFCM(feat_dim=feat_dim)
        channels = self.head.out_channels
        self.output_level = output_level
        self.tdnn = FusedTDNNLayer(
            channels,
            init_channels,
            5,
            stride=2,
            dilation=1,
            padding=-1,
        )
        channels = init_channels

        self.blocks = []
        self.transits = []
        block_specs = list(zip((12, 24, 16), (3, 3, 3), (1, 2, 2)))
        for i, (num_layers, kernel_size, dilation) in enumerate(block_specs):
            block = FusedCAMDenseTDNNBlock(
                num_layers=num_layers,
                in_channels=channels,
                out_channels=growth_rate,
                bn_channels=bn_size * growth_rate,
                kernel_size=kernel_size,
                dilation=dilation,
            )
            self.blocks.append(block)
            channels = channels + num_layers * growth_rate
            self.transits.append(
                FusedTransitLayer(
                    channels,
                    channels // 2,
                    bias=i == len(block_specs) - 1,
                )
            )
            channels //= 2

        self.out_nonlinear = [lambda x: nn.relu(x)]
        if output_level == "segment":
            self.stats = StatsPool()
            self.dense = DenseLayer(
                channels * 2, embedding_size, config_str="batchnorm_"
            )

    def __call__(self, x: mx.array) -> mx.array:
        x = mx.swapaxes(x, 1, 2)
        x = self.head(x)
        x = self.tdnn(x)
        for block, transit in zip(self.blocks, self.transits):
            x = block(x)
            x = transit(x)
        x = mx.swapaxes(x, 1, 2)
        x = _apply_layers(x, self.out_nonlinear)
        x = mx.swapaxes(x, 1, 2)
        if self.output_level == "segment":
            x = self.stats(x)
            x = self.dense(x)
            if x.ndim == 3 and x.shape[-1] == 1:
                x = mx.squeeze(x, -1)
        return x

    def inference(self, audio: mx.array) -> mx.array:
        if audio.ndim == 1:
            audio = mx.expand_dims(audio, 0)
        features = []
        for i in range(audio.shape[0]):
            fbank = kaldi_fbank(audio[i], num_mel_bins=80)
            fbank = fbank - mx.mean(fbank, axis=0, keepdims=True)
            features.append(fbank)
        max_len = max(f.shape[0] for f in features)
        padded = []
        for fbank in features:
            if fbank.shape[0] < max_len:
                pad = mx.zeros(
                    (max_len - fbank.shape[0], fbank.shape[1]), dtype=fbank.dtype
                )
                fbank = mx.concatenate([fbank, pad], axis=0)
            padded.append(fbank)
        return self(mx.stack(padded))
