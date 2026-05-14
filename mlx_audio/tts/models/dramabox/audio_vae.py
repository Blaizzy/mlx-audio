from __future__ import annotations

import math
from enum import Enum

import mlx.core as mx
import mlx.nn as nn

from .latent import AudioLatentShape, AudioPatchifier

LATENT_DOWNSAMPLE_FACTOR = 4


class CausalityAxis(Enum):
    NONE = None
    WIDTH = "width"
    HEIGHT = "height"
    WIDTH_COMPATIBILITY = "width-compatibility"


class NormType(Enum):
    GROUP = "group"
    PIXEL = "pixel"


class PixelNorm(nn.Module):
    def __init__(self, dim: int = 1, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return x * mx.rsqrt(
            mx.mean(mx.square(x), axis=self.dim, keepdims=True) + self.eps
        )


def build_normalization_layer(channels: int, norm_type: NormType = NormType.PIXEL):
    if norm_type is NormType.PIXEL:
        return PixelNorm(dim=1, eps=1e-6)
    return nn.GroupNorm(32, channels, eps=1e-6)


def _pair(value):
    return value if isinstance(value, tuple) else (value, value)


class Conv2dNCHW(nn.Module):
    """NCHW-facing wrapper around MLX's NHWC Conv2d."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        if in_channels % groups != 0:
            raise ValueError(
                f"The number of input channels ({in_channels}) must be "
                f"divisible by the number of groups ({groups})"
            )
        kernel_size, stride, padding = map(_pair, (kernel_size, stride, padding))
        scale = math.sqrt(1 / (in_channels * kernel_size[0] * kernel_size[1]))
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(out_channels, *kernel_size, in_channels // groups),
        )
        if bias:
            self.bias = mx.zeros((out_channels,))
        self.stride = stride
        self.padding = padding
        self.dilation = _pair(dilation)
        self.groups = groups

    def __call__(self, x: mx.array) -> mx.array:
        x = x.transpose(0, 2, 3, 1)
        x = mx.conv2d(
            x,
            self.weight,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        if "bias" in self:
            x = x + self.bias
        return x.transpose(0, 3, 1, 2)


class CausalConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        causality_axis: CausalityAxis = CausalityAxis.HEIGHT,
    ):
        super().__init__()
        kernel_size = _pair(kernel_size)
        dilation = _pair(dilation)
        pad_h = (kernel_size[0] - 1) * dilation[0]
        pad_w = (kernel_size[1] - 1) * dilation[1]
        if causality_axis is CausalityAxis.NONE:
            self.padding = (
                pad_w // 2,
                pad_w - pad_w // 2,
                pad_h // 2,
                pad_h - pad_h // 2,
            )
        elif causality_axis in (CausalityAxis.WIDTH, CausalityAxis.WIDTH_COMPATIBILITY):
            self.padding = (pad_w, 0, pad_h // 2, pad_h - pad_h // 2)
        elif causality_axis is CausalityAxis.HEIGHT:
            self.padding = (pad_w // 2, pad_w - pad_w // 2, pad_h, 0)
        else:
            raise ValueError(f"Invalid causality_axis: {causality_axis}")
        self.conv = Conv2dNCHW(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def __call__(self, x: mx.array) -> mx.array:
        left, right, top, bottom = self.padding
        x = mx.pad(x, [(0, 0), (0, 0), (top, bottom), (left, right)])
        return self.conv(x)


def make_conv2d(
    in_channels: int,
    out_channels: int,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] = 1,
    dilation: int | tuple[int, int] = 1,
    groups: int = 1,
    bias: bool = True,
    causality_axis: CausalityAxis | None = None,
):
    if causality_axis is not None:
        return CausalConv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            causality_axis=causality_axis,
        )
    padding = (
        kernel_size // 2
        if isinstance(kernel_size, int)
        else tuple(k // 2 for k in kernel_size)
    )
    return Conv2dNCHW(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
    )


class ResnetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        dropout: float = 0.0,
        norm_type: NormType = NormType.PIXEL,
        causality_axis: CausalityAxis = CausalityAxis.HEIGHT,
    ):
        super().__init__()
        del dropout
        out_channels = in_channels if out_channels is None else out_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm1 = build_normalization_layer(in_channels, norm_type)
        self.conv1 = make_conv2d(
            in_channels, out_channels, 3, causality_axis=causality_axis
        )
        self.norm2 = build_normalization_layer(out_channels, norm_type)
        self.conv2 = make_conv2d(
            out_channels, out_channels, 3, causality_axis=causality_axis
        )
        if in_channels != out_channels:
            self.nin_shortcut = make_conv2d(
                in_channels, out_channels, 1, causality_axis=causality_axis
            )
        else:
            self.nin_shortcut = None

    def __call__(self, x: mx.array) -> mx.array:
        h = nn.silu(self.norm1(x))
        h = self.conv1(h)
        h = nn.silu(self.norm2(h))
        h = self.conv2(h)
        if self.nin_shortcut is not None:
            x = self.nin_shortcut(x)
        return x + h


class Downsample(nn.Module):
    def __init__(self, channels: int, causality_axis: CausalityAxis):
        super().__init__()
        self.causality_axis = causality_axis
        self.conv = Conv2dNCHW(channels, channels, kernel_size=3, stride=2, padding=0)

    def __call__(self, x: mx.array) -> mx.array:
        if self.causality_axis is CausalityAxis.NONE:
            pad = (0, 1, 0, 1)
        elif self.causality_axis is CausalityAxis.WIDTH:
            pad = (2, 0, 0, 1)
        elif self.causality_axis is CausalityAxis.HEIGHT:
            pad = (0, 1, 2, 0)
        elif self.causality_axis is CausalityAxis.WIDTH_COMPATIBILITY:
            pad = (1, 0, 0, 1)
        else:
            raise ValueError(f"Invalid causality_axis: {self.causality_axis}")
        left, right, top, bottom = pad
        x = mx.pad(x, [(0, 0), (0, 0), (top, bottom), (left, right)])
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int, causality_axis: CausalityAxis):
        super().__init__()
        self.causality_axis = causality_axis
        self.conv = make_conv2d(channels, channels, 3, causality_axis=causality_axis)

    def __call__(self, x: mx.array) -> mx.array:
        x = mx.repeat(mx.repeat(x, repeats=2, axis=2), repeats=2, axis=3)
        x = self.conv(x)
        if self.causality_axis is CausalityAxis.HEIGHT:
            x = x[:, :, 1:, :]
        elif self.causality_axis is CausalityAxis.WIDTH:
            x = x[:, :, :, 1:]
        return x


class MidBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        dropout: float,
        norm_type: NormType,
        causality_axis: CausalityAxis,
    ):
        super().__init__()
        self.block_1 = ResnetBlock(
            channels, channels, dropout, norm_type, causality_axis
        )
        self.block_2 = ResnetBlock(
            channels, channels, dropout, norm_type, causality_axis
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = self.block_1(x)
        return self.block_2(x)


class Stage(nn.Module):
    def __init__(
        self,
        block: list[ResnetBlock],
        downsample: Downsample | None = None,
        upsample: Upsample | None = None,
    ):
        super().__init__()
        self.block = block
        self.downsample = downsample
        self.upsample = upsample


class PerChannelStatistics(nn.Module):
    def __init__(self, latent_channels: int = 128):
        super().__init__()
        self.std_of_means = mx.ones((latent_channels,))
        self.mean_of_means = mx.zeros((latent_channels,))

    def un_normalize(self, x: mx.array) -> mx.array:
        return x * self.std_of_means.astype(x.dtype) + self.mean_of_means.astype(
            x.dtype
        )

    def normalize(self, x: mx.array) -> mx.array:
        return (x - self.mean_of_means.astype(x.dtype)) / self.std_of_means.astype(
            x.dtype
        )


class AudioEncoder(nn.Module):
    def __init__(
        self,
        ch: int = 128,
        ch_mult: tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        in_channels: int = 2,
        z_channels: int = 8,
        double_z: bool = True,
        dropout: float = 0.0,
        norm_type: NormType = NormType.PIXEL,
        causality_axis: CausalityAxis = CausalityAxis.HEIGHT,
        sample_rate: int = 16000,
        mel_hop_length: int = 160,
        n_fft: int = 1024,
        mel_bins: int = 64,
    ):
        super().__init__()
        self.per_channel_statistics = PerChannelStatistics(latent_channels=ch)
        self.sample_rate = sample_rate
        self.mel_hop_length = mel_hop_length
        self.n_fft = n_fft
        self.mel_bins = mel_bins
        self.patchifier = AudioPatchifier()
        self.conv_in = make_conv2d(in_channels, ch, 3, causality_axis=causality_axis)
        self.down = []
        in_ch_mult = (1, *tuple(ch_mult))
        block_in = ch
        for level, mult in enumerate(ch_mult):
            block_in = ch * in_ch_mult[level]
            block_out = ch * mult
            blocks = []
            for _ in range(num_res_blocks):
                blocks.append(
                    ResnetBlock(block_in, block_out, dropout, norm_type, causality_axis)
                )
                block_in = block_out
            downsample = (
                Downsample(block_in, causality_axis)
                if level != len(ch_mult) - 1
                else None
            )
            self.down.append(Stage(block=blocks, downsample=downsample))
        self.mid = MidBlock(block_in, dropout, norm_type, causality_axis)
        self.norm_out = build_normalization_layer(block_in, norm_type)
        self.conv_out = make_conv2d(
            block_in,
            2 * z_channels if double_z else z_channels,
            3,
            causality_axis=causality_axis,
        )
        self.double_z = double_z

    def __call__(self, spectrogram: mx.array) -> mx.array:
        h = self.conv_in(spectrogram)
        for stage in self.down:
            for block in stage.block:
                h = block(h)
            if stage.downsample is not None:
                h = stage.downsample(h)
        h = self.mid(h)
        h = self.conv_out(nn.silu(self.norm_out(h)))
        means = mx.split(h, 2, axis=1)[0] if self.double_z else h
        latent_shape = AudioLatentShape(
            means.shape[0], means.shape[1], means.shape[2], means.shape[3]
        )
        patched = self.patchifier.patchify(means)
        normalized = self.per_channel_statistics.normalize(patched)
        return self.patchifier.unpatchify(normalized, latent_shape)


class AudioDecoder(nn.Module):
    def __init__(
        self,
        ch: int = 128,
        out_ch: int = 2,
        ch_mult: tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        resolution: int = 256,
        z_channels: int = 8,
        dropout: float = 0.0,
        norm_type: NormType = NormType.PIXEL,
        causality_axis: CausalityAxis = CausalityAxis.HEIGHT,
        sample_rate: int = 16000,
        mel_hop_length: int = 160,
        mel_bins: int = 64,
    ):
        super().__init__()
        self.per_channel_statistics = PerChannelStatistics(latent_channels=ch)
        self.sample_rate = sample_rate
        self.mel_hop_length = mel_hop_length
        self.mel_bins = mel_bins
        self.out_ch = out_ch
        self.causality_axis = causality_axis
        self.patchifier = AudioPatchifier()
        base_channels = ch * ch_mult[-1]
        self.conv_in = make_conv2d(
            z_channels, base_channels, 3, causality_axis=causality_axis
        )
        self.mid = MidBlock(base_channels, dropout, norm_type, causality_axis)
        self.up = [None] * len(ch_mult)
        block_in = base_channels
        for level in reversed(range(len(ch_mult))):
            block_out = ch * ch_mult[level]
            blocks = []
            for _ in range(num_res_blocks + 1):
                blocks.append(
                    ResnetBlock(block_in, block_out, dropout, norm_type, causality_axis)
                )
                block_in = block_out
            upsample = Upsample(block_in, causality_axis) if level != 0 else None
            self.up[level] = Stage(block=blocks, upsample=upsample)
        self.norm_out = build_normalization_layer(block_in, norm_type)
        self.conv_out = make_conv2d(block_in, out_ch, 3, causality_axis=causality_axis)

    def _denormalize_latents(
        self, sample: mx.array
    ) -> tuple[mx.array, AudioLatentShape]:
        latent_shape = AudioLatentShape(
            sample.shape[0], sample.shape[1], sample.shape[2], sample.shape[3]
        )
        patched = self.patchifier.patchify(sample)
        denormalized = self.per_channel_statistics.un_normalize(patched)
        sample = self.patchifier.unpatchify(denormalized, latent_shape)
        target_frames = latent_shape.frames * LATENT_DOWNSAMPLE_FACTOR
        if self.causality_axis is not CausalityAxis.NONE:
            target_frames = max(target_frames - (LATENT_DOWNSAMPLE_FACTOR - 1), 1)
        target_shape = AudioLatentShape(
            latent_shape.batch,
            self.out_ch,
            target_frames,
            self.mel_bins,
        )
        return sample, target_shape

    def _adjust_output_shape(
        self, decoded: mx.array, target_shape: AudioLatentShape
    ) -> mx.array:
        decoded = decoded[
            :,
            : target_shape.channels,
            : min(decoded.shape[2], target_shape.frames),
            : min(decoded.shape[3], target_shape.mel_bins),
        ]
        time_pad = target_shape.frames - decoded.shape[2]
        freq_pad = target_shape.mel_bins - decoded.shape[3]
        if time_pad > 0 or freq_pad > 0:
            decoded = mx.pad(
                decoded,
                [(0, 0), (0, 0), (0, max(time_pad, 0)), (0, max(freq_pad, 0))],
            )
        return decoded[
            :, : target_shape.channels, : target_shape.frames, : target_shape.mel_bins
        ]

    def __call__(self, sample: mx.array) -> mx.array:
        sample, target_shape = self._denormalize_latents(sample)
        h = self.conv_in(sample)
        h = self.mid(h)
        for level in reversed(range(len(self.up))):
            stage = self.up[level]
            for block in stage.block:
                h = block(h)
            if stage.upsample is not None:
                h = stage.upsample(h)
        h = self.conv_out(nn.silu(self.norm_out(h)))
        return self._adjust_output_shape(h, target_shape)


class AudioVAE(nn.Module):
    def __init__(
        self,
        ch: int = 128,
        ch_mult: tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        in_channels: int = 2,
        out_ch: int = 2,
        z_channels: int = 8,
        double_z: bool = True,
        dropout: float = 0.0,
        norm_type: NormType = NormType.PIXEL,
        causality_axis: CausalityAxis = CausalityAxis.HEIGHT,
        sample_rate: int = 16000,
        mel_hop_length: int = 160,
        n_fft: int = 1024,
        mel_bins: int = 64,
    ):
        super().__init__()
        self.encoder = AudioEncoder(
            ch=ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            in_channels=in_channels,
            z_channels=z_channels,
            double_z=double_z,
            dropout=dropout,
            norm_type=norm_type,
            causality_axis=causality_axis,
            sample_rate=sample_rate,
            mel_hop_length=mel_hop_length,
            n_fft=n_fft,
            mel_bins=mel_bins,
        )
        self.decoder = AudioDecoder(
            ch=ch,
            out_ch=out_ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            z_channels=z_channels,
            dropout=dropout,
            norm_type=norm_type,
            causality_axis=causality_axis,
            sample_rate=sample_rate,
            mel_hop_length=mel_hop_length,
            mel_bins=mel_bins,
        )

    def encode(self, spectrogram: mx.array) -> mx.array:
        return self.encoder(spectrogram)

    def decode(self, latent: mx.array) -> mx.array:
        return self.decoder(latent)
