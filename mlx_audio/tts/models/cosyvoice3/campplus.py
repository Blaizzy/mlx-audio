# Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)
"""
CosyVoice3 CAMPPlus speaker embedding model (pure MLX).

This version matches the CosyVoice3 ONNX export structure where BatchNorm
is fused into Conv for the head and initial TDNN layers.
"""

import re

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten


def _povey_window(size: int) -> mx.array:
    n = mx.arange(size)
    hann = 0.5 - 0.5 * mx.cos(2 * mx.pi * n / (size - 1))
    return mx.power(hann, 0.85)


def _next_power_of_2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def kaldi_fbank(
    audio: mx.array,
    sample_rate: int = 16000,
    num_mel_bins: int = 80,
    frame_length: float = 25.0,
    frame_shift: float = 10.0,
) -> mx.array:
    """Kaldi-style fbank feature extraction."""
    from mlx_audio.tts.models.chatterbox.s3gen.xvector import kaldi_fbank as _kaldi_fbank

    return _kaldi_fbank(audio, sample_rate, num_mel_bins, frame_length, frame_shift)


def statistics_pooling(x: mx.array, axis: int = -1) -> mx.array:
    mean = mx.mean(x, axis=axis)
    std = mx.sqrt(mx.var(x, axis=axis) + 1e-5)
    return mx.concatenate([mean, std], axis=-1)


def conv1d_pytorch_format(x: mx.array, conv_layer) -> mx.array:
    """Conv1d with PyTorch (B,C,T) format input/output."""
    x = mx.swapaxes(x, 1, 2)  # (B, C, T) -> (B, T, C)
    x = conv_layer(x)
    x = mx.swapaxes(x, 1, 2)  # (B, T, C) -> (B, C, T)
    return x


class BasicResBlock(nn.Module):
    """ResBlock with fused BatchNorm (Conv2d bias=True, no separate BN)."""

    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=(stride, 1), padding=1, bias=True
        )
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.shortcut = []
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = [
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=(stride, 1),
                    bias=True,
                ),
            ]

    def __call__(self, x: mx.array) -> mx.array:
        out = nn.relu(self.conv1(x))
        out = self.conv2(out)
        shortcut = x
        for layer in self.shortcut:
            shortcut = layer(shortcut)
        return nn.relu(out + shortcut)


class FCM(nn.Module):
    """Feature Context Module with fused BatchNorm."""

    def __init__(self, num_blocks=[2, 2], m_channels=32, feat_dim=80):
        super().__init__()
        self.in_planes = m_channels
        self.conv1 = nn.Conv2d(1, m_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.layer1 = self._make_layer(m_channels, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(m_channels, num_blocks[1], stride=2)
        self.conv2 = nn.Conv2d(
            m_channels, m_channels, kernel_size=3, stride=(2, 1), padding=1, bias=True
        )
        self.out_channels = m_channels * (feat_dim // 8)

    def _make_layer(self, planes: int, num_blocks: int, stride: int):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicResBlock(self.in_planes, planes, s))
            self.in_planes = planes * BasicResBlock.expansion
        return layers

    def __call__(self, x: mx.array) -> mx.array:
        x = mx.expand_dims(x, -1)  # (B, F, T, 1) = (B, H, W, C)
        out = nn.relu(self.conv1(x))
        for layer in self.layer1:
            out = layer(out)
        for layer in self.layer2:
            out = layer(out)
        out = nn.relu(self.conv2(out))
        B, H, W, C = out.shape
        out = mx.transpose(out, (0, 3, 1, 2))  # (B, C, H, W)
        out = mx.reshape(out, (B, C * H, W))
        return out


class CAMLayer(nn.Module):
    """Context-Aware Masking layer."""

    def __init__(
        self, bn_channels: int, out_channels: int,
        kernel_size: int, stride: int, padding: int,
        dilation: int, bias: bool, reduction: int = 2,
    ):
        super().__init__()
        self.linear_local = nn.Conv1d(
            bn_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation, bias=bias,
        )
        self.linear1 = nn.Conv1d(bn_channels, bn_channels // reduction, 1)
        self.linear2 = nn.Conv1d(bn_channels // reduction, out_channels, 1)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, C, T)
        y = conv1d_pytorch_format(x, self.linear_local)
        context = mx.mean(x, axis=-1, keepdims=True) + self.seg_pooling(x)
        context = nn.relu(conv1d_pytorch_format(context, self.linear1))
        m = nn.sigmoid(conv1d_pytorch_format(context, self.linear2))
        return y * m

    def seg_pooling(self, x: mx.array, seg_len: int = 100) -> mx.array:
        B, C, T = x.shape
        n_segs = (T + seg_len - 1) // seg_len
        pad_len = n_segs * seg_len - T
        if pad_len > 0:
            x_padded = mx.concatenate([x, mx.zeros((B, C, pad_len))], axis=-1)
        else:
            x_padded = x
        x_reshaped = mx.reshape(x_padded, (B, C, n_segs, seg_len))
        seg = mx.mean(x_reshaped, axis=-1)  # (B, C, n_segs)
        seg = mx.expand_dims(seg, -1)
        seg = mx.broadcast_to(seg, (B, C, n_segs, seg_len))
        seg = mx.reshape(seg, (B, C, -1))
        return seg[:, :, :T]


class CAMDenseTDNNLayer(nn.Module):
    """Dense TDNN layer matching CosyVoice3 ONNX structure."""

    def __init__(
        self, in_channels: int, out_channels: int, bn_channels: int,
        kernel_size: int, dilation: int = 1,
    ):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        # nonlinear1: BatchNorm + ReLU (NOT fused, has params)
        self.nonlinear1 = nn.BatchNorm(in_channels)
        # linear1: bottleneck conv (WITH bias in CosyVoice3)
        self.linear1 = nn.Conv1d(in_channels, bn_channels, 1, bias=True)
        # nonlinear2: ReLU only (no BN, no parameters)
        # CAM layer
        self.cam_layer = CAMLayer(
            bn_channels, out_channels, kernel_size,
            stride=1, padding=padding, dilation=dilation, bias=False,
        )

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, C, T)
        x = mx.swapaxes(x, 1, 2)  # -> (B, T, C)
        x = nn.relu(self.nonlinear1(x))
        x = self.linear1(x)
        x = nn.relu(x)  # nonlinear2: ReLU before CAM layer
        x = mx.swapaxes(x, 1, 2)  # -> (B, C, T)
        x = self.cam_layer(x)
        return x


class CAMDenseTDNNBlock(nn.Module):

    def __init__(
        self, num_layers: int, in_channels: int, out_channels: int,
        bn_channels: int, kernel_size: int, dilation: int = 1,
    ):
        super().__init__()
        self.layers = []
        for i in range(num_layers):
            self.layers.append(CAMDenseTDNNLayer(
                in_channels=in_channels + i * out_channels,
                out_channels=out_channels,
                bn_channels=bn_channels,
                kernel_size=kernel_size,
                dilation=dilation,
            ))

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = mx.concatenate([x, layer(x)], axis=1)
        return x


class TransitLayer(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super().__init__()
        self.nonlinear = nn.BatchNorm(in_channels)
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        x = mx.swapaxes(x, 1, 2)  # (B, C, T) -> (B, T, C)
        x = nn.relu(self.nonlinear(x))
        x = self.linear(x)
        x = mx.swapaxes(x, 1, 2)  # -> (B, C, T)
        return x


class CAMPPlus(nn.Module):
    """CosyVoice3 CAMPPlus speaker embedding model.

    Matches the ONNX export structure with fused BatchNorm in head/TDNN.
    """

    def __init__(
        self,
        feat_dim: int = 80,
        embedding_size: int = 192,
        growth_rate: int = 32,
        bn_size: int = 4,
        init_channels: int = 128,
    ):
        super().__init__()
        self.head = FCM(feat_dim=feat_dim)
        channels = self.head.out_channels  # 320

        # Initial TDNN with fused BN (Conv1d + ReLU, no separate BN)
        self.tdnn = nn.Conv1d(channels, init_channels, 5, stride=2, padding=2, bias=True)
        channels = init_channels  # 128

        # Dense TDNN blocks
        self.blocks = []
        self.transits = []
        for i, (num_layers, kernel_size, dilation) in enumerate(
            zip((12, 24, 16), (3, 3, 3), (1, 2, 2))
        ):
            self.blocks.append(CAMDenseTDNNBlock(
                num_layers=num_layers,
                in_channels=channels,
                out_channels=growth_rate,
                bn_channels=bn_size * growth_rate,
                kernel_size=kernel_size,
                dilation=dilation,
            ))
            channels = channels + num_layers * growth_rate

            # transit3 (i=2) has bias=True in CosyVoice3
            bias = (i == 2)
            self.transits.append(TransitLayer(channels, channels // 2, bias=bias))
            channels //= 2

        # Dense output layer
        self.dense_linear = nn.Conv1d(channels * 2, embedding_size, 1, bias=False)
        self.dense_bn = nn.BatchNorm(embedding_size, affine=False)

    def sanitize(self, weights: dict) -> dict:
        """Convert ONNX weight names to MLX model structure."""
        new_weights = {}
        curr_weights = dict(tree_flatten(self.parameters()))

        for key, value in weights.items():
            new_key = key

            if "num_batches_tracked" in key:
                continue

            # === Head (FCM) name mapping ===
            # head.layer1.layer1.0. -> head.layer1.0. (remove redundant layer name)
            new_key = re.sub(r"head\.layer(\d+)\.layer\d+\.(\d+)\.", r"head.layer\1.\2.", new_key)
            # head.layer1.0.shortcut.shortcut.0. -> head.layer1.0.shortcut.0.
            new_key = re.sub(r"\.shortcut\.shortcut\.(\d+)\.", r".shortcut.\1.", new_key)

            # === xvector structure ===
            new_key = re.sub(
                r"xvector\.block(\d+)\.",
                lambda m: f"blocks.{int(m.group(1))-1}.",
                new_key,
            )
            new_key = re.sub(
                r"xvector\.transit(\d+)\.",
                lambda m: f"transits.{int(m.group(1))-1}.",
                new_key,
            )
            new_key = new_key.replace("xvector.tdnn.linear.", "tdnn.")
            new_key = new_key.replace("xvector.dense.linear.", "dense_linear.")
            new_key = new_key.replace("xvector.dense.nonlinear.batchnorm.", "dense_bn.")

            # === DenseBlock layers ===
            new_key = re.sub(
                r"\.tdnnd(\d+)\.", lambda m: f".layers.{int(m.group(1))-1}.", new_key
            )

            # === NonLinear structure ===
            # .nonlinear1.batchnorm. -> .nonlinear1. (direct BatchNorm, not in list)
            new_key = re.sub(r"\.nonlinear1\.batchnorm\.", r".nonlinear1.", new_key)
            # transit .nonlinear.batchnorm. -> .nonlinear.
            new_key = new_key.replace(".nonlinear.batchnorm.", ".nonlinear.")

            # === Conv weight transposition ===
            if "weight" in new_key and value.ndim == 4:
                # Conv2d: PyTorch (O,I,H,W) -> MLX (O,H,W,I)
                if new_key in curr_weights and value.shape != curr_weights[new_key].shape:
                    value = value.transpose(0, 2, 3, 1)
            elif "weight" in new_key and value.ndim == 3:
                # Conv1d: PyTorch (O,I,K) -> MLX (O,K,I)
                if new_key in curr_weights and value.shape != curr_weights[new_key].shape:
                    value = value.swapaxes(1, 2)

            new_weights[new_key] = value

        return new_weights

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: Input features (B, T, F) where F is feat_dim
        Returns:
            Speaker embeddings (B, embedding_size)
        """
        x = mx.swapaxes(x, 1, 2)  # (B, T, F) -> (B, F, T)
        x = self.head(x)

        # TDNN with fused BN: just Conv1d + ReLU
        x = mx.swapaxes(x, 1, 2)  # (B, C, T) -> (B, T, C)
        x = nn.relu(self.tdnn(x))
        x = mx.swapaxes(x, 1, 2)  # -> (B, C, T)

        # Dense blocks with transitions
        for block, transit in zip(self.blocks, self.transits):
            x = block(x)
            x = transit(x)

        # out_nonlinear: just ReLU (no BN in CosyVoice3)
        x = nn.relu(x)

        # Statistics pooling
        x = mx.swapaxes(x, 1, 2)  # (B, C, T) -> (B, T, C)
        mean = mx.mean(x, axis=1)
        std = mx.sqrt(mx.var(x, axis=1) + 1e-5)
        x = mx.concatenate([mean, std], axis=-1)  # (B, 2*C)

        # Dense layer
        x = mx.expand_dims(x, 1)  # (B, 1, 2*C)
        x = self.dense_linear(x)  # (B, 1, emb)
        x = self.dense_bn(x)
        x = mx.squeeze(x, 1)  # (B, emb)

        return x
