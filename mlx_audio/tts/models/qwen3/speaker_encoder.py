# Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

from typing import Dict, List

import mlx.core as mx
import mlx.nn as nn

from .config import Qwen3TTSSpeakerEncoderConfig


class TimeDelayNetBlock(nn.Module):
    """TDNN block with 1D convolution and ReLU activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) * dilation // 2,  # Same padding equivalent
        )
        self.dilation = dilation

    def __call__(self, x: mx.array) -> mx.array:
        # x: [batch, channels, time]
        # For dilated conv, we need to handle it manually since MLX doesn't support dilation
        if self.dilation > 1:
            # Dilate input by inserting zeros
            batch, channels, time = x.shape
            kernel_size = self.conv.weight.shape[1]
            # Pad input appropriately for dilated convolution
            pad_size = (kernel_size - 1) * self.dilation // 2
            x = mx.pad(x, [(0, 0), (0, 0), (pad_size, pad_size)], mode="reflect")
        out = self.conv(x)
        return nn.relu(out)


class Res2NetBlock(nn.Module):
    """Res2Net block for multi-scale feature extraction."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale: int = 8,
        kernel_size: int = 3,
        dilation: int = 1,
    ):
        super().__init__()
        self.scale = scale
        in_channel = in_channels // scale
        hidden_channel = out_channels // scale

        self.blocks = [
            TimeDelayNetBlock(
                in_channel,
                hidden_channel,
                kernel_size=kernel_size,
                dilation=dilation,
            )
            for _ in range(scale - 1)
        ]

    def __call__(self, x: mx.array) -> mx.array:
        # x: [batch, channels, time]
        # Split along channel dimension
        chunks = mx.split(x, self.scale, axis=1)
        outputs = []

        output_part = None
        for i, chunk in enumerate(chunks):
            if i == 0:
                output_part = chunk
            elif i == 1:
                output_part = self.blocks[i - 1](chunk)
            else:
                output_part = self.blocks[i - 1](chunk + output_part)
            outputs.append(output_part)

        return mx.concatenate(outputs, axis=1)


class SqueezeExcitationBlock(nn.Module):
    """Squeeze-and-excitation block for channel attention."""

    def __init__(self, in_channels: int, se_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=se_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.conv2 = nn.Conv1d(
            in_channels=se_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def __call__(self, x: mx.array) -> mx.array:
        # Global average pooling
        x_mean = mx.mean(x, axis=2, keepdims=True)
        # SE path
        se = nn.relu(self.conv1(x_mean))
        se = mx.sigmoid(self.conv2(se))
        return x * se


class SqueezeExcitationRes2NetBlock(nn.Module):
    """TDNN-Res2Net-TDNN-SE block used in ECAPA-TDNN."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        res2net_scale: int = 8,
        se_channels: int = 128,
        kernel_size: int = 3,
        dilation: int = 1,
    ):
        super().__init__()
        self.out_channels = out_channels

        self.tdnn1 = TimeDelayNetBlock(
            in_channels, out_channels, kernel_size=1, dilation=1
        )
        self.res2net_block = Res2NetBlock(
            out_channels, out_channels, res2net_scale, kernel_size, dilation
        )
        self.tdnn2 = TimeDelayNetBlock(
            out_channels, out_channels, kernel_size=1, dilation=1
        )
        self.se_block = SqueezeExcitationBlock(out_channels, se_channels, out_channels)

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        x = self.tdnn1(x)
        x = self.res2net_block(x)
        x = self.tdnn2(x)
        x = self.se_block(x)
        return x + residual


class AttentiveStatisticsPooling(nn.Module):
    """Attentive statistics pooling layer."""

    def __init__(self, channels: int, attention_channels: int = 128):
        super().__init__()
        self.eps = 1e-12
        self.tdnn = TimeDelayNetBlock(channels * 3, attention_channels, 1, 1)
        self.conv = nn.Conv1d(
            in_channels=attention_channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def __call__(self, x: mx.array) -> mx.array:
        # x: [batch, channels, time]
        batch, channels, seq_length = x.shape

        # Compute mean and std
        mean = mx.mean(x, axis=2, keepdims=True)
        std = mx.sqrt(mx.var(x, axis=2, keepdims=True) + self.eps)

        # Expand to match sequence length
        mean_expanded = mx.broadcast_to(mean, (batch, channels, seq_length))
        std_expanded = mx.broadcast_to(std, (batch, channels, seq_length))

        # Concatenate features
        attention = mx.concatenate([x, mean_expanded, std_expanded], axis=1)

        # Apply attention
        attention = self.tdnn(attention)
        attention = mx.tanh(attention)
        attention = self.conv(attention)
        attention = mx.softmax(attention, axis=2)

        # Compute weighted mean and std
        mean = mx.sum(attention * x, axis=2, keepdims=True)
        var = mx.sum(attention * (x - mean) ** 2, axis=2, keepdims=True)
        std = mx.sqrt(var.clip(self.eps, None))

        # Concatenate mean and std
        pooled = mx.concatenate([mean, std], axis=1)
        return pooled


class Qwen3TTSSpeakerEncoder(nn.Module):
    """ECAPA-TDNN speaker encoder for Qwen3-TTS."""

    def __init__(self, config: Qwen3TTSSpeakerEncoderConfig):
        super().__init__()
        self.config = config
        self.channels = config.enc_channels

        # Build blocks
        self.blocks = []

        # Initial TDNN layer
        self.blocks.append(
            TimeDelayNetBlock(
                config.mel_dim,
                config.enc_channels[0],
                config.enc_kernel_sizes[0],
                config.enc_dilations[0],
            )
        )

        # SE-Res2Net layers
        for i in range(1, len(config.enc_channels) - 1):
            self.blocks.append(
                SqueezeExcitationRes2NetBlock(
                    config.enc_channels[i - 1],
                    config.enc_channels[i],
                    res2net_scale=config.enc_res2net_scale,
                    se_channels=config.enc_se_channels,
                    kernel_size=config.enc_kernel_sizes[i],
                    dilation=config.enc_dilations[i],
                )
            )

        # Multi-layer feature aggregation
        self.mfa = TimeDelayNetBlock(
            config.enc_channels[-1],
            config.enc_channels[-1],
            config.enc_kernel_sizes[-1],
            config.enc_dilations[-1],
        )

        # Attentive Statistical Pooling
        self.asp = AttentiveStatisticsPooling(
            config.enc_channels[-1],
            attention_channels=config.enc_attention_channels,
        )

        # Final linear transformation
        self.fc = nn.Conv1d(
            in_channels=config.enc_channels[-1] * 2,
            out_channels=config.enc_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: Mel spectrogram [batch, time, mel_dim]

        Returns:
            Speaker embedding [batch, enc_dim]
        """
        # Transpose to [batch, channels, time]
        x = mx.transpose(x, (0, 2, 1))

        hidden_states_list = []
        for layer in self.blocks:
            x = layer(x)
            hidden_states_list.append(x)

        # Multi-layer feature aggregation (concatenate SE-Res2Net outputs)
        x = mx.concatenate(hidden_states_list[1:], axis=1)
        x = self.mfa(x)

        # Attentive Statistical Pooling
        x = self.asp(x)

        # Final linear transformation
        x = self.fc(x)

        # Squeeze time dimension
        x = mx.squeeze(x, axis=-1)
        return x

    @staticmethod
    def sanitize(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Sanitize weights from PyTorch to MLX format."""
        sanitized = {}
        for k, v in weights.items():
            if not k.startswith("speaker_encoder."):
                continue

            # Remove prefix
            new_key = k.replace("speaker_encoder.", "")

            # Handle Conv1d weights: PyTorch [out, in, kernel] -> MLX [out, kernel, in]
            if "conv.weight" in new_key and len(v.shape) == 3:
                v = mx.transpose(v, (0, 2, 1))
            elif "fc.weight" in new_key and len(v.shape) == 3:
                v = mx.transpose(v, (0, 2, 1))

            sanitized[new_key] = v

        return sanitized
