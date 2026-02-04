# Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

import math
from typing import Dict, List

import mlx.core as mx
import mlx.nn as nn


class Snake1d(nn.Module):

    def __init__(self, channels: int, logscale: bool = True):
        super().__init__()
        self.alpha = mx.zeros((1, channels, 1))
        self.beta = mx.zeros((1, channels, 1))
        self.logscale = logscale

    def __call__(self, x: mx.array) -> mx.array:
        # x: [batch, channels, time]
        # Apply exp() to alpha/beta when logscale=True (matches diffusers)
        alpha = mx.exp(self.alpha) if self.logscale else self.alpha
        beta = mx.exp(self.beta) if self.logscale else self.beta
        return x + (1.0 / (beta + 1e-9)) * mx.power(mx.sin(alpha * x), 2)


class WeightNormConv1d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # Weight normalization: W = g * V / ||V||
        self.weight_g = mx.ones((out_channels, 1, 1))
        self.weight_v = (
            mx.random.normal((out_channels, in_channels, kernel_size)) * 0.02
        )

        if bias:
            self.bias = mx.zeros((out_channels,))
        else:
            self.bias = None

    def __call__(self, x: mx.array) -> mx.array:
        # x: [batch, channels, time]
        # Compute normalized weight
        norm = mx.sqrt(mx.sum(self.weight_v**2, axis=(1, 2), keepdims=True) + 1e-12)
        weight = self.weight_g * self.weight_v / norm

        # Transpose for MLX conv: [batch, time, channels]
        x = x.transpose(0, 2, 1)

        # MLX conv1d expects weight as [out, kernel, in]
        weight_mlx = weight.transpose(0, 2, 1)

        # Apply convolution
        y = mx.conv1d(
            x,
            weight_mlx,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )

        # Add bias
        if self.bias is not None:
            y = y + self.bias

        # Transpose back: [batch, channels, time]
        return y.transpose(0, 2, 1)


class WeightNormConvTranspose1d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weight_g = mx.ones((in_channels, 1, 1))
        self.weight_v = (
            mx.random.normal((in_channels, out_channels, kernel_size)) * 0.02
        )

        if bias:
            self.bias = mx.zeros((out_channels,))
        else:
            self.bias = None

    def __call__(self, x: mx.array) -> mx.array:
        # x: [batch, channels, time]
        batch_size, in_ch, in_len = x.shape

        # Compute normalized weight: [in, out, kernel]
        norm = mx.sqrt(mx.sum(self.weight_v**2, axis=(1, 2), keepdims=True) + 1e-12)
        weight = self.weight_g * self.weight_v / norm

        # Transposed conv1d output length: (in_len - 1) * stride + kernel_size
        # After removing 2*padding, final length is: (in_len - 1) * stride + kernel_size - 2*padding

        # x: [batch, in_ch, in_len] -> [batch, in_len, in_ch]
        x_t = x.transpose(0, 2, 1)

        # weight: [in_ch, out_ch, kernel] -> [in_ch, out_ch * kernel]
        weight_reshaped = weight.reshape(
            self.in_channels, self.out_channels * self.kernel_size
        )

        # Compute: [batch, in_len, out_ch * kernel]
        y = x_t @ weight_reshaped

        # Reshape to [batch, in_len, out_ch, kernel]
        y = y.reshape(batch_size, in_len, self.out_channels, self.kernel_size)

        # Transpose: [batch, out_ch, in_len, kernel]
        y = y.transpose(0, 2, 1, 3)

        # Full output length before padding removal
        full_out_len = (in_len - 1) * self.stride + self.kernel_size

        overlap = self.kernel_size - self.stride

        # Split kernel contributions
        first_part = y[:, :, :, : self.stride]  # [batch, out_ch, in_len, stride]
        second_part = y[:, :, :, self.stride :]  # [batch, out_ch, in_len, overlap]

        # Flatten: [batch, out_ch, in_len * stride]
        first_flat = first_part.reshape(
            batch_size, self.out_channels, in_len * self.stride
        )

        if overlap > 0:
            second_flat = second_part.reshape(
                batch_size, self.out_channels, in_len * overlap
            )

            first_padded = mx.concatenate(
                [
                    first_flat,
                    mx.zeros((batch_size, self.out_channels, overlap), dtype=x.dtype),
                ],
                axis=2,
            )

            # Pad second_flat at the start (no overlap before first input)
            second_padded = mx.concatenate(
                [
                    mx.zeros(
                        (batch_size, self.out_channels, self.stride), dtype=x.dtype
                    ),
                    second_flat,
                ],
                axis=2,
            )

            # Trim to match full_out_len
            first_padded = first_padded[:, :, :full_out_len]
            second_padded = second_padded[:, :, :full_out_len]

            output = first_padded + second_padded
        else:
            output = first_flat

        # Trim padding
        if self.padding > 0:
            end_idx = full_out_len - self.padding
            output = output[:, :, self.padding : end_idx]

        # Add bias
        if self.bias is not None:
            output = output + self.bias[None, :, None]

        return output


class ResidualUnit(nn.Module):

    def __init__(self, channels: int, kernel_size: int = 7, dilation: int = 1):
        super().__init__()
        self.snake1 = Snake1d(channels)
        self.conv1 = WeightNormConv1d(
            channels,
            channels,
            kernel_size,
            padding=(kernel_size - 1) * dilation // 2,
            dilation=dilation,
        )
        self.snake2 = Snake1d(channels)
        self.conv2 = WeightNormConv1d(channels, channels, 1)

    def __call__(self, x: mx.array) -> mx.array:
        y = self.snake1(x)
        y = self.conv1(y)
        y = self.snake2(y)
        y = self.conv2(y)
        # Handle potential padding mismatch
        padding = (x.shape[-1] - y.shape[-1]) // 2
        if padding > 0:
            x = x[..., padding:-padding]
        return x + y


class EncoderBlock(nn.Module):
    """Encoder block for Oobleck VAE."""

    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.res_unit1 = ResidualUnit(in_channels, dilation=1)
        self.res_unit2 = ResidualUnit(in_channels, dilation=3)
        self.res_unit3 = ResidualUnit(in_channels, dilation=9)
        self.snake1 = Snake1d(in_channels)
        self.conv1 = WeightNormConv1d(
            in_channels,
            out_channels,
            kernel_size=stride * 2,
            stride=stride,
            padding=math.ceil(stride / 2),
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = self.res_unit1(x)
        x = self.res_unit2(x)
        x = self.res_unit3(x)
        x = self.snake1(x)
        x = self.conv1(x)
        return x


class OobleckEncoder(nn.Module):
    """Oobleck VAE Encoder for encoding audio to latents.

    The encoder outputs 2*latent_dim channels representing mean and scale
    of a diagonal Gaussian distribution.
    """

    def __init__(
        self,
        audio_channels: int = 2,
        channels: int = 128,
        latent_channels: int = 128,  # Output channels (mean + scale)
        channel_multiples: List[int] = None,
        downsampling_ratios: List[int] = None,
    ):
        super().__init__()

        if channel_multiples is None:
            channel_multiples = [1, 2, 4, 8, 16]
        if downsampling_ratios is None:
            downsampling_ratios = [2, 4, 4, 6, 10]

        self.audio_channels = audio_channels
        self.channels = channels
        self.latent_channels = latent_channels

        # Input convolution
        self.conv1 = WeightNormConv1d(
            audio_channels, channels, kernel_size=7, padding=3
        )

        # Encoder blocks
        channel_multiples_with_1 = [1] + channel_multiples
        self.block = []
        for i, stride in enumerate(downsampling_ratios):
            in_ch = channels * channel_multiples_with_1[i]
            out_ch = channels * channel_multiples_with_1[i + 1]
            self.block.append(EncoderBlock(in_ch, out_ch, stride))

        # Output layers - output encoder_hidden_size channels (mean + scale)
        d_model = channels * channel_multiples[-1]
        self.snake1 = Snake1d(d_model)
        self.conv2 = WeightNormConv1d(
            d_model, latent_channels, kernel_size=3, padding=1
        )

    def __call__(self, x: mx.array) -> mx.array:
        """Encode audio to latent parameters (mean and scale).

        Args:
            x: Audio tensor of shape [batch, channels, samples] or [batch, samples, channels]

        Returns:
            Latent parameters of shape [batch, latent_channels, time]
            where latent_channels = 2 * latent_dim (mean + scale)
        """
        # Ensure channels-first format
        if x.shape[-1] == self.audio_channels:
            x = x.transpose(0, 2, 1)  # [batch, samples, ch] -> [batch, ch, samples]

        x = self.conv1(x)

        for block in self.block:
            x = block(x)

        x = self.snake1(x)
        x = self.conv2(x)

        return x


class DecoderBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
    ):
        super().__init__()
        # Snake activation before transposed conv
        self.snake1 = Snake1d(in_channels)

        # Transposed conv for upsampling
        self.conv_t1 = WeightNormConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size=stride * 2,
            stride=stride,
            padding=stride // 2,
        )

        # Residual units with different dilations
        self.res_unit1 = ResidualUnit(out_channels, dilation=1)
        self.res_unit2 = ResidualUnit(out_channels, dilation=3)
        self.res_unit3 = ResidualUnit(out_channels, dilation=9)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.snake1(x)
        x = self.conv_t1(x)
        x = self.res_unit1(x)
        x = self.res_unit2(x)
        x = self.res_unit3(x)
        return x


class OobleckDecoder(nn.Module):

    def __init__(
        self,
        input_channels: int = 64,
        channels: int = 128,
        audio_channels: int = 2,
        channel_multiples: List[int] = None,
        downsampling_ratios: List[int] = None,
    ):
        super().__init__()

        if channel_multiples is None:
            channel_multiples = [1, 2, 4, 8, 16]
        if downsampling_ratios is None:
            downsampling_ratios = [2, 4, 4, 6, 10]

        self.input_channels = input_channels
        self.channels = channels
        self.audio_channels = audio_channels

        # Reverse for decoder (upsampling)
        channel_multiples = list(reversed(channel_multiples))
        downsampling_ratios = list(reversed(downsampling_ratios))

        # Input convolution (named conv1 to match PyTorch weights)
        in_ch = input_channels
        out_ch = channels * channel_multiples[0]
        self.conv1 = WeightNormConv1d(in_ch, out_ch, kernel_size=7, padding=3)

        # Decoder blocks
        self.block = []
        for i, (mult, stride) in enumerate(zip(channel_multiples, downsampling_ratios)):
            in_ch = channels * mult
            out_ch = (
                channels * channel_multiples[i + 1]
                if i + 1 < len(channel_multiples)
                else channels
            )
            self.block.append(DecoderBlock(in_ch, out_ch, stride))

        # Output layers (named snake1/conv2 to match PyTorch weights)
        self.snake1 = Snake1d(channels)
        self.conv2 = WeightNormConv1d(
            channels, audio_channels, kernel_size=7, padding=3, bias=False
        )

    def __call__(self, x: mx.array) -> mx.array:

        # Ensure channels-first format
        if x.shape[-1] == self.input_channels:
            x = x.transpose(0, 2, 1)  # [batch, time, ch] -> [batch, ch, time]

        x = self.conv1(x)

        for block in self.block:
            x = block(x)

        x = self.snake1(x)
        x = self.conv2(x)

        # Note: PyTorch's Oobleck decoder does NOT apply tanh
        # The audio values should already be in a reasonable range

        return x


class AutoencoderOobleck(nn.Module):

    def __init__(
        self,
        audio_channels: int = 2,
        channel_multiples: List[int] = None,
        decoder_channels: int = 128,
        decoder_input_channels: int = 64,
        downsampling_ratios: List[int] = None,
        encoder_hidden_size: int = 128,
        sampling_rate: int = 48000,
    ):
        super().__init__()

        if channel_multiples is None:
            channel_multiples = [1, 2, 4, 8, 16]
        if downsampling_ratios is None:
            downsampling_ratios = [2, 4, 4, 6, 10]

        self.sampling_rate = sampling_rate
        self.hop_length = math.prod(downsampling_ratios)
        self.audio_channels = audio_channels
        self.latent_channels = decoder_input_channels
        self.encoder_hidden_size = encoder_hidden_size

        # Encoder for audio-to-latent conversion
        # Output encoder_hidden_size channels (mean + scale = 2 * latent_dim)
        self.encoder = OobleckEncoder(
            audio_channels=audio_channels,
            channels=decoder_channels,  # Match decoder's internal channel width
            latent_channels=encoder_hidden_size,  # 128 = 64 mean + 64 scale
            channel_multiples=channel_multiples,
            downsampling_ratios=downsampling_ratios,
        )

        # Decoder for latent-to-audio conversion
        self.decoder = OobleckDecoder(
            input_channels=decoder_input_channels,
            channels=decoder_channels,
            audio_channels=audio_channels,
            channel_multiples=channel_multiples,
            downsampling_ratios=downsampling_ratios,
        )

    def encode(self, audio: mx.array, sample: bool = True) -> mx.array:
        """Encode audio to latent representation.

        The encoder outputs a diagonal Gaussian distribution (mean + log_scale).
        By default, this returns the mean (deterministic sampling).

        Args:
            audio: Audio tensor of shape [batch, channels, samples]
                   or [batch, samples, channels]
            sample: If True, return mean (deterministic). If False, return
                    both mean and scale as a tuple.

        Returns:
            Latent tensor of shape [batch, time, latent_dim] (if sample=True)
            or tuple of (mean, scale) tensors (if sample=False)
        """
        # Encode to latent parameters: [batch, encoder_hidden_size, time]
        latent_params = self.encoder(audio)

        # Split into mean and log_scale (each latent_channels/2 = 64 dimensions)
        # Shape: [batch, 128, time] -> [batch, 64, time] + [batch, 64, time]
        mean, log_scale = mx.split(latent_params, 2, axis=1)

        if sample:
            # Return mean (deterministic sampling, equivalent to mode)
            # Transpose to [batch, time, latent_dim] for consistency
            return mean.transpose(0, 2, 1)
        else:
            # Return distribution parameters
            scale = mx.exp(log_scale)
            return mean.transpose(0, 2, 1), scale.transpose(0, 2, 1)

    def decode(self, latents: mx.array) -> mx.array:
        """Decode latents to audio.

        Args:
            latents: Latent tensor of shape [batch, time, latent_dim]
                     or [batch, latent_dim, time]

        Returns:
            Audio tensor of shape [batch, channels, samples]
        """
        return self.decoder(latents)

    @staticmethod
    def sanitize(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Convert PyTorch VAE weights to MLX format."""
        sanitized = {}

        for key, value in weights.items():
            new_key = key

            # Handle weight normalization - weight_v needs transposition
            # PyTorch Conv1d: [out, in, kernel]
            # MLX expects: [out, in, kernel] for weight_v (we transpose in forward)
            if "weight_v" in key and len(value.shape) == 3:
                # Keep as [out, in, kernel] - we'll handle in forward pass
                pass

            # Snake alpha/beta should be [1, channels, 1]
            if "snake" in key and ("alpha" in key or "beta" in key):
                if len(value.shape) == 1:
                    value = value[None, :, None]

            sanitized[new_key] = value

        return sanitized

    def load_weights(self, weights: Dict[str, mx.array], strict: bool = False):
        """Load weights into the model."""
        from mlx.utils import tree_unflatten

        # Sanitize all weights
        sanitized_weights = self.sanitize(weights)

        # Load decoder weights
        decoder_weights = {
            k: v for k, v in sanitized_weights.items() if k.startswith("decoder.")
        }
        if decoder_weights:
            nested = tree_unflatten(list(decoder_weights.items()))
            self.update(nested)

        # Load encoder weights if available
        encoder_weights = {
            k: v for k, v in sanitized_weights.items() if k.startswith("encoder.")
        }
        if encoder_weights:
            nested = tree_unflatten(list(encoder_weights.items()))
            self.update(nested)
