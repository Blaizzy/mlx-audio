import math
from typing import Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn


class FastConvTranspose1d(nn.Module):

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
        # Weight stored as [in_ch, out_ch, K] for efficient matmul
        self.weight = mx.zeros((in_channels, out_channels, kernel_size))
        if bias:
            self.bias = mx.zeros((out_channels,))

    def __call__(self, x: mx.array) -> mx.array:
        # x: [B, L, in_ch] (NLC)
        batch_size, in_len, _ = x.shape

        # [in_ch, out_ch, K] -> [in_ch, out_ch * K]
        w = self.weight.reshape(self.in_channels, self.out_channels * self.kernel_size)

        # [B, L, in_ch] @ [in_ch, out_ch * K] -> [B, L, out_ch * K]
        y = x @ w

        # [B, L, out_ch, K]
        y = y.reshape(batch_size, in_len, self.out_channels, self.kernel_size)

        # [B, out_ch, L, K]
        y = y.transpose(0, 2, 1, 3)

        full_out_len = (in_len - 1) * self.stride + self.kernel_size
        overlap = self.kernel_size - self.stride

        first_part = y[:, :, :, : self.stride]
        first_flat = first_part.reshape(
            batch_size, self.out_channels, in_len * self.stride
        )

        if overlap > 0:
            second_part = y[:, :, :, self.stride :]
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
            second_padded = mx.concatenate(
                [
                    mx.zeros(
                        (batch_size, self.out_channels, self.stride), dtype=x.dtype
                    ),
                    second_flat,
                ],
                axis=2,
            )

            first_padded = first_padded[:, :, :full_out_len]
            second_padded = second_padded[:, :, :full_out_len]
            output = first_padded + second_padded
        else:
            output = first_flat

        if self.padding > 0:
            end_idx = full_out_len - self.padding
            output = output[:, :, self.padding : end_idx]

        if "bias" in self:
            output = output + self.bias[None, :, None]

        # [B, out_ch, L_out] -> [B, L_out, out_ch] (NLC)
        return output.transpose(0, 2, 1)


class Snake1d(nn.Module):

    def __init__(self, channels: int, logscale: bool = True):
        super().__init__()
        self.alpha = mx.zeros(channels)
        self.beta = mx.zeros(channels)
        self.logscale = logscale

    def __call__(self, x: mx.array) -> mx.array:
        alpha = mx.exp(self.alpha) if self.logscale else self.alpha
        beta = mx.exp(self.beta) if self.logscale else self.beta
        return x + mx.reciprocal(beta + 1e-9) * mx.power(mx.sin(alpha * x), 2)


class ResidualUnit(nn.Module):

    def __init__(self, dimension: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.snake1 = Snake1d(dimension)
        self.conv1 = nn.Conv1d(
            dimension, dimension, kernel_size=7, dilation=dilation, padding=pad
        )
        self.snake2 = Snake1d(dimension)
        self.conv2 = nn.Conv1d(dimension, dimension, kernel_size=1)

    def __call__(self, hidden_state: mx.array) -> mx.array:
        output = self.conv1(self.snake1(hidden_state))
        output = self.conv2(self.snake2(output))
        padding = (hidden_state.shape[1] - output.shape[1]) // 2
        if padding > 0:
            hidden_state = hidden_state[:, padding:-padding, :]
        return hidden_state + output


class EncoderBlock(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, stride: int = 1):
        super().__init__()
        self.res_unit1 = ResidualUnit(input_dim, dilation=1)
        self.res_unit2 = ResidualUnit(input_dim, dilation=3)
        self.res_unit3 = ResidualUnit(input_dim, dilation=9)
        self.snake1 = Snake1d(input_dim)
        self.conv1 = nn.Conv1d(
            input_dim,
            output_dim,
            kernel_size=2 * stride,
            stride=stride,
            padding=math.ceil(stride / 2),
        )

    def __call__(self, hidden_state: mx.array) -> mx.array:
        hidden_state = self.res_unit1(hidden_state)
        hidden_state = self.res_unit2(hidden_state)
        hidden_state = self.snake1(self.res_unit3(hidden_state))
        hidden_state = self.conv1(hidden_state)
        return hidden_state


class DecoderBlock(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, stride: int = 1):
        super().__init__()
        self.snake1 = Snake1d(input_dim)
        self.conv_t1 = FastConvTranspose1d(
            input_dim,
            output_dim,
            kernel_size=2 * stride,
            stride=stride,
            padding=stride // 2,
        )
        self.res_unit1 = ResidualUnit(output_dim, dilation=1)
        self.res_unit2 = ResidualUnit(output_dim, dilation=3)
        self.res_unit3 = ResidualUnit(output_dim, dilation=9)

    def __call__(self, hidden_state: mx.array) -> mx.array:
        hidden_state = self.snake1(hidden_state)
        hidden_state = self.conv_t1(hidden_state)
        hidden_state = self.res_unit1(hidden_state)
        hidden_state = self.res_unit2(hidden_state)
        hidden_state = self.res_unit3(hidden_state)
        return hidden_state


class OobleckEncoder(nn.Module):

    def __init__(
        self,
        encoder_hidden_size: int,
        audio_channels: int,
        downsampling_ratios: List[int],
        channel_multiples: List[int],
    ):
        super().__init__()
        cm = [1] + list(channel_multiples)

        self.conv1 = nn.Conv1d(
            audio_channels, encoder_hidden_size, kernel_size=7, padding=3
        )

        self.block = []
        for i, stride in enumerate(downsampling_ratios):
            self.block.append(
                EncoderBlock(
                    input_dim=encoder_hidden_size * cm[i],
                    output_dim=encoder_hidden_size * cm[i + 1],
                    stride=stride,
                )
            )

        d_model = encoder_hidden_size * cm[-1]
        self.snake1 = Snake1d(d_model)
        self.conv2 = nn.Conv1d(d_model, encoder_hidden_size, kernel_size=3, padding=1)

    def __call__(self, hidden_state: mx.array) -> mx.array:
        hidden_state = self.conv1(hidden_state)
        for module in self.block:
            hidden_state = module(hidden_state)
        hidden_state = self.snake1(hidden_state)
        hidden_state = self.conv2(hidden_state)
        return hidden_state


class OobleckDecoder(nn.Module):

    def __init__(
        self,
        channels: int,
        input_channels: int,
        audio_channels: int,
        upsampling_ratios: List[int],
        channel_multiples: List[int],
    ):
        super().__init__()
        strides = upsampling_ratios
        cm = [1] + list(channel_multiples)

        self.conv1 = nn.Conv1d(
            input_channels, channels * cm[-1], kernel_size=7, padding=3
        )

        self.block = []
        for i, stride in enumerate(strides):
            self.block.append(
                DecoderBlock(
                    input_dim=channels * cm[len(strides) - i],
                    output_dim=channels * cm[len(strides) - i - 1],
                    stride=stride,
                )
            )

        self.snake1 = Snake1d(channels)
        self.conv2 = nn.Conv1d(
            channels, audio_channels, kernel_size=7, padding=3, bias=False
        )

    def __call__(self, hidden_state: mx.array) -> mx.array:
        hidden_state = self.conv1(hidden_state)
        for layer in self.block:
            hidden_state = layer(hidden_state)
        hidden_state = self.snake1(hidden_state)
        hidden_state = self.conv2(hidden_state)
        return hidden_state


class AutoencoderOobleck(nn.Module):

    def __init__(
        self,
        audio_channels: int = 2,
        channel_multiples: Optional[List[int]] = None,
        decoder_channels: int = 128,
        decoder_input_channels: int = 64,
        downsampling_ratios: Optional[List[int]] = None,
        encoder_hidden_size: int = 128,
        sampling_rate: int = 48000,
    ):
        super().__init__()
        if downsampling_ratios is None:
            downsampling_ratios = [2, 4, 4, 6, 10]
        if channel_multiples is None:
            channel_multiples = [1, 2, 4, 8, 16]

        self.sampling_rate = sampling_rate
        self.hop_length = math.prod(downsampling_ratios)
        self.audio_channels = audio_channels
        self.latent_channels = decoder_input_channels
        self.encoder_hidden_size = encoder_hidden_size

        self.encoder = OobleckEncoder(
            encoder_hidden_size=encoder_hidden_size,
            audio_channels=audio_channels,
            downsampling_ratios=downsampling_ratios,
            channel_multiples=channel_multiples,
        )
        self.decoder = OobleckDecoder(
            channels=decoder_channels,
            input_channels=decoder_input_channels,
            audio_channels=audio_channels,
            upsampling_ratios=downsampling_ratios[::-1],
            channel_multiples=channel_multiples,
        )

    def encode(self, audio: mx.array, sample: bool = True) -> mx.array:
        # audio: [B, channels, samples] -> NLC [B, samples, channels]
        if audio.shape[1] == self.audio_channels and audio.shape[-1] != self.audio_channels:
            audio = audio.transpose(0, 2, 1)

        h = self.encoder(audio)
        mean, _scale = mx.split(h, 2, axis=-1)

        if sample:
            return mean
        scale = mx.exp(_scale)
        return mean, scale

    def decode(self, latents: mx.array) -> mx.array:
        # latents: [B, T, latent_dim] NLC from ace_step.py
        audio_nlc = self.decoder(latents)
        # -> [B, channels, samples] for ace_step.py
        return audio_nlc.transpose(0, 2, 1)

    @staticmethod
    def _fuse_weight_norm(g: mx.array, v: mx.array) -> mx.array:
        v_flat = v.reshape(v.shape[0], -1)
        norm = mx.sqrt(mx.sum(v_flat * v_flat, axis=1, keepdims=True)).reshape(g.shape)
        return g * v / (norm + 1e-9)

    @staticmethod
    def sanitize(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        sanitized = {}
        processed = set()
        all_keys = sorted(weights.keys())

        for key in all_keys:
            if key in processed:
                continue

            if key.endswith(".weight_g"):
                base = key[: -len(".weight_g")]
                v_key = base + ".weight_v"
                if v_key not in weights:
                    processed.add(key)
                    continue

                w = AutoencoderOobleck._fuse_weight_norm(weights[key], weights[v_key])

                if "conv_t1" in base:
                    # FastConvTranspose1d stores [in, out, K] — keep as-is
                    pass
                else:
                    # Conv1d: PT [out, in, K] -> MLX [out, K, in]
                    w = w.transpose(0, 2, 1)

                sanitized[base + ".weight"] = w
                processed.add(key)
                processed.add(v_key)
                continue

            if key.endswith(".weight_v"):
                continue

            if key.endswith(".alpha") or key.endswith(".beta"):
                sanitized[key] = weights[key].squeeze()
                processed.add(key)
                continue

            sanitized[key] = weights[key]
            processed.add(key)

        return sanitized

    def load_weights(self, weights: Dict[str, mx.array], strict: bool = False):
        from mlx.utils import tree_unflatten

        sanitized = self.sanitize(weights)

        decoder_weights = {
            k: v for k, v in sanitized.items() if k.startswith("decoder.")
        }
        if decoder_weights:
            self.update(tree_unflatten(list(decoder_weights.items())))

        encoder_weights = {
            k: v for k, v in sanitized.items() if k.startswith("encoder.")
        }
        if encoder_weights:
            self.update(tree_unflatten(list(encoder_weights.items())))
