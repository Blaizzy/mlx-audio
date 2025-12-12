import math
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .config import AudioVAEConfig


class CausalConv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        padding: int = 0,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,  # We handle padding manually
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.pad_val = padding
        self._dilation = dilation
        self._kernel_size = kernel_size

    def __call__(self, x):
        # Causal padding: pad (kernel_size - 1) * dilation on the left
        if self.pad_val > 0:
            # Pad on the left (beginning of time)
            # x is (N, T, C)
            x_pad = mx.pad(x, ((0, 0), (self.pad_val * 2, 0), (0, 0)))
            return super().__call__(x_pad)
        return super().__call__(x)


class CausalTransposeConv1d(nn.ConvTranspose1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        bias: bool = True,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,  # We handle padding manually
            bias=bias,
        )
        self.pad_val = padding
        self.output_padding = output_padding

    def __call__(self, x):
        # PyTorch: return super().forward(x)[..., : -(self.__padding * 2 - self.__output_padding)]
        # This trims from the right.

        # MLX ConvTranspose1d output shape calculation is standard.
        # We need to reproduce the slicing behavior.
        y = super().__call__(x)

        trim = self.pad_val * 2 - self.output_padding
        if trim > 0:
            y = y[:, :-trim, :]
        return y


class Snake1d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.alpha = mx.ones((1, 1, channels))  # (1, 1, C) to broadcast over N and T

    def __call__(self, x):
        # x: (N, T, C)
        # alpha: (1, 1, C)
        shape = x.shape
        x = x + (1.0 / (self.alpha + 1e-9)) * mx.sin(self.alpha * x) ** 2
        return x


class CausalResidualUnit(nn.Module):
    def __init__(
        self, dim: int = 16, dilation: int = 1, kernel: int = 7, groups: int = 1
    ):
        super().__init__()
        # pad = ((7 - 1) * dilation) // 2
        pad = ((kernel - 1) * dilation) // 2

        self.snake1 = Snake1d(dim)
        self.conv1 = CausalConv1d(
            dim, dim, kernel_size=kernel, dilation=dilation, padding=pad, groups=groups
        )
        self.snake2 = Snake1d(dim)
        self.conv2 = CausalConv1d(dim, dim, kernel_size=1)

    def __call__(self, x):
        res = x
        x = self.snake1(x)
        x = self.conv1(x)
        x = self.snake2(x)
        x = self.conv2(x)

        # Verification of padding logic from PyTorch:
        # pad = (x.shape[-1] - y.shape[-1]) // 2  <-- this line in PyTorch checks something specific?
        # Actually in PyTorch `forward`:
        # y = self.block(x)
        # pad = (x.shape[-1] - y.shape[-1]) // 2
        # assert pad == 0

        # Since we implemented CausalConv1d to preserve length (mostly), we should be fine.
        # The PyTorch CausalConv1d pads by pad*2 on left, then valid convolution?
        # Wait, PyTorch Conv1d defaults validation is 'valid' if padding=0? No, default is 0 padding.
        # If we supply padding in init, it's symmetric. But CausalConv1d uses explicit F.pad.
        # So the inner Conv1d sees padded input.

        return res + x


class CausalEncoderBlock(nn.Module):
    def __init__(
        self,
        output_dim: int = 16,
        input_dim: int = None,
        stride: int = 1,
        groups: int = 1,
    ):
        super().__init__()
        input_dim = input_dim or output_dim // 2

        self.res1 = CausalResidualUnit(input_dim, dilation=1, groups=groups)
        self.res2 = CausalResidualUnit(input_dim, dilation=3, groups=groups)
        self.res3 = CausalResidualUnit(input_dim, dilation=9, groups=groups)
        self.snake = Snake1d(input_dim)
        self.conv = CausalConv1d(
            input_dim,
            output_dim,
            kernel_size=2 * stride,
            stride=stride,
            padding=math.ceil(stride / 2),
        )

    def __call__(self, x):
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.snake(x)
        x = self.conv(x)
        return x


class CausalEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        latent_dim: int = 32,
        strides: List[int] = [2, 4, 8, 8],
        depthwise: bool = False,
    ):
        super().__init__()

        self.conv_in = CausalConv1d(1, d_model, kernel_size=7, padding=3)

        self.blocks = []
        curr_dim = d_model
        for stride in strides:
            next_dim = curr_dim * 2
            groups = next_dim // 2 if depthwise else 1
            self.blocks.append(
                CausalEncoderBlock(
                    output_dim=next_dim,
                    input_dim=curr_dim,
                    stride=stride,
                    groups=groups,
                )
            )
            curr_dim = next_dim

        self.blocks = nn.Sequential(*self.blocks)  # Just a list of blocks

        groups = curr_dim if depthwise else 1

        self.fc_mu = CausalConv1d(curr_dim, latent_dim, kernel_size=3, padding=1)
        # self.fc_logvar = CausalConv1d(curr_dim, latent_dim, kernel_size=3, padding=1) # Not needed for inference usually

    def __call__(self, x):
        x = self.conv_in(x)
        for block in self.blocks.layers:  # MLX Sequential usually has layers
            x = block(x)
        mu = self.fc_mu(x)
        # logvar = self.fc_logvar(x)
        return mu


class NoiseBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear = CausalConv1d(dim, dim, kernel_size=1, bias=False)

    def __call__(self, x):
        # We probably don't need noise during inference?
        # PyTorch implementation adds noise * h.
        # But this is likely for VAE training.
        # Check standard usage: standard VAE separates randomness to z sampling.
        # But here it's added to features.
        # If seed is fixed, it's deterministic.
        # For pure reconstruction, maybe we can skip?
        # But existing weights expect it.
        # Wait, if I am doing inference, should I add random noise?
        # Usually no.
        # The reference `use_noise_block` defaults to False in Config.
        # And in `AudioVAEConfig` it defaults to False.
        # So I might not encounter it.
        # I'll implement it just in case.
        B, T, C = x.shape
        noise = mx.random.normal((B, T, 1))  # Incorrect shape in my head?
        # PyTorch: (B, C, T). noise: (B, 1, T).
        # MLX: (B, T, C). noise: (B, T, 1)?
        noise = mx.random.normal((B, T, 1)).astype(x.dtype)
        h = self.linear(x)
        n = noise * h
        return x + n


class CausalDecoderBlock(nn.Module):
    def __init__(
        self,
        input_dim: int = 16,
        output_dim: int = 8,
        stride: int = 1,
        groups: int = 1,
        use_noise_block: bool = False,
    ):
        super().__init__()

        self.snake = Snake1d(input_dim)
        self.conv_t = CausalTransposeConv1d(
            input_dim,
            output_dim,
            kernel_size=2 * stride,
            stride=stride,
            padding=math.ceil(stride / 2),
            output_padding=stride % 2,
        )

        self.noise = NoiseBlock(output_dim) if use_noise_block else None

        self.res1 = CausalResidualUnit(output_dim, dilation=1, groups=groups)
        self.res2 = CausalResidualUnit(output_dim, dilation=3, groups=groups)
        self.res3 = CausalResidualUnit(output_dim, dilation=9, groups=groups)

    def __call__(self, x):
        x = self.snake(x)
        x = self.conv_t(x)
        if self.noise:
            x = self.noise(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        return x


class CausalDecoder(nn.Module):
    def __init__(
        self,
        input_channel: int,
        channels: int,
        rates: List[int],
        depthwise: bool = False,
        d_out: int = 1,
        use_noise_block: bool = False,
    ):
        super().__init__()

        if depthwise:
            self.conv_in = nn.Sequential(
                CausalConv1d(
                    input_channel,
                    input_channel,
                    kernel_size=7,
                    padding=3,
                    groups=input_channel,
                ),
                CausalConv1d(input_channel, channels, kernel_size=1),
            )
        else:
            self.conv_in = CausalConv1d(
                input_channel, channels, kernel_size=7, padding=3
            )

        self.blocks = []
        for i, stride in enumerate(rates):
            input_dim = channels // (2**i)
            output_dim = channels // (2 ** (i + 1))
            groups = output_dim if depthwise else 1
            self.blocks.append(
                CausalDecoderBlock(
                    input_dim, output_dim, stride, groups, use_noise_block
                )
            )

        self.blocks = nn.Sequential(*self.blocks)
        final_dim = channels // (
            2 ** len(rates)
        )  # This matches the last output_dim in loop

        self.snake_out = Snake1d(final_dim)
        self.conv_out = CausalConv1d(final_dim, d_out, kernel_size=7, padding=3)

    def __call__(self, x):
        x = self.conv_in(x)
        for block in self.blocks.layers:
            x = block(x)
        x = self.snake_out(x)
        x = self.conv_out(x)
        return mx.tanh(x)


class AudioVAE(nn.Module):
    def __init__(self, config: AudioVAEConfig):
        super().__init__()
        self.config = config

        encoder_dim = config.encoder_dim
        encoder_rates = config.encoder_rates
        latent_dim = config.latent_dim
        decoder_dim = config.decoder_dim
        decoder_rates = config.decoder_rates
        self.hop_length = np.prod(encoder_rates)

        self.decoder_rates = decoder_rates
        self.encoder = CausalEncoder(
            encoder_dim, latent_dim, encoder_rates, depthwise=True
        )
        self.decoder = CausalDecoder(
            latent_dim,
            decoder_dim,
            decoder_rates,
            depthwise=True,
            d_out=1,
            use_noise_block=False,
        )
        self.sample_rate = config.sample_rate

    def encode(self, x, sample_rate: Optional[int] = None):
        if x.ndim == 2:
            x = x[:, :, None]  # Add channel dim
        if (
            x.shape[1] < x.shape[2]
        ):  # If channels < sequence length, it's in PyTorch format
            x = x.transpose(0, 2, 1)  # (N, C, T) -> (N, T, C)
        x = self.preprocess(x, sample_rate)
        z = self.encoder(x)
        return z

    def decode(self, z):
        # z: (N, T, C)
        out = self.decoder(z)
        return out.squeeze(-1)  # (N, T)

    def preprocess(self, audio_data, sample_rate):
        if sample_rate is None:
            sample_rate = self.sample_rate
        assert sample_rate == self.sample_rate
        pad_to = self.hop_length
        length = audio_data.shape[-1]
        right_pad = math.ceil(length / pad_to) * pad_to - length
        audio_data = mx.pad(audio_data, ((0, 0), (0, right_pad), (0, 0)))

        return audio_data

    def sanitize(self, weights):

        # 0. Filter out fc_logvar immediately (not used in inference, only fc_mu)
        weights = {k: v for k, v in weights.items() if "fc_logvar" not in k}

        # 1. Fuse weight_norm
        fused_weights = {}
        keys = list(weights.keys())
        processed_keys = set()

        for k in keys:
            if k in processed_keys:
                continue

            if k.endswith(".weight_g"):
                base = k[:-9]
                v_key = base + ".weight_v"
                if v_key in weights:
                    g = weights[k]
                    v = weights[v_key]
                    v_flat = v.reshape(v.shape[0], -1)
                    norm = mx.linalg.norm(v_flat, axis=1).reshape(g.shape)
                    w = g * (v / (norm + 1e-9))
                    fused_weights[base + ".weight"] = w
                    processed_keys.add(k)
                    processed_keys.add(v_key)
                    continue
            if k.endswith(".weight_v"):
                continue

            fused_weights[k] = weights[k]

        # 2. Remap keys
        remapped_weights = {}

        num_dec_blocks = len(self.decoder_rates)

        for k, v in fused_weights.items():
            parts = k.split(".")
            new_parts = []

            # Encoder
            if parts[0] == "encoder":
                if parts[1] == "block":
                    idx = int(parts[2])
                    if idx == 0:
                        new_parts = ["encoder", "conv_in"] + parts[3:]
                    else:
                        new_parts = [
                            "encoder",
                            "blocks",
                            "layers",
                            str(idx - 1),
                        ] + parts[3:]
                else:
                    new_parts = parts  # e.g. fc_mu

            # Decoder
            elif parts[0] == "decoder":
                if parts[1] == "model":
                    idx = int(parts[2])
                    # Assuming depthwise=True (default)
                    if idx == 0:
                        new_parts = ["decoder", "conv_in", "layers", "0"] + parts[3:]
                    elif idx == 1:
                        new_parts = ["decoder", "conv_in", "layers", "1"] + parts[3:]
                    elif 2 <= idx < 2 + num_dec_blocks:
                        new_parts = [
                            "decoder",
                            "blocks",
                            "layers",
                            str(idx - 2),
                        ] + parts[3:]
                    elif idx == 2 + num_dec_blocks:
                        new_parts = ["decoder", "snake_out"] + parts[3:]
                    elif idx == 2 + num_dec_blocks + 1:
                        new_parts = ["decoder", "conv_out"] + parts[3:]
                    else:
                        new_parts = parts  # Shouldn't happen
                else:
                    new_parts = parts

            else:
                new_parts = parts

            # Internal Block Remapping
            # Pattern: ... blocks.layers.X.block.Y ... -> ... blocks.layers.X.component ...
            # We iterate looking for "block" in substitutions

            # Check for CausalResidualUnit internal mapping (snake1 etc)
            # It usually appears inside EncoderBlock/DecoderBlock which ALSO use .block.Y
            # So we have nested structure.
            # E.g. encoder.blocks.layers.0.block.0.block.1.weight
            # Mapped: encoder.blocks.layers.0.res1.conv1.weight

            # Let's process the suffix parts for mapping.
            # We need to handle this iteratively or carefully.

            # Let's start with current new_parts.
            final_parts = []
            i = 0
            while i < len(new_parts):
                p = new_parts[i]

                # Encoder Block Internals
                # Context: we are inside [encoder, blocks, layers, N] or [decoder, blocks, layers, N]
                if (
                    p == "block"
                    and i + 1 < len(new_parts)
                    and new_parts[i + 1].isdigit()
                ):
                    idx = int(new_parts[i + 1])
                    suffix_idx = i + 2

                    # Determine context (Encoder or Decoder block?)
                    # Look at preceding parts.
                    is_encoder_block = (
                        "encoder" in new_parts[:i] and "blocks" in new_parts[:i]
                    )
                    is_decoder_block = (
                        "decoder" in new_parts[:i] and "blocks" in new_parts[:i]
                    )
                    # Also ResidualUnit uses 'block'

                    # We need to distinguish between Block level and Residual level
                    # ResidualUnit is nested INSIDE Block.
                    # PyTorch Block -> .block.0 (Residual) -> .block.1 (Conv)

                    if is_encoder_block and len(final_parts) == 4:
                        # encoder.blocks.layers.N.block.M
                        mapping = {
                            0: "res1",
                            1: "res2",
                            2: "res3",
                            3: "snake",
                            4: "conv",
                        }
                        final_parts.append(mapping.get(idx, f"unknown_{idx}"))
                        i += 2
                        continue

                    if is_decoder_block and len(final_parts) == 4:
                        # decoder.blocks.layers.N.block.M
                        mapping = {
                            0: "snake",
                            1: "conv_t",
                            2: "res1",
                            3: "res2",
                            4: "res3",
                        }
                        final_parts.append(mapping.get(idx, f"unknown_{idx}"))
                        i += 2
                        continue

                    # PyTorch ResidualUnit uses `self.block` Sequential.
                    # M: 0->snake1, 1->conv1, 2->snake2, 3->conv2
                    mapping = {0: "snake1", 1: "conv1", 2: "snake2", 3: "conv2"}
                    if idx in mapping:
                        final_parts.append(mapping[idx])
                        i += 2
                        continue

                final_parts.append(p)
                i += 1

            new_key = ".".join(final_parts)
            remapped_weights[new_key] = v

        # 3. Transpose based on model structure
        from mlx.utils import tree_flatten

        final_weights = {}
        model_params = dict(tree_flatten(self.parameters()))

        for k, w in remapped_weights.items():
            is_conv = "conv" in k and w.ndim == 3
            is_transpose = "conv_t" in k or "upsample" in k
            is_snake_alpha = "snake" in k and "alpha" in k and w.ndim == 3

            # Check if this is a 3D weight that needs transposition by comparing with model shape
            if k in model_params and w.ndim == 3 and model_params[k].ndim == 3:
                expected_shape = model_params[k].shape
                # If shapes don't match but transposed version does, transpose
                if w.shape != expected_shape:
                    # Try different transpose patterns
                    if w.transpose(0, 2, 1).shape == expected_shape:
                        w = w.transpose(0, 2, 1)
                    elif w.transpose(1, 2, 0).shape == expected_shape:
                        w = w.transpose(1, 2, 0)

            final_weights[k] = w

        return final_weights
