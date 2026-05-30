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
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.pad_val = padding
        self._dilation = dilation
        self._kernel_size = kernel_size

    def __call__(self, x):
        if self.pad_val > 0:
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
            padding=0,
            bias=bias,
        )
        self.pad_val = padding
        self.output_padding = output_padding

    def __call__(self, x):
        y = super().__call__(x)
        trim = self.pad_val * 2 - self.output_padding
        if trim > 0:
            y = y[:, :-trim, :]
        return y


class Snake1d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.alpha = mx.ones((1, 1, channels))

    def __call__(self, x):
        x = x + (1.0 / (self.alpha + 1e-9)) * mx.sin(self.alpha * x) ** 2
        return x


class CausalResidualUnit(nn.Module):
    def __init__(
        self, dim: int = 16, dilation: int = 1, kernel: int = 7, groups: int = 1
    ):
        super().__init__()
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

        self.blocks = nn.Sequential(*self.blocks)

        self.fc_mu = CausalConv1d(curr_dim, latent_dim, kernel_size=3, padding=1)

    def __call__(self, x):
        x = self.conv_in(x)
        for block in self.blocks.layers:
            x = block(x)
        mu = self.fc_mu(x)
        return mu


class NoiseBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear = CausalConv1d(dim, dim, kernel_size=1, bias=False)

    def __call__(self, x):
        B, T, C = x.shape
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
        self.input_channels = input_dim

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


class SampleRateConditionLayer(nn.Module):
    """Applies sample-rate-conditioned scale and bias per decoder block."""

    def __init__(
        self,
        input_dim: int,
        sr_bin_buckets: int = None,
        cond_type: str = "scale_bias",
        cond_dim: int = 128,
        out_layer: bool = False,
    ):
        super().__init__()
        self.cond_type = cond_type

        if cond_type in ("scale_bias", "scale_bias_init"):
            self.scale_embed = nn.Embedding(sr_bin_buckets, input_dim)
            self.bias_embed = nn.Embedding(sr_bin_buckets, input_dim)
        elif cond_type == "add":
            self.cond_embed = nn.Embedding(sr_bin_buckets, input_dim)
        elif cond_type == "concat":
            self.cond_embed = nn.Embedding(sr_bin_buckets, cond_dim)
            out_layer_in_dim = input_dim + cond_dim
        else:
            raise ValueError(f"Invalid cond_type: {cond_type}")

        if out_layer:
            out_layer_in_dim = (
                (input_dim + cond_dim) if cond_type == "concat" else input_dim
            )
            self.out_snake = Snake1d(out_layer_in_dim)
            self.out_conv = CausalConv1d(out_layer_in_dim, input_dim, kernel_size=1)
            self.has_out_layer = True
        else:
            self.has_out_layer = False

    def __call__(self, x, sr_cond):
        # x: (B, T, C), sr_cond: (B,) integer bucket index
        if self.cond_type in ("scale_bias", "scale_bias_init"):
            # scale/bias: (B, C) -> (B, 1, C)
            scale = self.scale_embed(sr_cond)[:, None, :]
            bias = self.bias_embed(sr_cond)[:, None, :]
            x = x * scale + bias
        elif self.cond_type == "add":
            x = x + self.cond_embed(sr_cond)[:, None, :]
        elif self.cond_type == "concat":
            cond = self.cond_embed(sr_cond)[:, None, :]
            cond = mx.broadcast_to(cond, (x.shape[0], x.shape[1], cond.shape[-1]))
            x = mx.concatenate([x, cond], axis=-1)

        if self.has_out_layer:
            x = self.out_snake(x)
            x = self.out_conv(x)

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
        sr_bin_boundaries: Optional[List[int]] = None,
        cond_type: str = "scale_bias",
        cond_dim: int = 128,
        cond_out_layer: bool = False,
    ):
        super().__init__()
        self.sr_bin_boundaries = sr_bin_boundaries

        # First conv layer(s)
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

        # Decoder blocks
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
        final_dim = channels // (2 ** len(rates))

        self.snake_out = Snake1d(final_dim)
        self.conv_out = CausalConv1d(final_dim, d_out, kernel_size=7, padding=3)

        # Sample rate conditioning
        if sr_bin_boundaries is not None:
            self._sr_boundaries = mx.array(sr_bin_boundaries, dtype=mx.int32)
            sr_bin_buckets = len(sr_bin_boundaries) + 1

            # Build conditioning layers parallel to decoder blocks
            # Only CausalDecoderBlocks get conditioning; other layers get None
            self.sr_cond_layers = []
            for block in self.blocks.layers:
                if isinstance(block, CausalDecoderBlock):
                    self.sr_cond_layers.append(
                        SampleRateConditionLayer(
                            input_dim=block.input_channels,
                            sr_bin_buckets=sr_bin_buckets,
                            cond_type=cond_type,
                            cond_dim=cond_dim,
                            out_layer=cond_out_layer,
                        )
                    )
                else:
                    self.sr_cond_layers.append(None)
        else:
            self._sr_boundaries = None
            self.sr_cond_layers = []

    def get_sr_idx(self, sr):
        """Bucket a sample rate into an index using boundaries."""
        if self._sr_boundaries is None:
            return mx.array([0], dtype=mx.int32)
        # Manual bucketize: count how many boundaries sr exceeds
        idx = mx.sum(sr >= self._sr_boundaries).astype(mx.int32)
        return idx.reshape(1)

    def __call__(self, x, sr_cond=None):
        x = self.conv_in(x)

        if self._sr_boundaries is not None and sr_cond is not None:
            sr_idx = self.get_sr_idx(sr_cond)
            for block, cond_layer in zip(self.blocks.layers, self.sr_cond_layers):
                if cond_layer is not None:
                    x = cond_layer(x, sr_idx)
                x = block(x)
        else:
            for block in self.blocks.layers:
                x = block(x)

        x = self.snake_out(x)
        x = self.conv_out(x)
        return mx.tanh(x)


class AudioVAE(nn.Module):
    """AudioVAE V2 with asymmetric encode/decode sample rates and SR conditioning."""

    def __init__(self, config: AudioVAEConfig):
        super().__init__()
        self.config = config

        self.hop_length = np.prod(config.encoder_rates)
        self.latent_dim = config.latent_dim
        self.sample_rate = config.sample_rate
        self.out_sample_rate = config.out_sample_rate
        self.chunk_size = math.prod(config.encoder_rates)
        self.decode_chunk_size = math.prod(config.decoder_rates)

        self.encoder = CausalEncoder(
            config.encoder_dim,
            config.latent_dim,
            config.encoder_rates,
            depthwise=config.depthwise,
        )
        self.decoder = CausalDecoder(
            config.latent_dim,
            config.decoder_dim,
            config.decoder_rates,
            depthwise=config.depthwise,
            d_out=1,
            use_noise_block=config.use_noise_block,
            sr_bin_boundaries=config.sr_bin_boundaries,
            cond_type=config.cond_type,
            cond_dim=config.cond_dim,
            cond_out_layer=config.cond_out_layer,
        )

    def encode(self, x, sample_rate: Optional[int] = None):
        if x.ndim == 2:
            x = x[:, :, None]
        if x.shape[1] < x.shape[2]:
            x = x.transpose(0, 2, 1)
        x = self.preprocess(x, sample_rate)
        z = self.encoder(x)
        return z

    def decode(self, z, sr_cond=None):
        # z: (N, T, C)
        if sr_cond is None:
            sr_cond = mx.array([self.out_sample_rate], dtype=mx.int32)
        out = self.decoder(z, sr_cond=sr_cond)
        return out.squeeze(-1)

    def preprocess(self, audio_data, sample_rate):
        if sample_rate is None:
            sample_rate = self.sample_rate
        assert sample_rate == self.sample_rate
        pad_to = self.hop_length
        length = audio_data.shape[1]
        right_pad = math.ceil(length / pad_to) * pad_to - length
        if right_pad > 0:
            audio_data = mx.pad(audio_data, ((0, 0), (0, right_pad), (0, 0)))
        return audio_data

    def sanitize(self, weights):
        # 0. Filter out fc_logvar (not used in inference)
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
        decoder_rates = self.config.decoder_rates
        num_dec_blocks = len(decoder_rates)

        for k, v in fused_weights.items():
            parts = k.split(".")
            new_parts = []

            # Encoder remapping
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
                    new_parts = parts

            # Decoder remapping
            elif parts[0] == "decoder":
                if parts[1] == "model":
                    idx = int(parts[2])
                    # depthwise=True: idx 0,1 are conv_in layers
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
                        new_parts = parts
                elif parts[1] == "sr_cond_model":
                    # PyTorch sr_cond_model indices match full model ModuleList
                    # With depthwise=True, decoder blocks start at idx 2
                    # Map sr_cond_model.{pytorch_idx} to sr_cond_layers.{pytorch_idx - offset}
                    pt_idx = int(parts[2])
                    offset = 2 if self.config.depthwise else 1
                    mlx_idx = pt_idx - offset
                    new_parts = ["decoder", "sr_cond_layers", str(mlx_idx)] + parts[3:]
                elif parts[1] == "sr_bin_boundaries":
                    # Buffer - store as decoder attribute
                    remapped_weights["decoder._sr_boundaries"] = v
                    continue
                else:
                    new_parts = parts
            else:
                new_parts = parts

            # Sub-block remapping (block.N -> named components)
            final_parts = []
            i = 0
            while i < len(new_parts):
                p = new_parts[i]

                if (
                    p == "block"
                    and i + 1 < len(new_parts)
                    and new_parts[i + 1].isdigit()
                ):
                    idx = int(new_parts[i + 1])

                    is_encoder_block = (
                        "encoder" in new_parts[:i] and "blocks" in new_parts[:i]
                    )
                    is_decoder_block = (
                        "decoder" in new_parts[:i] and "blocks" in new_parts[:i]
                    )

                    if is_encoder_block and len(final_parts) == 4:
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
                        mapping = {
                            0: "snake",
                            1: "conv_t",
                            2: "res1",
                            3: "res2",
                            4: "res3",
                        }
                        # Check for noise block offset
                        final_parts.append(mapping.get(idx, f"unknown_{idx}"))
                        i += 2
                        continue

                    # ResidualUnit inner block: 0->snake1, 1->conv1, 2->snake2, 3->conv2
                    mapping = {0: "snake1", 1: "conv1", 2: "snake2", 3: "conv2"}
                    if idx in mapping:
                        final_parts.append(mapping[idx])
                        i += 2
                        continue

                final_parts.append(p)
                i += 1

            new_key = ".".join(final_parts)
            remapped_weights[new_key] = v

        # 3. Fix shapes by comparing with model parameters
        from mlx.utils import tree_flatten

        final_weights = {}
        model_params = dict(tree_flatten(self.parameters()))

        for k, w in remapped_weights.items():
            if k in model_params and w.ndim == 3 and model_params[k].ndim == 3:
                expected_shape = model_params[k].shape
                if w.shape != expected_shape:
                    if w.transpose(0, 2, 1).shape == expected_shape:
                        w = w.transpose(0, 2, 1)
                    elif w.transpose(1, 2, 0).shape == expected_shape:
                        w = w.transpose(1, 2, 0)

            final_weights[k] = w

        return final_weights
