# Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

import math
from typing import Dict, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .config import AcousticTokenizerConfig


def _as_indices(sample_indices, batch_size: int) -> list:
    """Convert optional sample indices to a python int list."""
    if sample_indices is None:
        return list(range(batch_size))
    if isinstance(sample_indices, mx.array):
        if sample_indices.ndim == 0:
            return [int(sample_indices.item())]
        try:
            return [int(x) for x in sample_indices.tolist()]
        except AttributeError:
            return [int(x.item()) for x in sample_indices]
    if isinstance(sample_indices, (list, tuple)):
        return [int(i) for i in sample_indices]
    return [int(sample_indices)]


def _extra_padding_for_conv1d(
    length: int, kernel_size: int, stride: int, padding_total: int = 0
) -> int:
    """Match upstream SConv1d stride-alignment padding."""
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return int(ideal_length - length)


class ConvRMSNorm(nn.Module):
    """RMSNorm for convolutional features (B, C, T) format."""

    def __init__(self, dim: int, eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = mx.ones((dim,))
        else:
            self.weight = None

    def _norm(self, x: mx.array) -> mx.array:
        return x * mx.rsqrt(mx.mean(x**2, axis=-1, keepdims=True) + self.eps)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, C, T) -> transpose to (B, T, C) for normalization
        x = mx.transpose(x, (0, 2, 1))
        output = self._norm(x.astype(mx.float32)).astype(x.dtype)
        if self.weight is not None:
            output = output * self.weight
        # Transpose back to (B, C, T)
        return mx.transpose(output, (0, 2, 1))


class CausalConv1d(nn.Module):
    """Causal 1D convolution with padding on the left.

    Input/output format: (B, C, T) - batch, channels, time (PyTorch convention)
    MLX Conv1d expects: (B, T, C) - batch, time, channels
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups

        # Calculate padding for causal convolution
        self.padding = (kernel_size - 1) * dilation
        self.context_size = max(0, self.padding - (stride - 1))
        self._layer_id = None

        # Use MLX Conv1d with groups parameter
        # For grouped conv, MLX weight shape is (C_out, K, C_in/groups)
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,  # We handle padding manually
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    @property
    def layer_id(self) -> str:
        if self._layer_id is None:
            self._layer_id = f"sconv1d_{id(self)}"
        return self._layer_id

    def __call__(
        self,
        x: mx.array,
        cache=None,
        sample_indices=None,
        use_cache: bool = False,
        debug: bool = False,
    ) -> mx.array:
        del debug
        if not use_cache or cache is None:
            # x: (B, C, T) - input in PyTorch format
            padding_total = self.context_size
            extra_padding = _extra_padding_for_conv1d(
                int(x.shape[2]),
                self.kernel_size,
                self.stride,
                padding_total,
            )

            # Transpose to MLX format: (B, C, T) -> (B, T, C)
            x = mx.transpose(x, (0, 2, 1))

            # Match upstream causal non-streaming padding:
            # left pad uses context_size, right pad uses stride-alignment extra padding.
            if padding_total > 0 or extra_padding > 0:
                x = mx.pad(x, [(0, 0), (padding_total, extra_padding), (0, 0)])

            # Apply conv - MLX expects (B, T, C)
            x = self.conv(x)

            # Transpose back to PyTorch format: (B, T, C) -> (B, C, T)
            x = mx.transpose(x, (0, 2, 1))
            return x

        # Streaming path: maintain per-sample input context in cache.
        x_t = mx.transpose(x, (0, 2, 1))  # (B, T, C)
        batch_size = int(x_t.shape[0])
        idxs = _as_indices(sample_indices, batch_size)
        if len(idxs) != batch_size:
            raise ValueError("sample_indices must match batch size when use_cache=True.")

        outputs = []
        for i, sample_idx in enumerate(idxs):
            cur_x = x_t[i : i + 1]
            prev_ctx = cache.get_single(self.layer_id, sample_idx)
            if prev_ctx is None:
                prev_ctx = mx.zeros(
                    (1, self.context_size, self.in_channels), dtype=cur_x.dtype
                )
            full_x = (
                mx.concatenate([prev_ctx, cur_x], axis=1)
                if int(prev_ctx.shape[1]) > 0
                else cur_x
            )
            out_i = self.conv(full_x)
            outputs.append(out_i)

            if self.context_size > 0:
                if int(full_x.shape[1]) >= self.context_size:
                    new_ctx = full_x[:, -self.context_size :, :]
                else:
                    new_ctx = full_x
                cache.set_single(self.layer_id, sample_idx, new_ctx)

        y_t = mx.concatenate(outputs, axis=0)  # (B, T', C)
        return mx.transpose(y_t, (0, 2, 1))  # (B, C, T')


class CausalConvTranspose1d(nn.Module):
    """Causal transposed 1D convolution for upsampling.

    Input/output format: (B, C, T) - batch, channels, time (PyTorch convention)
    MLX ConvTranspose1d expects: (B, T, C) - batch, time, channels
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        bias: bool = True,
        trim_right_ratio: float = 1.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.trim_right_ratio = trim_right_ratio

        # Calculate padding
        self.padding_total = kernel_size - stride
        self.context_size = kernel_size - 1
        self._layer_id = None

        # Use MLX ConvTranspose1d
        self.convtr = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            bias=bias,
        )

    @property
    def layer_id(self) -> str:
        if self._layer_id is None:
            self._layer_id = f"sconvtr1d_{id(self)}"
        return self._layer_id

    def __call__(
        self,
        x: mx.array,
        cache=None,
        sample_indices=None,
        use_cache: bool = False,
        debug: bool = False,
    ) -> mx.array:
        del debug
        # Non-streaming path.
        if not use_cache or cache is None:
            # x: (B, C, T) - input in PyTorch format
            # Transpose to MLX format: (B, C, T) -> (B, T, C)
            x = mx.transpose(x, (0, 2, 1))

            # Apply transposed conv
            x = self.convtr(x)

            # Transpose back to PyTorch format: (B, T, C) -> (B, C, T)
            x = mx.transpose(x, (0, 2, 1))

            # Trim padding for causal (on time dimension, now axis 2)
            padding_right = math.ceil(self.padding_total * self.trim_right_ratio)
            padding_left = self.padding_total - padding_right

            if padding_left > 0:
                x = x[:, :, padding_left:]
            if padding_right > 0:
                x = x[:, :, :-padding_right]

            return x

        # Streaming path.
        x_t = mx.transpose(x, (0, 2, 1))  # (B, T, C_in)
        batch_size = int(x_t.shape[0])
        idxs = _as_indices(sample_indices, batch_size)
        if len(idxs) != batch_size:
            raise ValueError("sample_indices must match batch size when use_cache=True.")

        padding_right = math.ceil(self.padding_total * self.trim_right_ratio)
        padding_left = self.padding_total - padding_right

        out_chunks = []
        for i, sample_idx in enumerate(idxs):
            cur_x = x_t[i : i + 1]
            prev_in = cache.get_single(self.layer_id, sample_idx)
            if prev_in is None:
                prev_in = mx.zeros((1, 0, self.in_channels), dtype=cur_x.dtype)

            full_in = (
                mx.concatenate([prev_in, cur_x], axis=1)
                if int(prev_in.shape[1]) > 0
                else cur_x
            )
            full_out = self.convtr(full_in)  # (1, T, C_out)
            full_out = mx.transpose(full_out, (0, 2, 1))  # (1, C_out, T)

            if padding_left > 0:
                full_out = full_out[:, :, padding_left:]
            if padding_right > 0:
                full_out = full_out[:, :, :-padding_right]

            if int(prev_in.shape[1]) > 0:
                expected_new = int(cur_x.shape[1]) * int(self.stride)
                if int(full_out.shape[2]) >= expected_new:
                    out_i = full_out[:, :, -expected_new:]
                else:
                    out_i = full_out
            else:
                out_i = full_out
            out_chunks.append(out_i)

            if self.context_size > 0:
                if int(full_in.shape[1]) > self.context_size:
                    new_cache = full_in[:, -self.context_size :, :]
                else:
                    new_cache = full_in
                cache.set_single(self.layer_id, sample_idx, new_cache)

        max_len = max(int(o.shape[2]) for o in out_chunks) if out_chunks else 0
        if max_len <= 0:
            return mx.zeros((batch_size, self.out_channels, 0), dtype=x.dtype)

        padded = []
        for out in out_chunks:
            pad = max_len - int(out.shape[2])
            if pad > 0:
                out = mx.pad(out, [(0, 0), (0, 0), (0, pad)])
            padded.append(out)
        return mx.concatenate(padded, axis=0)


class DepthwiseConv(nn.Module):
    """Depthwise separable convolution wrapped in a conv module."""

    def __init__(
        self,
        dim: int,
        kernel_size: int = 7,
        causal: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.causal = causal

        # Wrapped in another conv module (to match HF structure: mixer.conv.conv.conv)
        self.conv = CausalConv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            groups=dim,
            bias=bias,
        )

    def __call__(
        self,
        x: mx.array,
        cache=None,
        sample_indices=None,
        use_cache: bool = False,
        debug: bool = False,
    ) -> mx.array:
        return self.conv(
            x,
            cache=cache,
            sample_indices=sample_indices,
            use_cache=use_cache,
            debug=debug,
        )


class Mixer(nn.Module):
    """Mixer module wrapping depthwise conv."""

    def __init__(
        self, dim: int, kernel_size: int = 7, causal: bool = True, bias: bool = True
    ):
        super().__init__()
        self.conv = DepthwiseConv(dim, kernel_size, causal, bias)

    def __call__(
        self,
        x: mx.array,
        cache=None,
        sample_indices=None,
        use_cache: bool = False,
        debug: bool = False,
    ) -> mx.array:
        return self.conv(
            x,
            cache=cache,
            sample_indices=sample_indices,
            use_cache=use_cache,
            debug=debug,
        )


class FeedForward(nn.Module):
    """Feed-forward network with SiLU activation.

    Note: Uses linear1/linear2 naming to match HuggingFace weights.
    """

    def __init__(self, dim: int, mult: float = 4.0, bias: bool = True):
        super().__init__()
        hidden_dim = int(dim * mult)
        self.linear1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.linear2 = nn.Linear(hidden_dim, dim, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.linear1(x)
        x = nn.gelu(x)
        x = self.linear2(x)
        return x


class Block1D(nn.Module):
    """1D convolutional block with depthwise conv and FFN."""

    def __init__(
        self,
        dim: int,
        layernorm: str = "RMSNorm",  # kept for config compatibility
        eps: float = 1e-6,
        causal: bool = True,
        bias: bool = True,
        layer_scale_init_value: float = 1e-6,
    ):
        super().__init__()
        _ = layernorm
        self.dim = dim

        # Normalization
        self.norm = ConvRMSNorm(dim, eps=eps)
        self.ffn_norm = ConvRMSNorm(dim, eps=eps)

        # Mixer (depthwise conv)
        self.mixer = Mixer(dim, kernel_size=7, causal=causal, bias=bias)

        # FFN
        self.ffn = FeedForward(dim, mult=4.0, bias=bias)

        # Layer scale - stored as parameters (gamma, ffn_gamma)
        if layer_scale_init_value > 0:
            self.gamma = mx.ones((dim,)) * layer_scale_init_value
            self.ffn_gamma = mx.ones((dim,)) * layer_scale_init_value
        else:
            self.gamma = None
            self.ffn_gamma = None

    def __call__(
        self,
        x: mx.array,
        cache=None,
        sample_indices=None,
        use_cache: bool = False,
        debug: bool = False,
    ) -> mx.array:
        # x: (B, C, T)

        # Mixer path
        residual = x
        x = self.norm(x)
        x = self.mixer(
            x,
            cache=cache,
            sample_indices=sample_indices,
            use_cache=use_cache,
            debug=debug,
        )
        if self.gamma is not None:
            x = x * mx.expand_dims(self.gamma, axis=(0, 2))
        x = residual + x

        # FFN path
        residual = x
        x = self.ffn_norm(x)
        # Transpose for FFN: (B, C, T) -> (B, T, C)
        x = mx.transpose(x, (0, 2, 1))
        x = self.ffn(x)
        # Transpose back: (B, T, C) -> (B, C, T)
        x = mx.transpose(x, (0, 2, 1))
        if self.ffn_gamma is not None:
            x = x * mx.expand_dims(self.ffn_gamma, axis=(0, 2))
        x = residual + x

        return x


class StemConv(nn.Module):
    """Stem convolution layer wrapped in Sequential structure to match HF."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        bias: bool = True,
    ):
        super().__init__()
        self.conv = CausalConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=bias,
        )

    def __call__(
        self,
        x: mx.array,
        cache=None,
        sample_indices=None,
        use_cache: bool = False,
        debug: bool = False,
    ) -> mx.array:
        return self.conv(
            x,
            cache=cache,
            sample_indices=sample_indices,
            use_cache=use_cache,
            debug=debug,
        )


class UpsampleLayer(nn.Module):
    """Upsample layer with transposed convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        bias: bool = True,
    ):
        super().__init__()
        self.convtr = CausalConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias,
        )

    def __call__(
        self,
        x: mx.array,
        cache=None,
        sample_indices=None,
        use_cache: bool = False,
        debug: bool = False,
    ) -> mx.array:
        return self.convtr(
            x,
            cache=cache,
            sample_indices=sample_indices,
            use_cache=use_cache,
            debug=debug,
        )


class HeadConv(nn.Module):
    """Output head convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        bias: bool = True,
    ):
        super().__init__()
        self.conv = CausalConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=bias,
        )

    def __call__(
        self,
        x: mx.array,
        cache=None,
        sample_indices=None,
        use_cache: bool = False,
        debug: bool = False,
    ) -> mx.array:
        return self.conv(
            x,
            cache=cache,
            sample_indices=sample_indices,
            use_cache=use_cache,
            debug=debug,
        )


class DownsampleLayer(nn.Module):
    """Downsample layer with strided causal convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        bias: bool = True,
    ):
        super().__init__()
        self.conv = CausalConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias,
        )

    def __call__(
        self,
        x: mx.array,
        cache=None,
        sample_indices=None,
        use_cache: bool = False,
        debug: bool = False,
    ) -> mx.array:
        return self.conv(
            x,
            cache=cache,
            sample_indices=sample_indices,
            use_cache=use_cache,
            debug=debug,
        )


class TokenizerEncoderOutput:
    """Encoder output wrapper with optional sampling helper."""

    def __init__(self, mean: mx.array, std=None):
        self.mean = mean
        self.std = std

    def sample(self, dist_type: str = "fix"):
        if dist_type == "fix" and self.std is not None:
            std = self.std
            if not isinstance(std, mx.array):
                std = mx.array(std, dtype=self.mean.dtype)
            x = self.mean + std * mx.random.normal(self.mean.shape, dtype=self.mean.dtype)
            return x, std
        if dist_type == "gaussian" and self.std is not None:
            std_scale = float(self.std) / 0.8
            batch_size = self.mean.shape[0]
            std = mx.random.normal((batch_size,), dtype=self.mean.dtype) * std_scale
            while std.ndim < self.mean.ndim:
                std = mx.expand_dims(std, axis=-1)
            x = self.mean + std * mx.random.normal(self.mean.shape, dtype=self.mean.dtype)
            return x, std
        return self.mean, self.std


class TokenizerStreamingCache:
    """Streaming cache for tokenizer states keyed by (layer_id, sample_idx)."""

    def __init__(self):
        self.cache: Dict[Tuple[str, int], mx.array] = {}

    @staticmethod
    def _indices(sample_indices) -> list:
        if sample_indices is None:
            return []
        if isinstance(sample_indices, mx.array):
            if sample_indices.ndim == 0:
                return [int(sample_indices.item())]
            try:
                return [int(x) for x in sample_indices.tolist()]
            except AttributeError:
                return [int(x.item()) for x in sample_indices]
        if isinstance(sample_indices, (list, tuple)):
            return [int(i) for i in sample_indices]
        return [int(sample_indices)]

    def get_single(self, layer_id: str, sample_idx: int) -> Optional[mx.array]:
        return self.cache.get((layer_id, int(sample_idx)))

    def set_single(self, layer_id: str, sample_idx: int, state: mx.array):
        self.cache[(layer_id, int(sample_idx))] = state

    def set_to_zero(self, sample_indices):
        idxs = set(self._indices(sample_indices))
        for (layer_id, sample_idx), state in list(self.cache.items()):
            if sample_idx in idxs:
                self.cache[(layer_id, sample_idx)] = mx.zeros_like(state)

    def clear(self, layer_id: Optional[str] = None, sample_indices=None):
        if layer_id is None and sample_indices is None:
            self.cache.clear()
            return
        idxs = None if sample_indices is None else set(self._indices(sample_indices))
        for key in list(self.cache.keys()):
            key_layer, key_idx = key
            if layer_id is not None and key_layer != layer_id:
                continue
            if idxs is not None and key_idx not in idxs:
                continue
            self.cache.pop(key, None)


class TokenizerEncoder(nn.Module):
    """Encoder that converts waveform audio to acoustic latents."""

    def __init__(self, config: AcousticTokenizerConfig):
        super().__init__()

        self.dimension = config.vae_dim
        self.channels = config.channels
        self.n_filters = config.encoder_n_filters
        # Match upstream: downsampling uses reversed ratios.
        self.ratios = list(reversed(config.encoder_ratios))
        self.depths = (
            [int(d) for d in config.encoder_depths.split("-")]
            if isinstance(config.encoder_depths, str)
            else list(config.encoder_depths)
        )
        self.n_stages = len(self.depths)

        self.downsample_layers = []
        self.downsample_layers.append(
            [
                StemConv(
                    in_channels=self.channels,
                    out_channels=self.n_filters,
                    kernel_size=7,
                    bias=config.conv_bias,
                )
            ]
        )

        for i in range(len(self.ratios)):
            in_ch = self.n_filters * (2**i)
            out_ch = self.n_filters * (2 ** (i + 1))
            ratio = self.ratios[i]
            self.downsample_layers.append(
                [
                    DownsampleLayer(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        kernel_size=ratio * 2,
                        stride=ratio,
                        bias=config.conv_bias,
                    )
                ]
            )

        self.stages = []
        for i in range(self.n_stages):
            in_ch = self.n_filters * (2**i)
            stage_blocks = []
            for _ in range(self.depths[i]):
                stage_blocks.append(
                    Block1D(
                        dim=in_ch,
                        layernorm=config.layernorm,
                        eps=config.layernorm_eps,
                        causal=config.causal,
                        bias=config.conv_bias,
                        layer_scale_init_value=config.layer_scale_init_value,
                    )
                )
            self.stages.append(stage_blocks)

        self.norm = (
            nn.Identity()
            if config.disable_last_norm
            else ConvRMSNorm(in_ch, eps=config.layernorm_eps)
        )
        self.head = HeadConv(
            in_channels=in_ch,
            out_channels=self.dimension,
            kernel_size=7,
            bias=config.conv_bias,
        )

    def __call__(
        self,
        x: mx.array,
        cache=None,
        sample_indices=None,
        use_cache: bool = False,
        debug: bool = False,
    ) -> mx.array:
        # x: (B, 1, T)
        for i in range(self.n_stages):
            x = self.downsample_layers[i][0](
                x,
                cache=cache,
                sample_indices=sample_indices,
                use_cache=use_cache,
                debug=debug,
            )
            for block in self.stages[i]:
                x = block(
                    x,
                    cache=cache,
                    sample_indices=sample_indices,
                    use_cache=use_cache,
                    debug=debug,
                )

        x = self.norm(x)
        x = self.head(
            x,
            cache=cache,
            sample_indices=sample_indices,
            use_cache=use_cache,
            debug=debug,
        )  # (B, D, T')
        return mx.transpose(x, (0, 2, 1))  # (B, T', D)


class TokenizerDecoder(nn.Module):
    """Decoder that converts latent representations back to audio.

    Architecture matches HuggingFace VibeVoice structure:
    - upsample_layers[0] is stem conv
    - upsample_layers[1-6] are transposed convolutions
    - stages[0-6] are transformer blocks
    - head is output convolution
    """

    def __init__(self, config: AcousticTokenizerConfig):
        super().__init__()

        self.dimension = config.vae_dim
        self.channels = config.channels
        self.n_filters = (
            config.decoder_n_filters
            if config.decoder_n_filters
            else config.encoder_n_filters
        )

        # Use decoder ratios or fallback to encoder ratios
        self.ratios = (
            config.decoder_ratios if config.decoder_ratios else config.encoder_ratios
        )

        # Parse depths - should be reversed encoder depths for decoder
        if config.decoder_depths:
            if isinstance(config.decoder_depths, str):
                self.depths = [int(d) for d in config.decoder_depths.split("-")]
            else:
                self.depths = config.decoder_depths
        else:
            if isinstance(config.encoder_depths, str):
                encoder_depths = [int(d) for d in config.encoder_depths.split("-")]
            else:
                encoder_depths = config.encoder_depths
            self.depths = list(reversed(encoder_depths))

        self.causal = config.causal
        self.n_stages = len(self.depths)

        # Upsample layers - wrapped in list structure to match HF naming
        # HF: upsample_layers.X.0.conv or upsample_layers.X.0.convtr
        self.upsample_layers = []

        # First upsample layer is stem conv (upsample_layers.0.0.conv)
        stem_out_ch = self.n_filters * (2 ** (self.n_stages - 1))
        self.upsample_layers.append(
            [
                StemConv(
                    in_channels=self.dimension,
                    out_channels=stem_out_ch,
                    kernel_size=7,
                    bias=config.conv_bias,
                )
            ]
        )

        # Remaining upsample layers are transposed convolutions
        for i in range(len(self.ratios)):
            in_ch = self.n_filters * (2 ** (self.n_stages - 1 - i))
            out_ch = (
                self.n_filters * (2 ** (self.n_stages - 2 - i))
                if i < len(self.ratios) - 1
                else self.n_filters
            )

            self.upsample_layers.append(
                [
                    UpsampleLayer(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        kernel_size=self.ratios[i] * 2,
                        stride=self.ratios[i],
                        bias=config.conv_bias,
                    )
                ]
            )

        # Transformer stages
        self.stages = []
        for i in range(self.n_stages):
            in_ch = self.n_filters * (2 ** (self.n_stages - 1 - i))
            stage_blocks = []
            for _ in range(self.depths[i]):
                stage_blocks.append(
                    Block1D(
                        dim=in_ch,
                        layernorm=config.layernorm,
                        eps=config.layernorm_eps,
                        causal=config.causal,
                        bias=config.conv_bias,
                        layer_scale_init_value=config.layer_scale_init_value,
                    )
                )
            self.stages.append(stage_blocks)

        # Output head
        self.head = HeadConv(
            in_channels=self.n_filters,
            out_channels=config.channels,
            kernel_size=7,
            bias=config.conv_bias,
        )

    def __call__(
        self,
        x: mx.array,
        cache=None,
        sample_indices=None,
        use_cache: bool = False,
        debug: bool = False,
    ) -> mx.array:
        """
        Args:
            x: Latent tensor of shape (B, T, D) or (B, D, T)

        Returns:
            Audio tensor of shape (B, 1, T')
        """
        # Ensure x is in (B, D, T) format
        if x.shape[1] != self.dimension:
            x = mx.transpose(x, (0, 2, 1))

        # Apply stem (first upsample layer)
        x = self.upsample_layers[0][0](
            x,
            cache=cache,
            sample_indices=sample_indices,
            use_cache=use_cache,
            debug=debug,
        )

        # Process through stages and upsampling
        for i in range(self.n_stages):
            # Apply stage blocks
            for block in self.stages[i]:
                x = block(
                    x,
                    cache=cache,
                    sample_indices=sample_indices,
                    use_cache=use_cache,
                    debug=debug,
                )

            # Apply upsampling (skip first upsample which was stem)
            if i + 1 < len(self.upsample_layers):
                x = self.upsample_layers[i + 1][0](
                    x,
                    cache=cache,
                    sample_indices=sample_indices,
                    use_cache=use_cache,
                    debug=debug,
                )

        # Output head
        x = self.head(
            x,
            cache=cache,
            sample_indices=sample_indices,
            use_cache=use_cache,
            debug=debug,
        )

        return x


class AcousticTokenizer(nn.Module):
    """VibeVoice acoustic tokenizer (decoder only for inference)."""

    def __init__(self, config: AcousticTokenizerConfig):
        super().__init__()
        self.config = config
        self.fix_std = config.fix_std
        self.std_dist_type = config.std_dist_type

        self.encoder = TokenizerEncoder(config)
        self.decoder = TokenizerDecoder(config)

    def encode(self, audio: mx.array) -> TokenizerEncoderOutput:
        """Convert audio to latent representations.

        Args:
            audio: Audio tensor of shape (B, 1, T), (B, T), or (T,)

        Returns:
            TokenizerEncoderOutput with mean latents of shape (B, T', D)
        """
        if audio.ndim == 1:
            audio = audio[None, None, :]
        elif audio.ndim == 2:
            audio = audio[:, None, :]
        latents = self.encoder(audio)
        return TokenizerEncoderOutput(mean=latents, std=self.fix_std)

    def decode(
        self,
        latents: mx.array,
        cache: Optional[TokenizerStreamingCache] = None,
        sample_indices=None,
        use_cache: bool = False,
        debug: bool = False,
    ) -> mx.array:
        """Convert latent representations to audio.

        Args:
            latents: Latent tensor of shape (B, T, D) where D = vae_dim
            cache: Optional streaming cache
            sample_indices: Optional sample indices for cache state
            use_cache: Whether to decode incrementally using cache
            debug: Debug flag

        Returns:
            Audio tensor of shape (B, 1, T')
        """
        return self.decoder(
            latents,
            cache=cache,
            sample_indices=sample_indices,
            use_cache=use_cache,
            debug=debug,
        )

    def __call__(self, latents: mx.array) -> mx.array:
        """Alias for decode."""
        return self.decode(latents)
