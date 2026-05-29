from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np

LRELU_SLOPE = 0.1


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


def _sinc(x: np.ndarray) -> np.ndarray:
    return np.where(x == 0, 1.0, np.sin(math.pi * x) / math.pi / x)


def kaiser_sinc_filter1d(
    cutoff: float, half_width: float, kernel_size: int
) -> mx.array:
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2
    delta_f = 4 * half_width
    amplitude = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if amplitude > 50.0:
        beta = 0.1102 * (amplitude - 8.7)
    elif amplitude >= 21.0:
        beta = 0.5842 * (amplitude - 21) ** 0.4 + 0.07886 * (amplitude - 21.0)
    else:
        beta = 0.0
    window = np.kaiser(kernel_size, beta)
    time = (
        np.arange(-half_size, half_size, dtype=np.float32) + 0.5
        if even
        else np.arange(kernel_size, dtype=np.float32) - half_size
    )
    if cutoff == 0:
        filt = np.zeros_like(time)
    else:
        filt = 2 * cutoff * window * _sinc(2 * cutoff * time)
        filt /= filt.sum()
    return mx.array(filt.reshape(1, 1, kernel_size), dtype=mx.float32)


class Conv1dNCL(nn.Module):
    """NCL-facing wrapper around MLX's NLC Conv1d."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        if in_channels % groups != 0:
            raise ValueError(
                f"The number of input channels ({in_channels}) must be "
                f"divisible by the number of groups ({groups})"
            )
        scale = math.sqrt(1 / (in_channels * kernel_size))
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(out_channels, kernel_size, in_channels // groups),
        )
        if bias:
            self.bias = mx.zeros((out_channels,))
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def __call__(self, x: mx.array) -> mx.array:
        y = mx.conv1d(
            x.transpose(0, 2, 1),
            self.weight,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        if "bias" in self:
            y = y + self.bias
        return y.transpose(0, 2, 1)


class ConvTranspose1dNCL(nn.Module):
    """NCL-facing wrapper around MLX's NLC ConvTranspose1d."""

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
        super().__init__()
        scale = math.sqrt(1 / (in_channels * kernel_size))
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(out_channels, kernel_size, in_channels),
        )
        if bias:
            self.bias = mx.zeros((out_channels,))
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def __call__(self, x: mx.array) -> mx.array:
        y = mx.conv_transpose1d(
            x.transpose(0, 2, 1),
            self.weight,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
        )
        if "bias" in self:
            y = y + self.bias
        return y.transpose(0, 2, 1)


class LowPassFilter1d(nn.Module):
    def __init__(
        self,
        cutoff: float = 0.5,
        half_width: float = 0.6,
        stride: int = 1,
        padding: bool = True,
        padding_mode: str = "edge",
        kernel_size: int = 12,
    ):
        super().__init__()
        if cutoff < 0.0:
            raise ValueError("Minimum cutoff must be larger than zero.")
        if cutoff > 0.5:
            raise ValueError("A cutoff above 0.5 does not make sense.")
        self.kernel_size = kernel_size
        self.even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(self.even)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        self.filter = kaiser_sinc_filter1d(cutoff, half_width, kernel_size)

    def __call__(self, x: mx.array) -> mx.array:
        _, n_channels, _ = x.shape
        if self.padding:
            x = mx.pad(
                x,
                [(0, 0), (0, 0), (self.pad_left, self.pad_right)],
                mode=self.padding_mode,
            )
        weight = mx.broadcast_to(
            self.filter.astype(x.dtype), (n_channels, 1, self.kernel_size)
        ).transpose(0, 2, 1)
        y = mx.conv1d(
            x.transpose(0, 2, 1),
            weight,
            stride=self.stride,
            padding=0,
            groups=n_channels,
        )
        return y.transpose(0, 2, 1)


class UpSample1d(nn.Module):
    def __init__(
        self,
        ratio: int = 2,
        kernel_size: int | None = None,
        persistent: bool = True,
        window_type: str = "kaiser",
    ):
        super().__init__()
        self.persistent = persistent
        self.ratio = ratio
        self.stride = ratio
        if window_type == "hann":
            rolloff = 0.99
            lowpass_filter_width = 6
            width = math.ceil(lowpass_filter_width / rolloff)
            self.kernel_size = 2 * width * ratio + 1
            self.pad = width
            self.pad_left = 2 * width * ratio
            self.pad_right = self.kernel_size - ratio
            time_axis = (np.arange(self.kernel_size) / ratio - width) * rolloff
            time_clamped = np.clip(
                time_axis, -lowpass_filter_width, lowpass_filter_width
            )
            window = np.cos(time_clamped * math.pi / lowpass_filter_width / 2) ** 2
            sinc = np.sinc(time_axis) * window * rolloff / ratio
            filter_data = sinc.reshape(1, 1, -1).astype(np.float32)
        else:
            self.kernel_size = (
                int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
            )
            self.pad = self.kernel_size // ratio - 1
            self.pad_left = (
                self.pad * self.stride + (self.kernel_size - self.stride) // 2
            )
            self.pad_right = (
                self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
            )
            filter_data = np.array(
                kaiser_sinc_filter1d(
                    cutoff=0.5 / ratio,
                    half_width=0.6 / ratio,
                    kernel_size=self.kernel_size,
                )
            )
        if persistent:
            self.filter = mx.array(filter_data, dtype=mx.float32)
        else:
            self._filter_np = filter_data

    def __call__(self, x: mx.array) -> mx.array:
        _, n_channels, _ = x.shape
        x = mx.pad(x, [(0, 0), (0, 0), (self.pad, self.pad)], mode="edge")
        filter_value = (
            self.filter if self.persistent else mx.array(self._filter_np, dtype=x.dtype)
        )
        weight = mx.broadcast_to(
            filter_value.astype(x.dtype), (n_channels, 1, self.kernel_size)
        ).transpose(0, 2, 1)
        y = mx.conv_general(
            x.transpose(0, 2, 1),
            weight,
            stride=1,
            padding=self.kernel_size - 1,
            input_dilation=self.stride,
            groups=n_channels,
        ).transpose(0, 2, 1)
        y = self.ratio * y
        right = -self.pad_right if self.pad_right else None
        return y[..., self.pad_left : right]


class DownSample1d(nn.Module):
    def __init__(self, ratio: int = 2, kernel_size: int | None = None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = (
            int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        )
        self.lowpass = LowPassFilter1d(
            cutoff=0.5 / ratio,
            half_width=0.6 / ratio,
            stride=ratio,
            kernel_size=self.kernel_size,
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.lowpass(x)


class Activation1d(nn.Module):
    def __init__(
        self,
        activation: nn.Module,
        up_ratio: int = 2,
        down_ratio: int = 2,
        up_kernel_size: int = 12,
        down_kernel_size: int = 12,
    ):
        super().__init__()
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    def __call__(self, x: mx.array) -> mx.array:
        target_length = x.shape[-1]
        y = self.downsample(self.act(self.upsample(x)))
        if y.shape[-1] > target_length:
            y = y[..., :target_length]
        elif y.shape[-1] < target_length:
            y = mx.pad(y, [(0, 0), (0, 0), (0, target_length - y.shape[-1])])
        return y


class Snake(nn.Module):
    def __init__(
        self,
        in_features: int,
        alpha: float = 1.0,
        alpha_trainable: bool = True,
        alpha_logscale: bool = True,
    ):
        super().__init__()
        del alpha_trainable
        self.alpha_logscale = alpha_logscale
        self.alpha = (
            mx.zeros((in_features,))
            if alpha_logscale
            else mx.ones((in_features,)) * alpha
        )
        self.eps = 1e-9

    def __call__(self, x: mx.array) -> mx.array:
        alpha = self.alpha.astype(x.dtype)[None, :, None]
        if self.alpha_logscale:
            alpha = mx.exp(alpha)
        return x + (1.0 / (alpha + self.eps)) * mx.square(mx.sin(x * alpha))


class SnakeBeta(nn.Module):
    def __init__(
        self,
        in_features: int,
        alpha: float = 1.0,
        alpha_trainable: bool = True,
        alpha_logscale: bool = True,
    ):
        super().__init__()
        del alpha_trainable
        self.alpha_logscale = alpha_logscale
        self.alpha = (
            mx.zeros((in_features,))
            if alpha_logscale
            else mx.ones((in_features,)) * alpha
        )
        self.beta = (
            mx.zeros((in_features,))
            if alpha_logscale
            else mx.ones((in_features,)) * alpha
        )
        self.eps = 1e-9

    def __call__(self, x: mx.array) -> mx.array:
        alpha = self.alpha.astype(x.dtype)[None, :, None]
        beta = self.beta.astype(x.dtype)[None, :, None]
        if self.alpha_logscale:
            alpha = mx.exp(alpha)
            beta = mx.exp(beta)
        return x + (1.0 / (beta + self.eps)) * mx.square(mx.sin(x * alpha))


class AMPBlock1(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple[int, int, int] = (1, 3, 5),
        activation: str = "snake",
    ):
        super().__init__()
        act_cls = SnakeBeta if activation == "snakebeta" else Snake
        self.convs1 = [
            Conv1dNCL(
                channels,
                channels,
                kernel_size,
                1,
                dilation=d,
                padding=get_padding(kernel_size, d),
            )
            for d in dilation
        ]
        self.convs2 = [
            Conv1dNCL(
                channels,
                channels,
                kernel_size,
                1,
                dilation=1,
                padding=get_padding(kernel_size, 1),
            )
            for _ in self.convs1
        ]
        self.acts1 = [Activation1d(act_cls(channels)) for _ in self.convs1]
        self.acts2 = [Activation1d(act_cls(channels)) for _ in self.convs2]

    def __call__(self, x: mx.array) -> mx.array:
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, self.acts1, self.acts2):
            xt = c1(a1(x))
            xt = c2(a2(xt))
            x = x + xt
        return x


class ResBlock1(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple[int, int, int] = (1, 3, 5),
    ):
        super().__init__()
        self.convs1 = [
            Conv1dNCL(
                channels,
                channels,
                kernel_size,
                1,
                dilation=d,
                padding=get_padding(kernel_size, d),
            )
            for d in dilation
        ]
        self.convs2 = [
            Conv1dNCL(
                channels,
                channels,
                kernel_size,
                1,
                dilation=1,
                padding=get_padding(kernel_size, 1),
            )
            for _ in self.convs1
        ]

    def __call__(self, x: mx.array) -> mx.array:
        for conv1, conv2 in zip(self.convs1, self.convs2):
            xt = nn.leaky_relu(x, LRELU_SLOPE)
            xt = conv1(xt)
            xt = nn.leaky_relu(xt, LRELU_SLOPE)
            xt = conv2(xt)
            x = x + xt
        return x


class Vocoder(nn.Module):
    def __init__(
        self,
        resblock_kernel_sizes: list[int] | None = None,
        upsample_rates: list[int] | None = None,
        upsample_kernel_sizes: list[int] | None = None,
        resblock_dilation_sizes: list[list[int]] | None = None,
        upsample_initial_channel: int = 1024,
        resblock: str = "1",
        output_sampling_rate: int = 24000,
        activation: str = "snake",
        use_tanh_at_final: bool = True,
        apply_final_activation: bool = True,
        use_bias_at_final: bool = True,
        in_channels: int = 128,
        out_channels: int = 2,
    ):
        super().__init__()
        resblock_kernel_sizes = resblock_kernel_sizes or [3, 7, 11]
        upsample_rates = upsample_rates or [6, 5, 2, 2, 2]
        upsample_kernel_sizes = upsample_kernel_sizes or [16, 15, 8, 4, 4]
        resblock_dilation_sizes = resblock_dilation_sizes or [
            [1, 3, 5],
            [1, 3, 5],
            [1, 3, 5],
        ]
        self.output_sampling_rate = output_sampling_rate
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.use_tanh_at_final = use_tanh_at_final
        self.apply_final_activation = apply_final_activation
        self.is_amp = resblock == "AMP1"
        self.conv_pre = Conv1dNCL(
            in_channels,
            upsample_initial_channel,
            kernel_size=7,
            stride=1,
            padding=3,
        )
        resblock_cls = ResBlock1 if resblock == "1" else AMPBlock1
        self.ups = [
            ConvTranspose1dNCL(
                upsample_initial_channel // (2**i),
                upsample_initial_channel // (2 ** (i + 1)),
                kernel_size,
                stride,
                padding=(kernel_size - stride) // 2,
            )
            for i, (stride, kernel_size) in enumerate(
                zip(upsample_rates, upsample_kernel_sizes)
            )
        ]
        final_channels = upsample_initial_channel // (2 ** len(upsample_rates))
        self.resblocks = []
        for i in range(len(upsample_rates)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for kernel_size, dilations in zip(
                resblock_kernel_sizes, resblock_dilation_sizes
            ):
                if self.is_amp:
                    self.resblocks.append(
                        resblock_cls(
                            ch, kernel_size, tuple(dilations), activation=activation
                        )
                    )
                else:
                    self.resblocks.append(
                        resblock_cls(ch, kernel_size, tuple(dilations))
                    )
        self.act_post = Activation1d(SnakeBeta(final_channels)) if self.is_amp else None
        self.conv_post = Conv1dNCL(
            final_channels,
            out_channels,
            kernel_size=7,
            stride=1,
            padding=3,
            bias=use_bias_at_final,
        )

    def __call__(self, x: mx.array) -> mx.array:
        if x.ndim == 4:
            x = x.transpose(0, 1, 3, 2)
            b, s, c, t = x.shape
            x = x.reshape(b, s * c, t)
        elif x.ndim == 3:
            x = x.transpose(0, 2, 1)
        else:
            raise ValueError(f"Expected 3D or 4D mel spectrogram, got {x.shape}")
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            if not self.is_amp:
                x = nn.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            start = i * self.num_kernels
            end = start + self.num_kernels
            block_outputs = mx.stack(
                [self.resblocks[idx](x) for idx in range(start, end)]
            )
            x = mx.mean(block_outputs, axis=0)
        x = self.act_post(x) if self.is_amp else nn.leaky_relu(x, LRELU_SLOPE)
        x = self.conv_post(x)
        if self.apply_final_activation:
            x = mx.tanh(x) if self.use_tanh_at_final else mx.clip(x, -1, 1)
        return x


class _STFTFn(nn.Module):
    def __init__(self, filter_length: int, hop_length: int, win_length: int):
        super().__init__()
        self.hop_length = hop_length
        self.win_length = win_length
        n_freqs = filter_length // 2 + 1
        self.forward_basis = mx.zeros((n_freqs * 2, 1, filter_length))
        self.inverse_basis = mx.zeros((n_freqs * 2, 1, filter_length))

    def __call__(self, y: mx.array) -> tuple[mx.array, mx.array]:
        if y.ndim == 2:
            y = y[:, None, :]
        left_pad = max(0, self.win_length - self.hop_length)
        y = mx.pad(y, [(0, 0), (0, 0), (left_pad, 0)])
        weight = self.forward_basis.astype(y.dtype).transpose(0, 2, 1)
        spec = mx.conv1d(
            y.transpose(0, 2, 1),
            weight,
            stride=self.hop_length,
            padding=0,
        ).transpose(0, 2, 1)
        n_freqs = spec.shape[1] // 2
        real, imag = spec[:, :n_freqs], spec[:, n_freqs:]
        magnitude = mx.sqrt(mx.square(real) + mx.square(imag))
        phase = mx.arctan2(imag.astype(mx.float32), real.astype(mx.float32)).astype(
            real.dtype
        )
        return magnitude, phase


class MelSTFT(nn.Module):
    def __init__(
        self,
        filter_length: int,
        hop_length: int,
        win_length: int,
        n_mel_channels: int,
    ):
        super().__init__()
        self.stft_fn = _STFTFn(filter_length, hop_length, win_length)
        n_freqs = filter_length // 2 + 1
        self.mel_basis = mx.zeros((n_mel_channels, n_freqs))

    def mel_spectrogram(
        self, y: mx.array
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        magnitude, phase = self.stft_fn(y)
        energy = mx.linalg.norm(magnitude, axis=1)
        mel = mx.matmul(self.mel_basis.astype(magnitude.dtype)[None], magnitude)
        log_mel = mx.log(mx.clip(mel, 1e-5, None))
        return log_mel, magnitude, phase, energy


class VocoderWithBWE(nn.Module):
    def __init__(
        self,
        vocoder: Vocoder,
        bwe_generator: Vocoder,
        mel_stft: MelSTFT,
        input_sampling_rate: int,
        output_sampling_rate: int,
        hop_length: int,
    ):
        super().__init__()
        self.vocoder = vocoder
        self.bwe_generator = bwe_generator
        self.mel_stft = mel_stft
        self.input_sampling_rate = input_sampling_rate
        self.output_sampling_rate = output_sampling_rate
        self.hop_length = hop_length
        self.resampler = UpSample1d(
            ratio=output_sampling_rate // input_sampling_rate,
            persistent=False,
            window_type="hann",
        )

    @property
    def conv_pre(self):
        return self.vocoder.conv_pre

    @property
    def conv_post(self):
        return self.vocoder.conv_post

    def _compute_mel(self, audio: mx.array) -> mx.array:
        batch, n_channels, _ = audio.shape
        flat = audio.reshape(batch * n_channels, -1)
        mel, _, _, _ = self.mel_stft.mel_spectrogram(flat)
        return mel.reshape(batch, n_channels, mel.shape[1], mel.shape[2])

    def __call__(self, mel_spec: mx.array) -> mx.array:
        input_dtype = mel_spec.dtype
        x = self.vocoder(mel_spec.astype(mx.float32))
        _, _, length_low_rate = x.shape
        output_length = (
            length_low_rate * self.output_sampling_rate // self.input_sampling_rate
        )
        remainder = length_low_rate % self.hop_length
        if remainder != 0:
            x = mx.pad(x, [(0, 0), (0, 0), (0, self.hop_length - remainder)])
        mel = self._compute_mel(x)
        residual = self.bwe_generator(mel.transpose(0, 1, 3, 2))
        skip = self.resampler(x)
        length = min(residual.shape[-1], skip.shape[-1])
        return mx.clip(residual[..., :length] + skip[..., :length], -1, 1)[
            ..., :output_length
        ].astype(input_dtype)


def build_dramabox_vocoder() -> VocoderWithBWE:
    vocoder = Vocoder(
        resblock_kernel_sizes=[3, 7, 11],
        upsample_rates=[5, 2, 2, 2, 2, 2],
        upsample_kernel_sizes=[11, 4, 4, 4, 4, 4],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_initial_channel=1536,
        resblock="AMP1",
        output_sampling_rate=16000,
        activation="snakebeta",
        use_tanh_at_final=False,
        use_bias_at_final=False,
    )
    bwe_generator = Vocoder(
        resblock_kernel_sizes=[3, 7, 11],
        upsample_rates=[6, 5, 2, 2, 2],
        upsample_kernel_sizes=[12, 11, 4, 4, 4],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_initial_channel=512,
        resblock="AMP1",
        output_sampling_rate=48000,
        activation="snakebeta",
        use_tanh_at_final=False,
        use_bias_at_final=False,
    )
    mel_stft = MelSTFT(
        filter_length=512,
        hop_length=80,
        win_length=512,
        n_mel_channels=64,
    )
    return VocoderWithBWE(
        vocoder=vocoder,
        bwe_generator=bwe_generator,
        mel_stft=mel_stft,
        input_sampling_rate=16000,
        output_sampling_rate=48000,
        hop_length=80,
    )
