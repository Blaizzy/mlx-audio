"""CosyVoice3 HiFT vocoder (mel spectrogram -> waveform).

Uses a harmonic-plus-noise excitation source, causal convolution blocks, and
inverse STFT reconstruction. Non-streaming whole-utterance synthesis is
supported.
"""

import math
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.tts.models.chatterbox.s3gen.hifigan import (
    Snake,
    hann_window_periodic,
    istft,
    stft,
)

from .config import HiFTConfig


def _linear_interpolate_align_false(x: mx.array, new_size: int) -> mx.array:
    """Interpolate along the last axis using pixel-center coordinates."""
    T = x.shape[-1]
    if new_size == T:
        return x
    dst = mx.arange(new_size).astype(x.dtype)
    src = (dst + 0.5) * (T / new_size) - 0.5
    src = mx.clip(src, 0, T - 1)
    idx_low = mx.floor(src).astype(mx.int32)
    idx_high = mx.minimum(idx_low + 1, T - 1)
    w = src - idx_low.astype(x.dtype)
    low = mx.take(x, idx_low, axis=-1)
    high = mx.take(x, idx_high, axis=-1)
    return low + w * (high - low)


# --------------------------------------------------------------------------- #
# causal conv building blocks
# --------------------------------------------------------------------------- #
class CausalConv1d(nn.Module):
    """1-D convolution with one-sided padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        causal_type: str = "left",
    ):
        super().__init__()
        assert causal_type in ("left", "right")
        scale = math.sqrt(1 / (in_channels * kernel_size))
        self.weight = mx.random.uniform(
            low=-scale, high=scale, shape=(out_channels, kernel_size, in_channels)
        )
        self.bias = mx.zeros((out_channels,))
        self.dilation = dilation
        self.causal_type = causal_type
        self.causal_padding = dilation * (kernel_size - 1)

    def __call__(self, x: mx.array) -> mx.array:
        """x: (B, T, C) channel-last."""
        if self.causal_padding:
            widths = (
                [(0, 0), (self.causal_padding, 0), (0, 0)]
                if self.causal_type == "left"
                else [(0, 0), (0, self.causal_padding), (0, 0)]
            )
            x = mx.pad(x, widths)
        y = mx.conv1d(x, self.weight, stride=1, padding=0, dilation=self.dilation)
        return y + self.bias


class CausalConv1dDownSample(nn.Module):
    """Strided convolution with left-context padding."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int):
        super().__init__()
        scale = math.sqrt(1 / (in_channels * kernel_size))
        self.weight = mx.random.uniform(
            low=-scale, high=scale, shape=(out_channels, kernel_size, in_channels)
        )
        self.bias = mx.zeros((out_channels,))
        self.stride = stride
        self.causal_padding = stride - 1

    def __call__(self, x: mx.array) -> mx.array:
        x = mx.pad(x, [(0, 0), (self.causal_padding, 0), (0, 0)])
        y = mx.conv1d(x, self.weight, stride=self.stride, padding=0)
        return y + self.bias


class CausalConv1dUpsample(nn.Module):
    """Nearest-neighbor upsampling followed by causal convolution."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int):
        super().__init__()
        scale = math.sqrt(1 / (in_channels * kernel_size))
        self.weight = mx.random.uniform(
            low=-scale, high=scale, shape=(out_channels, kernel_size, in_channels)
        )
        self.bias = mx.zeros((out_channels,))
        self.stride = stride
        self.causal_padding = kernel_size - 1

    def __call__(self, x: mx.array) -> mx.array:
        """x: (B, T, C) channel-last."""
        x = mx.repeat(x, self.stride, axis=1)  # nearest-neighbor upsample
        x = mx.pad(x, [(0, 0), (self.causal_padding, 0), (0, 0)])
        y = mx.conv1d(x, self.weight, stride=1, padding=0)
        return y + self.bias


class ResBlock(nn.Module):
    """Causal ResBlock: Snake -> CausalConv1d(left) -> Snake -> CausalConv1d(left)."""

    def __init__(self, channels: int, kernel_size: int, dilations: List[int]):
        super().__init__()
        self.convs1 = [
            CausalConv1d(channels, channels, kernel_size, dilation=d, causal_type="left")
            for d in dilations
        ]
        self.convs2 = [
            CausalConv1d(channels, channels, kernel_size, dilation=1, causal_type="left")
            for _ in dilations
        ]
        self.activations1 = [Snake(channels, alpha_logscale=False) for _ in dilations]
        self.activations2 = [Snake(channels, alpha_logscale=False) for _ in dilations]

    def __call__(self, x: mx.array) -> mx.array:
        """x: (B, C, T) channel-first."""
        for i in range(len(self.convs1)):
            xt = self.activations1[i](x)
            xt = mx.swapaxes(xt, 1, 2)
            xt = self.convs1[i](xt)
            xt = mx.swapaxes(xt, 1, 2)
            xt = self.activations2[i](xt)
            xt = mx.swapaxes(xt, 1, 2)
            xt = self.convs2[i](xt)
            xt = mx.swapaxes(xt, 1, 2)
            x = xt + x
        return x


class CausalConvRNNF0Predictor(nn.Module):
    """5-layer causal Conv1d + ELU stack -> Linear classifier.

    condnet[0] uses right-context (kernel 4), condnet[1:] use left-context
    (kernel 3) — matches ``cosyvoice...CausalConvRNNF0Predictor``.
    """

    def __init__(self, num_class: int = 1, in_channels: int = 80, cond_channels: int = 512):
        super().__init__()
        self.condnet = [
            CausalConv1d(in_channels, cond_channels, kernel_size=4, causal_type="right"),
            CausalConv1d(cond_channels, cond_channels, kernel_size=3, causal_type="left"),
            CausalConv1d(cond_channels, cond_channels, kernel_size=3, causal_type="left"),
            CausalConv1d(cond_channels, cond_channels, kernel_size=3, causal_type="left"),
            CausalConv1d(cond_channels, cond_channels, kernel_size=3, causal_type="left"),
        ]
        self.classifier = nn.Linear(cond_channels, num_class)

    def __call__(self, x: mx.array) -> mx.array:
        """x: (B, C, T) -> f0 (B, T)."""
        x = mx.swapaxes(x, 1, 2)  # (B, T, C)
        for conv in self.condnet:
            x = nn.elu(conv(x))
        x = self.classifier(x)
        x = mx.squeeze(x, axis=-1)
        return mx.abs(x)


# --------------------------------------------------------------------------- #
# causal NSF source
# --------------------------------------------------------------------------- #
class CausalSineGen(nn.Module):
    """Causal harmonic excitation generator for the vocoder."""

    def __init__(
        self,
        samp_rate: int,
        upsample_scale: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        noise_std: float = 0.003,
        voiced_threshold: float = 0,
    ):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold
        self.upsample_scale = upsample_scale
        # Random initial phases for harmonic components.
        rand_ini = mx.random.uniform(shape=(1, harmonic_num + 1))
        self.rand_ini = mx.concatenate([mx.zeros((1, 1)), rand_ini[:, 1:]], axis=1)

    def _f02uv(self, f0: mx.array) -> mx.array:
        return (f0 > self.voiced_threshold).astype(mx.float32)

    def __call__(self, f0: mx.array) -> Tuple[mx.array, mx.array, mx.array]:
        """f0: (B, T, 1) -> sine_waves/uv/noise, each (B, T, H+1)/(B, T, 1)."""
        harmonics = mx.arange(1, self.harmonic_num + 2).reshape(1, 1, -1)
        fn = f0 * harmonics  # (B, T, H+1)

        T = fn.shape[1]
        rad_values = (fn / self.sampling_rate) % 1  # (B, T, H+1)
        rad_values = rad_values.at[:, 0, :].add(self.rand_ini)

        T_down = max(1, T // self.upsample_scale)
        rad_t = mx.swapaxes(rad_values, 1, 2)  # (B, H+1, T)
        rad_down_t = _linear_interpolate_align_false(rad_t, T_down)  # (B, H+1, T_down)
        rad_down = mx.swapaxes(rad_down_t, 1, 2)  # (B, T_down, H+1)

        phase_down = mx.cumsum(rad_down, axis=1) * 2 * math.pi  # (B, T_down, H+1)
        phase_down_t = mx.swapaxes(phase_down, 1, 2) * self.upsample_scale  # (B, H+1, T_down)
        phase_t = mx.repeat(phase_down_t, self.upsample_scale, axis=-1)  # nearest upsample

        diff = T - phase_t.shape[-1]
        if diff > 0:
            phase_t = mx.pad(phase_t, [(0, 0), (0, 0), (0, diff)])
        elif diff < 0:
            phase_t = phase_t[:, :, :T]
        phase = mx.swapaxes(phase_t, 1, 2)  # (B, T, H+1)

        sine_waves = mx.sin(phase) * self.sine_amp
        uv = self._f02uv(f0)  # (B, T, 1)
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * mx.random.uniform(
            shape=sine_waves.shape, key=mx.random.key(0)
        )
        sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise


class CausalSourceModuleHnNSF(nn.Module):
    """Merges CausalSineGen harmonics into a single excitation via tanh(linear(.))."""

    def __init__(
        self,
        sampling_rate: int,
        upsample_scale: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        add_noise_std: float = 0.003,
        voiced_threshod: float = 0,
    ):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        self.l_sin_gen = CausalSineGen(
            sampling_rate, upsample_scale, harmonic_num, sine_amp, add_noise_std, voiced_threshod
        )
        self.l_linear = nn.Linear(harmonic_num + 1, 1)

    def __call__(self, x: mx.array) -> Tuple[mx.array, mx.array, mx.array]:
        """x: (B, T, 1) -> sine_merge (B, T, 1), noise (B, T, 1), uv (B, T, 1)."""
        sine_wavs, uv, _ = self.l_sin_gen(x)
        sine_merge = mx.tanh(self.l_linear(sine_wavs))
        noise = mx.random.normal(shape=uv.shape) * self.sine_amp / 3
        return sine_merge, noise, uv


# --------------------------------------------------------------------------- #
# generator
# --------------------------------------------------------------------------- #
class CausalHiFTGenerator(nn.Module):
    """v3 HiFT vocoder: causal NSF sine source + causal ISTFTNet decoder."""

    def __init__(self, config: HiFTConfig):
        super().__init__()
        self.config = config
        self.out_channels = 1
        self.nb_harmonics = config.nb_harmonics
        self.sampling_rate = config.sampling_rate
        self.istft_params = config.istft_params
        self.lrelu_slope = 0.1
        self.audio_limit = 0.99

        self.num_kernels = len(config.resblock_kernel_sizes)
        self.num_upsamples = len(config.upsample_rates)

        upsample_scale = math.prod(config.upsample_rates) * config.istft_params["hop_len"]
        self.f0_upsample_scale = upsample_scale

        self.m_source = CausalSourceModuleHnNSF(
            sampling_rate=config.sampling_rate,
            upsample_scale=upsample_scale,
            harmonic_num=config.nb_harmonics,
            sine_amp=config.nsf_alpha,
            add_noise_std=config.nsf_sigma,
            voiced_threshod=config.nsf_voiced_threshold,
        )

        self.conv_pre_look_right = config.conv_pre_look_right
        self.conv_pre = CausalConv1d(
            config.in_channels,
            config.base_channels,
            config.conv_pre_look_right + 1,
            dilation=1,
            causal_type="right",
        )

        self.ups = [
            CausalConv1dUpsample(
                config.base_channels // (2**i), config.base_channels // (2 ** (i + 1)), k, u
            )
            for i, (u, k) in enumerate(
                zip(config.upsample_rates, config.upsample_kernel_sizes)
            )
        ]

        downsample_rates = [1] + config.upsample_rates[::-1][:-1]
        downsample_cum_rates = []
        cum = 1
        for r in downsample_rates:
            cum *= r
            downsample_cum_rates.append(cum)

        self.source_downs = []
        self.source_resblocks = []
        for i, (u, k, d) in enumerate(
            zip(
                downsample_cum_rates[::-1],
                config.source_resblock_kernel_sizes,
                config.source_resblock_dilation_sizes,
            )
        ):
            ch = config.base_channels // (2 ** (i + 1))
            if u == 1:
                self.source_downs.append(
                    CausalConv1d(
                        config.istft_params["n_fft"] + 2, ch, 1, dilation=1, causal_type="left"
                    )
                )
            else:
                self.source_downs.append(
                    CausalConv1dDownSample(config.istft_params["n_fft"] + 2, ch, u * 2, u)
                )
            self.source_resblocks.append(ResBlock(ch, k, d))

        self.resblocks = []
        for i in range(len(self.ups)):
            ch = config.base_channels // (2 ** (i + 1))
            for k, d in zip(config.resblock_kernel_sizes, config.resblock_dilation_sizes):
                self.resblocks.append(ResBlock(ch, k, d))

        final_ch = config.base_channels // (2 ** len(self.ups))
        self.conv_post = CausalConv1d(
            final_ch, config.istft_params["n_fft"] + 2, 7, dilation=1, causal_type="left"
        )

        self.stft_window = hann_window_periodic(config.istft_params["n_fft"])
        self.f0_predictor = CausalConvRNNF0Predictor(
            in_channels=config.in_channels, cond_channels=config.base_channels
        )

    # ------------------------------------------------------------------ #
    def _f0_upsample(self, f0: mx.array) -> mx.array:
        return mx.repeat(f0, self.f0_upsample_scale, axis=2)

    def _stft(self, x: mx.array) -> tuple:
        return stft(x, self.istft_params["n_fft"], self.istft_params["hop_len"], self.stft_window)

    def _istft(self, magnitude: mx.array, phase: mx.array) -> mx.array:
        return istft(
            magnitude, phase, self.istft_params["n_fft"], self.istft_params["hop_len"], self.stft_window
        )

    def decode(self, x: mx.array, s: mx.array) -> mx.array:
        """x: mel (B, in_channels, T); s: source (B, 1, T_wave). -> waveform (B, T_wave)."""
        s_stft_real, s_stft_imag = self._stft(s.squeeze(1))
        s_stft = mx.concatenate([s_stft_real, s_stft_imag], axis=1)

        x = mx.swapaxes(x, 1, 2)  # (B, T, C)
        x = self.conv_pre(x)
        x = mx.swapaxes(x, 1, 2)  # (B, C, T)

        for i in range(self.num_upsamples):
            x = nn.leaky_relu(x, negative_slope=self.lrelu_slope)
            x = mx.swapaxes(x, 1, 2)
            x = self.ups[i](x)
            x = mx.swapaxes(x, 1, 2)

            if i == self.num_upsamples - 1:
                x = mx.concatenate([x[:, :, 1:2], x], axis=2)  # reflection pad (1, 0)

            si = mx.swapaxes(s_stft, 1, 2)
            si = self.source_downs[i](si)
            si = mx.swapaxes(si, 1, 2)
            si = self.source_resblocks[i](si)
            x = x + si

            start_idx = i * self.num_kernels
            x = mx.mean(
                mx.stack(
                    [self.resblocks[start_idx + j](x) for j in range(self.num_kernels)], axis=0
                ),
                axis=0,
            )

        x = nn.leaky_relu(x)
        x = mx.swapaxes(x, 1, 2)
        x = self.conv_post(x)
        x = mx.swapaxes(x, 1, 2)

        n_fft_half = self.istft_params["n_fft"] // 2 + 1
        magnitude = mx.exp(x[:, :n_fft_half, :])
        phase = mx.sin(x[:, n_fft_half:, :])

        x = self._istft(magnitude, phase)
        x = mx.clip(x, -self.audio_limit, self.audio_limit)
        return x

    def __call__(self, speech_feat: mx.array) -> Tuple[mx.array, mx.array]:
        f0 = self.f0_predictor(speech_feat)  # (B, T)
        s = self._f0_upsample(mx.expand_dims(f0, 1))  # (B, 1, T*scale)
        s = mx.swapaxes(s, 1, 2)  # (B, T*scale, 1)
        s, _, _ = self.m_source(s)
        s = mx.swapaxes(s, 1, 2)  # (B, 1, T*scale)
        generated_speech = self.decode(x=speech_feat, s=s)
        return generated_speech, s

    def inference(
        self, speech_feat: mx.array, cache_source: Optional[mx.array] = None
    ) -> Tuple[mx.array, mx.array]:
        return self(speech_feat)

    # ------------------------------------------------------------------ #
    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        from .convert import convert_hift_weights

        return convert_hift_weights(weights)
