from __future__ import annotations

import math
from collections.abc import Mapping

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.dsp import ISTFTCache, hanning, stft

from .config import NemotronVoiceChatCodecConfig


class CausalConv1dCache:
    """Per-layer causal-convolution and spectrogram overlap state."""

    def __init__(self):
        self.cache: dict[int | str, mx.array] = {}

    def update(
        self,
        states: mx.array,
        layer_id: int | str,
        padding: int,
        *,
        padding_value: float = 0.0,
        flush: bool = False,
    ) -> mx.array:
        if padding < 0:
            raise ValueError("padding must be non-negative")
        if padding == 0:
            return states

        previous = self.cache.get(layer_id)
        if previous is None:
            previous = mx.full(
                (states.shape[0], padding, states.shape[2]),
                padding_value,
                dtype=states.dtype,
            )
        elif previous.shape != (states.shape[0], padding, states.shape[2]):
            raise ValueError(
                f"cache entry {layer_id!r} has shape {previous.shape}, expected "
                f"{(states.shape[0], padding, states.shape[2])}"
            )

        padded = mx.concatenate([previous, states], axis=1)
        self.cache[layer_id] = padded[:, -padding:]
        if flush:
            self.cache.pop(layer_id, None)
        return padded

    def clear(self) -> None:
        self.cache.clear()


def _spectrogram(waveform: mx.array, n_fft: int, hop_length: int) -> mx.array:
    """Adapt the shared 1-D STFT to VoiceChat's batched padded layout."""

    if waveform.ndim != 2:
        raise ValueError(
            f"waveform must have shape (batch, samples), got {waveform.shape}"
        )
    if waveform.shape[-1] == 0:
        raise ValueError("waveform must contain at least one sample")

    pad_left = (n_fft - hop_length) // 2
    pad_right = n_fft - hop_length - pad_left
    window = hanning(n_fft, periodic=True).astype(mx.float32)
    spectra = []
    for signal in waveform:
        padded = mx.pad(signal.astype(mx.float32), (pad_left, pad_right))
        if padded.shape[-1] < n_fft:
            padded = mx.pad(padded, (0, n_fft - padded.shape[-1]))
        spectra.append(
            stft(
                padded,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=n_fft,
                window=window,
                center=False,
            ).T
        )
    return mx.stack(spectra)


class ChannelLayerNorm(nn.Module):
    """Layer normalization over channels for ``(batch, time, channels)``."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((channels,))
        self.bias = mx.zeros((channels,))
        self.eps = eps

    def __call__(self, inputs: mx.array) -> mx.array:
        mean = mx.mean(inputs, axis=-1, keepdims=True)
        variance = mx.var(inputs, axis=-1, keepdims=True)
        normalized = (inputs - mean) * mx.rsqrt(variance + self.eps)
        return normalized * self.weight + self.bias


class ConvNeXtBlock1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int, layer_id: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.layer_id = layer_id
        self.dwconv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            groups=channels,
            bias=True,
        )
        self.norm = ChannelLayerNorm(channels)
        self.pwconv1 = nn.Conv1d(channels, 4 * channels, 1, bias=True)
        self.pwconv2 = nn.Conv1d(4 * channels, channels, 1, bias=True)

    def __call__(
        self,
        inputs: mx.array,
        *,
        cache: CausalConv1dCache | None = None,
        flush: bool = False,
    ) -> mx.array:
        residual = inputs
        if cache is None:
            hidden = mx.pad(inputs, ((0, 0), (self.kernel_size - 1, 0), (0, 0)))
        else:
            hidden = cache.update(
                inputs,
                self.layer_id,
                self.kernel_size - 1,
                flush=flush,
            )
        hidden = self.dwconv(hidden)
        hidden = self.norm(hidden)
        hidden = self.pwconv1(hidden)
        hidden = nn.gelu(hidden)
        hidden = self.pwconv2(hidden)
        return residual + hidden


class AudioEncoder(nn.Module):
    def __init__(self, config: NemotronVoiceChatCodecConfig):
        super().__init__()
        channels = [
            config.base_channels * multiplier
            for multiplier in config.channel_multipliers
        ]
        if len(channels) != len(config.downsample_rates):
            raise ValueError(
                "channel_multipliers and downsample_rates must have equal lengths"
            )

        layers: list[nn.Module] = [
            nn.Conv1d(config.stft_channels, channels[0], 1, bias=False)
        ]
        block_index = 0
        for index, (stage_channels, rate) in enumerate(
            zip(channels, config.downsample_rates)
        ):
            for _ in range(config.blocks_per_stage):
                layers.append(
                    ConvNeXtBlock1d(
                        stage_channels,
                        config.block_kernel_size,
                        block_index,
                    )
                )
                block_index += 1
            next_channels = (
                channels[index + 1] if index + 1 < len(channels) else config.latent_dim
            )
            layers.append(
                nn.Conv1d(
                    stage_channels,
                    next_channels,
                    rate,
                    stride=rate,
                    bias=False,
                )
            )
        self.layers = layers

    def __call__(self, inputs: mx.array) -> mx.array:
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs


class AudioDecoder(nn.Module):
    def __init__(self, config: NemotronVoiceChatCodecConfig):
        super().__init__()
        channels = [
            config.base_channels * multiplier
            for multiplier in config.channel_multipliers
        ]
        reversed_channels = list(reversed(channels))
        reversed_rates = list(reversed(config.downsample_rates))

        layers: list[nn.Module] = []
        source_channels = config.latent_dim
        block_index = 0
        for stage_channels, rate in zip(reversed_channels, reversed_rates):
            layers.append(
                nn.ConvTranspose1d(
                    source_channels,
                    stage_channels,
                    rate,
                    stride=rate,
                    bias=False,
                )
            )
            for _ in range(config.blocks_per_stage):
                layers.append(
                    ConvNeXtBlock1d(
                        stage_channels,
                        config.block_kernel_size,
                        block_index,
                    )
                )
                block_index += 1
            source_channels = stage_channels
        layers.append(nn.Conv1d(channels[0], config.stft_channels, 1, bias=False))
        self.layers = layers

    def __call__(
        self,
        inputs: mx.array,
        *,
        cache: CausalConv1dCache | None = None,
        flush: bool = False,
    ) -> mx.array:
        for layer in self.layers:
            if isinstance(layer, ConvNeXtBlock1d):
                inputs = layer(inputs, cache=cache, flush=flush)
            else:
                inputs = layer(inputs)
        return inputs


class ScalarVariance(nn.Module):
    def __init__(self):
        super().__init__()
        self.variance = mx.array(1.0, dtype=mx.float32)


class ProbabilisticResidualVectorQuantizer(nn.Module):
    def __init__(self, config: NemotronVoiceChatCodecConfig):
        super().__init__()
        self.mus_list = [
            mx.zeros((config.codebook_size, config.latent_dim))
            for _ in range(config.num_quantizers)
        ]
        # Variances are training/generative-path parameters. Keeping their exact
        # hierarchy lets the embedded codec strictly load every checkpoint tensor.
        self.variance_list = [ScalarVariance() for _ in range(config.num_quantizers)]

    def encode(self, latents: mx.array) -> mx.array:
        """Quantize ``(B, T, D)`` latents into ``(B, Q, T)`` code IDs."""

        residual = latents
        codes = []
        for means in self.mus_list:
            distances = (
                mx.sum(residual * residual, axis=-1, keepdims=True)
                - 2.0 * (residual @ means.T)
                + mx.sum(means * means, axis=-1)[None, None, :]
            )
            indices = mx.argmin(distances, axis=-1)
            codes.append(indices)
            residual = residual - means[indices]
        return mx.stack(codes, axis=1)

    def decode(self, codes: mx.array) -> mx.array:
        """Decode ``(B, Q, T)`` code IDs into ``(B, T, D)`` latents."""

        if codes.ndim != 3:
            raise ValueError(f"codes must have shape (B, Q, T), got {codes.shape}")
        if codes.shape[1] > len(self.mus_list):
            raise ValueError(
                f"received {codes.shape[1]} quantizers, maximum is {len(self.mus_list)}"
            )
        latents = mx.zeros(
            (codes.shape[0], codes.shape[2], self.mus_list[0].shape[-1]),
            dtype=self.mus_list[0].dtype,
        )
        for index in range(codes.shape[1]):
            latents = latents + self.mus_list[index][codes[:, index, :]]
        return latents


class NemotronVoiceChatCodec(nn.Module):
    """22.05 kHz, 31-codebook codec embedded in NemotronLabs VoiceChat."""

    def __init__(self, config: NemotronVoiceChatCodecConfig | None = None):
        super().__init__()
        self.config = config or NemotronVoiceChatCodecConfig()
        self.encoder = AudioEncoder(self.config)
        self.decoder = AudioDecoder(self.config)
        self.prvq = ProbabilisticResidualVectorQuantizer(self.config)
        self._istft_cache = ISTFTCache()

    @property
    def sample_rate(self) -> int:
        return self.config.sample_rate

    @property
    def frame_rate(self) -> float:
        return self.config.frame_rate

    @property
    def waveform_to_token_ratio(self) -> int:
        return self.config.waveform_to_token_ratio

    def encode_latents(self, waveform: mx.array) -> mx.array:
        if waveform.ndim == 3:
            if waveform.shape[1] != 1:
                raise ValueError("only mono waveforms are supported")
            waveform = waveform[:, 0, :]
        spectrum = _spectrogram(
            waveform,
            self.config.n_fft,
            self.config.hop_length,
        )
        features = mx.concatenate([spectrum.real, spectrum.imag], axis=1)
        return self.encoder(features.transpose(0, 2, 1))

    def decode_latents(
        self,
        latents: mx.array,
        *,
        cache: CausalConv1dCache | None = None,
        flush: bool = False,
    ) -> mx.array:
        features = self.decoder(latents, cache=cache, flush=flush).transpose(0, 2, 1)
        num_bins = self.config.n_fft // 2 + 1
        magnitude_logits = features[:, :num_bins, :]
        phase = features[:, num_bins:, :]

        max_magnitude = 100.0
        magnitude = max_magnitude * mx.exp(
            -nn.softplus(-magnitude_logits + math.log(max_magnitude))
        )
        real = magnitude * mx.cos(phase)
        imag = magnitude * mx.sin(phase)
        # DC and Nyquist bins of an rFFT are real-valued.
        imag = mx.concatenate(
            [
                mx.zeros_like(imag[:, :1]),
                imag[:, 1:-1],
                mx.zeros_like(imag[:, -1:]),
            ],
            axis=1,
        )
        if cache is not None:
            # Four spectrogram frames are sufficient for both sides of the
            # 16-sample Hann window at a hop of four.  Prepending the previous
            # frames and trimming the corresponding waveform context makes each
            # call return exactly one sample-continuous codec chunk.
            half_spec_padding = math.ceil(
                ((self.config.n_fft - self.config.hop_length) // 2)
                / self.config.hop_length
            )
            spec_padding = half_spec_padding * 2
            real = cache.update(
                real.transpose(0, 2, 1),
                "istft_real",
                spec_padding,
                flush=flush,
            ).transpose(0, 2, 1)
            imag = cache.update(
                imag.transpose(0, 2, 1),
                "istft_imag",
                spec_padding,
                flush=flush,
            ).transpose(0, 2, 1)
            if flush:
                tail = mx.zeros(
                    (real.shape[0], real.shape[1], half_spec_padding),
                    dtype=real.dtype,
                )
                real = mx.concatenate([real, tail], axis=2)
                imag = mx.concatenate([imag, tail], axis=2)
        window = hanning(self.config.n_fft, periodic=True).astype(mx.float32)
        waveform = self._istft_cache.istft(
            real,
            imag,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.n_fft,
            window=window,
            center=False,
            constrain_value_range=True,
        )
        pad_left = (self.config.n_fft - self.config.hop_length) // 2
        pad_right = self.config.n_fft - self.config.hop_length - pad_left
        waveform = waveform[:, pad_left : waveform.shape[-1] - pad_right]
        if cache is not None:
            half_wave_padding = half_spec_padding * self.config.hop_length
            waveform = waveform[:, half_wave_padding:-half_wave_padding]
        return waveform[:, None, :]

    def encode(self, waveform: mx.array) -> mx.array:
        return self.prvq.encode(self.encode_latents(waveform))

    def decode(
        self,
        codes: mx.array,
        *,
        cache: CausalConv1dCache | None = None,
        flush: bool = False,
    ) -> mx.array:
        return self.decode_latents(
            self.prvq.decode(codes),
            cache=cache,
            flush=flush,
        )

    def decode_step(
        self,
        codes: mx.array,
        cache: CausalConv1dCache,
        *,
        flush: bool = False,
    ) -> mx.array:
        """Decode newly generated ``(B, Q, T)`` codes without replaying history."""

        if cache is None:
            raise ValueError("decode_step requires a per-stream cache")
        return self.decode(codes, cache=cache, flush=flush)

    def sanitize(
        self, weights: Mapping[str, mx.array], prefix: str = ""
    ) -> dict[str, mx.array]:
        """Convert NeMo convolution layouts while preserving strict key names."""

        normalized_prefix = prefix.rstrip(".")
        if normalized_prefix:
            normalized_prefix += "."

        converted: dict[str, mx.array] = {}
        for source_key, value in weights.items():
            if normalized_prefix and not source_key.startswith(normalized_prefix):
                continue
            key = source_key[len(normalized_prefix) :]
            key = key.replace("prvq._variance_list.", "prvq.variance_list.")
            is_weight = key.endswith(".weight") and value.ndim == 3
            if is_weight and key.startswith("decoder.layers."):
                layer_index = int(key.split(".")[2])
                # Decoder stage starts are ConvTranspose1d; all other 3D decoder
                # weights are ordinary Conv1d kernels.
                stage_width = self.config.blocks_per_stage + 1
                stage_region = stage_width * len(self.config.downsample_rates)
                if (
                    layer_index < stage_region
                    and layer_index % stage_width == 0
                    and "dwconv" not in key
                ):
                    value = value.transpose(1, 2, 0)
                else:
                    value = value.transpose(0, 2, 1)
            elif is_weight:
                value = value.transpose(0, 2, 1)
            converted[key] = value
        return converted


Model = NemotronVoiceChatCodec
