# Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from scipy.signal import get_window


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    """Calculate padding for same output size."""
    return int((kernel_size * dilation - dilation) / 2)


class Conv1dPT(nn.Module):
    """
    Conv1d wrapper that accepts PyTorch format (B, C, T) input.
    Internally transposes to MLX format (B, T, C), applies conv, transposes back.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, C, T) -> (B, T, C)
        x = x.transpose(0, 2, 1)
        x = self.conv(x)
        # (B, T, C) -> (B, C, T)
        return x.transpose(0, 2, 1)


class ConvTranspose1dPT(nn.Module):
    """
    ConvTranspose1d wrapper that accepts PyTorch format (B, C, T) input.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ):
        super().__init__()
        self.conv = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, C, T) -> (B, T, C)
        x = x.transpose(0, 2, 1)
        x = self.conv(x)
        # (B, T, C) -> (B, C, T)
        return x.transpose(0, 2, 1)


class Snake(nn.Module):

    def __init__(
        self,
        in_features: int,
        alpha: float = 1.0,
        alpha_trainable: bool = True,
        alpha_logscale: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.alpha_logscale = alpha_logscale

        if alpha_logscale:
            self.alpha = mx.zeros(in_features) * alpha
        else:
            self.alpha = mx.ones(in_features) * alpha

        self.no_div_by_zero = 1e-9

    def __call__(self, x: mx.array) -> mx.array:

        alpha = mx.reshape(self.alpha, (1, -1, 1))

        if self.alpha_logscale:
            alpha = mx.exp(alpha)

        no_div_by_zero = 1e-9
        min_alpha = 1e-4

        alpha_sign = mx.sign(alpha)
        alpha_abs = mx.abs(alpha)

        alpha_clamped = alpha_sign * mx.maximum(alpha_abs, min_alpha)

        alpha_clamped = mx.where(alpha_abs < no_div_by_zero, min_alpha, alpha_clamped)

        return x + (1.0 / alpha_clamped) * mx.power(mx.sin(x * alpha), 2)


class ResBlock(nn.Module):
    """Residual block with dilated convolutions."""

    def __init__(
        self,
        channels: int = 512,
        kernel_size: int = 3,
        dilations: List[int] = [1, 3, 5],
    ):
        super().__init__()

        self.convs1 = []
        self.convs2 = []
        self.activations1 = []
        self.activations2 = []

        for dilation in dilations:
            padding = get_padding(kernel_size, dilation)
            self.convs1.append(
                Conv1dPT(
                    channels, channels, kernel_size, padding=padding, dilation=dilation
                )
            )
            self.convs2.append(
                Conv1dPT(
                    channels, channels, kernel_size, padding=get_padding(kernel_size, 1)
                )
            )
            self.activations1.append(Snake(channels))
            self.activations2.append(Snake(channels))

    def __call__(self, x: mx.array) -> mx.array:
        for conv1, conv2, act1, act2 in zip(
            self.convs1, self.convs2, self.activations1, self.activations2
        ):
            xt = act1(x)
            xt = conv1(xt)
            xt = act2(xt)
            xt = conv2(xt)
            x = xt + x
        return x


class SineGen(nn.Module):
    """Sine waveform generator for source excitation."""

    def __init__(
        self,
        samp_rate: int,
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

    def __call__(self, f0: mx.array) -> Tuple[mx.array, mx.array, mx.array]:
        """
        Generate sine waveform from F0.

        Args:
            f0: Fundamental frequency (B, 1, T)

        Returns:
            sine_waves: Generated sine waves
            uv: Voiced/unvoiced mask
            noise: Noise component
        """
        B, _, T = f0.shape

        # Create harmonic frequency matrix using broadcasting
        # harmonics: [1, 2, 3, ..., harmonic_num+1]
        harmonics = mx.arange(1, self.harmonic_num + 2)[None, :, None]  # (1, H, 1)
        F_mat = f0 * harmonics / self.sampling_rate  # (B, H, T)

        # Phase accumulation
        theta_mat = 2 * np.pi * mx.cumsum(F_mat, axis=-1)
        theta_mat = theta_mat % (2 * np.pi)

        # Random phase offset (zero for fundamental, random for harmonics)
        if self.harmonic_num > 0:
            random_phases = mx.random.uniform(
                low=-np.pi, high=np.pi, shape=(B, self.harmonic_num, 1)
            )
            zero_phase = mx.zeros((B, 1, 1))
            phase_vec = mx.concatenate([zero_phase, random_phases], axis=1)
        else:
            phase_vec = mx.zeros((B, 1, 1))

        # Generate sine waves
        sine_waves = self.sine_amp * mx.sin(theta_mat + phase_vec)

        # Voiced/unvoiced mask
        uv = (f0 > self.voiced_threshold).astype(mx.float32)

        # Add noise for unvoiced regions
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * mx.random.normal(sine_waves.shape)

        # Combine: voiced regions use sine, unvoiced use noise
        sine_waves = sine_waves * uv + noise

        return sine_waves, uv, noise


class SourceModule(nn.Module):
    """Source module for harmonic-plus-noise synthesis."""

    def __init__(
        self,
        sampling_rate: int,
        upsample_scale: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        add_noise_std: float = 0.003,
        voiced_threshold: float = 10,
    ):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std

        self.l_sin_gen = SineGen(
            sampling_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshold
        )

        # Linear layer to merge harmonics
        self.l_linear = nn.Linear(harmonic_num + 1, 1)

    def __call__(self, f0: mx.array) -> Tuple[mx.array, mx.array, mx.array]:
        """
        Args:
            f0: F0 input (B, T, 1)

        Returns:
            sine_merge: Merged sine source
            noise: Noise source
            uv: Voiced/unvoiced mask
        """
        # Generate sine waves
        sine_wavs, uv, _ = self.l_sin_gen(f0.transpose(0, 2, 1))
        sine_wavs = sine_wavs.transpose(0, 2, 1)  # (B, T, harmonics)
        uv = uv.transpose(0, 2, 1)

        # Merge harmonics
        sine_merge = mx.tanh(self.l_linear(sine_wavs))

        # Generate noise source
        noise = mx.random.normal(uv.shape) * self.sine_amp / 3

        return sine_merge, noise, uv


def elu(x: mx.array, alpha: float = 1.0) -> mx.array:
    """ELU activation: x if x > 0, alpha * (exp(x) - 1) otherwise."""
    return mx.where(x > 0, x, alpha * (mx.exp(x) - 1))


class F0Predictor(nn.Module):
    """
    F0 predictor from mel-spectrogram.
    Matches PyTorch ConvRNNF0Predictor structure with condnet (5 conv layers with ELU) + classifier.

    The classifier outputs values that are made non-negative with abs().
    The model predicts F0 in Hz directly (no scaling needed).
    """

    def __init__(
        self, in_channels: int = 80, hidden_channels: int = 512, num_layers: int = 5
    ):
        super().__init__()

        # condnet is a list of Conv layers (indices 0, 2, 4, 6, 8 in PyTorch Sequential)
        # with ELU activations at indices 1, 3, 5, 7, 9 which we apply inline
        self.condnet = []
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else hidden_channels
            self.condnet.append(
                Conv1dPT(in_ch, hidden_channels, kernel_size=3, padding=1)
            )

        # classifier: final linear layer to predict F0
        self.classifier = nn.Linear(hidden_channels, 1)

    def __call__(self, mel: mx.array) -> mx.array:
        """
        Args:
            mel: Mel-spectrogram (B, 80, T)

        Returns:
            f0: Predicted F0 in Hz (B, T)
        """
        x = mel
        for conv in self.condnet:
            x = conv(x)
            x = elu(x)  # Use ELU activation like PyTorch original

        # x: (B, C, T) -> (B, T, C) for linear
        x = x.transpose(0, 2, 1)
        f0 = self.classifier(x)  # (B, T, 1)
        f0 = f0[:, :, 0]  # (B, T)

        # Apply abs() to ensure non-negative - model outputs F0 in Hz directly
        f0 = mx.abs(f0)

        return f0


class HiFTGenerator(nn.Module):
    """
    HiFTNet Generator: Neural Source Filter + ISTFTNet
    Converts mel-spectrograms to waveforms.
    """

    def __init__(
        self,
        in_channels: int = 80,
        base_channels: int = 512,
        nb_harmonics: int = 8,
        sampling_rate: int = 24000,
        nsf_alpha: float = 0.1,
        nsf_sigma: float = 0.003,
        nsf_voiced_threshold: float = 10,
        upsample_rates: List[int] = [8, 5, 3],
        upsample_kernel_sizes: List[int] = [16, 11, 7],
        resblock_kernel_sizes: List[int] = [3, 7, 11],
        resblock_dilation_sizes: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        source_resblock_kernel_sizes: List[int] = [7, 7, 11],
        source_resblock_dilation_sizes: List[List[int]] = [
            [1, 3, 5],
            [1, 3, 5],
            [1, 3, 5],
        ],
        istft_params: Dict[str, int] = {"n_fft": 16, "hop_len": 4},
        f0_predictor: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.sampling_rate = sampling_rate
        self.istft_params = istft_params
        self.audio_limit = 0.99
        self.num_upsamples = len(upsample_rates)
        self.num_kernels = len(resblock_kernel_sizes)

        # F0 predictor
        self.f0_predictor = f0_predictor if f0_predictor is not None else F0Predictor()

        # Source module
        upsample_scale = int(np.prod(upsample_rates)) * istft_params["hop_len"]
        self.m_source = SourceModule(
            sampling_rate=sampling_rate,
            upsample_scale=upsample_scale,
            harmonic_num=nb_harmonics,
            sine_amp=nsf_alpha,
            add_noise_std=nsf_sigma,
            voiced_threshold=nsf_voiced_threshold,
        )

        # F0 upsampling (using repeat instead of interpolate for MLX)
        self.f0_upsample_scale = upsample_scale

        # Initial conv
        self.conv_pre = Conv1dPT(in_channels, base_channels, kernel_size=7, padding=3)

        # Upsampling layers
        self.ups = []
        ch = base_channels
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                ConvTranspose1dPT(
                    ch // (2**i),
                    ch // (2 ** (i + 1)),
                    kernel_size=k,
                    stride=u,
                    padding=(k - u) // 2,
                )
            )

        # Source downsampling and resblocks
        self.source_downs = []
        self.source_resblocks = []
        downsample_rates = [1] + upsample_rates[::-1][:-1]
        downsample_cum = list(np.cumprod(downsample_rates))

        for i, (u, k, d) in enumerate(
            zip(
                downsample_cum[::-1],
                source_resblock_kernel_sizes,
                source_resblock_dilation_sizes,
            )
        ):
            out_ch = ch // (2 ** (i + 1))
            if u == 1:
                self.source_downs.append(
                    Conv1dPT(istft_params["n_fft"] + 2, out_ch, kernel_size=1)
                )
            else:
                self.source_downs.append(
                    Conv1dPT(
                        istft_params["n_fft"] + 2,
                        out_ch,
                        kernel_size=u * 2,
                        stride=u,
                        padding=u // 2,
                    )
                )
            self.source_resblocks.append(ResBlock(out_ch, k, d))

        # Main resblocks
        self.resblocks = []
        for i in range(len(self.ups)):
            res_ch = ch // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(ResBlock(res_ch, k, d))

        # Final conv
        final_ch = ch // (2 ** len(self.ups))
        self.conv_post = Conv1dPT(
            final_ch, istft_params["n_fft"] + 2, kernel_size=7, padding=3
        )

        # STFT window
        self.stft_window = mx.array(
            get_window("hann", istft_params["n_fft"], fftbins=True).astype(np.float32)
        )

    def _upsample_f0(self, f0: mx.array) -> mx.array:
        """Upsample F0 using repeat."""
        # f0: (B, T) -> (B, T * scale)
        f0 = f0[:, :, None]  # (B, T, 1)
        f0 = mx.repeat(f0, self.f0_upsample_scale, axis=1)  # (B, T*scale, 1)
        return f0

    def _stft(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        """
        Compute STFT using numpy for efficiency.

        Args:
            x: Input waveform (B, T)

        Returns:
            real: Real part of STFT (B, n_fft//2 + 1, T_frames)
            imag: Imaginary part of STFT (B, n_fft//2 + 1, T_frames)
        """
        n_fft = self.istft_params["n_fft"]
        hop_len = self.istft_params["hop_len"]

        # Convert to numpy for efficient STFT
        x_np = np.array(x)
        window_np = np.array(self.stft_window)

        B, T = x_np.shape

        # Number of frames
        n_frames = (T - n_fft) // hop_len + 1

        # Pad if necessary
        if n_frames < 1:
            # Pad input to at least n_fft length
            pad_len = n_fft - T
            x_np = np.pad(x_np, ((0, 0), (0, pad_len)))
            T = x_np.shape[1]
            n_frames = 1

        # Extract frames
        frames = np.zeros((B, n_frames, n_fft), dtype=np.float32)
        for i in range(n_frames):
            start = i * hop_len
            frames[:, i, :] = x_np[:, start : start + n_fft] * window_np

        # Apply FFT to each frame
        spectrum = np.fft.rfft(frames, n=n_fft, axis=-1)

        # Extract real and imaginary parts
        real = spectrum.real.astype(np.float32)
        imag = spectrum.imag.astype(np.float32)

        # Transpose to (B, freq, time)
        real = real.transpose(0, 2, 1)
        imag = imag.transpose(0, 2, 1)

        return mx.array(real), mx.array(imag)

    def _istft(self, magnitude: mx.array, phase: mx.array) -> mx.array:
        """
        Inverse STFT using numpy for efficient overlap-add.
        Matches PyTorch torch.istft with center=True (default).

        Args:
            magnitude: Magnitude spectrogram (B, n_fft//2 + 1, T)
            phase: Phase spectrogram (B, n_fft//2 + 1, T)

        Returns:
            audio: Reconstructed waveform (B, T_audio)
        """
        n_fft = self.istft_params["n_fft"]
        hop_len = self.istft_params["hop_len"]

        # Clamp magnitude for stability
        magnitude = mx.clip(magnitude, a_min=None, a_max=1e2)

        B, _, T = magnitude.shape

        # Reconstruct complex spectrogram
        real = magnitude * mx.cos(phase)
        imag = magnitude * mx.sin(phase)

        # Create complex array: shape (B, F, T) -> need (B, T, F) for irfft
        spec_real = real.transpose(0, 2, 1)  # (B, T, F)
        spec_imag = imag.transpose(0, 2, 1)  # (B, T, F)

        # Perform IRFFT on each frame using MLX
        frames = mx.fft.irfft(spec_real + 1j * spec_imag, n=n_fft, axis=-1)

        # Convert to numpy for efficient overlap-add
        frames_np = np.array(frames)  # (B, T, n_fft)
        window_np = np.array(self.stft_window)

        # Apply synthesis window
        frames_np = frames_np * window_np[None, None, :]

        # Overlap-add synthesis using numpy (much faster than MLX loops)
        output_length = (T - 1) * hop_len + n_fft
        audio_np = np.zeros((B, output_length), dtype=np.float32)
        window_sum = np.zeros(output_length, dtype=np.float32)

        for i in range(T):
            start = i * hop_len
            audio_np[:, start : start + n_fft] += frames_np[:, i, :]
            window_sum[start : start + n_fft] += window_np**2

        # Normalize by window overlap
        window_sum = np.maximum(window_sum, 1e-8)
        audio_np = audio_np / window_sum[None, :]

        # Trim center padding to match PyTorch torch.istft with center=True
        # Output length should be (T - 1) * hop_len for centered STFT
        pad = n_fft // 2
        expected_len = (T - 1) * hop_len
        audio_np = audio_np[:, pad : pad + expected_len]

        return mx.array(audio_np)

    def decode(self, x: mx.array, s: mx.array) -> mx.array:
        """
        Decode mel to waveform.

        Args:
            x: Mel-spectrogram (B, 80, T)
            s: Source signal (B, 1, T_audio)

        Returns:
            audio: Waveform (B, T_audio)
        """
        # Compute source STFT
        s_real, s_imag = self._stft(s[:, 0, :])
        s_stft = mx.concatenate([s_real, s_imag], axis=1)

        # Initial conv
        x = self.conv_pre(x)

        # Upsampling with source fusion
        for i in range(self.num_upsamples):
            x = nn.leaky_relu(x, negative_slope=0.1)
            x = self.ups[i](x)

            # Reflection padding at last upsample
            if i == self.num_upsamples - 1:
                x = mx.pad(x, [(0, 0), (0, 0), (1, 0)])

            # Source fusion
            si = self.source_downs[i](s_stft)
            si = self.source_resblocks[i](si)
            # Match lengths (take minimum to handle off-by-one from convolutions)
            min_len = min(x.shape[2], si.shape[2])
            x = x[:, :, :min_len] + si[:, :, :min_len]

            # ResBlocks
            xs = None
            for j in range(self.num_kernels):
                idx = i * self.num_kernels + j
                if xs is None:
                    xs = self.resblocks[idx](x)
                else:
                    xs = xs + self.resblocks[idx](x)
            x = xs / self.num_kernels

        x = nn.leaky_relu(x)
        x = self.conv_post(x)

        # Split into magnitude and phase
        n_fft = self.istft_params["n_fft"]
        magnitude = mx.exp(x[:, : n_fft // 2 + 1, :])
        phase = mx.sin(x[:, n_fft // 2 + 1 :, :])  # Apply sin() as in PyTorch original

        # ISTFT
        audio = self._istft(magnitude, phase)
        audio = mx.clip(audio, -self.audio_limit, self.audio_limit)

        return audio

    def __call__(self, mel: mx.array) -> Tuple[mx.array, mx.array]:
        """
        Generate waveform from mel-spectrogram.

        Args:
            mel: Mel-spectrogram (B, 80, T)

        Returns:
            audio: Generated waveform (B, T_audio)
            f0: Predicted F0
        """
        # Predict F0
        f0 = self.f0_predictor(mel)

        # Upsample F0
        f0_up = self._upsample_f0(f0)

        # Generate source
        s, _, _ = self.m_source(f0_up)
        s = s.transpose(0, 2, 1)  # (B, 1, T_audio)

        # Decode
        audio = self.decode(mel, s)

        return audio, f0

    def inference(
        self,
        speech_feat: mx.array,
        cache_source: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array]:
        """
        Inference function matching original API.

        Args:
            speech_feat: Mel-spectrogram (B, T, 80) - note: transposed from forward
            cache_source: Cached source for streaming

        Returns:
            audio: Generated waveform
            source: Source signal for caching
        """
        # Transpose to (B, 80, T)
        mel = speech_feat.transpose(0, 2, 1)

        # Predict F0
        f0 = self.f0_predictor(mel)

        # Upsample F0
        f0_up = self._upsample_f0(f0)

        # Generate source
        s, _, _ = self.m_source(f0_up)
        s = s.transpose(0, 2, 1)  # (B, 1, T_audio)

        # Use cache if provided
        if cache_source is not None and cache_source.shape[2] > 0:
            cache_len = cache_source.shape[2]
            s = mx.concatenate([cache_source, s[:, :, cache_len:]], axis=2)

        # Decode
        audio = self.decode(mel, s)

        return audio, s
