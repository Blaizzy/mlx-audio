# Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)


import math
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.dsp import ISTFTCache, hanning


class Snake(nn.Module):
    """
    Snake activation function.

    x + (1/alpha) * sin^2(alpha * x)

    CosyVoice uses alpha_logscale=False, so alpha is stored directly.
    """

    def __init__(self, channels: int, alpha: float = 1.0, alpha_logscale: bool = False):
        super().__init__()
        self.alpha_logscale = alpha_logscale
        # Initialize alpha: zeros if logscale (exp(0)=1), ones otherwise
        if alpha_logscale:
            self.alpha = mx.zeros((channels,)) * alpha
        else:
            self.alpha = mx.ones((channels,)) * alpha

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, T, C) - MLX format
        alpha = self.alpha[None, None, :]
        if self.alpha_logscale:
            alpha = mx.exp(alpha)
        return x + (1.0 / (alpha + 1e-9)) * mx.sin(x * alpha) ** 2


class CausalConv1d(nn.Module):
    """
    Causal 1D convolution with proper padding.

    Matches PyTorch CausalConv1d from CosyVoice:
    - causal_type='left': padding on left (standard causal, no future info)
    - causal_type='right': padding on right (look-ahead)
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
        causal_type: str = "left",
    ):
        super().__init__()
        assert stride == 1, "CausalConv1d only supports stride=1"
        assert causal_type in ["left", "right"]

        self.kernel_size = kernel_size
        self.dilation = dilation
        self.causal_type = causal_type

        # Causal padding formula from CosyVoice:
        # causal_padding = int((kernel_size * dilation - dilation) / 2) * 2 + (kernel_size + 1) % 2
        self.causal_padding = (
            int((kernel_size * dilation - dilation) / 2) * 2 + (kernel_size + 1) % 2
        )

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def __call__(self, x: mx.array) -> mx.array:
        # x is in (B, T, C) format for MLX
        if self.causal_padding > 0:
            if self.causal_type == "left":
                # Standard causal: pad on left
                x = mx.pad(x, [(0, 0), (self.causal_padding, 0), (0, 0)])
            else:
                # Look-ahead: pad on right
                x = mx.pad(x, [(0, 0), (0, self.causal_padding), (0, 0)])
        return self.conv(x)


class CausalConv1dUpsample(nn.Module):
    """
    Causal upsampling layer: Upsample (nearest) + Conv1d.

    This matches the CausalConv1dUpsample from CosyVoice which is:
    1. Nearest neighbor upsampling by scale_factor
    2. Causal Conv1d (left padding)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        bias: bool = True,
    ):
        super().__init__()
        self.stride = stride
        self.causal_padding = kernel_size - 1

        # Regular Conv1d (the upsampling is done via nearest neighbor, not conv transpose)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            bias=bias,
        )

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, T, C)
        B, T, C = x.shape

        # Nearest neighbor upsampling by repeating each timestep
        # This is equivalent to torch.nn.Upsample(scale_factor=stride, mode='nearest')
        x = mx.repeat(x, self.stride, axis=1)  # (B, T*stride, C)

        # Causal (left) padding
        if self.causal_padding > 0:
            x = mx.pad(x, [(0, 0), (self.causal_padding, 0), (0, 0)])

        # Apply conv
        x = self.conv(x)

        return x


class CausalConv1dDownSample(nn.Module):
    """
    Causal downsampling layer using strided convolution with causal (left) padding.

    This matches CausalConv1dDownSample from CosyVoice:
    - causal_padding = stride - 1
    - Left padding before conv
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        bias: bool = True,
    ):
        super().__init__()
        self.stride = stride
        self.causal_padding = stride - 1

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            bias=bias,
        )

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, T, C)
        # Causal (left) padding
        if self.causal_padding > 0:
            x = mx.pad(x, [(0, 0), (self.causal_padding, 0), (0, 0)])

        return self.conv(x)


class ResBlock(nn.Module):
    """Residual block with Snake activation for HiFi-GAN."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: Tuple[int, int, int] = (1, 3, 5),
    ):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size

        # Activations
        self.activations1 = [Snake(channels) for _ in range(3)]
        self.activations2 = [Snake(channels) for _ in range(3)]

        # Convolutions - first set with dilation
        self.convs1 = [
            CausalConv1d(channels, channels, kernel_size, dilation=d) for d in dilation
        ]

        # Convolutions - second set with dilation 1
        self.convs2 = [
            CausalConv1d(channels, channels, kernel_size, dilation=1) for _ in range(3)
        ]

    def __call__(self, x: mx.array) -> mx.array:
        for i in range(3):
            xt = self.activations1[i](x)
            xt = self.convs1[i](xt)
            xt = self.activations2[i](xt)
            xt = self.convs2[i](xt)
            x = x + xt
        return x


class SourceModuleHnNSF(nn.Module):
    """
    Source module for Neural Source Filter vocoder.

    Generates harmonic source signal from F0.
    Matches the original CosyVoice SineGen2 + SourceModuleHnNSF implementation.

    Key difference from naive implementation: uses interpolation-based phase computation
    to avoid numerical precision issues at high sample rates:
    1. Downsample rad_values by 1/upsample_scale
    2. Cumsum at lower rate
    3. Multiply phase by upsample_scale
    4. Interpolate back to original size
    """

    def __init__(
        self,
        sampling_rate: int = 24000,
        upsample_scale: int = 480,
        nb_harmonics: int = 8,
        nsf_alpha: float = 0.1,
        nsf_sigma: float = 0.003,
        nsf_voiced_threshold: float = 10,
    ):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.upsample_scale = upsample_scale
        self.nb_harmonics = nb_harmonics
        self.sine_amp = nsf_alpha  # amplitude of sine waves (default 0.1)
        self.noise_std = nsf_sigma  # noise std for voiced regions
        self.nsf_voiced_threshold = nsf_voiced_threshold
        self.dim = nb_harmonics + 1

        # Linear layer to combine harmonics
        self.l_linear = nn.Linear(nb_harmonics + 1, 1)

        # Fixed random initialization for causal inference (matching PyTorch)
        # This ensures deterministic output for the same input
        # rand_ini has shape (1, dim) with first element = 0
        self._rand_ini = None

        # Fixed noise for sine waves in causal inference (matching PyTorch SineGen2)
        # PyTorch: self.sine_waves = torch.rand(1, 300 * 24000, 9)
        self._sine_waves_noise = None

        # Fixed noise for causal inference (matching PyTorch SourceModuleHnNSF)
        # PyTorch: self.uv = torch.rand(1, 300 * 24000, 1)
        self._fixed_noise = None

    def __call__(
        self, x: mx.array, upsample_ratio: int
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """
        Generate source signal from F0.

        Implements SineGen2 logic inline (matching PyTorch CosyVoice exactly),
        then applies linear + tanh.

        Args:
            x: F0 values (B, 1, T)
            upsample_ratio: Ratio to upsample

        Returns:
            Tuple of (source_signal, uv_signal, noise_signal)
        """
        B, _, T = x.shape
        T_out = T * upsample_ratio

        # Upsample F0 (B, 1, T_out) using nearest neighbor
        f0 = self._upsample_nearest(x, T_out)

        # Get voiced/unvoiced mask
        uv = (f0 > self.nsf_voiced_threshold).astype(mx.float32)  # (B, 1, T_out)

        # --- SineGen2 logic (matching PyTorch exactly) ---
        # Transpose F0 to (B, T_out, 1) for processing
        f0_t = f0.transpose(0, 2, 1)  # (B, T_out, 1)

        # Generate F0 matrix for all harmonics: fn = f0 * [1, 2, 3, ..., dim]
        # Shape: (B, T_out, dim)
        harmonic_scales = mx.arange(1, self.dim + 1, dtype=mx.float32).reshape(1, 1, -1)
        fn = f0_t * harmonic_scales  # (B, T_out, dim)

        # Convert to rad/sample: rad_values = (fn / sr) % 1
        rad_values = (fn / self.sampling_rate) % 1

        # Add random initial phase ONLY at first timestep (t=0)
        # Fundamental component (index 0) always gets 0 phase
        # Use fixed random initialization for deterministic inference (matching PyTorch causal=True)
        if self._rand_ini is None:
            import numpy as np
            # Initialize with fixed seed for deterministic inference
            # Uses numpy random with fixed seed to ensure reproducibility
            np.random.seed(1)
            rand_init = np.random.rand(1, self.dim).astype(np.float32)
            rand_init[0, 0] = 0  # Fundamental component has 0 phase
            self._rand_ini = mx.array(rand_init)

        # Broadcast to batch size and add to first timestep
        rand_ini = mx.broadcast_to(self._rand_ini, (B, self.dim))
        # Add to first timestep only: rad_values[:, 0, :] += rand_ini
        first_timestep = rad_values[:, 0:1, :] + rand_ini[:, None, :]
        rad_values = mx.concatenate([first_timestep, rad_values[:, 1:, :]], axis=1)

        # --- Key SineGen2 optimization: interpolation-based cumsum ---
        # This avoids numerical precision issues at high sample rates

        # 1. Downsample rad_values to lower rate using linear interpolation
        # PyTorch: interpolate(rad_values.transpose(1,2), scale_factor=1/upsample_scale, mode="linear")
        T_down = T_out // self.upsample_scale
        rad_values_down = self._interpolate_1d(rad_values, T_down, mode="linear")

        # 2. Cumsum at lower rate, multiply by 2*pi
        phase_down = mx.cumsum(rad_values_down, axis=1) * 2 * math.pi

        # 3. Multiply by upsample_scale BEFORE upsampling (matching PyTorch exactly)
        # Then upsample with "nearest" mode (causal=True in CosyVoice3)
        phase_scaled = phase_down * self.upsample_scale
        phase = self._interpolate_1d(phase_scaled, T_out, mode="nearest")

        # Generate sine waves
        sine_waves = self.sine_amp * mx.sin(phase)  # (B, T_out, dim)

        # UV mask: (B, T_out, 1)
        uv_t = uv.transpose(0, 2, 1)

        # Noise amplitude: voiced uses noise_std, unvoiced uses sine_amp/3
        noise_amp = uv_t * self.noise_std + (1 - uv_t) * self.sine_amp / 3

        # Use fixed noise for causal inference (matching PyTorch SineGen2)
        # PyTorch: self.sine_waves = torch.rand(1, 300 * 24000, 9)
        if self._sine_waves_noise is None:
            import numpy as np
            # Initialize with fixed seed for deterministic inference
            np.random.seed(2)  # Different seed from rand_ini
            # Shape: (1, max_length, dim) matching PyTorch self.sine_waves
            sine_waves_noise = np.random.rand(1, 300 * 24000, self.dim).astype(np.float32)
            self._sine_waves_noise = mx.array(sine_waves_noise)

        # Use fixed noise slice instead of random noise
        noise = noise_amp * self._sine_waves_noise[:, :T_out, :]

        # Apply UV mask and add noise (unvoiced regions get noise only)
        sine_waves = sine_waves * uv_t + noise

        # --- SourceModuleHnNSF logic ---
        # Combine harmonics with linear layer
        source = self.l_linear(sine_waves)  # (B, T_out, 1)

        # Apply tanh to bound output
        source = mx.tanh(source)

        # Transpose to (B, 1, T_out)
        source = source.transpose(0, 2, 1)

        # Noise for noise branch (same shape as uv)
        # For causal inference, use FIXED noise (matching PyTorch SourceModuleHnNSF)
        # This prevents trembling/jitter in the output
        if self._fixed_noise is None:
            import numpy as np
            # Initialize fixed noise matching PyTorch structure: torch.rand(1, 300 * 24000, 1)
            np.random.seed(3)  # Different seed from rand_ini and sine_waves_noise
            fixed_noise = np.random.rand(1, 300 * 24000, 1).astype(np.float32)
            self._fixed_noise = mx.array(fixed_noise)

        # Use fixed noise scaled by sine_amp / 3
        output_noise = self._fixed_noise[:, :T_out, :].transpose(0, 2, 1) * self.sine_amp / 3

        return source, uv, output_noise

    def _upsample_nearest(self, x: mx.array, target_len: int) -> mx.array:
        """Nearest neighbor upsampling matching PyTorch F.interpolate mode='nearest'.

        For scale_factor upsampling, each input value is repeated scale_factor times.
        Formula: out[i] = in[floor(i * in_size / out_size)]
        """
        B, C, T = x.shape
        # Use same formula as _interpolate_1d for consistency
        indices = mx.floor(
            mx.arange(target_len, dtype=mx.float32) * T / target_len
        ).astype(mx.int32)
        indices = mx.minimum(indices, T - 1)
        return x[:, :, indices]

    def _interpolate_1d(
        self, x: mx.array, target_len: int, mode: str = "linear", align_corners: bool = False
    ) -> mx.array:
        """
        1D interpolation along time axis.

        Matches PyTorch F.interpolate behavior.

        Args:
            x: Input tensor (B, T, C)
            target_len: Target length
            mode: 'linear' or 'nearest'
            align_corners: If True, corners are aligned (like PyTorch align_corners=True)

        Returns:
            Interpolated tensor (B, target_len, C)
        """
        B, T, C = x.shape

        if mode == "nearest":
            # PyTorch nearest mode with align_corners=False
            # Each output position maps to: floor(out_idx * in_size / out_size)
            indices = mx.floor(
                mx.arange(target_len, dtype=mx.float32) * T / target_len
            ).astype(mx.int32)
            indices = mx.minimum(indices, T - 1)
            return x[:, indices, :]
        else:
            # Linear interpolation
            if align_corners:
                # align_corners=True: corners are aligned
                t_out = mx.linspace(0, T - 1, target_len)
            else:
                # align_corners=False (PyTorch default):
                # Coordinate mapping: (out_idx + 0.5) * in_size / out_size - 0.5
                t_out = (mx.arange(target_len, dtype=mx.float32) + 0.5) * T / target_len - 0.5
                # Clamp to valid range
                t_out = mx.clip(t_out, 0, T - 1)

            # For each output position, find the surrounding input positions
            idx_low = mx.floor(t_out).astype(mx.int32)
            idx_high = mx.minimum(idx_low + 1, T - 1)
            weight = t_out - idx_low.astype(mx.float32)

            # Gather and interpolate
            x_low = x[:, idx_low, :]  # (B, target_len, C)
            x_high = x[:, idx_high, :]  # (B, target_len, C)

            return x_low + weight[None, :, None] * (x_high - x_low)


class ConvRNNF0Predictor(nn.Module):
    """
    F0 predictor using convolutional layers.

    Matches CausalConvRNNF0Predictor from CosyVoice:
    - First conv: causal_type='right' (look-ahead)
    - Other convs: causal_type='left' (causal)
    """

    def __init__(
        self,
        num_class: int = 1,
        in_channels: int = 80,
        cond_channels: int = 512,
    ):
        super().__init__()
        # Condition network - uses CausalConv1d with proper causal types
        # First layer uses 'right' (look-ahead), rest use 'left' (causal)
        self.condnet = [
            CausalConv1d(in_channels, cond_channels, kernel_size=4, causal_type="right"),
            nn.ELU(),
            CausalConv1d(cond_channels, cond_channels, kernel_size=3, causal_type="left"),
            nn.ELU(),
            CausalConv1d(cond_channels, cond_channels, kernel_size=3, causal_type="left"),
            nn.ELU(),
            CausalConv1d(cond_channels, cond_channels, kernel_size=3, causal_type="left"),
            nn.ELU(),
            CausalConv1d(cond_channels, cond_channels, kernel_size=3, causal_type="left"),
            nn.ELU(),
        ]
        self.classifier = nn.Linear(cond_channels, num_class)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Predict F0 from mel-spectrogram.

        Args:
            x: Mel-spectrogram (B, T, mel_dim) - MLX format

        Returns:
            F0 prediction (B, T, 1) - same length as input due to causal padding
        """
        for layer in self.condnet:
            x = layer(x)
        x = self.classifier(x)  # (B, T, 1)
        x = mx.abs(x)  # F0 is positive
        return x


class CausalHiFTGenerator(nn.Module):
    """
    Causal HiFi-GAN with ISTFT for streaming speech synthesis.

    Matches CausalHiFTGenerator from CosyVoice.
    """

    def __init__(
        self,
        in_channels: int = 80,
        base_channels: int = 512,
        nb_harmonics: int = 8,  # For m_source (l_linear)
        sampling_rate: int = 24000,
        nsf_alpha: float = 0.1,
        nsf_sigma: float = 0.003,
        nsf_voiced_threshold: float = 10,
        upsample_rates: List[int] = [8, 5, 3],
        upsample_kernel_sizes: List[int] = [16, 11, 7],
        n_fft: int = 16,
        hop_len: int = 4,
        resblock_kernel_sizes: List[int] = [3, 7, 11],
        resblock_dilation_sizes: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        source_resblock_kernel_sizes: List[int] = [7, 7, 11],
        source_resblock_dilation_sizes: List[List[int]] = [
            [1, 3, 5],
            [1, 3, 5],
            [1, 3, 5],
        ],
        lrelu_slope: float = 0.1,
        audio_limit: float = 0.99,
        conv_pre_look_right: int = 4,
        gain: float = 1.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.sampling_rate = sampling_rate
        self.nb_harmonics = nb_harmonics
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.audio_limit = audio_limit
        self.gain = gain
        self.num_upsamples = len(upsample_rates)
        self.upsample_rates = upsample_rates

        # Calculate total upsample factor (including hop_len for audio rate)
        self.upsample_factor = hop_len
        for r in upsample_rates:
            self.upsample_factor *= r
        # upsample_factor = 8 * 5 * 3 * 4 = 480 (mel frames to audio samples)

        # F0 predictor
        self.f0_predictor = ConvRNNF0Predictor(
            num_class=1, in_channels=in_channels, cond_channels=base_channels
        )

        # Source module (uses nb_harmonics)
        self.m_source = SourceModuleHnNSF(
            sampling_rate=sampling_rate,
            upsample_scale=self.upsample_factor,  # Total upsample ratio for SineGen2
            nb_harmonics=nb_harmonics,
            nsf_alpha=nsf_alpha,
            nsf_sigma=nsf_sigma,
            nsf_voiced_threshold=nsf_voiced_threshold,
        )

        # Pre-convolution - uses causal_type='right' (look-ahead)
        self.conv_pre = CausalConv1d(
            in_channels, base_channels, kernel_size=conv_pre_look_right + 1, causal_type="right"
        )

        # Upsampling layers (using CausalConv1dUpsample: Upsample + Conv1d, not ConvTranspose1d)
        self.ups = []
        ch = base_channels
        for i, (rate, ksize) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                CausalConv1dUpsample(
                    ch,
                    ch // 2,
                    kernel_size=ksize,
                    stride=rate,
                )
            )
            ch = ch // 2

        # Residual blocks
        self.resblocks = []
        ch = base_channels
        num_kernels = len(resblock_kernel_sizes)
        for i in range(len(upsample_rates)):
            ch = ch // 2
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(ResBlock(ch, k, tuple(d)))

        # Source downsampling and processing (uses n_fft + 2 channels from STFT)
        # Calculate downsample rates to match PyTorch: cumulative product of reversed upsamples
        # For upsample_rates = [8, 5, 3]: downsample_cum_rates[::-1] = [15, 3, 1]
        downsample_rates = [1] + upsample_rates[::-1][:-1]  # [1, 3, 5]
        downsample_cum_rates = []
        cum = 1
        for r in downsample_rates:
            cum *= r
            downsample_cum_rates.append(cum)
        # Reverse for use in loop: [15, 3, 1]
        source_downsample_strides = downsample_cum_rates[::-1]

        source_ch = n_fft + 2  # STFT output channels (real + imag)
        self.source_downs = []
        self.source_resblocks = []
        ch = base_channels
        for i, (stride, src_ksize) in enumerate(
            zip(source_downsample_strides, source_resblock_kernel_sizes)
        ):
            ch = ch // 2
            # Use stride from cumulative downsample rates
            # Kernel size: 1 for stride==1, stride*2 for stride>1 (matches PyTorch)
            if stride == 1:
                # No downsampling - use CausalConv1d with kernel_size=1
                self.source_downs.append(
                    CausalConv1d(source_ch, ch, kernel_size=1, causal_type="left")
                )
            else:
                # Downsample with CausalConv1dDownSample, kernel_size = stride * 2
                self.source_downs.append(
                    CausalConv1dDownSample(source_ch, ch, kernel_size=stride * 2, stride=stride)
                )
            self.source_resblocks.append(
                ResBlock(ch, src_ksize, tuple(source_resblock_dilation_sizes[i]))
            )

        # Post-convolution for STFT - output channels = n_fft + 2 (mag + phase)
        # Uses causal_type='left' (default)
        self.conv_post = CausalConv1d(ch, n_fft + 2, kernel_size=7, causal_type="left")

        # ISTFT cache for efficient synthesis
        self._istft_cache = ISTFTCache()

    def __call__(self, mel: mx.array) -> mx.array:
        """
        Generate audio from mel-spectrogram.

        Args:
            mel: Mel-spectrogram (B, mel_dim, T)

        Returns:
            Audio waveform (B, T * upsample_factor)
        """
        # Transpose mel from (B, mel_dim, T) to (B, T, mel_dim) for MLX Conv1d
        mel = mel.transpose(0, 2, 1)

        # Predict F0 - expects (B, T, mel_dim), returns (B, T, 1)
        f0 = self.f0_predictor(mel)
        # Transpose F0 to (B, 1, T) for source module
        f0 = f0.transpose(0, 2, 1)

        # Generate source signal (time domain)
        source, uv, noise = self.m_source(f0, self.upsample_factor)

        # Convert source to frequency domain using STFT
        # source shape: (B, 1, T_audio)
        source_stft = self._stft(
            source.squeeze(1)
        )  # (B, C, T_stft) where C = 2 * n_freq
        # Transpose for MLX Conv1d: (B, C, T) -> (B, T, C)
        source_stft = source_stft.transpose(0, 2, 1)

        # Pre-conv - mel is already in (B, T, C) format
        x = self.conv_pre(mel)

        # Upsampling with source injection
        num_kernels = 3  # len(resblock_kernel_sizes)
        for i in range(self.num_upsamples):
            x = nn.leaky_relu(x, negative_slope=0.1)
            x = self.ups[i](x)

            # Apply reflection pad after last upsample (matches PyTorch)
            if i == self.num_upsamples - 1:
                # ReflectionPad1d((1, 0)) - pad 1 sample on left using reflection
                # x: (B, T, C) - reflect along time axis
                x = mx.concatenate([x[:, 1:2, :], x], axis=1)

            # Source injection - source_downs processes source STFT
            s_down = self.source_downs[i](source_stft)
            s_down = self.source_resblocks[i](s_down)

            # Crop or pad s_down to match x length
            x_len = x.shape[1]
            s_len = s_down.shape[1]
            if s_len > x_len:
                s_down = s_down[:, :x_len, :]
            elif s_len < x_len:
                pad_len = x_len - s_len
                s_down = mx.pad(s_down, [(0, 0), (0, pad_len), (0, 0)])

            x = x + s_down

            # Residual blocks
            xs = 0
            for j in range(num_kernels):
                xs = xs + self.resblocks[i * num_kernels + j](x)
            x = xs / num_kernels

        # Final leaky_relu uses PyTorch default negative_slope=0.01 (NOT self.lrelu_slope=0.1)
        x = nn.leaky_relu(x, negative_slope=0.01)
        x = self.conv_post(x)

        # Transpose to (B, C, T) for synthesis
        x = x.transpose(0, 2, 1)

        # Synthesize audio using ISTFT (no source multiplication)
        audio = self._istft(x)

        # Apply gain to normalize audio amplitude
        audio = audio * self.gain

        # Limit audio
        audio = mx.clip(audio, -self.audio_limit, self.audio_limit)

        return audio

    def _stft(self, x: mx.array) -> mx.array:
        """
        Compute Short-Time Fourier Transform (matching PyTorch torch.stft with center=True).

        Converts time-domain signal to frequency domain representation
        with real and imaginary parts concatenated.

        Args:
            x: Time-domain audio (B, T_audio)

        Returns:
            STFT representation (B, 2*n_freq, T_stft) where first n_freq channels
            are real and last n_freq channels are imaginary
        """
        B, T_audio = x.shape
        n_freq = self.n_fft // 2 + 1  # 9 for n_fft=16
        pad_amount = self.n_fft // 2

        # Apply center padding using reflect mode (matching PyTorch center=True)
        # Reflect pad: for input [a, b, c, d, e], pad_left=2 gives [c, b, a, b, c, d, e, ...]
        # MLX pad doesn't have reflect mode, so we implement it manually
        left_pad = x[:, 1 : pad_amount + 1][:, ::-1]  # Reflect left
        right_pad = x[:, -(pad_amount + 1) : -1][:, ::-1]  # Reflect right
        x_padded = mx.concatenate([left_pad, x, right_pad], axis=1)

        T_padded = x_padded.shape[1]

        # Compute number of frames
        n_frames = (T_padded - self.n_fft) // self.hop_len + 1

        # Create analysis window (Hann with fftbins=True, matching scipy.signal.get_window)
        # fftbins=True uses periodic Hann: 0.5 * (1 - cos(2*pi*i/n))
        window = 0.5 * (1 - mx.cos(2 * math.pi * mx.arange(self.n_fft) / self.n_fft))

        # Extract frames and apply window
        frames_list = []
        for t in range(n_frames):
            start = t * self.hop_len
            frame = x_padded[:, start : start + self.n_fft] * window[None, :]
            frames_list.append(frame)

        frames = mx.stack(frames_list, axis=1)  # (B, n_frames, n_fft)

        # Apply FFT
        spec = mx.fft.fft(frames, axis=-1)  # (B, n_frames, n_fft)

        # Take only positive frequencies
        spec = spec[:, :, :n_freq]  # (B, n_frames, n_freq)

        # Separate real and imaginary parts
        real = spec.real  # (B, n_frames, n_freq)
        imag = spec.imag  # (B, n_frames, n_freq)

        # Transpose to (B, n_freq, n_frames) and concatenate
        real = real.transpose(0, 2, 1)  # (B, n_freq, n_frames)
        imag = imag.transpose(0, 2, 1)  # (B, n_freq, n_frames)

        # Concatenate real and imaginary: (B, 2*n_freq, n_frames)
        stft_out = mx.concatenate([real, imag], axis=1)

        return stft_out

    def _generate_harmonics(self, f0: mx.array, upsample_ratio: int) -> mx.array:
        """Generate harmonic signals from F0 for source processing."""
        B, _, T = f0.shape
        T_out = T * upsample_ratio

        # Upsample F0
        indices = mx.floor(mx.linspace(0, T - 1, T_out)).astype(mx.int32)
        f0_up = f0[:, :, indices]  # (B, 1, T_out)

        # Generate harmonics - use source_harmonics for source processing
        omega = f0_up / self.sampling_rate

        harmonics = []
        for n in range(1, self.source_harmonics + 2):
            phase = mx.cumsum(omega * n * 2 * math.pi, axis=-1)
            harmonic = mx.sin(phase)
            harmonics.append(harmonic)

        return mx.concatenate(harmonics, axis=1)  # (B, source_harmonics+1, T_out)

    def _generate_harmonics_at_rate(self, f0: mx.array, rate: int) -> mx.array:
        """Generate harmonic signals from F0 at specified rate."""
        B, _, T = f0.shape
        T_out = T * rate

        if rate > 1:
            indices = mx.floor(mx.linspace(0, T - 1, T_out)).astype(mx.int32)
            f0_up = f0[:, :, indices]
        else:
            f0_up = f0

        omega = f0_up / self.sampling_rate

        harmonics = []
        for n in range(1, self.source_harmonics + 2):
            phase = mx.cumsum(omega * n * 2 * math.pi, axis=-1)
            harmonic = mx.sin(phase)
            harmonics.append(harmonic)

        return mx.concatenate(harmonics, axis=1)  # (B, source_harmonics+1, T_out)

    def _upsample_harmonics(self, harmonics: mx.array, target_len: int) -> mx.array:
        """Upsample harmonics to target length using linear interpolation."""
        B, T, C = harmonics.shape
        if T == target_len:
            return harmonics

        # Linear interpolation
        indices = mx.linspace(0, T - 1, target_len)
        idx_floor = mx.floor(indices).astype(mx.int32)
        idx_ceil = mx.minimum(idx_floor + 1, T - 1)
        weights = indices - idx_floor.astype(mx.float32)

        h_floor = harmonics[:, idx_floor, :]
        h_ceil = harmonics[:, idx_ceil, :]
        return h_floor * (1 - weights[None, :, None]) + h_ceil * weights[None, :, None]

    def _istft(self, x: mx.array) -> mx.array:
        """
        Synthesize audio from network output using ISTFT.

        Uses the optimized ISTFTCache from mlx_audio.dsp.

        The conv_post output contains STFT magnitude and phase predictions.
        - magnitude = exp(x[:, :n_freq, :])
        - phase = sin(x[:, n_freq:, :])  # CosyVoice applies sin (comment says "redundant")
        - real = magnitude * cos(phase)
        - imag = magnitude * sin(phase)

        Args:
            x: STFT coefficients (B, C, T) where C = n_fft + 2
               First n_fft//2+1 channels: log magnitude
               Last n_fft//2+1 channels: phase input

        Returns:
            Audio waveform (B, T_audio)
        """
        B, C, T = x.shape
        n_freq = self.n_fft // 2 + 1  # 9 for n_fft=16

        # Split into magnitude and phase (matching original CosyVoice implementation)
        log_mag = x[:, :n_freq, :]  # (B, n_freq, T)
        phase_input = x[:, n_freq:, :]  # (B, n_freq, T)

        # Magnitude: exp of first half channels, clipped to prevent overflow
        magnitude = mx.exp(log_mag)
        magnitude = mx.clip(magnitude, a_min=None, a_max=1e2)

        # Phase: sin of second half channels (CosyVoice does this, comments say "redundant")
        phase = mx.sin(phase_input)

        # Create real and imaginary parts
        real = magnitude * mx.cos(phase)  # (B, n_freq, T)
        imag = magnitude * mx.sin(phase)  # (B, n_freq, T)

        # Use ISTFTCache for efficient batched ISTFT
        # Window: Hann window with periodic=True (fftbins=True equivalent)
        window = hanning(self.n_fft, periodic=True)

        # Calculate expected audio length: (T - 1) * hop_len
        # This matches torch.istft with center=True
        expected_audio_length = (T - 1) * self.hop_len

        audio = self._istft_cache.istft(
            real_part=real,
            imag_part=imag,
            n_fft=self.n_fft,
            hop_length=self.hop_len,
            win_length=self.n_fft,
            window=window,
            center=True,
            audio_length=expected_audio_length,
        )

        return audio

    def inference(self, mel: mx.array) -> mx.array:
        """Inference wrapper."""
        return self(mel)
