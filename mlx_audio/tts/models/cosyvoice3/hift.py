"""
CosyVoice3 HIFT (HiFi-GAN with ISTFT) Vocoder implementation in MLX.

Based on: https://github.com/FunAudioLLM/CosyVoice
"""

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
    """Causal 1D convolution with proper padding."""

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
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def __call__(self, x: mx.array) -> mx.array:
        # x is in (B, T, C) format for MLX
        # Causal padding on the time dimension (axis 1)
        if self.padding > 0:
            x = mx.pad(x, [(0, 0), (self.padding, 0), (0, 0)])
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
    Matches the original CosyVoice SineGen + SourceModuleHnNSF implementation.
    """

    def __init__(
        self,
        sampling_rate: int = 24000,
        nb_harmonics: int = 8,
        nsf_alpha: float = 0.1,
        nsf_sigma: float = 0.003,
        nsf_voiced_threshold: float = 10,
    ):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.nb_harmonics = nb_harmonics
        self.sine_amp = nsf_alpha  # amplitude of sine waves (default 0.1)
        self.noise_std = nsf_sigma  # noise std for voiced regions
        self.nsf_voiced_threshold = nsf_voiced_threshold

        # Linear layer to combine harmonics
        self.l_linear = nn.Linear(nb_harmonics + 1, 1)

    def __call__(
        self, x: mx.array, upsample_ratio: int
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """
        Generate source signal from F0.

        Implements SineGen logic inline, then applies linear + tanh.

        Args:
            x: F0 values (B, 1, T)
            upsample_ratio: Ratio to upsample

        Returns:
            Tuple of (source_signal, uv_signal, noise_signal)
        """
        B, _, T = x.shape
        T_out = T * upsample_ratio

        # Upsample F0 (B, 1, T_out)
        f0 = self._upsample(x, T_out)

        # Get voiced/unvoiced mask
        uv = (f0 > self.nsf_voiced_threshold).astype(mx.float32)  # (B, 1, T_out)

        # --- SineGen logic ---
        # Generate F0 matrix for all harmonics (fundamental + nb_harmonics)
        # F_mat[:, i, :] = f0 * (i+1) / sampling_rate
        omega = f0 / self.sampling_rate  # (B, 1, T_out)

        # Generate harmonics with cumulative phase
        harmonics = []
        for n in range(self.nb_harmonics + 1):
            # Cumulative phase: 2*pi * cumsum(f0 * (n+1) / sr)
            phase = 2 * math.pi * mx.cumsum(omega * (n + 1), axis=-1)
            # Sine wave with amplitude
            harmonic = self.sine_amp * mx.sin(phase)
            harmonics.append(harmonic)

        # Stack harmonics: (B, nb_harmonics+1, T_out)
        sine_waves = mx.concatenate(harmonics, axis=1)

        # Transpose for processing: (B, T_out, nb_harmonics+1)
        sine_waves = sine_waves.transpose(0, 2, 1)

        # UV mask for noise: (B, T_out, 1)
        uv_t = uv.transpose(0, 2, 1)

        # Noise amplitude: voiced uses noise_std, unvoiced uses sine_amp/3
        noise_amp = uv_t * self.noise_std + (1 - uv_t) * self.sine_amp / 3
        noise = noise_amp * mx.random.normal(sine_waves.shape)

        # Apply UV mask and add noise
        sine_waves = sine_waves * uv_t + noise

        # --- SourceModuleHnNSF logic ---
        # Combine harmonics with linear layer
        source = self.l_linear(sine_waves)  # (B, T_out, 1)

        # Apply tanh to bound output
        source = mx.tanh(source)

        # Transpose to (B, 1, T_out)
        source = source.transpose(0, 2, 1)

        # Noise for noise branch (same shape as uv)
        output_noise = mx.random.normal(uv.shape) * self.sine_amp / 3

        return source, uv, output_noise

    def _upsample(self, x: mx.array, target_len: int) -> mx.array:
        """Nearest neighbor upsampling."""
        B, C, T = x.shape
        indices = mx.floor(mx.linspace(0, T - 1, target_len)).astype(mx.int32)
        return x[:, :, indices]


class F0PredictorConv(nn.Module):
    """Conv1d wrapper for F0 predictor - uses regular conv without causal padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
    ):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=0)

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv(x)


class ConvRNNF0Predictor(nn.Module):
    """F0 predictor using convolutional layers."""

    def __init__(
        self,
        num_class: int = 1,
        in_channels: int = 80,
        cond_channels: int = 512,
    ):
        super().__init__()
        # Condition network - uses regular Conv1d without padding (matches PyTorch)
        self.condnet = nn.Sequential(
            F0PredictorConv(in_channels, cond_channels, kernel_size=4),
            nn.ELU(),
            F0PredictorConv(cond_channels, cond_channels, kernel_size=3),
            nn.ELU(),
            F0PredictorConv(cond_channels, cond_channels, kernel_size=3),
            nn.ELU(),
            F0PredictorConv(cond_channels, cond_channels, kernel_size=3),
            nn.ELU(),
            F0PredictorConv(cond_channels, cond_channels, kernel_size=3),
            nn.ELU(),
        )
        self.classifier = nn.Linear(cond_channels, num_class)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Predict F0 from mel-spectrogram.

        Args:
            x: Mel-spectrogram (B, T, mel_dim) - MLX format

        Returns:
            F0 prediction (B, T', 1) - note: T' < T due to valid convolutions
        """
        x = self.condnet(x)  # (B, T', cond_channels)
        x = self.classifier(x)  # (B, T', 1)
        x = mx.abs(x)  # F0 is positive
        return x


class CausalHiFTGenerator(nn.Module):
    """
    Causal HiFi-GAN with ISTFT for streaming speech synthesis.
    """

    def __init__(
        self,
        in_channels: int = 80,
        base_channels: int = 512,
        nb_harmonics: int = 8,  # For m_source (l_linear)
        source_harmonics: int = 17,  # For source processing (source_downs)
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
        source_down_kernel_sizes: List[int] = [30, 6, 1],
        lrelu_slope: float = 0.1,
        audio_limit: float = 0.99,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.sampling_rate = sampling_rate
        self.nb_harmonics = nb_harmonics
        self.source_harmonics = source_harmonics
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.audio_limit = audio_limit

        # Calculate total upsample factor
        self.upsample_factor = 1
        for r in upsample_rates:
            self.upsample_factor *= r

        # F0 predictor
        self.f0_predictor = ConvRNNF0Predictor(
            num_class=1, in_channels=in_channels, cond_channels=base_channels
        )

        # Source module (uses nb_harmonics)
        self.m_source = SourceModuleHnNSF(
            sampling_rate=sampling_rate,
            nb_harmonics=nb_harmonics,
            nsf_alpha=nsf_alpha,
            nsf_sigma=nsf_sigma,
            nsf_voiced_threshold=nsf_voiced_threshold,
        )

        # Pre-convolution
        self.conv_pre = CausalConv1d(in_channels, base_channels, kernel_size=5)

        # Upsampling layers
        self.ups = []
        ch = base_channels
        for i, (rate, ksize) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                nn.ConvTranspose1d(
                    ch,
                    ch // 2,
                    kernel_size=ksize,
                    stride=rate,
                    padding=(ksize - rate) // 2,
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

        # Source downsampling and processing (uses source_harmonics)
        source_ch = source_harmonics + 1
        self.source_downs = []
        self.source_resblocks = []
        ch = base_channels
        for i, (rate, ksize, src_ksize) in enumerate(
            zip(upsample_rates, source_down_kernel_sizes, source_resblock_kernel_sizes)
        ):
            ch = ch // 2
            # Use provided kernel sizes for source_downs
            self.source_downs.append(
                nn.Conv1d(source_ch, ch, kernel_size=ksize, stride=rate, padding=rate)
            )
            self.source_resblocks.append(
                ResBlock(ch, src_ksize, tuple(source_resblock_dilation_sizes[i]))
            )

        # Post-convolution for STFT - output channels = n_fft + 2 (mag + phase)
        self.conv_post = CausalConv1d(ch, n_fft + 2, kernel_size=7)

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
        for i in range(len(self.ups)):
            x = nn.leaky_relu(x, negative_slope=0.1)
            x = self.ups[i](x)

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

        x = nn.leaky_relu(x, negative_slope=0.1)
        x = self.conv_post(x)

        # Transpose to (B, C, T) for synthesis
        x = x.transpose(0, 2, 1)

        # Synthesize audio using ISTFT (no source multiplication)
        audio = self._istft(x)

        # Limit audio
        audio = mx.clip(audio, -self.audio_limit, self.audio_limit)

        return audio

    def _stft(self, x: mx.array) -> mx.array:
        """
        Compute Short-Time Fourier Transform.

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

        # Compute number of frames
        n_frames = (T_audio - self.n_fft) // self.hop_len + 1

        # Create analysis window (Hann)
        window = 0.5 * (1 - mx.cos(2 * math.pi * mx.arange(self.n_fft) / self.n_fft))

        # Extract frames and apply window
        frames_list = []
        for t in range(n_frames):
            start = t * self.hop_len
            frame = x[:, start : start + self.n_fft] * window[None, :]
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
