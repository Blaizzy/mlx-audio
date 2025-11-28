# Ported from https://github.com/resemble-ai/chatterbox

import math
from typing import Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn


def hann_window_periodic(size: int) -> mx.array:
    """
    Create periodic Hann window (fftbins=True), matching scipy.signal.get_window('hann', size, fftbins=True).

    Uses size in denominator instead of size-1, which is the correct window for STFT.
    """
    return mx.array([0.5 * (1 - math.cos(2 * math.pi * n / size)) for n in range(size)])


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    """Calculate padding for 'same' convolution."""
    return int((kernel_size * dilation - dilation) / 2)


class Snake(nn.Module):
    """
    Snake activation function: x + (1/α) * sin²(αx)

    Periodic activation function from:
    https://arxiv.org/abs/2006.08195
    """

    def __init__(
        self,
        in_features: int,
        alpha: float = 1.0,
        alpha_trainable: bool = True,
        alpha_logscale: bool = False,
    ):
        """
        Args:
            in_features: Number of input features (channels)
            alpha: Initial alpha value (frequency parameter)
            alpha_trainable: Whether alpha should be trainable
            alpha_logscale: Whether to use log-scale for alpha
        """
        super().__init__()
        self.in_features = in_features
        self.alpha_logscale = alpha_logscale

        # Initialize alpha
        if alpha_logscale:
            # Log scale: initialized to zeros
            self.alpha = mx.zeros(in_features) * alpha
        else:
            # Linear scale: initialized to ones * alpha
            self.alpha = mx.ones(in_features) * alpha

        # Small constant to avoid division by zero
        self.no_div_by_zero = 1e-9

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass: x + (1/α) * sin²(αx)

        Args:
            x: Input tensor of shape (B, C, T)

        Returns:
            Output tensor of same shape
        """
        # Reshape alpha to align with x: (C,) -> (1, C, 1)
        alpha = mx.reshape(self.alpha, (1, -1, 1))

        # Apply log scale if needed
        if self.alpha_logscale:
            alpha = mx.exp(alpha)

        # Snake activation: x + (1/α) * sin²(αx)
        return x + (1.0 / (alpha + self.no_div_by_zero)) * mx.power(
            mx.sin(x * alpha), 2
        )


class ResBlock(nn.Module):
    """Residual block with Snake activation for HiFi-GAN."""

    def __init__(
        self,
        channels: int = 512,
        kernel_size: int = 3,
        dilations: List[int] = [1, 3, 5],
    ):
        super().__init__()
        self.channels = channels
        self.convs1 = []
        self.convs2 = []
        self.activations1 = []
        self.activations2 = []

        for dilation in dilations:
            # First conv in each dilation block
            self.convs1.append(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    stride=1,
                    padding=get_padding(kernel_size, dilation),
                    dilation=dilation,
                )
            )
            # Second conv (dilation=1)
            self.convs2.append(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    stride=1,
                    padding=get_padding(kernel_size, 1),
                )
            )
            # Snake activations
            self.activations1.append(Snake(channels, alpha_logscale=False))
            self.activations2.append(Snake(channels, alpha_logscale=False))

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: Input tensor (B, C, T)

        Returns:
            Output tensor (B, C, T)
        """
        for i in range(len(self.convs1)):
            xt = self.activations1[i](x)
            # Conv1d: (B, C, T) -> transpose -> conv -> transpose back
            xt = mx.swapaxes(xt, 1, 2)  # (B, C, T) -> (B, T, C)
            xt = self.convs1[i](xt)
            xt = mx.swapaxes(xt, 1, 2)  # (B, T, C) -> (B, C, T)
            xt = self.activations2[i](xt)
            xt = mx.swapaxes(xt, 1, 2)  # (B, C, T) -> (B, T, C)
            xt = self.convs2[i](xt)
            xt = mx.swapaxes(xt, 1, 2)  # (B, T, C) -> (B, C, T)
            x = xt + x  # Residual connection
        return x


class SineGen(nn.Module):
    """
    Sine wave generator for harmonic synthesis.

    Generates sine waves with harmonics based on F0 (fundamental frequency).
    """

    def __init__(
        self,
        samp_rate: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        noise_std: float = 0.003,
        voiced_threshold: float = 0,
    ):
        """
        Args:
            samp_rate: Sampling rate in Hz
            harmonic_num: Number of harmonic overtones
            sine_amp: Amplitude of sine waveform
            noise_std: Standard deviation of Gaussian noise
            voiced_threshold: F0 threshold for voiced/unvoiced classification
        """
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold

    def _f02uv(self, f0: mx.array) -> mx.array:
        """Convert F0 to voiced/unvoiced signal."""
        return (f0 > self.voiced_threshold).astype(mx.float32)

    def __call__(self, f0: mx.array) -> tuple:
        """
        Generate sine waves from F0.

        Args:
            f0: Fundamental frequency tensor (B, 1, T) in Hz

        Returns:
            Tuple of (sine_waves, uv, noise)
        """
        B, _, T = f0.shape

        # Create frequency matrix for harmonics using broadcasting (no loop)
        # harmonic_multipliers: [1, 2, 3, ..., harmonic_num+1] shape (1, H, 1)
        harmonic_multipliers = mx.arange(1, self.harmonic_num + 2).reshape(1, -1, 1)
        # f0 is (B, 1, T), result is (B, H, T)
        F_mat = f0 * harmonic_multipliers / self.sampling_rate

        # Calculate phase
        theta_mat = 2 * math.pi * (mx.cumsum(F_mat, axis=-1) % 1)

        # Random phase offset for each harmonic (no loop)
        phase_vec = mx.random.uniform(
            low=-math.pi, high=math.pi, shape=(B, self.harmonic_num + 1, 1)
        )
        # Zero out fundamental frequency phase offset (index 0) using where
        mask = mx.arange(self.harmonic_num + 1).reshape(1, -1, 1) > 0
        phase_vec = mx.where(mask, phase_vec, 0.0)

        # Generate sine waveforms
        sine_waves = self.sine_amp * mx.sin(theta_mat + phase_vec)

        # Generate voiced/unvoiced signal
        uv = self._f02uv(f0)

        # Noise: larger for unvoiced, smaller for voiced
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * mx.random.normal(shape=sine_waves.shape)

        # Voiced regions use sine, unvoiced use noise
        sine_waves = sine_waves * uv + noise

        return sine_waves, uv, noise


class SourceModuleHnNSF(nn.Module):
    """
    Neural Source Filter (NSF) module for harmonic and noise generation.

    Combines multiple sine harmonics into a single excitation signal.
    """

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

        # Sine generator for harmonics
        self.l_sin_gen = SineGen(
            sampling_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshod
        )

        # Linear layer to merge harmonics
        self.l_linear = nn.Linear(harmonic_num + 1, 1)

    def __call__(self, x: mx.array) -> tuple:
        """
        Generate harmonic and noise sources from F0.

        Args:
            x: F0 tensor (B, T, 1)

        Returns:
            Tuple of (sine_merge, noise, uv)
        """
        # Generate sine harmonics
        sine_wavs, uv, _ = self.l_sin_gen(mx.swapaxes(x, 1, 2))
        sine_wavs = mx.swapaxes(sine_wavs, 1, 2)
        uv = mx.swapaxes(uv, 1, 2)

        # Merge harmonics
        sine_merge = mx.tanh(self.l_linear(sine_wavs))

        # Source for noise branch, in the same shape as uv
        noise = mx.random.normal(shape=uv.shape) * self.sine_amp / 3

        return sine_merge, noise, uv


def stft(x: mx.array, n_fft: int, hop_length: int, window: mx.array) -> tuple:
    """
    Short-Time Fourier Transform.

    Args:
        x: Input signal (B, T)
        n_fft: FFT size
        hop_length: Hop length
        window: Window function

    Returns:
        Tuple of (real, imag) each of shape (B, n_fft//2+1, num_frames)
    """
    B, T = x.shape

    # Pad signal
    pad_length = n_fft // 2
    x_padded = mx.pad(x, [(0, 0), (pad_length, pad_length)])

    # Calculate number of frames
    num_frames = (x_padded.shape[1] - n_fft) // hop_length + 1

    # Create frames using vectorized slicing (avoid Python loop)
    # Generate all frame start indices at once
    frame_starts = mx.arange(num_frames) * hop_length  # (num_frames,)
    sample_offsets = mx.arange(n_fft)  # (n_fft,)
    # All indices: (num_frames, n_fft) where indices[f, s] = frame_starts[f] + s
    all_indices = frame_starts[:, None] + sample_offsets[None, :]  # (num_frames, n_fft)
    # Gather frames: x_padded is (B, T), gather along axis 1
    # Result shape: (B, num_frames, n_fft)
    frames = mx.take(x_padded, all_indices.flatten(), axis=1).reshape(
        B, num_frames, n_fft
    )
    # Transpose to (B, n_fft, num_frames)
    frames = mx.swapaxes(frames, 1, 2)

    # Apply window
    window_expanded = mx.reshape(window, (1, -1, 1))
    frames = frames * window_expanded

    # FFT
    # MLX doesn't have a direct rfft, so we use fft and take the first half
    fft_result = mx.fft.fft(frames, axis=1)

    # Take positive frequencies
    real = mx.real(fft_result[:, : n_fft // 2 + 1, :])
    imag = mx.imag(fft_result[:, : n_fft // 2 + 1, :])

    return real, imag


def istft(
    magnitude: mx.array, phase: mx.array, n_fft: int, hop_length: int, window: mx.array
) -> mx.array:
    """
    Inverse Short-Time Fourier Transform (pure MLX implementation).

    Args:
        magnitude: Magnitude (B, n_fft//2+1, num_frames)
        phase: Phase (B, n_fft//2+1, num_frames)
        n_fft: FFT size
        hop_length: Hop length
        window: Window function

    Returns:
        Reconstructed signal (B, T)
    """
    # Clip magnitude
    magnitude = mx.clip(magnitude, a_min=None, a_max=1e2)

    # Convert to complex
    real = magnitude * mx.cos(phase)
    imag = magnitude * mx.sin(phase)

    # Create full spectrum (conjugate symmetric)
    B, F, num_frames = real.shape

    # Mirror the spectrum for negative frequencies
    real_mirror = real[:, 1:-1, :][:, ::-1, :]
    imag_mirror = imag[:, 1:-1, :][:, ::-1, :]
    real_full = mx.concatenate([real, real_mirror], axis=1)
    imag_full = mx.concatenate([imag, -imag_mirror], axis=1)

    # Combine into complex
    spectrum = real_full + 1j * imag_full

    # Inverse FFT
    frames = mx.fft.ifft(spectrum, axis=1)
    frames = mx.real(frames)  # Take real part

    # Apply window
    window_expanded = mx.reshape(window, (1, -1, 1))
    frames = frames * window_expanded

    # Pure MLX overlap-add
    output_length = (num_frames - 1) * hop_length + n_fft

    # Create index arrays for scatter-add
    frame_offsets = mx.arange(num_frames) * hop_length
    sample_indices = mx.arange(n_fft)
    # indices[f, s] = frame_offsets[f] + sample_indices[s]
    indices = frame_offsets[:, None] + sample_indices[None, :]  # (num_frames, n_fft)
    indices_flat = indices.flatten()  # (num_frames * n_fft,)

    # Window squared for normalization - compute once (same for all batches)
    window_sq = window**2
    window_sum = mx.zeros(output_length)
    window_updates = mx.tile(window_sq, (num_frames,))
    window_sum = window_sum.at[indices_flat].add(window_updates)
    window_sum = mx.maximum(window_sum, 1e-8)

    # Vectorized overlap-add for all batch items at once
    # frames is (B, n_fft, num_frames), need (B, num_frames, n_fft) for indexing
    frame_data = mx.swapaxes(frames, 1, 2)  # (B, num_frames, n_fft)
    updates = frame_data.reshape(B, -1)  # (B, num_frames * n_fft)

    # Use vmap-style vectorized scatter-add via index expansion
    # Expand indices to (B, num_frames * n_fft) - same indices for all batches
    output = mx.zeros((B, output_length))
    # Process with a single batched at[] operation using advanced indexing
    # batch_indices: [0,0,...,0, 1,1,...,1, ...] for each element
    batch_indices = mx.repeat(mx.arange(B), num_frames * n_fft)
    flat_indices = mx.tile(indices_flat, (B,))
    # Scatter add using 2D indexing
    flat_output = output.flatten()
    # Convert 2D indices to 1D: batch * output_length + sample_idx
    linear_indices = batch_indices * output_length + flat_indices
    flat_output = flat_output.at[linear_indices].add(updates.flatten())
    output = flat_output.reshape(B, output_length)

    # Normalize with precomputed window_sum (broadcasts across batch)
    output = output / window_sum

    # Remove padding
    pad_length = n_fft // 2
    output = output[:, pad_length:-pad_length]

    return output


class HiFTGenerator(nn.Module):
    """
    HiFi-GAN with Neural Source Filter (HiFT-Net) generator.

    Combines mel-spectrogram upsampling with neural source filter
    for high-quality speech synthesis using ISTFT.

    Reference: https://arxiv.org/abs/2309.09493
    """

    def __init__(
        self,
        in_channels: int = 80,
        base_channels: int = 512,
        nb_harmonics: int = 8,
        sampling_rate: int = 22050,
        nsf_alpha: float = 0.1,
        nsf_sigma: float = 0.003,
        nsf_voiced_threshold: float = 10,
        upsample_rates: List[int] = [8, 8],
        upsample_kernel_sizes: List[int] = [16, 16],
        istft_params: Dict[str, int] = {"n_fft": 16, "hop_len": 4},
        resblock_kernel_sizes: List[int] = [3, 7, 11],
        resblock_dilation_sizes: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        source_resblock_kernel_sizes: List[int] = [7, 11],
        source_resblock_dilation_sizes: List[List[int]] = [[1, 3, 5], [1, 3, 5]],
        lrelu_slope: float = 0.1,
        audio_limit: float = 0.99,
        f0_predictor: Optional[nn.Module] = None,
    ):
        """
        Args:
            in_channels: Number of input mel channels
            base_channels: Base number of channels
            nb_harmonics: Number of harmonics for NSF
            sampling_rate: Audio sampling rate
            nsf_alpha: NSF sine amplitude
            nsf_sigma: NSF noise std
            nsf_voiced_threshold: F0 threshold for voiced/unvoiced
            upsample_rates: Upsampling factors
            upsample_kernel_sizes: Upsampling kernel sizes
            istft_params: ISTFT parameters (n_fft, hop_len)
            resblock_kernel_sizes: ResBlock kernel sizes
            resblock_dilation_sizes: ResBlock dilations
            source_resblock_kernel_sizes: Source ResBlock kernel sizes
            source_resblock_dilation_sizes: Source ResBlock dilations
            lrelu_slope: LeakyReLU negative slope
            audio_limit: Audio clipping limit
            f0_predictor: Optional F0 prediction module
        """
        super().__init__()

        self.out_channels = 1
        self.nb_harmonics = nb_harmonics
        self.sampling_rate = sampling_rate
        self.istft_params = istft_params
        self.lrelu_slope = lrelu_slope
        self.audio_limit = audio_limit

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        upsample_scale = math.prod(upsample_rates) * istft_params["hop_len"]

        # Neural Source Filter
        self.m_source = SourceModuleHnNSF(
            sampling_rate=sampling_rate,
            upsample_scale=upsample_scale,
            harmonic_num=nb_harmonics,
            sine_amp=nsf_alpha,
            add_noise_std=nsf_sigma,
            voiced_threshod=nsf_voiced_threshold,
        )

        # F0 upsampler
        self.f0_upsample_scale = upsample_scale

        # Pre-convolution
        self.conv_pre = nn.Conv1d(in_channels, base_channels, 7, stride=1, padding=3)

        # Upsampling layers
        self.ups = []
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                nn.ConvTranspose1d(
                    base_channels // (2**i),
                    base_channels // (2 ** (i + 1)),
                    k,
                    stride=u,
                    padding=(k - u) // 2,
                )
            )

        # Source downsampling and resblocks for frequency domain fusion
        self.source_downs = []
        self.source_resblocks = []
        downsample_rates = [1] + upsample_rates[::-1][:-1]
        # Compute cumulative product manually (replaces np.cumprod)
        downsample_cum_rates = []
        cum_prod = 1
        for rate in downsample_rates:
            cum_prod *= rate
            downsample_cum_rates.append(cum_prod)

        for i, (u, k, d) in enumerate(
            zip(
                downsample_cum_rates[::-1],
                source_resblock_kernel_sizes,
                source_resblock_dilation_sizes,
            )
        ):
            if u == 1:
                self.source_downs.append(
                    nn.Conv1d(
                        istft_params["n_fft"] + 2,
                        base_channels // (2 ** (i + 1)),
                        1,
                        stride=1,
                    )
                )
            else:
                self.source_downs.append(
                    nn.Conv1d(
                        istft_params["n_fft"] + 2,
                        base_channels // (2 ** (i + 1)),
                        u * 2,
                        stride=u,
                        padding=u // 2,
                    )
                )
            self.source_resblocks.append(
                ResBlock(base_channels // (2 ** (i + 1)), k, d)
            )

        # Residual blocks after each upsampling
        self.resblocks = []
        for i in range(len(self.ups)):
            ch = base_channels // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(ResBlock(ch, k, d))

        # Post-convolution (outputs to frequency domain)
        ch = base_channels // (2 ** len(self.ups))
        self.conv_post = nn.Conv1d(
            ch, istft_params["n_fft"] + 2, 7, stride=1, padding=3
        )

        # STFT window (Hann window - periodic, matching scipy fftbins=True)
        self.stft_window = hann_window_periodic(istft_params["n_fft"])

        self.f0_predictor = f0_predictor

    def _f0_upsample(self, f0: mx.array) -> mx.array:
        """Upsample F0 using nearest neighbor interpolation."""
        # f0 shape: (B, 1, T)
        # Use mx.repeat for efficient nearest-neighbor upsampling
        return mx.repeat(f0, self.f0_upsample_scale, axis=2)

    def _stft(self, x: mx.array) -> tuple:
        """Perform STFT on input signal."""
        return stft(
            x,
            self.istft_params["n_fft"],
            self.istft_params["hop_len"],
            self.stft_window,
        )

    def _istft(self, magnitude: mx.array, phase: mx.array) -> mx.array:
        """Perform inverse STFT."""
        return istft(
            magnitude,
            phase,
            self.istft_params["n_fft"],
            self.istft_params["hop_len"],
            self.stft_window,
        )

    def decode(self, x: mx.array, s: mx.array) -> mx.array:
        """
        Decode mel-spectrogram to waveform.

        Args:
            x: Mel-spectrogram (B, C, T)
            s: Source signal (B, 1, T_s)

        Returns:
            Generated waveform (B, T)
        """
        # STFT of source signal
        s_stft_real, s_stft_imag = self._stft(s.squeeze(1))
        s_stft = mx.concatenate([s_stft_real, s_stft_imag], axis=1)

        # Pre-convolution: (B, C, T) -> transpose -> conv -> transpose back
        x = mx.swapaxes(x, 1, 2)  # (B, C, T) -> (B, T, C)
        x = self.conv_pre(x)
        x = mx.swapaxes(x, 1, 2)  # (B, T, C) -> (B, C, T)

        for i in range(self.num_upsamples):
            x = nn.leaky_relu(x, negative_slope=self.lrelu_slope)
            # ConvTranspose1d: (B, C, T) -> transpose -> conv -> transpose back
            x = mx.swapaxes(x, 1, 2)  # (B, C, T) -> (B, T, C)
            x = self.ups[i](x)
            x = mx.swapaxes(x, 1, 2)  # (B, T, C) -> (B, C, T)

            if i == self.num_upsamples - 1:
                # Reflection pad: pad 1 sample on left using reflection
                # For (1, 0) reflection: prepend x[:, :, 1:2] to the beginning
                x = mx.concatenate([x[:, :, 1:2], x], axis=2)

            # Source fusion: (B, C, T) -> transpose -> conv -> transpose back
            si = mx.swapaxes(s_stft, 1, 2)  # (B, C, T) -> (B, T, C)
            si = self.source_downs[i](si)
            si = mx.swapaxes(si, 1, 2)  # (B, T, C) -> (B, C, T)
            si = self.source_resblocks[i](si)
            x = x + si

            # Apply residual blocks
            xs = None
            for j in range(self.num_kernels):
                idx = i * self.num_kernels + j
                if xs is None:
                    xs = self.resblocks[idx](x)
                else:
                    xs = xs + self.resblocks[idx](x)
            x = xs / self.num_kernels

        x = nn.leaky_relu(x, negative_slope=self.lrelu_slope)
        # conv_post: (B, C, T) -> transpose -> conv -> transpose back
        x = mx.swapaxes(x, 1, 2)  # (B, C, T) -> (B, T, C)
        x = self.conv_post(x)
        x = mx.swapaxes(x, 1, 2)  # (B, T, C) -> (B, C, T)

        # Split into magnitude and phase
        n_fft_half = self.istft_params["n_fft"] // 2 + 1
        magnitude = mx.exp(x[:, :n_fft_half, :])
        phase = mx.sin(x[:, n_fft_half:, :])  # sin is redundancy, matches original

        # Inverse STFT
        x = self._istft(magnitude, phase)
        x = mx.clip(x, -self.audio_limit, self.audio_limit)

        return x

    def __call__(self, speech_feat: mx.array, cache_source: mx.array = None) -> tuple:
        """
        Generate waveform from mel-spectrogram.

        Args:
            speech_feat: Mel-spectrogram (B, C, T)
            cache_source: Cached source for streaming

        Returns:
            Tuple of (waveform, source)
        """
        if cache_source is None:
            cache_source = mx.zeros((1, 1, 0))

        # Predict F0 from mel
        f0 = self.f0_predictor(speech_feat)

        # Upsample F0
        s = self._f0_upsample(mx.expand_dims(f0, 1))
        s = mx.swapaxes(s, 1, 2)  # (B, T, 1)

        # Generate source from F0
        s, _, _ = self.m_source(s)
        s = mx.swapaxes(s, 1, 2)  # (B, 1, T)

        # Use cache to avoid glitch in streaming
        if cache_source.shape[2] != 0:
            cache_len = cache_source.shape[2]
            s = mx.concatenate([cache_source, s[:, :, cache_len:]], axis=2)

        # Decode mel + source to audio
        generated_speech = self.decode(x=speech_feat, s=s)

        return generated_speech, s

    def inference(self, speech_feat: mx.array, cache_source: mx.array = None) -> tuple:
        """Inference-mode forward pass."""
        return self(speech_feat, cache_source)
