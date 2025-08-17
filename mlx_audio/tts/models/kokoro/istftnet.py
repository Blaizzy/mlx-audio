import math
from typing import List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_audio.utils import istft, stft

from ..base import check_array_shape
from ..interpolate import interpolate


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


def compute_norm(
    x: mx.array,
    p: int,
    dim: Optional[Union[int, List[int]]] = None,
    keepdim: bool = False,
) -> mx.array:
    """
    Compute the p-norm of a tensor along specified dimensions.

    Args:
        x: Input array
        p: Order of the norm (1 or 2)
        dim: Dimension(s) along which to compute the norm
        keepdim: Whether to keep the reduced dimensions

    Returns:
        MLX array containing the computed norm
    """
    if p not in [1, 2]:
        raise ValueError("Only p-norms with p of 1 or 2 are supported")

    # Handle dimension input
    if dim is None:
        dim = tuple(range(x.ndim))
    elif isinstance(dim, int):
        dim = (dim,)

    if p == 1:
        # L1 norm
        return mx.sum(mx.abs(x), axis=dim, keepdims=keepdim)
    else:
        # L2 norm
        return mx.sqrt(mx.sum(x * x, axis=dim, keepdims=keepdim))


def weight_norm(
    weight_v: mx.array, weight_g: mx.array, dim: Optional[int] = None
) -> mx.array:
    """
    Applies weight normalization to the input tensor.

    Weight normalization reparameterizes weight vectors in a neural network
    as a magnitude scalar times a direction vector: w = g * v/||v||

    Args:
        weight_v: Weight direction tensor (v)
        weight_g: Weight magnitude tensor (g)
        dim: Dimension along which to normalize. If None, normalize over all dims
            except dim=-1

    Returns:
        Normalized weight tensor
    """
    rank = len(weight_v.shape)

    if dim is not None:
        # Adjust negative dim
        if dim < -1:
            dim += rank

        # Create list of axes to normalize over
        axes = list(range(rank))
        if dim != -1:
            axes.remove(dim)
    else:
        # Default behavior: normalize over all dimensions
        axes = list(range(rank))

    # Compute L2 norm of v along specified axes
    norm_v = compute_norm(weight_v, p=2, dim=axes, keepdim=True)

    # Normalize and scale by g: w = g * (v / ||v||)
    normalized_weight = weight_v / (
        norm_v + 1e-7
    )  # Add epsilon for numerical stability
    return normalized_weight * weight_g


class ConvWeighted(nn.Module):
    """Conv1d with weight normalization"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        encode: bool = False,
    ):
        super().__init__()

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # Initialize weight magnitude (g) and direction (v) vectors
        self.weight_g = mx.ones(
            (out_channels, 1, 1)
        )  # Scalar magnitude per output channel
        self.weight_v = mx.ones(
            (out_channels, kernel_size, in_channels)
        )  # Direction vectors

        self.bias = mx.zeros(in_channels if encode else out_channels) if bias else None

    def __call__(self, x, conv):

        weight = weight_norm(self.weight_v, self.weight_g, dim=0)

        if self.bias is not None:
            bias = self.bias.reshape(1, 1, -1)
        else:
            bias = None

        def apply_conv(x, weight_to_use):
            if self.bias is not None:
                return (
                    conv(
                        x,
                        weight_to_use,
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation,
                        groups=self.groups,
                    )
                    + bias
                )
            return conv(
                x,
                weight_to_use,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )

        try:
            # Check if channels last match or if groups > 1 for ConvTransposed1d
            if x.shape[-1] == weight.shape[-1] or self.groups > 1:
                # Input is channels first, use weight as-is
                return apply_conv(x, weight)
            else:
                # Input is channels last, need to transpose weight
                return apply_conv(x, weight.T)
        except Exception as e:
            print(f"Error: {e}")
            print(f"x.shape: {x.shape}, weight.shape: {weight.shape}")
            raise e


class _InstanceNorm(nn.Module):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        # Initialize parameters
        if self.affine:
            self.weight = mx.ones((num_features,))
            self.bias = mx.zeros((num_features,))
        else:
            self.weight = None
            self.bias = None

        if self.track_running_stats:
            self.running_mean = mx.zeros((num_features,))
            self.running_var = mx.ones((num_features,))
        else:
            self.running_mean = None
            self.running_var = None

    def _check_input_dim(self, input):
        raise NotImplementedError

    def _get_no_batch_dim(self):
        raise NotImplementedError

    def _handle_no_batch_input(self, input):
        # Add batch dimension, apply norm, then remove batch dimension
        expanded = mx.expand_dims(input, axis=0)
        result = self._apply_instance_norm(expanded)
        return mx.squeeze(result, axis=0)

    def _apply_instance_norm(self, input):
        # MLX doesn't have a direct instance_norm function like PyTorch
        # So we need to implement it manually

        # Get dimensions
        dims = list(range(input.ndim))
        feature_dim = dims[-self._get_no_batch_dim()]

        # Compute statistics along all dims except batch and feature dims
        reduce_dims = [d for d in dims if d != 0 and d != feature_dim]

        if self.training or not self.track_running_stats:
            # Compute mean and variance for normalization
            mean = mx.mean(input, axis=reduce_dims, keepdims=True)
            var = mx.var(input, axis=reduce_dims, keepdims=True)

            # Update running stats if tracking
            if self.track_running_stats and self.training:
                # Compute overall mean and variance (across batch too)
                overall_mean = mx.mean(mean, axis=0)
                overall_var = mx.mean(var, axis=0)

                # Update running statistics
                self.running_mean = (
                    1 - self.momentum
                ) * self.running_mean + self.momentum * overall_mean
                self.running_var = (
                    1 - self.momentum
                ) * self.running_var + self.momentum * overall_var
        else:
            # Use running statistics
            mean_shape = [1] * input.ndim
            mean_shape[feature_dim] = self.num_features
            var_shape = mean_shape.copy()

            mean = mx.reshape(self.running_mean, mean_shape)
            var = mx.reshape(self.running_var, var_shape)

        # Normalize
        x_norm = (input - mean) / mx.sqrt(var + self.eps)

        # Apply affine transform if needed
        if self.affine:
            weight_shape = [1] * input.ndim
            weight_shape[feature_dim] = self.num_features
            bias_shape = weight_shape.copy()

            weight = mx.reshape(self.weight, weight_shape)
            bias = mx.reshape(self.bias, bias_shape)

            return x_norm * weight + bias
        else:
            return x_norm

    def __call__(self, input):
        self._check_input_dim(input)

        feature_dim = input.ndim - self._get_no_batch_dim()
        if input.shape[feature_dim] != self.num_features:
            if self.affine:
                raise ValueError(
                    f"expected input's size at dim={feature_dim} to match num_features"
                    f" ({self.num_features}), but got: {input.shape[feature_dim]}."
                )
            else:
                print(
                    f"input's size at dim={feature_dim} does not match num_features. "
                    "You can silence this warning by not passing in num_features, "
                    "which is not used because affine=False"
                )

        if input.ndim == self._get_no_batch_dim():
            return self._handle_no_batch_input(input)

        return self._apply_instance_norm(input)


class InstanceNorm1d(_InstanceNorm):
    """Applies Instance Normalization over a 2D (unbatched) or 3D (batched) input.

    This implementation follows the algorithm described in the paper
    "Instance Normalization: The Missing Ingredient for Fast Stylization".

    Args:
        num_features: Number of features or channels (C) of the input
        eps: A value added to the denominator for numerical stability. Default: 1e-5
        momentum: The value used for the running_mean and running_var computation. Default: 0.1
        affine: When True, this module has learnable affine parameters. Default: False
        track_running_stats: When True, this module tracks running statistics. Default: False

    Shape:
        - Input: (N, C, L) or (C, L)
        - Output: Same shape as input

    Examples:
        >>> # Without Learnable Parameters
        >>> m = nn.InstanceNorm1d(100)
        >>> # With Learnable Parameters
        >>> m = nn.InstanceNorm1d(100, affine=True)
        >>> input = mx.random.normal((20, 100, 40))
        >>> output = m(input)
    """

    def _get_no_batch_dim(self):
        return 2

    def _check_input_dim(self, input):
        if input.ndim not in (2, 3):
            raise ValueError(f"expected 2D or 3D input (got {input.ndim}D input)")


class AdaIN1d(nn.Module):
    def __init__(self, style_dim: int, num_features: int):
        super().__init__()
        self.norm = InstanceNorm1d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def __call__(self, x: mx.array, s: mx.array) -> mx.array:
        h = self.fc(s)
        h = mx.expand_dims(h, axis=2)  # Equivalent to view(..., 1)
        gamma, beta = mx.split(h, 2, axis=1)
        x = (1 + gamma) * self.norm(x) + beta
        return x


class AdaINResBlock1(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: Tuple[int, int, int] = (1, 3, 5),
        style_dim: int = 64,
    ):
        super(AdaINResBlock1, self).__init__()
        self.convs1 = [
            ConvWeighted(
                channels,
                channels,
                kernel_size,
                stride=1,
                padding=get_padding(kernel_size, dilation[i]),
                dilation=dilation[i],
            )
            for i in range(3)
        ]
        self.convs2 = [
            ConvWeighted(
                channels,
                channels,
                kernel_size,
                stride=1,
                padding=get_padding(kernel_size, 1),
                dilation=1,
            )
            for _ in range(3)
        ]
        self.adain1 = [AdaIN1d(style_dim, channels) for _ in range(3)]
        self.adain2 = [AdaIN1d(style_dim, channels) for _ in range(3)]
        self.alpha1 = [mx.ones((1, channels, 1)) for _ in range(len(self.convs1))]
        self.alpha2 = [mx.ones((1, channels, 1)) for _ in range(len(self.convs2))]

    def __call__(self, x: mx.array, s: mx.array) -> mx.array:
        for c1, c2, n1, n2, a1, a2 in zip(
            self.convs1, self.convs2, self.adain1, self.adain2, self.alpha1, self.alpha2
        ):
            xt = n1(x, s)
            xt = xt + (1 / a1) * (mx.sin(a1 * xt) ** 2)  # Snake1D

            xt = xt.swapaxes(2, 1)
            xt = c1(xt, mx.conv1d)
            xt = xt.swapaxes(2, 1)

            xt = n2(xt, s)
            xt = xt + (1 / a2) * (mx.sin(a2 * xt) ** 2)  # Snake1D

            xt = xt.swapaxes(2, 1)
            xt = c2(xt, mx.conv1d)
            xt = xt.swapaxes(2, 1)

            x = xt + x
        return x


def mlx_angle(z, deg=False):
    z = mx.array(z)

    if z.dtype == mx.complex64:
        zimag = mx.imag(z)
        zreal = mx.real(z)
    else:
        zimag = mx.zeros_like(z)
        zreal = z

    a = mx.arctan2(zimag, zreal)

    if deg:
        a = a * (180.0 / math.pi)

    return a


def mlx_unwrap(p, discont=None, axis=-1, period=2 * math.pi):
    if discont is None:
        discont = period / 2

    discont = max(discont, period / 2)

    slice_indices = [slice(None)] * p.ndim

    slice_indices[axis] = slice(1, None)
    after_slice = tuple(slice_indices)

    slice_indices[axis] = slice(None, -1)
    before_slice = tuple(slice_indices)

    dd = p[after_slice] - p[before_slice]

    interval_high = period / 2
    interval_low = -interval_high

    ddmod = dd - period * mx.floor((dd - interval_low) / period)
    ddmod = mx.where(
        (mx.abs(dd - interval_high) < 1e-10) & (dd > 0), interval_high, ddmod
    )

    ph_correct = ddmod - dd
    ph_correct = mx.where(mx.abs(dd) < discont, 0, ph_correct)

    padding_shape = list(ph_correct.shape)
    padding_shape[axis] = 1
    zero_padding = mx.zeros(padding_shape)
    padded_corrections = mx.concatenate([zero_padding, ph_correct], axis=axis)
    cumulative_corrections = mx.cumsum(padded_corrections, axis=axis)

    return p + cumulative_corrections


class MLXSTFT:
    def __init__(
        self, filter_length=800, hop_length=200, win_length=800, window="hann"
    ):
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length

        self.window = window

    def transform(self, input_data):
        # Ensure 2D
        if input_data.ndim == 1:
            input_data = input_data[None, :]

        magnitudes = []
        phases = []

        for batch_idx in range(input_data.shape[0]):
            # Compute STFT
            x_stft = stft(
                input_data[batch_idx],
                n_fft=self.filter_length,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                center=True,
                pad_mode="reflect",
            ).transpose(1, 0)

            # Get magnitude
            magnitude = mx.abs(x_stft)

            # Get phase
            phase = mlx_angle(x_stft)

            magnitudes.append(magnitude)
            phases.append(phase)

        magnitudes = mx.stack(magnitudes, axis=0)
        phases = mx.stack(phases, axis=0)

        return magnitudes, phases

    def inverse(self, magnitude, phase):
        reconstructed = []

        for batch_idx in range(magnitude.shape[0]):
            # Unwrap phases for reconstruction
            phase_cont = mlx_unwrap(phase[batch_idx], axis=1)

            # Combine magnitude and phase
            real_part = magnitude[batch_idx] * mx.cos(phase_cont)
            imag_part = magnitude[batch_idx] * mx.sin(phase_cont)
            x_stft = real_part + 1j * imag_part

            # Inverse STFT
            audio = istft(
                x_stft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                center=True,
                length=None,
            )

            reconstructed.append(audio)

        reconstructed = mx.stack(reconstructed, axis=0)[:, None, :]

        return reconstructed

    def __call__(self, input_data: mx.array) -> mx.array:
        self.magnitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return mx.expand_dims(reconstruction, axis=-2)


class SineGen:
    def __init__(
        self,
        samp_rate: int,
        upsample_scale: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        noise_std: float = 0.003,
        voiced_threshold: float = 0,
        flag_for_pulse: bool = False,
    ):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold
        self.flag_for_pulse = flag_for_pulse
        self.upsample_scale = upsample_scale

    def _f02uv(self, f0: mx.array) -> mx.array:
        return mx.array(f0 > self.voiced_threshold, dtype=mx.float32)

    def _f02sine(self, f0_values: mx.array) -> mx.array:
        """f0_values: (batchsize, length, dim)
        where dim indicates fundamental tone and overtones
        """
        # convert to F0 in rad. The interger part n can be ignored
        # because 2 * np.pi * n doesn't affect phase
        rad_values = (f0_values / self.sampling_rate) % 1
        # initial phase noise (no noise for fundamental component)
        rand_ini = mx.random.normal((f0_values.shape[0], f0_values.shape[2]))
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini
        # instantanouse phase sine[t] = sin(2*pi \sum_i=1 ^{t} rad)
        if not self.flag_for_pulse:
            rad_values = interpolate(
                rad_values.transpose(0, 2, 1),
                scale_factor=1 / self.upsample_scale,
                mode="linear",
            ).transpose(0, 2, 1)
            phase = mx.cumsum(rad_values, axis=1) * 2 * mx.pi
            phase = interpolate(
                phase.transpose(0, 2, 1) * self.upsample_scale,
                scale_factor=self.upsample_scale,
                mode="linear",
            ).transpose(0, 2, 1)
            sines = mx.sin(phase)
        else:
            # If necessary, make sure that the first time step of every
            # voiced segments is sin(pi) or cos(0)
            # This is used for pulse-train generation
            # identify the last time step in unvoiced segments
            uv = self._f02uv(f0_values)
            uv_1 = mx.roll(uv, shifts=-1, axis=1)
            uv_1[:, -1, :] = 1
            u_loc = (uv < 1) * (uv_1 > 0)
            # get the instantanouse phase
            tmp_cumsum = mx.cumsum(rad_values, axis=1)
            # different batch needs to be processed differently
            for idx in range(f0_values.shape[0]):
                temp_sum = tmp_cumsum[idx, u_loc[idx, :, 0], :]
                temp_sum[1:, :] = temp_sum[1:, :] - temp_sum[0:-1, :]
                # stores the accumulation of i.phase within
                # each voiced segments
                tmp_cumsum[idx, :, :] = 0
                tmp_cumsum[idx, u_loc[idx, :, 0], :] = temp_sum
            # rad_values - tmp_cumsum: remove the accumulation of i.phase
            # within the previous voiced segment.
            i_phase = mx.cumsum(rad_values - tmp_cumsum, axis=1)
            # get the sines
            sines = mx.cos(i_phase * 2 * mx.pi)
        return sines

    def __call__(self, f0: mx.array) -> Tuple[mx.array, mx.array, mx.array]:
        f0_buf = mx.zeros((f0.shape[0], f0.shape[1], self.dim))

        # Fundamental component
        fn = f0 * mx.arange(1, self.harmonic_num + 2)[None, None, :]

        # Generate sine waveforms
        sine_waves = self._f02sine(fn) * self.sine_amp

        # Generate UV signal
        uv = self._f02uv(f0)

        # Generate noise
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * mx.random.normal(sine_waves.shape)

        sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise


class SourceModuleHnNSF(nn.Module):
    """SourceModule for hn-nsf
    SourceModule(sampling_rate, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshod=0)
    sampling_rate: sampling_rate in Hz
    harmonic_num: number of harmonic above F0 (default: 0)
    sine_amp: amplitude of sine source signal (default: 0.1)
    add_noise_std: std of additive Gaussian noise (default: 0.003)
        note that amplitude of noise in unvoiced is decided
        by sine_amp
    voiced_threshold: threhold to set U/V given F0 (default: 0)
    Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
    F0_sampled (batchsize, length, 1)
    Sine_source (batchsize, length, 1)
    noise_source (batchsize, length 1)
    uv (batchsize, length, 1)
    """

    def __init__(
        self,
        sampling_rate,
        upsample_scale,
        harmonic_num=0,
        sine_amp=0.1,
        add_noise_std=0.003,
        voiced_threshod=0,
    ):
        super(SourceModuleHnNSF, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        # to produce sine waveforms
        self.l_sin_gen = SineGen(
            sampling_rate,
            upsample_scale,
            harmonic_num,
            sine_amp,
            add_noise_std,
            voiced_threshod,
        )
        # to merge source harmonics into a single excitation
        self.l_linear = nn.Linear(harmonic_num + 1, 1)

    def __call__(self, x):
        """
        Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
        F0_sampled (batchsize, length, 1)
        Sine_source (batchsize, length, 1)
        noise_source (batchsize, length 1)
        """
        # source for harmonic branch
        sine_wavs, uv, _ = self.l_sin_gen(x)
        sine_merge = mx.tanh(self.l_linear(sine_wavs))
        # source for noise branch, in the same shape as uv
        noise = mx.random.normal(uv.shape) * self.sine_amp / 3
        return sine_merge, noise, uv


class ReflectionPad1d(nn.Module):
    def __init__(self, padding):
        super().__init__()
        self.padding = padding

    def __call__(self, x):
        return mx.pad(x, ((0, 0), (0, 0), (self.padding[0], self.padding[1])))


def leaky_relu(x, negative_slope=0.01):
    return mx.where(x > 0, x, x * negative_slope)


class Generator(nn.Module):
    def __init__(
        self,
        style_dim,
        resblock_kernel_sizes,
        upsample_rates,
        upsample_initial_channel,
        resblock_dilation_sizes,
        upsample_kernel_sizes,
        gen_istft_n_fft,
        gen_istft_hop_size,
    ):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        upsample_rates = mx.array(upsample_rates)
        self.m_source = SourceModuleHnNSF(
            sampling_rate=24000,
            upsample_scale=mx.prod(upsample_rates) * gen_istft_hop_size,
            harmonic_num=8,
            voiced_threshod=10,
        )
        self.f0_upsamp = nn.Upsample(
            scale_factor=mx.prod(upsample_rates) * gen_istft_hop_size
        )
        self.noise_convs = []
        self.noise_res = []
        self.ups = []
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                ConvWeighted(
                    upsample_initial_channel // (2 ** (i + 1)),
                    upsample_initial_channel // (2**i),
                    int(k),
                    int(u),
                    padding=int((k - u) // 2),
                    encode=True,
                )
            )
        self.resblocks = []
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(AdaINResBlock1(ch, k, d, style_dim))
            c_cur = upsample_initial_channel // (2 ** (i + 1))
            if i + 1 < len(upsample_rates):
                stride_f0 = int(mx.prod(upsample_rates[i + 1 :]))
                self.noise_convs.append(
                    nn.Conv1d(
                        gen_istft_n_fft + 2,
                        c_cur,
                        kernel_size=stride_f0 * 2,
                        stride=stride_f0,
                        padding=(stride_f0 + 1) // 2,
                    )
                )
                self.noise_res.append(AdaINResBlock1(c_cur, 7, [1, 3, 5], style_dim))
            else:
                self.noise_convs.append(
                    nn.Conv1d(gen_istft_n_fft + 2, c_cur, kernel_size=1)
                )
                self.noise_res.append(AdaINResBlock1(c_cur, 11, [1, 3, 5], style_dim))
        self.post_n_fft = gen_istft_n_fft
        self.conv_post = ConvWeighted(ch, self.post_n_fft + 2, 7, 1, padding=3)
        self.reflection_pad = ReflectionPad1d((1, 0))
        self.stft = MLXSTFT(
            filter_length=gen_istft_n_fft,
            hop_length=gen_istft_hop_size,
            win_length=gen_istft_n_fft,
        )

    def __call__(self, x, s, f0):
        f0 = self.f0_upsamp(f0[:, None].transpose(0, 2, 1))  # bs,n,t
        har_source, noi_source, uv = self.m_source(f0)
        har_source = mx.squeeze(har_source.transpose(0, 2, 1), axis=1)
        har_spec, har_phase = self.stft.transform(har_source)
        har = mx.concatenate([har_spec, har_phase], axis=1)
        har = har.swapaxes(2, 1)
        for i in range(self.num_upsamples):
            x = leaky_relu(x, negative_slope=0.1)
            x_source = self.noise_convs[i](har)
            x_source = x_source.swapaxes(2, 1)
            x_source = self.noise_res[i](x_source, s)

            x = x.swapaxes(2, 1)
            x = self.ups[i](x, mx.conv_transpose1d)
            x = x.swapaxes(2, 1)

            if i == self.num_upsamples - 1:
                x = self.reflection_pad(x)
            x = x + x_source

            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x, s)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x, s)
            x = xs / self.num_kernels

        x = leaky_relu(x, negative_slope=0.01)

        x = x.swapaxes(2, 1)
        x = self.conv_post(x, mx.conv1d)
        x = x.swapaxes(2, 1)

        spec = mx.exp(x[:, : self.post_n_fft // 2 + 1, :])
        phase = mx.sin(x[:, self.post_n_fft // 2 + 1 :, :])
        result = self.stft.inverse(spec, phase)
        return result


class UpSample1d(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type
        self.interpolate = nn.Upsample(
            scale_factor=2, mode="nearest", align_corners=True
        )

    def __call__(self, x):
        if self.layer_type == "none":
            return x
        else:
            return self.interpolate(x)


class AdainResBlk1d(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        style_dim=64,
        actv=nn.LeakyReLU(0.2),
        upsample="none",
        dropout_p=0.0,
        bias=False,
        conv_type=None,
    ):
        super().__init__()
        self.actv = actv
        self.dim_in = dim_in
        self.conv_type = conv_type
        self.upsample_type = upsample
        self.upsample = UpSample1d(upsample)
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)
        self.dropout = nn.Dropout(dropout_p)
        if upsample == "none":
            self.pool = nn.Identity()
        else:
            self.pool = ConvWeighted(
                1, dim_in, kernel_size=3, stride=2, padding=1, groups=dim_in
            )

    def _build_weights(self, dim_in, dim_out, style_dim):
        self.conv1 = ConvWeighted(dim_in, dim_out, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvWeighted(dim_out, dim_out, kernel_size=3, stride=1, padding=1)
        self.norm1 = AdaIN1d(style_dim, dim_in)
        self.norm2 = AdaIN1d(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = ConvWeighted(
                dim_in, dim_out, kernel_size=1, stride=1, padding=0, bias=False
            )

    def _shortcut(self, x):
        x = x.swapaxes(2, 1)
        x = self.upsample(x)
        x = x.swapaxes(2, 1)

        if self.learned_sc:
            x = x.swapaxes(2, 1)
            x = self.conv1x1(x, mx.conv1d)
            x = x.swapaxes(2, 1)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)

        # Manually implement grouped ConvTranspose1d since MLX doesn't support groups
        x = x.swapaxes(2, 1)
        x = self.pool(x, mx.conv_transpose1d) if self.upsample_type != "none" else x
        x = mx.pad(x, ((0, 0), (1, 0), (0, 0))) if self.upsample_type != "none" else x
        x = x.swapaxes(2, 1)

        x = x.swapaxes(2, 1)
        x = self.conv1(self.dropout(x), mx.conv1d)
        x = x.swapaxes(2, 1)

        x = self.norm2(x, s)
        x = self.actv(x)

        x = x.swapaxes(2, 1)
        x = self.conv2(x, mx.conv1d)
        x = x.swapaxes(2, 1)
        return x

    def __call__(self, x, s):
        out = self._residual(x, s)
        out = (out + self._shortcut(x)) / mx.sqrt(2)
        return out


class Decoder(nn.Module):
    def __init__(
        self,
        dim_in,
        style_dim,
        dim_out,
        resblock_kernel_sizes,
        upsample_rates,
        upsample_initial_channel,
        resblock_dilation_sizes,
        upsample_kernel_sizes,
        gen_istft_n_fft,
        gen_istft_hop_size,
    ):
        super().__init__()
        self.encode = AdainResBlk1d(dim_in + 2, 1024, style_dim, conv_type=mx.conv1d)
        self.decode = []
        self.decode.append(
            AdainResBlk1d(1024 + 2 + 64, 1024, style_dim, conv_type=mx.conv1d)
        )
        self.decode.append(
            AdainResBlk1d(1024 + 2 + 64, 1024, style_dim, conv_type=mx.conv1d)
        )
        self.decode.append(
            AdainResBlk1d(1024 + 2 + 64, 1024, style_dim, conv_type=mx.conv1d)
        )
        self.decode.append(
            AdainResBlk1d(
                1024 + 2 + 64, 512, style_dim, upsample=True, conv_type=mx.conv1d
            )
        )
        self.F0_conv = ConvWeighted(1, 1, kernel_size=3, stride=2, padding=1, groups=1)
        self.N_conv = ConvWeighted(1, 1, kernel_size=3, stride=2, padding=1, groups=1)
        self.asr_res = [ConvWeighted(512, 64, kernel_size=1, padding=0)]
        self.generator = Generator(
            style_dim,
            resblock_kernel_sizes,
            upsample_rates,
            upsample_initial_channel,
            resblock_dilation_sizes,
            upsample_kernel_sizes,
            gen_istft_n_fft,
            gen_istft_hop_size,
        )

    def __call__(self, asr, F0_curve, N, s):
        s = mx.array(s)
        F0 = self.F0_conv(F0_curve[:, None, :].swapaxes(2, 1), mx.conv1d).swapaxes(2, 1)
        N = self.N_conv(N[:, None, :].swapaxes(2, 1), mx.conv1d).swapaxes(2, 1)
        x = mx.concatenate([asr, F0, N], axis=1)
        x = self.encode(x, s)
        asr_res = self.asr_res[0](asr.swapaxes(2, 1), mx.conv1d).swapaxes(2, 1)
        res = True
        for block in self.decode:  # Working in MLX
            if res:
                x = mx.concatenate([x, asr_res, F0, N], axis=1)
            x = block(x, s)
            # Check if this block has upsampling
            if hasattr(block, "upsample_type") and block.upsample_type != "none":
                res = False
        x = self.generator(x, s, F0_curve)  # Working in MLX
        return x

    def sanitize(self, key, weights):
        sanitized_weights = None
        if "noise_convs" in key and key.endswith(".weight"):
            sanitized_weights = weights.transpose(0, 2, 1)

        elif "weight_v" in key:
            if check_array_shape(weights):
                sanitized_weights = weights
            else:
                sanitized_weights = weights.transpose(0, 2, 1)

        else:
            sanitized_weights = weights

        return sanitized_weights
