"""
MLX implementation of mel-filterbank feature extraction
Compatible with torchaudio.compliance.kaldi.fbank
"""

from typing import Tuple

import mlx.core as mx


def _next_power_of_2(x: int) -> int:
    """Returns the smallest power of 2 that is greater than x"""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def _get_strided(
    waveform: mx.array, window_size: int, window_shift: int, snip_edges: bool
) -> mx.array:
    """Given a waveform, returns frames by shifting window along the waveform."""
    num_samples = waveform.shape[0]

    if snip_edges:
        if num_samples < window_size:
            return mx.zeros((0, 0))
        else:
            m = 1 + (num_samples - window_size) // window_shift
    else:
        # Reflect padding
        m = (num_samples + (window_shift // 2)) // window_shift
        pad = window_size // 2 - window_shift // 2

        if pad > 0:
            # Reflect padding on both sides
            pad_left = waveform[1 : pad + 1][::-1]
            pad_right = waveform[-1 : -pad - 1 : -1] if pad > 1 else waveform[-1:0:-1]
            waveform = mx.concatenate([pad_left, waveform, pad_right])
        else:
            # Negative pad means we trim from the front
            pad_right = waveform[::-1]
            waveform = mx.concatenate([waveform[-pad:], pad_right])

    # Create frames using as_strided
    frames = mx.as_strided(waveform, shape=(m, window_size), strides=(window_shift, 1))

    return frames


def _feature_window_function(
    window_type: str, window_size: int, periodic: bool = False
) -> mx.array:
    """Returns a window function with the given type and size"""
    if window_type == "hanning":
        n = mx.arange(window_size)
        window = 0.5 - 0.5 * mx.cos(
            2 * mx.pi * n / (window_size if periodic else window_size - 1)
        )
    elif window_type == "hamming":
        n = mx.arange(window_size)
        window = 0.54 - 0.46 * mx.cos(
            2 * mx.pi * n / (window_size if periodic else window_size - 1)
        )
    elif window_type == "povey":
        # Like Hanning but goes to zero at edges
        n = mx.arange(window_size)
        hann = 0.5 - 0.5 * mx.cos(2 * mx.pi * n / (window_size - 1))
        window = mx.power(hann, 0.85)
    elif window_type == "rectangular":
        window = mx.ones(window_size)
    else:
        raise ValueError(f"Invalid window type: {window_type}")

    return window


def mel_scale(freq: mx.array) -> mx.array:
    """Convert frequency to mel scale"""
    return 1127.0 * mx.log(1.0 + freq / 700.0)


def inverse_mel_scale(mel_freq: mx.array) -> mx.array:
    """Convert mel scale to frequency"""
    return 700.0 * (mx.exp(mel_freq / 1127.0) - 1.0)


def get_mel_banks(
    num_bins: int,
    window_length_padded: int,
    sample_freq: float,
    low_freq: float,
    high_freq: float,
    vtln_low: float = 100.0,
    vtln_high: float = -500.0,
    vtln_warp_factor: float = 1.0,
) -> Tuple[mx.array, mx.array]:
    """
    Create mel filterbank matrix matching Kaldi's implementation

    Returns:
        (bins, center_freqs): Mel filterbank matrix and center frequencies
    """
    assert num_bins > 3, "Must have at least 3 mel bins"
    assert window_length_padded % 2 == 0

    num_fft_bins = window_length_padded // 2
    nyquist = 0.5 * sample_freq

    if high_freq <= 0.0:
        high_freq += nyquist

    assert (
        (0.0 <= low_freq < nyquist)
        and (0.0 < high_freq <= nyquist)
        and (low_freq < high_freq)
    )

    # FFT bin width
    fft_bin_width = sample_freq / window_length_padded
    mel_low_freq = float(mel_scale(mx.array(low_freq)))
    mel_high_freq = float(mel_scale(mx.array(high_freq)))

    # Divide by num_bins+1 because of end effects
    mel_freq_delta = (mel_high_freq - mel_low_freq) / (num_bins + 1)

    if vtln_high < 0.0:
        vtln_high += nyquist

    # Create mel points
    bin_idx = mx.arange(num_bins).reshape(-1, 1)
    left_mel = mel_low_freq + bin_idx * mel_freq_delta
    center_mel = mel_low_freq + (bin_idx + 1.0) * mel_freq_delta
    right_mel = mel_low_freq + (bin_idx + 2.0) * mel_freq_delta

    # No VTLN warping for now (vtln_warp_factor == 1.0)
    center_freqs = inverse_mel_scale(center_mel)

    # Create frequency grid for all FFT bins
    mel = mel_scale(fft_bin_width * mx.arange(num_fft_bins)).reshape(1, -1)

    # Calculate filter responses
    up_slope = (mel - left_mel) / (center_mel - left_mel)
    down_slope = (right_mel - mel) / (right_mel - center_mel)

    # Combine slopes, taking minimum and clamping to [0, 1]
    bins = mx.maximum(mx.zeros(1), mx.minimum(up_slope, down_slope))

    return bins, center_freqs.squeeze()


def compute_fbank(audio_in, args):
    """
    Compute mel-filterbank features matching torchaudio.compliance.kaldi.fbank

    Args:
        audio_in: Input audio (torch tensor or numpy array)
        args: Namespace with parameters

    Returns:
        MLX array of log mel-filterbank features
    """
    # Convert input to numpy then MLX
    if hasattr(audio_in, "numpy"):
        audio_np = audio_in.numpy()
    else:
        audio_np = audio_in

    # Handle batch dimension - take first channel
    if audio_np.ndim == 2:
        audio_np = audio_np[0]

    waveform = mx.array(audio_np)

    # Extract parameters
    sample_frequency = float(args.sampling_rate)
    frame_length = args.win_len / args.sampling_rate * 1000  # to milliseconds
    frame_shift = args.win_inc / args.sampling_rate * 1000  # to milliseconds
    num_mel_bins = args.num_mels
    window_type = args.win_type

    # Kaldi default parameters
    dither = 1.0  # Re-enabled with deterministic seed
    energy_floor = 1.0
    preemphasis_coefficient = 0.97
    raw_energy = True
    remove_dc_offset = True
    round_to_power_of_two = True
    snip_edges = True
    use_log_fbank = True
    use_power = True
    low_freq = 20.0
    high_freq = 0.0  # 0 means Nyquist

    # Window properties
    window_shift_samples = int(sample_frequency * frame_shift * 0.001)
    window_size = int(sample_frequency * frame_length * 0.001)
    padded_window_size = (
        _next_power_of_2(window_size) if round_to_power_of_two else window_size
    )

    # Get frames
    strided_input = _get_strided(
        waveform, window_size, window_shift_samples, snip_edges
    )

    if strided_input.shape[0] == 0:
        return mx.zeros((0, num_mel_bins))

    # Apply dither
    if dither != 0.0:
        # mx.random.seed(42) # Randomness enabled for production
        # Kaldi uses Gaussian dither
        rand_gauss = mx.random.normal(strided_input.shape) * dither
        strided_input = strided_input + rand_gauss
    else:
        pass  # No dither applied

    # Remove DC offset
    if remove_dc_offset:
        row_means = mx.mean(strided_input, axis=1, keepdims=True)
        strided_input = strided_input - row_means

    # Compute raw energy before preemphasis (not used in fbank output)
    if raw_energy:
        signal_log_energy = mx.log(mx.maximum(mx.sum(strided_input**2, axis=1), 1e-8))

    # Apply preemphasis
    if preemphasis_coefficient != 0.0:
        # Create preemphasis: first column stays same, others subtract scaled previous
        first_col = strided_input[:, 0:1]
        other_cols = (
            strided_input[:, 1:] - preemphasis_coefficient * strided_input[:, :-1]
        )
        strided_input = mx.concatenate([first_col, other_cols], axis=1)

    # Apply window function
    window = _feature_window_function(window_type, window_size)
    strided_input = strided_input * window

    # Pad to padded_window_size if needed
    if padded_window_size != window_size:
        padding = padded_window_size - window_size
        strided_input = mx.pad(strided_input, [(0, 0), (0, padding)])

    # Compute FFT - using complex FFT then taking magnitude
    fft_result = mx.fft.rfft(strided_input, n=padded_window_size, axis=1)
    spectrum = mx.abs(fft_result)

    # Power spectrum
    if use_power:
        spectrum = spectrum**2.0

    # Get mel filterbank
    mel_energies, _ = get_mel_banks(
        num_mel_bins, padded_window_size, sample_frequency, low_freq, high_freq
    )

    # Kaldi pads mel energies with a zero column on the right
    # mel_energies shape is (num_mel_bins, padded_window_size // 2)
    # but spectrum shape is (frames, padded_window_size // 2 + 1)
    mel_energies = mx.pad(mel_energies, [(0, 0), (0, 1)])

    # Apply mel filterbank
    # Spectrum is (frames, freq_bins), mel_energies is (num_mel_bins, freq_bins)
    # Result should be (frames, num_mel_bins)
    mel_features = mx.matmul(spectrum, mel_energies.T)

    # Apply log
    if use_log_fbank:
        mel_features = mx.log(mx.maximum(mel_features, 1e-8))

    return mel_features
