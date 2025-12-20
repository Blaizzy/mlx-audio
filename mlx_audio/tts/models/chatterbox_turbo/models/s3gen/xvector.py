# Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)
# Adapted from FunASR (https://github.com/alibaba-damo-academy/FunASR)
# MIT License

"""
CAMPPlus speaker encoder for x-vector extraction.
Used for speaker conditioning in S3Gen.
"""

from collections import OrderedDict
from typing import List, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def pad_list(xs: List[mx.array], pad_value: float = 0) -> mx.array:
    """Pad list of tensors to same length."""
    n_batch = len(xs)
    max_len = max(x.shape[0] for x in xs)

    # Create padded tensor
    pad_shape = (n_batch, max_len) + xs[0].shape[1:]
    pad = mx.full(pad_shape, pad_value, dtype=xs[0].dtype)

    for i, x in enumerate(xs):
        # Use slice assignment
        indices = mx.arange(x.shape[0])
        pad = pad.at[i, : x.shape[0]].set(x)

    return pad


def _mel_scale(freq: np.ndarray) -> np.ndarray:
    """Convert Hz to mel scale (Kaldi style)."""
    return 1127.0 * np.log(1.0 + freq / 700.0)


def _inverse_mel_scale(mel: np.ndarray) -> np.ndarray:
    """Convert mel to Hz (Kaldi style)."""
    return 700.0 * (np.exp(mel / 1127.0) - 1.0)


def _get_mel_banks_kaldi(
    num_bins: int,
    padded_window_size: int,
    sample_freq: float,
    low_freq: float,
    high_freq: float,
) -> np.ndarray:
    """
    Create mel filterbank matrix exactly matching Kaldi's get_mel_banks.

    Returns:
        Filterbank matrix of shape (num_bins, padded_window_size // 2)
    """
    num_fft_bins = padded_window_size // 2
    nyquist = 0.5 * sample_freq

    if high_freq <= 0.0:
        high_freq += nyquist

    # FFT bin width
    fft_bin_width = sample_freq / padded_window_size

    # Mel scale boundaries
    mel_low_freq = 1127.0 * np.log(1.0 + low_freq / 700.0)
    mel_high_freq = 1127.0 * np.log(1.0 + high_freq / 700.0)

    # Mel frequency delta
    mel_freq_delta = (mel_high_freq - mel_low_freq) / (num_bins + 1)

    # Create filterbank bins
    bins = np.zeros((num_bins, num_fft_bins))

    for i in range(num_bins):
        left_mel = mel_low_freq + i * mel_freq_delta
        center_mel = mel_low_freq + (i + 1) * mel_freq_delta
        right_mel = mel_low_freq + (i + 2) * mel_freq_delta

        for j in range(num_fft_bins):
            # Convert FFT bin to mel
            freq = fft_bin_width * j
            mel = 1127.0 * np.log(1.0 + freq / 700.0)

            # Triangular filter
            if mel > left_mel and mel <= center_mel:
                bins[i, j] = (mel - left_mel) / (center_mel - left_mel)
            elif mel > center_mel and mel < right_mel:
                bins[i, j] = (right_mel - mel) / (right_mel - center_mel)

    return bins


def _povey_window(length: int) -> np.ndarray:
    """Create Povey window (Hanning raised to power 0.85)."""
    n = np.arange(length)
    # Use periodic=False equivalent: divide by (length - 1)
    window = 0.5 - 0.5 * np.cos(2 * np.pi * n / (length - 1))
    return window**0.85


def extract_fbank_features(
    audio: mx.array,
    num_mel_bins: int = 80,
    sample_rate: int = 16000,
    frame_length_ms: float = 25.0,
    frame_shift_ms: float = 10.0,
    low_freq: float = 20.0,
    high_freq: float = 0.0,
    preemphasis_coeff: float = 0.97,
    remove_dc_offset: bool = True,
    use_power: bool = True,
    snip_edges: bool = True,
) -> mx.array:
    """
    Extract log-mel filterbank features matching Kaldi's fbank implementation exactly.

    This replicates torchaudio.compliance.kaldi.fbank behavior.

    Args:
        audio: Audio waveform (T,) or (B, T)
        num_mel_bins: Number of mel bins (default 80)
        sample_rate: Sample rate (default 16000)
        frame_length_ms: Frame length in milliseconds (default 25.0)
        frame_shift_ms: Frame shift in milliseconds (default 10.0)
        low_freq: Low frequency cutoff (default 20.0)
        high_freq: High frequency cutoff (0 = Nyquist)
        preemphasis_coeff: Preemphasis coefficient (default 0.97)
        remove_dc_offset: Remove DC offset per frame (default True)
        use_power: Use power spectrum instead of magnitude (default True)
        snip_edges: If True, output frames where the entire frame fits (default True)

    Returns:
        Features (B, T, num_mel_bins)
    """
    if audio.ndim == 1:
        audio = audio[None, :]

    # Convert to numpy for processing (use float64 for precision)
    audio_np = np.array(audio).astype(np.float64)

    # Frame parameters (exactly matching Kaldi)
    frame_length = int(sample_rate * frame_length_ms * 0.001)
    frame_shift = int(sample_rate * frame_shift_ms * 0.001)

    # Round to power of 2 for FFT
    padded_length = 1
    while padded_length < frame_length:
        padded_length *= 2

    # Create mel filterbank (exactly matching Kaldi)
    mel_banks = _get_mel_banks_kaldi(
        num_mel_bins, padded_length, sample_rate, low_freq, high_freq
    )
    # Pad right column with zeros to match FFT output size
    mel_banks = np.pad(mel_banks, ((0, 0), (0, 1)), mode="constant", constant_values=0)

    # Create Povey window
    window = _povey_window(frame_length)

    # Epsilon for numerical stability (matching torch.finfo(torch.float).eps)
    epsilon = np.finfo(np.float32).eps

    features_list = []

    for b in range(audio_np.shape[0]):
        wav = audio_np[b]

        # Calculate number of frames
        if snip_edges:
            if len(wav) < frame_length:
                num_frames = 0
            else:
                num_frames = 1 + (len(wav) - frame_length) // frame_shift
        else:
            num_frames = (len(wav) + frame_shift // 2) // frame_shift

        if num_frames == 0:
            features_list.append(np.zeros((1, num_mel_bins), dtype=np.float32))
            continue

        # Extract frames using strided approach (matching Kaldi's _get_strided)
        frames = np.zeros((num_frames, frame_length))
        for i in range(num_frames):
            start = i * frame_shift
            end = start + frame_length
            if end <= len(wav):
                frames[i] = wav[start:end]
            else:
                frames[i, : len(wav) - start] = wav[start:]

        # Step 1: Remove DC offset per frame (before preemphasis)
        if remove_dc_offset:
            frames = frames - frames.mean(axis=1, keepdims=True)

        # Step 2: Apply preemphasis PER FRAME (Kaldi applies it after framing)
        if preemphasis_coeff != 0.0:
            # For each frame, shift and apply preemphasis
            # strided_input[i,j] -= preemph * strided_input[i, max(0, j-1)]
            preemph_frames = np.zeros_like(frames)
            preemph_frames[:, 0] = frames[
                :, 0
            ]  # First sample unchanged (replicate padding)
            preemph_frames[:, 1:] = frames[:, 1:] - preemphasis_coeff * frames[:, :-1]
            frames = preemph_frames

        # Step 3: Apply window
        frames = frames * window

        # Step 4: Pad to FFT length
        padded_frames = np.zeros((num_frames, padded_length))
        padded_frames[:, :frame_length] = frames

        # Step 5: Compute FFT and get magnitude/power spectrum
        fft_out = np.fft.rfft(padded_frames, n=padded_length)
        spectrum = np.abs(fft_out)
        if use_power:
            spectrum = spectrum**2

        # Step 6: Apply mel filterbank
        mel_energies = np.dot(spectrum, mel_banks.T)

        # Step 7: Apply log with epsilon floor
        log_mel = np.log(np.maximum(mel_energies, epsilon))

        features_list.append(log_mel.astype(np.float32))

    # Pad to same length and stack
    max_len = max(f.shape[0] for f in features_list)
    padded = []
    for f in features_list:
        if f.shape[0] < max_len:
            pad_arr = np.zeros((max_len - f.shape[0], num_mel_bins), dtype=np.float32)
            f = np.concatenate([f, pad_arr], axis=0)
        padded.append(mx.array(f))

    return mx.stack(padded)


class BasicResBlock(nn.Module):
    """Basic residual block for FCM."""

    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=(stride, 1), padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm(planes)

        # Shortcut connection
        self.use_shortcut = stride != 1 or in_planes != self.expansion * planes
        if self.use_shortcut:
            self.shortcut_conv = nn.Conv2d(
                in_planes,
                self.expansion * planes,
                kernel_size=1,
                stride=(stride, 1),
                bias=False,
            )
            self.shortcut_bn = nn.BatchNorm(self.expansion * planes)

    def __call__(self, x: mx.array) -> mx.array:
        out = nn.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.use_shortcut:
            shortcut = self.shortcut_bn(self.shortcut_conv(x))
        else:
            shortcut = x

        out = out + shortcut
        out = nn.relu(out)
        return out


class FCM(nn.Module):
    """Feature Convolutional Module for CAMPPlus."""

    def __init__(self, m_channels: int = 32, feat_dim: int = 80):
        super().__init__()
        self.in_planes = m_channels

        self.conv1 = nn.Conv2d(
            1, m_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm(m_channels)

        # Layer 1: 2 residual blocks with stride 2
        self.layer1 = [
            BasicResBlock(m_channels, m_channels, stride=2),
            BasicResBlock(m_channels, m_channels, stride=1),
        ]

        # Layer 2: 2 residual blocks with stride 2
        self.layer2 = [
            BasicResBlock(m_channels, m_channels, stride=2),
            BasicResBlock(m_channels, m_channels, stride=1),
        ]

        self.conv2 = nn.Conv2d(
            m_channels, m_channels, kernel_size=3, stride=(2, 1), padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm(m_channels)

        self.out_channels = m_channels * (feat_dim // 8)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, T, F) -> permute to (B, F, T) then add channel dim
        # In PyTorch: x = x.permute(0, 2, 1) then x = x.unsqueeze(1)
        # MLX Conv2d uses NHWC, so we need (B, F, T, 1)
        x = x.transpose(0, 2, 1)  # (B, F, T)
        x = x[:, :, :, None]  # (B, F, T, 1)

        out = nn.relu(self.bn1(self.conv1(x)))

        for layer in self.layer1:
            out = layer(out)
        for layer in self.layer2:
            out = layer(out)

        out = nn.relu(self.bn2(self.conv2(out)))

        # out shape: (B, F/8, T, C) in NHWC
        # Reshape to: (B, C * F/8, T)
        B, F_reduced, T, C = out.shape
        out = out.transpose(0, 3, 1, 2).reshape(B, C * F_reduced, T)

        return out


class TDNNLayer(nn.Module):
    """Time-delay neural network layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        if padding < 0:
            assert kernel_size % 2 == 1
            padding = (kernel_size - 1) // 2 * dilation

        self.linear = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.bn = nn.BatchNorm(out_channels)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, C, T) - need to convert to (B, T, C) for MLX Conv1d
        x = x.transpose(0, 2, 1)  # (B, T, C)
        x = self.linear(x)  # MLX Conv1d: (B, T, C) -> (B, T', C')
        x = self.bn(x)
        x = nn.relu(x)
        x = x.transpose(0, 2, 1)  # Back to (B, C', T')
        return x


class CAMLayer(nn.Module):
    """Context-Aware Masking layer."""

    def __init__(
        self,
        bn_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
        bias: bool,
        reduction: int = 2,
    ):
        super().__init__()
        self.linear_local = nn.Conv1d(
            bn_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.linear1 = nn.Conv1d(bn_channels, bn_channels // reduction, 1)
        self.linear2 = nn.Conv1d(bn_channels // reduction, out_channels, 1)

    def seg_pooling(self, x: mx.array, seg_len: int = 100) -> mx.array:
        """Segment-based average pooling. x is (B, T, C)."""
        B, T, C = x.shape

        # Compute number of segments
        n_segs = (T + seg_len - 1) // seg_len

        # Pad to multiple of seg_len
        pad_len = n_segs * seg_len - T
        if pad_len > 0:
            x = mx.concatenate([x, mx.zeros((B, pad_len, C))], axis=1)

        # Reshape and compute mean
        x = x.reshape(B, n_segs, seg_len, C)
        seg = x.mean(axis=2)  # (B, n_segs, C)

        # Expand back
        seg = mx.repeat(seg[:, :, None, :], seg_len, axis=2)
        seg = seg.reshape(B, -1, C)[:, : T + pad_len, :]

        return seg[:, :T, :] if pad_len > 0 else seg

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, C, T) from caller - convert to (B, T, C) for MLX Conv1d
        x = x.transpose(0, 2, 1)  # (B, T, C)

        y = self.linear_local(x)  # (B, T', C')

        # Context: global mean + segment pooling
        context = x.mean(axis=1, keepdims=True) + self.seg_pooling(x)
        context = nn.relu(self.linear1(context))
        m = mx.sigmoid(self.linear2(context))

        result = y * m
        # Convert back to (B, C', T')
        return result.transpose(0, 2, 1)


class CAMDenseTDNNLayer(nn.Module):
    """CAM Dense TDNN layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bn_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        assert kernel_size % 2 == 1
        padding = (kernel_size - 1) // 2 * dilation

        self.bn1 = nn.BatchNorm(in_channels)
        self.linear1 = nn.Conv1d(in_channels, bn_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm(bn_channels)
        self.cam_layer = CAMLayer(
            bn_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, C, T)
        # Convert to (B, T, C) for BatchNorm and Conv1d
        x = x.transpose(0, 2, 1)  # (B, T, C)
        x = self.bn1(x)
        x = nn.relu(x)

        x = self.linear1(x)  # MLX Conv1d expects (B, T, C)

        x = self.bn2(x)
        x = nn.relu(x)

        # Convert back to (B, C, T) for CAMLayer
        x = x.transpose(0, 2, 1)
        x = self.cam_layer(x)
        return x


class CAMDenseTDNNBlock(nn.Module):
    """Block of CAM Dense TDNN layers."""

    def __init__(
        self,
        num_layers: int,
        in_channels: int,
        out_channels: int,
        bn_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.layers = []
        for i in range(num_layers):
            layer = CAMDenseTDNNLayer(
                in_channels=in_channels + i * out_channels,
                out_channels=out_channels,
                bn_channels=bn_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                bias=bias,
            )
            self.layers.append(layer)

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = mx.concatenate([x, layer(x)], axis=1)
        return x


class TransitLayer(nn.Module):
    """Transition layer between blocks."""

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.bn = nn.BatchNorm(in_channels)
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, C, T) -> (B, T, C) for processing
        x = x.transpose(0, 2, 1)  # (B, T, C)
        x = self.bn(x)
        x = nn.relu(x)
        x = self.linear(x)  # MLX Conv1d expects (B, T, C)
        x = x.transpose(0, 2, 1)  # Back to (B, C, T)
        return x


class StatsPool(nn.Module):
    """Statistics pooling layer."""

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, C, T)
        mean = x.mean(axis=2)
        std = x.std(axis=2)
        return mx.concatenate([mean, std], axis=1)


class DenseLayer(nn.Module):
    """Dense layer with batch normalization."""

    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super().__init__()
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)
        self.bn = nn.BatchNorm(out_channels, affine=False)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, C) for 2D, (B, C, T) for 3D
        if x.ndim == 2:
            # (B, C) -> (B, 1, C) for MLX Conv1d
            x = x[:, None, :]  # Add time dimension
            x = self.linear(x)  # (B, 1, C')
            x = x.squeeze(1)  # (B, C')
        else:
            # (B, C, T) -> (B, T, C) for MLX Conv1d
            x = x.transpose(0, 2, 1)
            x = self.linear(x)
            x = x.transpose(0, 2, 1)

        # BatchNorm expects (B, ..., C)
        if x.ndim == 2:
            x = self.bn(x)
        else:
            x_t = x.transpose(0, 2, 1)
            x_t = self.bn(x_t)
            x = x_t.transpose(0, 2, 1)

        return x


class CAMPPlus(nn.Module):
    """
    CAMPPlus speaker encoder for x-vector extraction.

    This model extracts speaker embeddings from audio for voice conditioning.
    """

    def __init__(
        self,
        feat_dim: int = 80,
        embedding_size: int = 192,
        growth_rate: int = 32,
        bn_size: int = 4,
        init_channels: int = 128,
    ):
        super().__init__()

        self.head = FCM(feat_dim=feat_dim)
        channels = self.head.out_channels

        # Initial TDNN layer
        self.tdnn = TDNNLayer(
            channels, init_channels, 5, stride=2, dilation=1, padding=-1
        )
        channels = init_channels

        # CAM Dense TDNN blocks
        block_configs = [
            (12, 3, 1),  # num_layers, kernel_size, dilation
            (24, 3, 2),
            (16, 3, 2),
        ]

        self.blocks = []
        self.transits = []

        for i, (num_layers, kernel_size, dilation) in enumerate(block_configs):
            block = CAMDenseTDNNBlock(
                num_layers=num_layers,
                in_channels=channels,
                out_channels=growth_rate,
                bn_channels=bn_size * growth_rate,
                kernel_size=kernel_size,
                dilation=dilation,
            )
            self.blocks.append(block)
            channels = channels + num_layers * growth_rate

            transit = TransitLayer(channels, channels // 2, bias=False)
            self.transits.append(transit)
            channels //= 2

        # Output layers
        self.out_bn = nn.BatchNorm(channels)
        self.stats = StatsPool()
        self.dense = DenseLayer(channels * 2, embedding_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass.

        Args:
            x: Input features (B, T, F) where F=feat_dim

        Returns:
            Speaker embeddings (B, embedding_size)
        """
        # FCM head
        x = self.head(x)  # (B, C, T)

        # TDNN
        x = self.tdnn(x)

        # CAM Dense blocks with transit layers
        for block, transit in zip(self.blocks, self.transits):
            x = block(x)
            x = transit(x)

        # Output processing
        x_t = x.transpose(0, 2, 1)
        x_t = self.out_bn(x_t)
        x_t = nn.relu(x_t)
        x = x_t.transpose(0, 2, 1)

        x = self.stats(x)
        x = self.dense(x)

        return x

    def inference(self, audio_list: List[mx.array]) -> mx.array:
        """
        Extract speaker embeddings from audio.

        Args:
            audio_list: List of audio waveforms at 16kHz

        Returns:
            Speaker embeddings (B, embedding_size)
        """
        # Stack audio
        audio = mx.stack([a if a.ndim == 1 else a[0] for a in audio_list])

        # Extract features
        features = extract_fbank_features(audio)

        # Apply CMN (Cepstral Mean Normalization) - subtract per-frequency-bin mean
        # This matches PyTorch: feature = feature - feature.mean(dim=0, keepdim=True)
        features = features - features.mean(axis=1, keepdims=True)

        # Forward pass
        return self(features)

    def sanitize(self, weights: dict) -> dict:
        """
        Sanitize PyTorch weights for MLX compatibility.

        Maps PyTorch weight names to MLX layer names.
        """

        import re

        new_weights = {}

        for key, value in weights.items():
            # Skip num_batches_tracked
            if "num_batches_tracked" in key:
                continue

            new_key = key

            # === xvector mappings ===
            if key.startswith("xvector."):
                new_key = key[8:]  # Remove "xvector." prefix

                # Map tdnn.nonlinear.batchnorm -> tdnn.bn
                new_key = new_key.replace("tdnn.nonlinear.batchnorm", "tdnn.bn")

                # Map block{N}.tdnnd{M} -> blocks.{N-1}.layers.{M-1}
                match = re.search(r"block(\d+)\.tdnnd(\d+)", new_key)
                if match:
                    block_idx = int(match.group(1)) - 1
                    layer_idx = int(match.group(2)) - 1
                    old = f"block{match.group(1)}.tdnnd{match.group(2)}"
                    new = f"blocks.{block_idx}.layers.{layer_idx}"
                    new_key = new_key.replace(old, new)

                # Map transit{N} -> transits.{N-1}
                match = re.search(r"transit(\d+)", new_key)
                if match:
                    transit_idx = int(match.group(1)) - 1
                    new_key = new_key.replace(
                        f"transit{match.group(1)}", f"transits.{transit_idx}"
                    )

                # Map nonlinear.batchnorm -> bn (for transit/tdnn)
                new_key = new_key.replace("nonlinear.batchnorm", "bn")

                # Map nonlinear1.batchnorm -> bn1
                new_key = new_key.replace("nonlinear1.batchnorm", "bn1")
                # Map nonlinear2.batchnorm -> bn2
                new_key = new_key.replace("nonlinear2.batchnorm", "bn2")

                # Map out_nonlinear.batchnorm -> out_bn
                new_key = new_key.replace("out_nonlinear.batchnorm", "out_bn")

                # Map dense.nonlinear.batchnorm -> dense.bn
                new_key = new_key.replace("dense.nonlinear.batchnorm", "dense.bn")

            # === head mappings ===
            elif key.startswith("head."):
                # Map shortcut.0 -> shortcut_conv
                new_key = new_key.replace("shortcut.0", "shortcut_conv")
                # Map shortcut.1 -> shortcut_bn
                new_key = new_key.replace("shortcut.1", "shortcut_bn")

            # Handle Conv weight transpose
            if "weight" in key and value.ndim >= 3:
                if value.ndim == 4:
                    if value.shape[2] == value.shape[3]:
                        value = mx.array(np.array(value).transpose(0, 2, 3, 1))

                elif value.ndim == 3:
                    if value.shape[2] <= 7 and value.shape[1] > value.shape[2]:
                        value = mx.array(np.array(value).transpose(0, 2, 1))

            new_weights[new_key] = value

        return new_weights
