"""Pure-MLX CAM++ speaker x-vector encoder for dots.tts.

Imports ONLY mlx + numpy — never torch. Two public surfaces:

  * ``kaldi_fbank(wav, sample_rate=16000) -> np.ndarray [T, 80]`` — a numpy
    reimplementation of the torchaudio ``Kaldi.fbank`` front-end (Povey window,
    pre-emphasis 0.97, per-frame DC removal, snip_edges framing, FFT 512, 80-mel
    filterbank low20/high8000, log floor) followed by the per-bin CMN that the
    upstream ``extract_speaker_fbank`` applies. Matches torch Kaldi to ~1e-5.
  * ``CAMPPlus`` — the MLX 3D-Speaker CAM++ trunk (FCM Conv2d front-end + dense
    TDNN trunk + stats pooling + final dense). ``model(fbank[B, T, 80]) -> [B, 512]``.

All BatchNorms are folded to a frozen affine ``y = (x - rm)/sqrt(rv + eps)*w + b``
(eps 1e-5) at load time (see ``loader.load_speaker``); the affine-free final BN
folds to ``(x - rm)/sqrt(rv + eps)``. Conv weights are stored in MLX layout
(Conv1d ``[out, k, in]``, Conv2d ``[out, kh, kw, in]``); activations flow
channels-last (NLC / NHWC).

tf32 note: MLX's fast conv/matmul round fp32 -> ~tf32 on Apple Silicon, costing
up to ~1e-1 on large inputs. CAM++ is gated on COSINE (>= 0.9995), which is robust
to that rounding, so the fast path is used throughout.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

# torchaudio Kaldi constants (compliance/kaldi.py).
_EPSILON = float(np.finfo(np.float32).eps)  # 1.1920928955078125e-07
_PREEMPH = 0.97
_POVEY_EXP = 0.85


def _next_power_of_2(x: int) -> int:
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def _mel_scale(freq):
    return 1127.0 * np.log(1.0 + freq / 700.0)


def _inverse_mel_scale(mel_freq):
    return 700.0 * (np.exp(mel_freq / 1127.0) - 1.0)


def _get_mel_banks(
    num_bins: int,
    window_length_padded: int,
    sample_freq: float,
    low_freq: float,
    high_freq: float,
) -> np.ndarray:
    """numpy port of ``torchaudio.compliance.kaldi.get_mel_banks`` (vtln_warp=1).

    Returns the ``[num_bins, num_fft_bins]`` triangular mel filterbank (fp64),
    where ``num_fft_bins = window_length_padded // 2``.
    """
    num_fft_bins = window_length_padded // 2
    nyquist = 0.5 * sample_freq
    if high_freq <= 0.0:
        high_freq += nyquist

    fft_bin_width = sample_freq / window_length_padded
    mel_low_freq = _mel_scale(low_freq)
    mel_high_freq = _mel_scale(high_freq)
    mel_freq_delta = (mel_high_freq - mel_low_freq) / (num_bins + 1)

    bin_idx = np.arange(num_bins).reshape(num_bins, 1).astype(np.float64)
    left_mel = mel_low_freq + bin_idx * mel_freq_delta
    center_mel = mel_low_freq + (bin_idx + 1.0) * mel_freq_delta
    right_mel = mel_low_freq + (bin_idx + 2.0) * mel_freq_delta

    mel = _mel_scale(fft_bin_width * np.arange(num_fft_bins)).reshape(1, num_fft_bins)

    up_slope = (mel - left_mel) / (center_mel - left_mel)
    down_slope = (right_mel - mel) / (right_mel - center_mel)
    bins = np.maximum(0.0, np.minimum(up_slope, down_slope))
    return bins  # [num_bins, num_fft_bins]


def kaldi_fbank(wav, sample_rate: int = 16000) -> np.ndarray:
    """Compute Kaldi-equivalent log-mel fbank + per-bin CMN, matching the oracle.

    Reproduces ``torchaudio.compliance.kaldi.fbank`` with the dots.tts speaker
    defaults (80 mel bins, 25 ms / 10 ms frames, Povey window, pre-emphasis 0.97,
    remove_dc_offset, snip_edges, FFT 512, low 20 Hz / high 8000 Hz, log floor,
    dither 0, use_energy=False) and then subtracts the per-bin (per-column) mean
    over time — the ``mean_norm`` step ``extract_speaker_fbank`` applies.

    Args:
        wav: 1-D waveform in [-1, 1] (NO int16 rescale), already at ``sample_rate``.
        sample_rate: must match ``wav`` (16000 for the speaker front-end).

    Returns:
        ``[num_frames, 80]`` fp32 numpy array (post-CMN).
    """
    x = np.asarray(wav, dtype=np.float64).reshape(-1)
    n_mels = 80
    low_freq = 20.0
    high_freq = 0.0  # <= 0 => Nyquist (8000 at 16 kHz), per Kaldi default
    window_size = int(sample_rate * 25.0 * 0.001)  # 400
    window_shift = int(sample_rate * 10.0 * 0.001)  # 160
    padded = _next_power_of_2(window_size)  # 512

    num_samples = x.shape[0]
    if num_samples < window_size:
        return np.zeros((0, n_mels), dtype=np.float32)

    # --- snip_edges framing: m = 1 + (n - window)//shift, frame i = x[i*shift : +window]
    m = 1 + (num_samples - window_size) // window_shift
    idx = np.arange(window_size)[None, :] + window_shift * np.arange(m)[:, None]
    frames = x[idx]  # [m, window_size]

    # --- remove_dc_offset: per-frame mean subtraction
    frames = frames - frames.mean(axis=1, keepdims=True)

    # --- pre-emphasis: x[:, j] -= 0.97 * x[:, max(0, j-1)] (first sample replicated)
    offset = np.concatenate([frames[:, :1], frames[:, :-1]], axis=1)
    frames = frames - _PREEMPH * offset

    # --- Povey window: hann(periodic=False)^0.85
    n = np.arange(window_size)
    hann = 0.5 - 0.5 * np.cos(2.0 * np.pi * n / (window_size - 1))
    window = hann**_POVEY_EXP
    frames = frames * window[None, :]

    # --- zero-pad to FFT size, rfft, power spectrum
    if padded != window_size:
        frames = np.pad(frames, ((0, 0), (0, padded - window_size)))
    spectrum = np.abs(np.fft.rfft(frames, n=padded, axis=1))  # [m, padded//2 + 1]
    spectrum = spectrum**2.0

    # --- mel filterbank (pad one zero column to match padded//2 + 1 bins), apply, log
    mel = _get_mel_banks(n_mels, padded, float(sample_rate), low_freq, high_freq)
    mel = np.pad(mel, ((0, 0), (0, 1)))  # [n_mels, padded//2 + 1]
    mel_energies = spectrum @ mel.T  # [m, n_mels]
    mel_energies = np.log(np.maximum(mel_energies, _EPSILON))

    # --- per-bin CMN over time (extract_speaker_fbank's mean_norm)
    mel_energies = mel_energies - mel_energies.mean(axis=0, keepdims=True)
    return mel_energies.astype(np.float32)


# --------------------------------------------------------------------------- #
# CAM++ trunk (MLX). All conv weights pre-transposed to MLX layout, all BN
# folded to frozen affine (scale, shift) at load time by loader.load_speaker.
# --------------------------------------------------------------------------- #


class _BN:
    """Frozen BatchNorm folded to an affine over the channel (last) axis: x*scale + shift."""

    def __init__(self, scale: mx.array, shift: mx.array):
        self.scale = scale  # [C]
        self.shift = shift  # [C]

    def __call__(self, x: mx.array) -> mx.array:
        return x * self.scale + self.shift


def _relu(x: mx.array) -> mx.array:
    return mx.maximum(x, 0)


class _Conv2d:
    """Conv2d over NHWC tensors. ``weight`` is MLX layout ``[out, kh, kw, in]``."""

    def __init__(
        self,
        weight: mx.array,
        bias: mx.array | None,
        *,
        stride: tuple[int, int] = (1, 1),
        padding: tuple[int, int] = (1, 1),
    ):
        self.weight = weight
        self.bias = bias
        self.stride = stride
        self.padding = padding

    def __call__(self, x: mx.array) -> mx.array:
        y = mx.conv2d(x, self.weight, stride=self.stride, padding=self.padding)
        if self.bias is not None:
            y = y + self.bias
        return y


class _Conv1d:
    """Conv1d over NLC tensors. ``weight`` is MLX layout ``[out, k, in]``.

    ``padding`` is the symmetric zero-pad applied to the time axis (matching
    torch Conv1d ``padding``); ``stride`` / ``dilation`` mirror torch.
    """

    def __init__(
        self,
        weight: mx.array,
        bias: mx.array | None,
        *,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
    ):
        self.weight = weight
        self.bias = bias
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def __call__(self, x: mx.array) -> mx.array:
        y = mx.conv1d(
            x,
            self.weight,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )
        if self.bias is not None:
            y = y + self.bias
        return y


class _BasicResBlock:
    """ResNet basic block over NHWC: conv1(stride,1)-bn1-relu, conv2-bn2, +shortcut, relu."""

    def __init__(self, conv1, bn1, conv2, bn2, shortcut_conv=None, shortcut_bn=None):
        self.conv1 = conv1
        self.bn1 = bn1
        self.conv2 = conv2
        self.bn2 = bn2
        self.shortcut_conv = shortcut_conv
        self.shortcut_bn = shortcut_bn

    def __call__(self, x: mx.array) -> mx.array:
        out = _relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.shortcut_conv is not None:
            sc = self.shortcut_bn(self.shortcut_conv(x))
        else:
            sc = x
        return _relu(out + sc)


class FCM:
    """Conv2d front-end folding the mel-height into channels.

    Input ``[B, F=80, T]`` (channels-first F-over-T) is treated as a single-channel
    image ``[B, 1, F, T]``, run through conv1/bn1/relu, two residual layers (each
    halving F), conv2/bn2/relu (F halved again, stride (2,1)), then the
    ``[B, C, H, W]`` feature is folded to ``[B, C*H, W]`` in C-major order.

    Internally everything runs NHWC ``[B, H=F, W=T, C]`` (MLX-native); the leading
    channel-1 dim becomes the trailing C=1 channel.
    """

    def __init__(self, conv1, bn1, layer1, layer2, conv2, bn2, out_channels: int):
        self.conv1 = conv1
        self.bn1 = bn1
        self.layer1 = layer1  # list[_BasicResBlock]
        self.layer2 = layer2
        self.conv2 = conv2
        self.bn2 = bn2
        self.out_channels = out_channels

    def __call__(self, x: mx.array) -> mx.array:
        # x: [B, F, T] (channels-first) -> NHWC single-channel image [B, F, T, 1]
        x = x[..., None]
        out = _relu(self.bn1(self.conv1(x)))
        for blk in self.layer1:
            out = blk(out)
        for blk in self.layer2:
            out = blk(out)
        out = _relu(self.bn2(self.conv2(out)))
        # out: NHWC [B, H, W, C]. Upstream torch had NCHW [B, C, H, W] and reshaped
        # to [B, C*H, W] (C-major). Reproduce by moving to [B, C, H, W] then folding.
        b, h, w, c = out.shape
        out = out.transpose(0, 3, 1, 2)  # [B, C, H, W]
        out = out.reshape(b, c * h, w)  # [B, C*H, W], C-major (matches torch)
        return out


class _TDNNLayer:
    """Conv1d -> BN -> ReLU over NLC tensors (the trunk's input projection)."""

    def __init__(self, conv: _Conv1d, bn: _BN):
        self.conv = conv
        self.bn = bn

    def __call__(self, x: mx.array) -> mx.array:
        return _relu(self.bn(self.conv(x)))


def _seg_pooling(x: mx.array, seg_len: int = 100) -> mx.array:
    """avg_pool1d(kernel=seg_len, stride=seg_len, ceil_mode=True) then expand back to T.

    ``x`` is NLC ``[B, T, C]``. With ceil_mode the last (partial) window averages
    only its valid samples; the pooled value is then repeated ``seg_len`` times and
    sliced back to ``T`` — reproducing torch's ``F.avg_pool1d(..., ceil_mode=True)``
    + ``expand``/``reshape``/slice exactly (the most error-prone op in CAM++).
    """
    b, t, c = x.shape
    n_full = t // seg_len
    pooled = []
    if n_full > 0:
        head = x[:, : n_full * seg_len, :].reshape(b, n_full, seg_len, c)
        pooled.append(head.mean(axis=2))  # [B, n_full, C]
    rem = t - n_full * seg_len
    if rem > 0:
        # ceil_mode: a trailing partial window averaged over its `rem` valid samples.
        tail = x[:, n_full * seg_len :, :].mean(axis=1, keepdims=True)  # [B, 1, C]
        pooled.append(tail)
    seg = mx.concatenate(pooled, axis=1)  # [B, n_seg, C]
    n_seg = seg.shape[1]
    # expand each segment value across seg_len, then slice back to T.
    seg = mx.broadcast_to(seg[:, :, None, :], (b, n_seg, seg_len, c))
    seg = seg.reshape(b, n_seg * seg_len, c)
    return seg[:, :t, :]


class _CAMLayer:
    """3D-Speaker CAM attention layer over NLC tensors."""

    def __init__(self, linear_local, linear1, linear2):
        self.linear_local = linear_local  # _Conv1d (k, dilated)
        self.linear1 = linear1  # _Conv1d (k=1)
        self.linear2 = linear2  # _Conv1d (k=1)

    def __call__(self, x: mx.array) -> mx.array:
        y = self.linear_local(x)
        context = x.mean(axis=1, keepdims=True) + _seg_pooling(x)
        context = _relu(self.linear1(context))
        m = mx.sigmoid(self.linear2(context))
        return y * m


class _CAMDenseTDNNLayer:
    """nonlinear1(BN-ReLU) -> linear1(k1) -> nonlinear2(BN-ReLU) -> cam_layer."""

    def __init__(self, bn1, relu1, linear1, bn2, relu2, cam_layer):
        # bn1/bn2 are _BN; relu flags folded as the get_nonlinear "batchnorm-relu".
        self.bn1 = bn1
        self.linear1 = linear1  # _Conv1d k=1
        self.bn2 = bn2
        self.cam_layer = cam_layer

    def __call__(self, x: mx.array) -> mx.array:
        x = self.linear1(_relu(self.bn1(x)))
        x = _relu(self.bn2(x))
        return self.cam_layer(x)


class _CAMDenseTDNNBlock:
    """Dense block: each layer's output is concatenated (channels-last) onto the input."""

    def __init__(self, layers: list[_CAMDenseTDNNLayer]):
        self.layers = layers

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = mx.concatenate([x, layer(x)], axis=2)  # concat over channels (NLC)
        return x


class _TransitLayer:
    """BN-ReLU -> Conv1d (k=1)."""

    def __init__(self, bn: _BN, linear: _Conv1d):
        self.bn = bn
        self.linear = linear

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear(_relu(self.bn(x)))


def _stats_pooling(x: mx.array) -> mx.array:
    """Mean ++ UNBIASED (N-1) std over time, concatenated. ``x`` NLC -> [B, 2C]."""
    t = x.shape[1]
    mean = x.mean(axis=1)  # [B, C]
    centered = x - mean[:, None, :]
    var = (centered * centered).sum(axis=1) / (t - 1)  # unbiased
    std = mx.sqrt(var)
    return mx.concatenate([mean, std], axis=1)


class CAMPPlus:
    """Pure-MLX 3D-Speaker CAM++ x-vector encoder.

    ``model(fbank)`` takes ``[B, T, 80]`` log-mel features and returns the
    ``[B, 512]`` speaker embedding (final BatchNorm-normalized, NOT L2-normalized).
    """

    def __init__(
        self,
        head: FCM,
        tdnn: _TDNNLayer,
        block1: _CAMDenseTDNNBlock,
        transit1: _TransitLayer,
        block2: _CAMDenseTDNNBlock,
        transit2: _TransitLayer,
        block3: _CAMDenseTDNNBlock,
        transit3: _TransitLayer,
        out_nonlinear: _BN,
        dense_linear: _Conv1d,
        dense_bn: _BN,
    ):
        self.head = head
        self.tdnn = tdnn
        self.block1 = block1
        self.transit1 = transit1
        self.block2 = block2
        self.transit2 = transit2
        self.block3 = block3
        self.transit3 = transit3
        self.out_nonlinear = out_nonlinear  # BN-ReLU
        self.dense_linear = dense_linear  # Conv1d k=1 (1024 -> 512)
        self.dense_bn = dense_bn  # affine-free BN over 512

    def __call__(self, x: mx.array) -> mx.array:
        # x: [B, T, F=80] -> permute to channels-first [B, F, T] for the FCM.
        x = x.transpose(0, 2, 1)  # [B, F, T]
        x = self.head(x)  # [B, C*H=320, W=T']  (channels-first)
        x = x.transpose(0, 2, 1)  # NLC [B, T', 320] for the conv trunk

        x = self.tdnn(x)
        x = self.block1(x)
        x = self.transit1(x)
        x = self.block2(x)
        x = self.transit2(x)
        x = self.block3(x)
        x = self.transit3(x)
        x = _relu(self.out_nonlinear(x))  # out_nonlinear = BN-ReLU

        x = _stats_pooling(x)  # [B, 1024]
        # DenseLayer: Conv1d over a length-1 sequence then affine-free BN.
        x = x[:, None, :]  # [B, 1, 1024]
        x = self.dense_linear(x)  # [B, 1, 512]
        x = self.dense_bn(x)  # affine-free BN
        return x[:, 0, :]  # [B, 512]
