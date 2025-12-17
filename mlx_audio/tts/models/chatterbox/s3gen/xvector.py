from collections import OrderedDict
from typing import Dict

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.utils import mel_filters, stft


def _povey_window(size: int) -> mx.array:
    # Hann window: 0.5 - 0.5 * cos(2*pi*n/(N-1))
    n = mx.arange(size)
    hann = 0.5 - 0.5 * mx.cos(2 * mx.pi * n / (size - 1))
    # Povey window: hann^0.85
    return mx.power(hann, 0.85)


def _next_power_of_2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def kaldi_fbank(
    audio: mx.array,
    sample_rate: int = 16000,
    num_mel_bins: int = 80,
    frame_length: float = 25.0,  # ms
    frame_shift: float = 10.0,  # ms
) -> mx.array:

    # Calculate frame parameters
    win_length = int(sample_rate * frame_length / 1000)  # 400 for 25ms @ 16kHz
    hop_length = int(sample_rate * frame_shift / 1000)  # 160 for 10ms @ 16kHz

    # Kaldi rounds n_fft to next power of 2
    n_fft = _next_power_of_2(win_length)  # 400 -> 512

    # Ensure 1D
    if audio.ndim > 1:
        audio = audio.squeeze()

    # Kaldi snip_edges=True: only process frames that fully fit in the signal
    # Number of frames: floor((signal_len - win_length) / hop_length) + 1
    signal_len = audio.shape[0]
    num_frames = (signal_len - win_length) // hop_length + 1
    if num_frames < 1:
        num_frames = 1

    # Create Povey window
    window = _povey_window(win_length)

    # Vectorized frame extraction using mx.take (no Python loop)
    if num_frames > 0:
        # Create indices for all frames at once: (num_frames, win_length)
        frame_starts = mx.arange(num_frames) * hop_length  # (num_frames,)
        frame_offsets = mx.arange(win_length)  # (win_length,)
        # Broadcast to get all indices: (num_frames, win_length)
        indices = frame_starts[:, None] + frame_offsets[None, :]

        # Extract all frames at once
        frames = mx.take(audio, indices.flatten()).reshape(num_frames, win_length)

        # Remove DC offset per frame (vectorized)
        frames = frames - mx.mean(frames, axis=1, keepdims=True)

        # Apply pre-emphasis (vectorized): frame[1:] - 0.97 * frame[:-1]
        preemph = 0.97
        frames = mx.concatenate(
            [frames[:, :1], frames[:, 1:] - preemph * frames[:, :-1]], axis=1
        )

        # Apply window (vectorized)
        frames = frames * window[None, :]
    else:
        # Edge case: audio too short - single frame
        frame = audio
        if frame.shape[0] < win_length:
            frame = mx.concatenate([frame, mx.zeros((win_length - frame.shape[0],))])
        frame = frame[:win_length]
        frame = frame - mx.mean(frame)
        frame = mx.concatenate([frame[:1], frame[1:] - 0.97 * frame[:-1]])
        frame = frame * window
        frames = frame[None, :]  # (1, win_length)

    # Zero-pad to n_fft for FFT
    if win_length < n_fft:
        pad_amount = n_fft - win_length
        frames = mx.concatenate(
            [frames, mx.zeros((frames.shape[0], pad_amount))], axis=1
        )

    # FFT
    spec = mx.fft.rfft(frames)  # (num_frames, n_fft//2 + 1)

    # Power spectrum
    power_spec = mx.abs(spec) ** 2  # (num_frames, n_fft//2 + 1)

    # Mel filterbank (Kaldi uses HTK mel scale)
    filters = mel_filters(
        sample_rate=sample_rate,
        n_fft=n_fft,
        n_mels=num_mel_bins,
        f_min=20.0,
        f_max=sample_rate / 2,
        norm=None,
        mel_scale="htk",
    )

    # Apply filterbank: (num_frames, F) @ (F, M) -> (num_frames, M)
    mel_spec = power_spec @ filters.T

    # Log compression with small floor (Kaldi uses std::numeric_limits<float>::epsilon())
    # Note: energy_floor=1.0 applies to signal energy for dither, not mel bins
    fbank = mx.log(mx.maximum(mel_spec, 1.1920929e-07))

    return fbank


class BasicResBlock(nn.Module):

    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        # PyTorch uses stride=(stride, 1) - stride in H dim only
        # MLX Conv2d expects NHWC format
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=(stride, 1), padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm(planes)

        self.shortcut = []
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = [
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=(stride, 1),
                    bias=False,
                ),
                nn.BatchNorm(self.expansion * planes),
            ]

    def __call__(self, x: mx.array) -> mx.array:
        out = nn.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Shortcut connection
        shortcut = x
        for layer in self.shortcut:
            shortcut = layer(shortcut)
        out = out + shortcut

        return nn.relu(out)


class FCM(nn.Module):

    def __init__(
        self, block=BasicResBlock, num_blocks=[2, 2], m_channels=32, feat_dim=80
    ):
        super().__init__()
        self.in_planes = m_channels
        self.conv1 = nn.Conv2d(
            1, m_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm(m_channels)

        self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, m_channels, num_blocks[0], stride=2)

        # PyTorch uses stride=(2, 1) - stride 2 in H dim only
        self.conv2 = nn.Conv2d(
            m_channels, m_channels, kernel_size=3, stride=(2, 1), padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm(m_channels)
        self.out_channels = m_channels * (feat_dim // 8)

    def _make_layer(self, block, planes: int, num_blocks: int, stride: int):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return layers

    def __call__(self, x: mx.array) -> mx.array:

        x = mx.expand_dims(x, -1)  # (B, F, T, 1) = (B, H, W, C)
        out = nn.relu(self.bn1(self.conv1(x)))

        for layer in self.layer1:
            out = layer(out)
        for layer in self.layer2:
            out = layer(out)

        out = nn.relu(self.bn2(self.conv2(out)))

        B, H, W, C = out.shape

        out = mx.transpose(out, (0, 3, 1, 2))  # (B, C, H, W)
        out = mx.reshape(out, (B, C * H, W))
        return out


def get_nonlinear(config_str: str, channels: int):
    layers = []
    for name in config_str.split("-"):
        if name == "relu":
            layers.append(lambda x: nn.relu(x))
        elif name == "prelu":
            layers.append(nn.PReLU(channels))
        elif name == "batchnorm":
            layers.append(nn.BatchNorm(channels))
        elif name == "batchnorm_":
            layers.append(nn.BatchNorm(channels, affine=False))
        else:
            raise ValueError(f"Unexpected module: {name}")
    return layers


def statistics_pooling(x: mx.array, axis: int = -1, keepdim: bool = False) -> mx.array:
    mean = mx.mean(x, axis=axis, keepdims=keepdim)
    std = mx.sqrt(mx.var(x, axis=axis, keepdims=keepdim) + 1e-5)
    stats = mx.concatenate([mean, std], axis=-1 if not keepdim else axis)
    return stats


def conv1d_pytorch_format(x: mx.array, conv_layer) -> mx.array:
    # MLX Conv1d expects (B, T, C)
    x = mx.swapaxes(x, 1, 2)  # (B, C, T) -> (B, T, C)
    x = conv_layer(x)  # (B, T', C')
    x = mx.swapaxes(x, 1, 2)  # (B, T', C') -> (B, C', T')
    return x


class StatsPool(nn.Module):

    def __call__(self, x: mx.array) -> mx.array:
        return statistics_pooling(x)


class TDNNLayer(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = False,
        config_str: str = "batchnorm-relu",
    ):
        super().__init__()
        if padding < 0:
            assert kernel_size % 2 == 1, f"Expected odd kernel size, got {kernel_size}"
            padding = (kernel_size - 1) // 2 * dilation

        self.linear = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def __call__(self, x: mx.array) -> mx.array:
        # Input x: (B, C, T) - PyTorch format
        # MLX Conv1d expects (B, T, C)
        x = mx.swapaxes(x, 1, 2)  # (B, C, T) -> (B, T, C)
        x = self.linear(x)  # (B, T', C')
        # Apply nonlinear in MLX format (B, T, C) - BatchNorm expects features last
        for layer in self.nonlinear:
            x = layer(x) if callable(layer) else layer(x)
        # Convert back to PyTorch format
        x = mx.swapaxes(x, 1, 2)  # (B, T', C') -> (B, C', T')
        return x


class CAMLayer(nn.Module):

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

    def __call__(self, x: mx.array) -> mx.array:
        # Input x: (B, C, T) - PyTorch format
        y = conv1d_pytorch_format(x, self.linear_local)
        context = mx.mean(x, axis=-1, keepdims=True) + self.seg_pooling(x)
        context = nn.relu(conv1d_pytorch_format(context, self.linear1))
        m = nn.sigmoid(conv1d_pytorch_format(context, self.linear2))
        return y * m

    def seg_pooling(
        self, x: mx.array, seg_len: int = 100, stype: str = "avg"
    ) -> mx.array:
        B, C, T = x.shape

        # Calculate number of segments (ceil mode)
        n_segs = (T + seg_len - 1) // seg_len

        # Pad x to be divisible by seg_len
        pad_len = n_segs * seg_len - T
        if pad_len > 0:
            x_padded = mx.concatenate([x, mx.zeros((B, C, pad_len))], axis=-1)
        else:
            x_padded = x

        # Reshape to (B, C, n_segs, seg_len)
        x_reshaped = mx.reshape(x_padded, (B, C, n_segs, seg_len))

        # Pool each segment
        if stype == "avg":
            seg = mx.mean(x_reshaped, axis=-1)  # (B, C, n_segs)
        elif stype == "max":
            seg = mx.max(x_reshaped, axis=-1)  # (B, C, n_segs)
        else:
            raise ValueError("Wrong segment pooling type.")

        # Expand back: (B, C, n_segs) -> (B, C, n_segs, seg_len) -> (B, C, n_segs*seg_len)
        seg = mx.expand_dims(seg, -1)  # (B, C, n_segs, 1)
        seg = mx.broadcast_to(seg, (B, C, n_segs, seg_len))  # (B, C, n_segs, seg_len)
        seg = mx.reshape(seg, (B, C, -1))  # (B, C, n_segs*seg_len)

        # Truncate to original length
        seg = seg[:, :, :T]
        return seg


class CAMDenseTDNNLayer(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bn_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = False,
        config_str: str = "batchnorm-relu",
        memory_efficient: bool = False,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, f"Expected odd kernel size, got {kernel_size}"
        padding = (kernel_size - 1) // 2 * dilation

        self.nonlinear1 = get_nonlinear(config_str, in_channels)
        self.linear1 = nn.Conv1d(in_channels, bn_channels, 1, bias=False)
        self.nonlinear2 = get_nonlinear(config_str, bn_channels)
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

        x = mx.swapaxes(x, 1, 2)  # (B, C, T) -> (B, T, C)

        # Apply nonlinear1 (BatchNorm) in MLX format
        for layer in self.nonlinear1:
            x = layer(x) if callable(layer) else layer(x)

        # Apply linear1 (Conv1d) - stays in MLX format
        x = self.linear1(x)

        # Apply nonlinear2 (BatchNorm) in MLX format
        for layer in self.nonlinear2:
            x = layer(x) if callable(layer) else layer(x)

        # Convert back to PyTorch format for CAM layer
        x = mx.swapaxes(x, 1, 2)  # (B, T, C) -> (B, C, T)

        # CAM layer expects PyTorch format
        x = self.cam_layer(x)
        return x


class CAMDenseTDNNBlock(nn.Module):

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
        config_str: str = "batchnorm-relu",
        memory_efficient: bool = False,
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
                config_str=config_str,
                memory_efficient=memory_efficient,
            )
            self.layers.append(layer)

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = mx.concatenate([x, layer(x)], axis=1)
        return x


class TransitLayer(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        config_str: str = "batchnorm-relu",
    ):
        super().__init__()
        self.nonlinear = get_nonlinear(config_str, in_channels)
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:

        x = mx.swapaxes(x, 1, 2)  # (B, C, T) -> (B, T, C)

        for layer in self.nonlinear:
            x = layer(x) if callable(layer) else layer(x)

        x = self.linear(x)

        x = mx.swapaxes(x, 1, 2)  # (B, T, C) -> (B, C, T)
        return x


class DenseLayer(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = False,
        config_str: str = "batchnorm-relu",
    ):
        super().__init__()
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def __call__(self, x: mx.array) -> mx.array:
        if len(x.shape) == 2:
            x = mx.expand_dims(x, 1)  # (B, C) -> (B, 1, C)
            x = self.linear(x)  # (B, 1, C') - MLX Conv1d
            # Apply nonlinear in MLX format (B, T, C)
            for layer in self.nonlinear:
                x = layer(x) if callable(layer) else layer(x)
            x = mx.squeeze(x, 1)  # (B, C')
        else:
            # 3D input: (B, C, T) - PyTorch format
            # Convert to MLX format
            x = mx.swapaxes(x, 1, 2)  # (B, C, T) -> (B, T, C)
            x = self.linear(x)  # MLX Conv1d

            # Apply nonlinear in MLX format
            for layer in self.nonlinear:
                x = layer(x) if callable(layer) else layer(x)

            # Convert back to PyTorch format
            x = mx.swapaxes(x, 1, 2)  # (B, T, C) -> (B, C, T)

        return x


class CAMPPlus(nn.Module):

    def __init__(
        self,
        feat_dim: int = 80,
        embedding_size: int = 192,
        growth_rate: int = 32,
        bn_size: int = 4,
        init_channels: int = 128,
        config_str: str = "batchnorm-relu",
        memory_efficient: bool = True,
        output_level: str = "segment",
        **kwargs,
    ):
        super().__init__()

        self.head = FCM(feat_dim=feat_dim)
        channels = self.head.out_channels
        self.output_level = output_level

        # Build xvector network
        self.tdnn = TDNNLayer(
            channels,
            init_channels,
            5,
            stride=2,
            dilation=1,
            padding=-1,
            config_str=config_str,
        )
        channels = init_channels

        # Dense TDNN blocks
        self.blocks = []
        self.transits = []

        for i, (num_layers, kernel_size, dilation) in enumerate(
            zip((12, 24, 16), (3, 3, 3), (1, 2, 2))
        ):
            block = CAMDenseTDNNBlock(
                num_layers=num_layers,
                in_channels=channels,
                out_channels=growth_rate,
                bn_channels=bn_size * growth_rate,
                kernel_size=kernel_size,
                dilation=dilation,
                config_str=config_str,
                memory_efficient=memory_efficient,
            )
            self.blocks.append(block)
            channels = channels + num_layers * growth_rate

            transit = TransitLayer(
                channels, channels // 2, bias=False, config_str=config_str
            )
            self.transits.append(transit)
            channels //= 2

        self.out_nonlinear = get_nonlinear(config_str, channels)

        if output_level == "segment":
            self.stats = StatsPool()
            self.dense = DenseLayer(
                channels * 2, embedding_size, config_str="batchnorm_"
            )

    def sanitize(self, weights: dict) -> dict:

        import re

        from mlx.utils import tree_flatten

        new_weights = {}

        # Get expected shapes from model for idempotent transposition
        curr_weights = dict(tree_flatten(self.parameters()))

        for key, value in weights.items():
            new_key = key

            # Skip num_batches_tracked
            if "num_batches_tracked" in key:
                continue

            # === Name mapping for xvector structure ===
            # xvector.block1 -> blocks.0, xvector.block2 -> blocks.1, etc.
            new_key = re.sub(
                r"xvector\.block(\d+)\.",
                lambda m: f"blocks.{int(m.group(1))-1}.",
                new_key,
            )
            # xvector.transit1 -> transits.0, etc.
            new_key = re.sub(
                r"xvector\.transit(\d+)\.",
                lambda m: f"transits.{int(m.group(1))-1}.",
                new_key,
            )
            # xvector.tdnn -> tdnn (remove xvector prefix)
            new_key = new_key.replace("xvector.tdnn.", "tdnn.")
            # xvector.dense -> dense
            new_key = new_key.replace("xvector.dense.", "dense.")
            # xvector.out_nonlinear -> out_nonlinear
            new_key = new_key.replace("xvector.out_nonlinear.", "out_nonlinear.")

            # === DenseBlock internal structure ===
            # .tdnnd1. -> .layers.0., .tdnnd2. -> .layers.1., etc.
            new_key = re.sub(
                r"\.tdnnd(\d+)\.", lambda m: f".layers.{int(m.group(1))-1}.", new_key
            )

            # === NonLinear structure ===
            # .nonlinear1.batchnorm. -> .nonlinear1.0. (BatchNorm in Sequential)
            new_key = re.sub(
                r"\.nonlinear(\d+)\.batchnorm\.", r".nonlinear\1.0.", new_key
            )
            # .nonlinear.batchnorm. -> .nonlinear.0.
            new_key = new_key.replace(".nonlinear.batchnorm.", ".nonlinear.0.")
            # out_nonlinear.batchnorm. -> out_nonlinear.0. (at start or with dot)
            new_key = new_key.replace(".out_nonlinear.batchnorm.", ".out_nonlinear.0.")
            if new_key.startswith("out_nonlinear.batchnorm."):
                new_key = new_key.replace(
                    "out_nonlinear.batchnorm.", "out_nonlinear.0.", 1
                )

            # === Conv weight transposition (idempotent) ===
            # Only transpose if shape doesn't match expected MLX format
            # Handle both "conv" layers and "shortcut" layers (which are also Conv2d)
            if "weight" in new_key and value.ndim == 4:
                # Conv2d: PyTorch (O,I,H,W) -> MLX (O,H,W,I)
                if (
                    new_key in curr_weights
                    and value.shape != curr_weights[new_key].shape
                ):
                    value = value.transpose(0, 2, 3, 1)
            elif "weight" in new_key and value.ndim == 3:
                # Conv1d: PyTorch (O,I,K) -> MLX (O,K,I)
                if (
                    new_key in curr_weights
                    and value.shape != curr_weights[new_key].shape
                ):
                    value = value.swapaxes(1, 2)

            new_weights[new_key] = value

        return new_weights

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: Input features (B, T, F) where F is feat_dim

        Returns:
            Speaker embeddings (B, embedding_size) if segment-level
        """
        x = mx.swapaxes(x, 1, 2)  # (B, T, F) => (B, F, T) - PyTorch format
        x = self.head(x)
        x = self.tdnn(x)

        # Dense blocks with transitions
        for block, transit in zip(self.blocks, self.transits):
            x = block(x)
            x = transit(x)

        # Output nonlinearity - convert to MLX format for BatchNorm
        x = mx.swapaxes(x, 1, 2)  # (B, C, T) -> (B, T, C)
        for layer in self.out_nonlinear:
            x = layer(x) if callable(layer) else layer(x)
        x = mx.swapaxes(x, 1, 2)  # (B, T, C) -> (B, C, T)

        if self.output_level == "segment":
            x = self.stats(x)
            x = self.dense(x)
            # Remove last dimension if needed
            if x.ndim == 3 and x.shape[-1] == 1:
                x = mx.squeeze(x, -1)

        return x

    def inference(self, audio: mx.array) -> mx.array:
        """
        Inference on raw audio waveform.

        Args:
            audio: Audio waveform (B, T) or (T,) at 16kHz

        Returns:
            Speaker embeddings (B, embedding_size)
        """
        # Handle batched or single audio
        if audio.ndim == 1:
            audio = mx.expand_dims(audio, 0)

        # Extract fbank features for each audio in batch
        features = []
        for i in range(audio.shape[0]):
            fbank = kaldi_fbank(audio[i], num_mel_bins=80)
            # Mean normalization (as in original Chatterbox)
            fbank = fbank - mx.mean(fbank, axis=0, keepdims=True)
            features.append(fbank)

        # Pad to same length
        max_len = max(f.shape[0] for f in features)
        padded_features = []
        for f in features:
            if f.shape[0] < max_len:
                pad = mx.zeros((max_len - f.shape[0], f.shape[1]))
                f = mx.concatenate([f, pad], axis=0)
            padded_features.append(f)

        # Stack to batch: (B, T, F)
        batch_features = mx.stack(padded_features)

        return self(batch_features)
