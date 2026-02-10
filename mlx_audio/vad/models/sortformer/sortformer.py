"""
Sortformer: Speaker Diarization model ported from NVIDIA NeMo.

Architecture:
  1. FastConformer Encoder (fc_encoder): Conv subsampling + Conformer layers
     with relative positional attention
  2. Transformer Encoder (tf_encoder): BART-style encoder layers with
     learned positional embeddings
  3. Sortformer Modules: Linear projection + feedforward + sigmoid output
     for N speakers
"""

import math
import time
from dataclasses import dataclass
from typing import Dict, Generator, Iterable, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from scipy import signal as scipy_signal

from mlx_audio.audio_io import read as audio_read
from mlx_audio.dsp import hanning, mel_filters, stft

from .config import FCEncoderConfig, ModelConfig, ModulesConfig, TFEncoderConfig

# NeMo constants
_LOG_GUARD = 2**-24  # NeMo's log_zero_guard_value
_NORM_CONSTANT = 1e-5  # NeMo's normalization epsilon


# =============================================================================
# Feature Extraction
# =============================================================================


def preemphasis_filter(waveform: mx.array, coeff: float = 0.97) -> mx.array:
    """Apply preemphasis filter: y[n] = x[n] - coeff * x[n-1]."""
    return mx.concatenate(
        [waveform[..., :1], waveform[..., 1:] - coeff * waveform[..., :-1]], axis=-1
    )


def extract_mel_features(
    waveform: mx.array,
    sample_rate: int = 16000,
    n_fft: int = 512,
    hop_length: int = 160,
    win_length: int = 400,
    n_mels: int = 80,
    preemphasis_coeff: float = 0.97,
    normalize: str = "per_feature",
    pad_to: int = 16,
) -> mx.array:
    """Extract log-mel spectrogram features matching NeMo's FilterbankFeatures.

    Args:
        waveform: (num_samples,) or (batch, num_samples)
        normalize: "per_feature" for per-mel-bin normalization, None to skip
        pad_to: pad output frames to a multiple of this value (0 to disable)
        Returns: (batch, n_mels, num_frames) matching NeMo convention
    """
    if waveform.ndim == 1:
        waveform = waveform[None, :]  # Add batch dim

    # Apply preemphasis
    waveform = preemphasis_filter(waveform, preemphasis_coeff)

    batch_size = waveform.shape[0]

    # NeMo uses librosa mel filters with norm="slaney" and slaney mel scale
    # (librosa defaults: htk=False → slaney scale, norm="slaney" per NeMo default)
    mel_fb = mel_filters(
        sample_rate=sample_rate,
        n_fft=n_fft,
        n_mels=n_mels,
        f_min=0,
        f_max=None,
        norm="slaney",
        mel_scale="slaney",
    )  # (n_mels, n_fft//2+1)

    # Center-padded window matching PyTorch's torch.stft behavior:
    # When win_length < n_fft, PyTorch center-pads the window with zeros
    window = hanning(win_length)  # symmetric (periodic=False)
    if win_length < n_fft:
        left = (n_fft - win_length) // 2
        right = n_fft - win_length - left
        window = mx.concatenate([mx.zeros((left,)), window, mx.zeros((right,))])

    all_features = []
    for b in range(batch_size):
        # STFT with constant padding (matching NeMo's default pad_mode)
        spec = stft(
            waveform[b],
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=True,
            pad_mode="constant",
        )  # (num_frames, n_fft//2+1) complex

        # Power spectrum: |STFT|^2 (magnitude then square, same as mag_power=2.0)
        power = mx.abs(spec) ** 2  # (num_frames, n_fft//2+1)

        # Apply mel filterbank: power @ mel_fb^T -> (num_frames, n_mels)
        mel_spec = power @ mel_fb.T  # (num_frames, n_mels)

        # Log with NeMo's guard value
        mel_spec = mx.log(mel_spec + _LOG_GUARD)

        all_features.append(mel_spec.T)  # (n_mels, num_frames)

    features = mx.stack(all_features)  # (batch, n_mels, num_frames)

    # Per-feature normalization (NeMo normalize="per_feature")
    # Normalize each mel bin to zero mean and unit variance across time
    if normalize == "per_feature":
        # features: (batch, n_mels, num_frames)
        mean = mx.mean(features, axis=2, keepdims=True)  # (batch, n_mels, 1)
        # Bessel's correction: divide by (N-1)
        var = mx.sum((features - mean) ** 2, axis=2, keepdims=True) / (
            features.shape[2] - 1
        )
        std = mx.sqrt(var)
        features = (features - mean) / (std + _NORM_CONSTANT)

    # Pad to multiple of pad_to frames (NeMo default: 16)
    if pad_to > 0:
        num_frames = features.shape[2]
        remainder = num_frames % pad_to
        if remainder > 0:
            pad_size = pad_to - remainder
            features = mx.pad(features, [(0, 0), (0, 0), (0, pad_size)])

    return features


# =============================================================================
# FastConformer Encoder Components
# =============================================================================


class ConvSubsampling(nn.Module):
    """Depthwise-striding convolutional subsampling (factor=8).

    NeMo dw_striding layout (indices match the nn.Sequential indices):
      0: Conv2d(1, 256, 3, stride=2, padding=1)          - regular conv
      1: activation (no weights)
      2: Conv2d(256, 256, 3, stride=2, padding=1, groups=256) - depthwise
      3: Conv2d(256, 256, 1)                               - pointwise
      4: activation (no weights)
      5: Conv2d(256, 256, 3, stride=2, padding=1, groups=256) - depthwise
      6: Conv2d(256, 256, 1)                               - pointwise
    """

    def __init__(self, config: FCEncoderConfig):
        super().__init__()
        feat_in = config.num_mel_bins
        conv_channels = config.subsampling_conv_channels
        feat_out = config.hidden_size
        ks = config.subsampling_conv_kernel_size
        stride = config.subsampling_conv_stride
        pad = (ks - 1) // 2

        # Layer 0: regular conv (1 -> conv_channels, stride=2)
        self.layers_0 = nn.Conv2d(
            1, conv_channels, kernel_size=ks, stride=stride, padding=pad
        )
        # Layer 2: depthwise conv (conv_channels -> conv_channels, stride=2, groups=conv_channels)
        self.layers_2 = nn.Conv2d(
            conv_channels,
            conv_channels,
            kernel_size=ks,
            stride=stride,
            padding=pad,
            groups=conv_channels,
        )
        # Layer 3: pointwise conv (conv_channels -> conv_channels, kernel=1)
        self.layers_3 = nn.Conv2d(conv_channels, conv_channels, kernel_size=1)
        # Layer 5: depthwise conv (conv_channels -> conv_channels, stride=2, groups=conv_channels)
        self.layers_5 = nn.Conv2d(
            conv_channels,
            conv_channels,
            kernel_size=ks,
            stride=stride,
            padding=pad,
            groups=conv_channels,
        )
        # Layer 6: pointwise conv (conv_channels -> conv_channels, kernel=1)
        self.layers_6 = nn.Conv2d(conv_channels, conv_channels, kernel_size=1)

        # Linear projection: conv_channels * (feat_in // 8) -> feat_out
        linear_in = conv_channels * (feat_in // 8)
        if feat_in % 8 != 0:
            linear_in = conv_channels * math.ceil(feat_in / 8)
        self.linear = nn.Linear(linear_in, feat_out)

    def __call__(self, x: mx.array, lengths: mx.array) -> Tuple[mx.array, mx.array]:
        """
        Args:
            x: (batch, feat_dim, time) - mel spectrogram
            lengths: (batch,) - lengths in frames
        Returns:
            x: (batch, time//8, hidden_size)
            lengths: (batch,) - subsampled lengths
        """
        # NeMo Conv2d input: (B, C=1, H=time, W=feat) in NCHW
        # MLX Conv2d input: (B, H=time, W=feat, C=1) in NHWC
        # Input x is (batch, feat_dim, time), so transpose to (batch, time, feat_dim)
        x = mx.transpose(x, axes=(0, 2, 1))  # (B, time, feat_dim)
        x = mx.expand_dims(x, axis=-1)  # (B, time, feat_dim, 1)

        # Stage 1: regular conv + ReLU
        x = nn.relu(self.layers_0(x))
        # Stage 2: depthwise + pointwise + ReLU
        x = nn.relu(self.layers_3(self.layers_2(x)))
        # Stage 3: depthwise + pointwise + ReLU
        x = nn.relu(self.layers_6(self.layers_5(x)))

        # x is (batch, time//8, feat_dim//8, conv_channels) in NHWC
        # NeMo flattens as (batch, time//8, conv_channels * feat_dim//8)
        # Transpose: (b, t, f, c) -> (b, t, c, f) then reshape
        b, t, f, c = x.shape
        x = mx.transpose(x, axes=(0, 1, 3, 2))  # (b, t, c, f)
        x = x.reshape(b, t, c * f)
        x = self.linear(x)

        # Update lengths using NeMo's calc_length formula:
        # output = floor((input + 2*padding - kernel_size) / stride) + 1
        # With padding=1, kernel_size=3, stride=2: floor((L - 1) / 2) + 1
        for _ in range(3):  # 3 stages of stride-2
            lengths = mx.floor((lengths - 1) / 2).astype(mx.int32) + 1

        return x, lengths


class RelPositionalEncoding(nn.Module):
    """Relative positional encoding for Conformer (Transformer-XL style)."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

    def __call__(self, x: mx.array) -> mx.array:
        """Generate relative positional encoding.
        Args:
            x: (batch, time, d_model)
        Returns:
            pos_emb: (1, 2*time-1, d_model)
        """
        seq_len = x.shape[1]
        positions = mx.arange(seq_len - 1, -seq_len, -1, dtype=mx.float32)
        # positions shape: (2*seq_len - 1,)

        dim = mx.arange(0, self.d_model, 2, dtype=mx.float32)
        div_term = mx.exp(dim * -(math.log(10000.0) / self.d_model))

        # (2*seq_len-1, d_model//2)
        angles = positions[:, None] * div_term[None, :]
        pe = mx.zeros((positions.shape[0], self.d_model))
        pe = pe.at[:, 0::2].add(mx.sin(angles))
        pe = pe.at[:, 1::2].add(mx.cos(angles))
        return pe[None, :, :]  # (1, 2*seq_len-1, d_model)


class RelPositionMultiHeadAttention(nn.Module):
    """Multi-head attention with relative positional encoding (Transformer-XL)."""

    def __init__(self, config: FCEncoderConfig):
        super().__init__()
        n_feat = config.hidden_size
        n_head = config.num_attention_heads
        self.h = n_head
        self.d_k = n_feat // n_head
        self.s_d_k = math.sqrt(self.d_k)

        self.q_proj = nn.Linear(n_feat, n_feat, bias=config.attention_bias)
        self.k_proj = nn.Linear(n_feat, n_feat, bias=config.attention_bias)
        self.v_proj = nn.Linear(n_feat, n_feat, bias=config.attention_bias)
        self.o_proj = nn.Linear(n_feat, n_feat, bias=config.attention_bias)
        self.relative_k_proj = nn.Linear(n_feat, n_feat, bias=False)

        # Learnable biases for content and position attention
        self.bias_u = mx.zeros((n_head, self.d_k))
        self.bias_v = mx.zeros((n_head, self.d_k))

    def rel_shift(self, x: mx.array) -> mx.array:
        """Compute relative positional encoding shift."""
        b, h, qlen, pos_len = x.shape
        # Pad left
        x = mx.pad(x, [(0, 0), (0, 0), (0, 0), (1, 0)])
        x = x.reshape(b, h, pos_len + 1, qlen)
        x = x[:, :, 1:, :].reshape(b, h, qlen, pos_len)
        return x

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: Optional[mx.array] = None,
        pos_emb: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Args:
            query, key, value: (batch, time, d_model)
            mask: (batch, 1, time, time) or None
            pos_emb: (1, 2*time-1, d_model)
        Returns:
            output: (batch, time, d_model)
        """
        n_batch = query.shape[0]

        q = self.q_proj(query).reshape(n_batch, -1, self.h, self.d_k)
        k = self.k_proj(key).reshape(n_batch, -1, self.h, self.d_k)
        v = self.v_proj(value).reshape(n_batch, -1, self.h, self.d_k)

        # Transpose to (batch, head, time, d_k)
        q = mx.transpose(q, axes=(0, 2, 1, 3))
        k = mx.transpose(k, axes=(0, 2, 1, 3))
        v = mx.transpose(v, axes=(0, 2, 1, 3))

        # For relative position: q needs to be (batch, time, head, d_k) first
        q_t = mx.transpose(q, axes=(0, 2, 1, 3))  # (batch, time, head, d_k)

        # Positional encoding
        p = self.relative_k_proj(pos_emb).reshape(1, -1, self.h, self.d_k)
        p = mx.transpose(p, axes=(0, 2, 1, 3))  # (1, head, 2*time-1, d_k)

        # q + bias_u for content attention, q + bias_v for position attention
        q_with_bias_u = mx.transpose(
            q_t + self.bias_u, axes=(0, 2, 1, 3)
        )  # (batch, head, time, d_k)
        q_with_bias_v = mx.transpose(
            q_t + self.bias_v, axes=(0, 2, 1, 3)
        )  # (batch, head, time, d_k)

        # Content attention: matrix_ac
        matrix_ac = q_with_bias_u @ mx.transpose(k, axes=(0, 1, 3, 2))

        # Position attention: matrix_bd
        matrix_bd = q_with_bias_v @ mx.transpose(p, axes=(0, 1, 3, 2))
        matrix_bd = self.rel_shift(matrix_bd)

        # Trim to match content attention size
        matrix_bd = matrix_bd[:, :, :, : matrix_ac.shape[-1]]

        scores = (matrix_ac + matrix_bd) / self.s_d_k

        if mask is not None:
            scores = mx.where(mask, mx.array(-1e4), scores)

        attn = mx.softmax(scores, axis=-1)
        if mask is not None:
            attn = mx.where(mask, mx.array(0.0), attn)

        x = attn @ v  # (batch, head, time, d_k)
        x = mx.transpose(x, axes=(0, 2, 1, 3)).reshape(n_batch, -1, self.h * self.d_k)
        return self.o_proj(x)


class ConformerFeedForward(nn.Module):
    """Conformer feed-forward module."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear2(nn.silu(self.linear1(x)))


class ConformerConvolution(nn.Module):
    """Conformer convolution module with GLU, depthwise conv, and batch norm."""

    def __init__(self, config: FCEncoderConfig):
        super().__init__()
        d_model = config.hidden_size
        kernel_size = config.conv_kernel_size

        # Pointwise conv1: d_model -> 2*d_model (for GLU)
        self.pointwise_conv1 = nn.Conv1d(d_model, d_model * 2, kernel_size=1, bias=True)
        # Depthwise conv
        self.depthwise_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=d_model,
            bias=True,
        )
        # BatchNorm (stored as running stats for inference)
        self.norm = BatchNorm1d(d_model)
        # Pointwise conv2: d_model -> d_model
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: (batch, time, d_model)
        Returns:
            x: (batch, time, d_model)
        """
        # (batch, time, d_model) -> (batch, d_model, time) via transpose
        # MLX Conv1d expects (batch, time, channels), so we work in that space
        # But the PyTorch model transposes to (batch, channels, time) for convs
        # MLX Conv1d: input (N, L, C_in) -> output (N, L_out, C_out)

        x = self.pointwise_conv1(x)  # (batch, time, 2*d_model)

        # GLU: split and gate
        x1, x2 = mx.split(x, 2, axis=-1)
        x = x1 * mx.sigmoid(x2)  # (batch, time, d_model)

        x = self.depthwise_conv(x)  # (batch, time, d_model)
        x = self.norm(x)
        x = nn.silu(x)
        x = self.pointwise_conv2(x)  # (batch, time, d_model)

        return x


class BatchNorm1d(nn.Module):
    """Batch normalization using stored running statistics (inference mode only)."""

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.weight = mx.ones((num_features,))
        self.bias = mx.zeros((num_features,))
        self.running_mean = mx.zeros((num_features,))
        self.running_var = mx.ones((num_features,))

    def __call__(self, x: mx.array) -> mx.array:
        """Apply batch norm using running stats.
        Args:
            x: (batch, time, features)
        Returns:
            x: (batch, time, features)
        """
        return (x - self.running_mean) / mx.sqrt(
            self.running_var + self.eps
        ) * self.weight + self.bias


class ConformerLayer(nn.Module):
    """Single Conformer encoder layer.

    Structure: FF1 -> Self-Attn -> Conv -> FF2 -> LayerNorm
    """

    def __init__(self, config: FCEncoderConfig):
        super().__init__()
        d_model = config.hidden_size
        d_ff = config.intermediate_size
        self.fc_factor = 0.5

        # Feed-forward 1
        self.norm_feed_forward1 = nn.LayerNorm(d_model)
        self.feed_forward1 = ConformerFeedForward(d_model, d_ff)

        # Self-attention
        self.norm_self_att = nn.LayerNorm(d_model)
        self.self_attn = RelPositionMultiHeadAttention(config)

        # Convolution
        self.norm_conv = nn.LayerNorm(d_model)
        self.conv = ConformerConvolution(config)

        # Feed-forward 2
        self.norm_feed_forward2 = nn.LayerNorm(d_model)
        self.feed_forward2 = ConformerFeedForward(d_model, d_ff)

        # Output norm
        self.norm_out = nn.LayerNorm(d_model)

    def __call__(
        self,
        x: mx.array,
        pos_emb: mx.array,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Args:
            x: (batch, time, d_model)
            pos_emb: (1, 2*time-1, d_model)
            mask: optional attention mask
        """
        # FF1
        residual = x
        x = self.norm_feed_forward1(x)
        x = self.feed_forward1(x)
        residual = residual + x * self.fc_factor

        # Self-attention
        x = self.norm_self_att(residual)
        x = self.self_attn(x, x, x, mask=mask, pos_emb=pos_emb)
        residual = residual + x

        # Conv
        x = self.norm_conv(residual)
        x = self.conv(x)
        residual = residual + x

        # FF2
        x = self.norm_feed_forward2(residual)
        x = self.feed_forward2(x)
        residual = residual + x * self.fc_factor

        # Output norm
        x = self.norm_out(residual)
        return x


class FastConformerEncoder(nn.Module):
    """FastConformer encoder with conv subsampling and Conformer layers."""

    def __init__(self, config: FCEncoderConfig):
        super().__init__()
        self.config = config
        self.subsampling = ConvSubsampling(config)
        self.layers = [ConformerLayer(config) for _ in range(config.num_hidden_layers)]
        self.pos_enc = RelPositionalEncoding(
            config.hidden_size, config.max_position_embeddings
        )
        self.scale_input = config.scale_input

    def pre_encode(
        self, audio_signal: mx.array, length: mx.array
    ) -> Tuple[mx.array, mx.array]:
        """Run only ConvSubsampling (first stage, used for streaming).

        Args:
            audio_signal: (batch, n_mels, time) - mel spectrogram
            length: (batch,) - lengths in mel frames
        Returns:
            x: (batch, time//8, hidden_size) - pre-encoded embeddings
            lengths: (batch,) - subsampled lengths
        """
        return self.subsampling(audio_signal, length)

    def encode(
        self, embeddings: mx.array, lengths: mx.array
    ) -> Tuple[mx.array, mx.array]:
        """Run Conformer layers on pre-encoded embeddings (bypass_pre_encode).

        Args:
            embeddings: (batch, time, hidden_size) - pre-encoded embeddings
            lengths: (batch,) - valid lengths
        Returns:
            x: (batch, hidden_size, time) - encoder output (channels first)
            lengths: (batch,) - unchanged
        """
        x = embeddings
        if self.scale_input:
            x = x * (self.config.hidden_size**0.5)

        pos_emb = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, pos_emb)

        x = mx.transpose(x, axes=(0, 2, 1))
        return x, lengths

    def __call__(
        self, audio_signal: mx.array, length: mx.array
    ) -> Tuple[mx.array, mx.array]:
        """Full forward: ConvSubsampling + Conformer layers.

        Args:
            audio_signal: (batch, n_mels, time) - mel spectrogram
            length: (batch,) - lengths in frames
        Returns:
            x: (batch, hidden_size, time//8) - encoder output (channels first)
            lengths: (batch,) - subsampled lengths
        """
        x, lengths = self.pre_encode(audio_signal, length)
        return self.encode(x, lengths)


# =============================================================================
# Transformer Encoder Components (BART-style)
# =============================================================================


class TransformerAttention(nn.Module):
    """Standard multi-head attention for the Transformer encoder."""

    def __init__(self, config: TFEncoderConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = config.encoder_attention_heads
        self.head_dim = self.embed_dim // self.num_heads

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.scale = self.head_dim**-0.5

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        B, T, _ = query.shape

        q = (
            self.q_proj(query)
            .reshape(B, T, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        k = (
            self.k_proj(key)
            .reshape(B, -1, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        v = (
            self.v_proj(value)
            .reshape(B, -1, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )

        scores = (q * self.scale) @ k.transpose(0, 1, 3, 2)

        if mask is not None:
            scores = scores + mask

        attn = mx.softmax(scores, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, self.embed_dim)
        return self.out_proj(out)


class TransformerEncoderLayer(nn.Module):
    """Single Transformer encoder layer (post-LN, BART-style)."""

    def __init__(self, config: TFEncoderConfig):
        super().__init__()
        self.self_attn = TransformerAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(
            config.d_model, eps=config.layer_norm_eps
        )
        self.fc1 = nn.Linear(config.d_model, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, config.d_model)
        self.final_layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.activation_fn = nn.relu

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        """Post-LN: Attn -> Add -> LN -> FFN -> Add -> LN"""
        residual = x
        x = self.self_attn(x, x, x, mask=mask)
        x = residual + x
        x = self.self_attn_layer_norm(x)

        residual = x
        x = self.activation_fn(self.fc1(x))
        x = self.fc2(x)
        x = residual + x
        x = self.final_layer_norm(x)

        return x


class TransformerEncoder(nn.Module):
    """Transformer encoder with learned positional embeddings."""

    def __init__(self, config: TFEncoderConfig):
        super().__init__()
        self.config = config
        self.embed_positions = nn.Embedding(config.max_source_positions, config.d_model)
        self.layers = [
            TransformerEncoderLayer(config) for _ in range(config.encoder_layers)
        ]

    def __call__(
        self,
        encoder_states: mx.array,
        encoder_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Args:
            encoder_states: (batch, time, d_model)
            encoder_mask: (batch, time) - True where valid
        Returns:
            output: (batch, time, d_model)
        """
        seq_len = encoder_states.shape[1]

        # Add positional embeddings
        positions = mx.arange(seq_len)
        pos_emb = self.embed_positions(positions)
        x = encoder_states + pos_emb

        # Create attention mask from encoder_mask
        attn_mask = None
        if encoder_mask is not None:
            # (batch, 1, 1, time) - mask for padding positions
            attn_mask = (~encoder_mask)[:, None, None, :].astype(mx.float32) * -1e4

        for layer in self.layers:
            x = layer(x, mask=attn_mask)

        return x


# =============================================================================
# Sortformer Modules
# =============================================================================


class SortformerModules(nn.Module):
    """Sortformer output modules: projection + feedforward + speaker sigmoid."""

    def __init__(self, config: ModulesConfig):
        super().__init__()
        self.n_spk = config.num_speakers
        self.fc_d_model = config.fc_d_model
        self.tf_d_model = config.tf_d_model

        # Projection from FC encoder dim to TF encoder dim
        self.encoder_proj = nn.Linear(config.fc_d_model, config.tf_d_model)

        # Speaker output layers
        self.first_hidden_to_hidden = nn.Linear(config.tf_d_model, config.tf_d_model)
        self.single_hidden_to_spks = nn.Linear(config.tf_d_model, config.num_speakers)
        self.hidden_to_spks = nn.Linear(2 * config.tf_d_model, config.num_speakers)

    def forward_speaker_sigmoids(self, hidden_out: mx.array) -> mx.array:
        """Compute speaker probabilities.
        Args:
            hidden_out: (batch, time, tf_d_model)
        Returns:
            preds: (batch, time, num_speakers)
        """
        hidden_out = nn.relu(hidden_out)
        hidden_out = self.first_hidden_to_hidden(hidden_out)
        hidden_out = nn.relu(hidden_out)
        spk_preds = self.single_hidden_to_spks(hidden_out)
        preds = mx.sigmoid(spk_preds)
        return preds

    @staticmethod
    def length_to_mask(lengths: mx.array, max_length: int) -> mx.array:
        """Convert lengths to boolean mask.
        Args:
            lengths: (batch,)
            max_length: int
        Returns:
            mask: (batch, max_length) - True where valid
        """
        arange = mx.arange(max_length)
        return arange[None, :] < lengths[:, None]


# =============================================================================
# Diarization Output
# =============================================================================


@dataclass
class DiarizationSegment:
    """A single diarization segment."""

    start: float
    end: float
    speaker: int


@dataclass
class DiarizationOutput:
    """Output from the diarization model."""

    segments: List[DiarizationSegment]
    speaker_probs: Optional[mx.array] = None
    num_speakers: int = 0
    total_time: float = 0.0
    state: Optional["StreamingState"] = None

    @property
    def text(self) -> str:
        """Format as RTTM-like text output."""
        lines = []
        for seg in self.segments:
            duration = seg.end - seg.start
            lines.append(
                f"SPEAKER audio 1 {seg.start:.3f} {duration:.3f} <NA> <NA> speaker_{seg.speaker} <NA> <NA>"
            )
        return "\n".join(lines)


@dataclass
class StreamingState:
    """State maintained between streaming diarization chunks.

    The streaming architecture maintains two buffers of pre-encoded embeddings
    (after ConvSubsampling, before Conformer layers):

    - **spkcache** (speaker cache): Long-term context, compressed when full
      by keeping the most informative frames based on prediction scores.
    - **fifo**: Recent context buffer. Oldest frames roll into spkcache
      when the FIFO overflows.

    Each streaming step processes ``[spkcache + fifo + new_chunk]`` through the
    full Conformer + Transformer encoder, but only emits predictions for the
    new chunk.
    """

    spkcache: mx.array  # (1, cache_frames, emb_dim)
    spkcache_preds: mx.array  # (1, cache_frames, n_spk)
    fifo: mx.array  # (1, fifo_frames, emb_dim)
    fifo_preds: mx.array  # (1, fifo_frames, n_spk)
    frames_processed: int  # total diarization frames emitted so far

    @property
    def spkcache_len(self) -> int:
        return self.spkcache.shape[1]

    @property
    def fifo_len(self) -> int:
        return self.fifo.shape[1]


# =============================================================================
# Main Model
# =============================================================================


class Model(nn.Module):
    """Sortformer speaker diarization model.

    Architecture:
        1. Feature extraction (mel spectrogram)
        2. FastConformer encoder (conv subsampling + conformer layers)
        3. Transformer encoder (BART-style)
        4. Sortformer modules (feedforward + sigmoid output)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # FastConformer encoder
        self.fc_encoder = FastConformerEncoder(config.fc_encoder_config)

        # Transformer encoder
        self.tf_encoder = TransformerEncoder(config.tf_encoder_config)

        # Sortformer modules
        self.sortformer_modules = SortformerModules(config.modules_config)

        self._processor_config = config.processor_config

    def __call__(
        self,
        audio_signal: mx.array,
        audio_signal_length: mx.array,
    ) -> mx.array:
        """Full forward pass.
        Args:
            audio_signal: (batch, n_mels, time) - mel features
            audio_signal_length: (batch,) - feature lengths
        Returns:
            preds: (batch, diar_frame_count, num_speakers)
        """
        # FastConformer encoder
        emb_seq, emb_seq_length = self.fc_encoder(audio_signal, audio_signal_length)
        # emb_seq: (batch, hidden_size, time//8)

        # Transpose to (batch, time, hidden_size)
        emb_seq = mx.transpose(emb_seq, axes=(0, 2, 1))

        # Project to transformer dimension if needed
        if self.sortformer_modules.encoder_proj is not None:
            emb_seq = self.sortformer_modules.encoder_proj(emb_seq)

        # Create mask
        encoder_mask = SortformerModules.length_to_mask(
            emb_seq_length, emb_seq.shape[1]
        )

        # Transformer encoder
        trans_emb_seq = self.tf_encoder(
            encoder_states=emb_seq, encoder_mask=encoder_mask
        )

        # Speaker sigmoids
        preds = self.sortformer_modules.forward_speaker_sigmoids(trans_emb_seq)

        # Apply mask
        preds = preds * encoder_mask[:, :, None]

        return preds

    def generate(
        self,
        audio: Union[str, np.ndarray, mx.array],
        *,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        min_duration: float = 0.0,
        merge_gap: float = 0.0,
        verbose: bool = False,
    ) -> DiarizationOutput:
        """Run speaker diarization on audio.

        Args:
            audio: Path to audio file, numpy array, or mx.array
            sample_rate: Sample rate of input audio
            threshold: Speaker activity threshold (0-1)
            min_duration: Minimum segment duration in seconds
            merge_gap: Maximum gap to merge consecutive segments
            verbose: Print progress information

        Returns:
            DiarizationOutput with speaker segments and probabilities
        """
        start_time = time.time()

        waveform, sample_rate = self._load_audio(audio, sample_rate)
        proc = self._processor_config

        # Trim leading/trailing silence using energy-based VAD.
        # Per-feature normalization is distorted by long silence regions
        # (NeMo's pipeline uses a VAD model for this; we use a simple energy check).
        waveform, trim_offset = self._trim_silence(waveform, proc.sampling_rate)
        trim_offset_sec = trim_offset / proc.sampling_rate

        waveform = mx.array(waveform)

        # Peak normalization (NeMo style: eps=1e-3)
        waveform = (1.0 / (mx.max(mx.abs(waveform)) + 1e-3)) * waveform

        # Extract features
        features = extract_mel_features(
            waveform,
            sample_rate=proc.sampling_rate,
            n_fft=proc.n_fft,
            hop_length=proc.hop_length,
            win_length=proc.win_length,
            n_mels=proc.feature_size,
            preemphasis_coeff=proc.preemphasis,
        )
        # features: (1, n_mels, num_frames)

        feature_lengths = mx.array([features.shape[2]])

        if verbose:
            print(f"Audio: {waveform.shape[-1] / proc.sampling_rate:.2f}s")
            if trim_offset > 0:
                print(f"Trimmed {trim_offset_sec:.2f}s leading silence")
            print(f"Features: {features.shape}")

        # Forward pass
        preds = self(features, feature_lengths)
        mx.eval(preds)

        # preds: (1, diar_frames, num_speakers)
        preds_np = np.array(preds[0])

        # Post-process predictions to segments
        subsampling_factor = self.config.fc_encoder_config.subsampling_factor
        frame_duration = (proc.hop_length * subsampling_factor) / proc.sampling_rate

        segments = self._preds_to_segments(
            preds_np,
            frame_duration=frame_duration,
            threshold=threshold,
            min_duration=min_duration,
            merge_gap=merge_gap,
        )

        # Shift segment timestamps back to original audio timeline
        if trim_offset > 0:
            segments = [
                DiarizationSegment(
                    start=seg.start + trim_offset_sec,
                    end=seg.end + trim_offset_sec,
                    speaker=seg.speaker,
                )
                for seg in segments
            ]

        # Count unique speakers
        active_speakers = set(seg.speaker for seg in segments)

        elapsed = time.time() - start_time

        if verbose:
            print(
                f"Found {len(segments)} segments with {len(active_speakers)} speakers"
            )
            print(f"Processing time: {elapsed:.2f}s")

        return DiarizationOutput(
            segments=segments,
            speaker_probs=preds[0],
            num_speakers=len(active_speakers),
            total_time=elapsed,
        )

    # =====================================================================
    # Streaming API
    # =====================================================================

    def init_streaming_state(self) -> StreamingState:
        """Create an empty streaming state.

        Returns:
            A fresh StreamingState with empty speaker cache and FIFO.
        """
        emb_dim = self.config.fc_encoder_config.hidden_size
        n_spk = self.config.modules_config.num_speakers
        empty_emb = mx.zeros((1, 0, emb_dim))
        empty_pred = mx.zeros((1, 0, n_spk))
        return StreamingState(
            spkcache=empty_emb,
            spkcache_preds=empty_pred,
            fifo=empty_emb,
            fifo_preds=empty_pred,
            frames_processed=0,
        )

    def streaming_step(
        self,
        chunk_features: mx.array,
        chunk_length: mx.array,
        state: StreamingState,
    ) -> Tuple[mx.array, StreamingState]:
        """Process one chunk of mel features through the streaming pipeline.

        Each call pre-encodes the chunk, concatenates it with the cached
        context ``[spkcache + fifo + chunk]``, runs the full encoder, and
        returns predictions for the *new chunk only*.

        Args:
            chunk_features: ``(1, n_mels, chunk_mel_frames)`` mel features.
            chunk_length: ``(1,)`` valid length in mel frames.
            state: Current :class:`StreamingState`.

        Returns:
            ``(chunk_preds, new_state)`` where ``chunk_preds`` is an mx.array
            of shape ``(chunk_diar_frames, n_spk)`` with speaker
            probabilities for this chunk.
        """
        # Step 1: Pre-encode chunk through ConvSubsampling
        chunk_embs, chunk_emb_lengths = self.fc_encoder.pre_encode(
            chunk_features, chunk_length
        )
        # chunk_embs: (1, chunk_diar_frames, emb_dim)
        chunk_diar_len = int(chunk_emb_lengths[0].item())
        chunk_embs = chunk_embs[:, :chunk_diar_len, :]

        # Step 2: Concatenate [spkcache, fifo, chunk]
        parts = []
        if state.spkcache_len > 0:
            parts.append(state.spkcache)
        if state.fifo_len > 0:
            parts.append(state.fifo)
        parts.append(chunk_embs)

        all_embs = mx.concatenate(parts, axis=1)  # (1, total, emb_dim)
        total_len = all_embs.shape[1]
        all_lengths = mx.array([total_len])

        # Step 3: Run Conformer layers on full context
        fc_out, _ = self.fc_encoder.encode(all_embs, all_lengths)
        # fc_out: (1, emb_dim, total_len)
        fc_out = mx.transpose(fc_out, axes=(0, 2, 1))  # (1, total, emb_dim)

        # Step 4: encoder_proj + Transformer + speaker sigmoids
        if self.sortformer_modules.encoder_proj is not None:
            fc_out = self.sortformer_modules.encoder_proj(fc_out)

        encoder_mask = SortformerModules.length_to_mask(all_lengths, total_len)
        trans_out = self.tf_encoder(fc_out, encoder_mask)
        all_preds = self.sortformer_modules.forward_speaker_sigmoids(trans_out)
        all_preds = all_preds * encoder_mask[:, :, None]

        # Step 5: Extract predictions for the new chunk only
        chunk_start = state.spkcache_len + state.fifo_len
        chunk_preds = all_preds[:, chunk_start : chunk_start + chunk_diar_len, :]

        # Also extract updated predictions for spkcache and fifo
        # (the model re-attends over the full context each step)
        updated_cache_preds = all_preds[:, : state.spkcache_len, :]
        updated_fifo_preds = all_preds[
            :, state.spkcache_len : state.spkcache_len + state.fifo_len, :
        ]

        # Eval all outputs to materialize data and release the forward-pass
        # graph (attention matrices, encoder intermediates, etc.).
        mx.eval(chunk_preds, chunk_embs, updated_cache_preds, updated_fifo_preds)

        # Step 6: Update streaming state
        new_state = self._update_streaming_state(
            state,
            chunk_embs,
            chunk_preds,
            updated_cache_preds,
            updated_fifo_preds,
        )

        return chunk_preds[0], new_state

    def generate_stream(
        self,
        audio: Union[str, np.ndarray, mx.array, Iterable[np.ndarray]],
        *,
        state: Optional[StreamingState] = None,
        sample_rate: int = 16000,
        chunk_duration: float = 5.0,
        threshold: float = 0.5,
        min_duration: float = 0.0,
        merge_gap: float = 0.0,
        spkcache_max: int = 188,
        fifo_max: int = 188,
        verbose: bool = False,
    ) -> Generator[DiarizationOutput, None, None]:
        """Process audio in chunks, yielding diarization results incrementally.

        Supports three modes:

        1. **File / full array** (no ``state``): loads audio, extracts features
           with global normalization, and processes in fixed-duration chunks.
        2. **Iterable of chunks** (no ``state``): each chunk is independently
           normalized and processed, simulating real-time streaming.
        3. **Single chunk + state**: processes one chunk through the streaming
           pipeline and yields a single result with the updated ``state``
           attached (``result.state``).

        Args:
            audio: Audio input — one of:

                - ``str``: path to an audio file
                - ``np.ndarray`` or ``mx.array``: full waveform (or a single
                  chunk when ``state`` is provided)
                - ``Iterable[np.ndarray]``: pre-built audio chunks

            state: Optional streaming state. When provided with a single
                array, processes that one chunk and attaches the updated
                state to the yielded result's ``state`` field. Use
                :meth:`init_streaming_state` to create the initial state.
            sample_rate: Sample rate of input audio.
            chunk_duration: Duration of each chunk in seconds (ignored when
                ``audio`` is an iterable of chunks or ``state`` is provided).
            threshold: Speaker activity threshold (0-1).
            min_duration: Minimum segment duration in seconds.
            merge_gap: Maximum gap to merge consecutive segments.
            spkcache_max: Maximum speaker cache size in diarization frames.
            fifo_max: Maximum FIFO size in diarization frames.
            verbose: Print progress information.

        Yields:
            :class:`DiarizationOutput` for each chunk. When ``state`` is
            provided, the yielded result includes ``result.state`` with
            the updated :class:`StreamingState`.
        """
        # --- Single chunk + state: delegate to feed ---
        if state is not None and isinstance(audio, (np.ndarray, mx.array)):
            result, new_state = self.feed(
                audio,
                state,
                sample_rate=sample_rate,
                threshold=threshold,
                min_duration=min_duration,
                merge_gap=merge_gap,
                spkcache_max=spkcache_max,
                fifo_max=fifo_max,
            )
            result.state = new_state
            yield result
            return

        # --- Iterable of chunks: per-chunk normalization path ---
        if not isinstance(audio, (str, np.ndarray, mx.array)):
            yield from self._stream_from_chunks(
                audio,
                sample_rate=sample_rate,
                threshold=threshold,
                min_duration=min_duration,
                merge_gap=merge_gap,
                spkcache_max=spkcache_max,
                fifo_max=fifo_max,
                verbose=verbose,
            )
            return

        # --- Audio loading (same as generate) ---
        waveform, sample_rate = self._load_audio(audio, sample_rate)
        proc = self._processor_config
        waveform, trim_offset = self._trim_silence(waveform, proc.sampling_rate)
        trim_offset_sec = trim_offset / proc.sampling_rate

        waveform = mx.array(waveform)
        waveform = (1.0 / (mx.max(mx.abs(waveform)) + 1e-3)) * waveform

        # --- Feature extraction (global normalization) ---
        features = extract_mel_features(
            waveform,
            sample_rate=proc.sampling_rate,
            n_fft=proc.n_fft,
            hop_length=proc.hop_length,
            win_length=proc.win_length,
            n_mels=proc.feature_size,
            preemphasis_coeff=proc.preemphasis,
        )
        # features: (1, n_mels, total_mel_frames)
        total_mel_frames = features.shape[2]

        subsampling_factor = self.config.fc_encoder_config.subsampling_factor
        frame_duration = (proc.hop_length * subsampling_factor) / proc.sampling_rate

        # Chunk size in mel frames, aligned to subsampling factor
        chunk_mel = (
            round(
                chunk_duration
                * proc.sampling_rate
                / proc.hop_length
                / subsampling_factor
            )
            * subsampling_factor
        )
        chunk_mel = max(chunk_mel, subsampling_factor)

        if verbose:
            audio_dur = waveform.shape[-1] / proc.sampling_rate
            n_chunks = math.ceil(total_mel_frames / chunk_mel)
            print(
                f"Streaming: {audio_dur:.2f}s audio in {n_chunks} chunks "
                f"({chunk_duration:.1f}s each)"
            )

        # --- Streaming loop ---
        state = self.init_streaming_state()
        offset_mel = 0
        chunk_idx = 0

        while offset_mel < total_mel_frames:
            end_mel = min(offset_mel + chunk_mel, total_mel_frames)
            chunk_feat = features[:, :, offset_mel:end_mel]
            chunk_len = mx.array([chunk_feat.shape[2]])

            chunk_preds, state = self.streaming_step(chunk_feat, chunk_len, state)

            # Time offset for this chunk's diarization frames
            chunk_time_offset = offset_mel / proc.sampling_rate * proc.hop_length
            # Actually: each mel frame = hop_length samples, and diar frames
            # are subsampled by 8, so the time offset is:
            chunk_time_offset = (offset_mel * proc.hop_length) / proc.sampling_rate

            segments = self._preds_to_segments(
                chunk_preds,
                frame_duration=frame_duration,
                threshold=threshold,
                min_duration=min_duration,
                merge_gap=merge_gap,
            )

            # Shift to absolute timeline
            segments = [
                DiarizationSegment(
                    start=seg.start + chunk_time_offset + trim_offset_sec,
                    end=seg.end + chunk_time_offset + trim_offset_sec,
                    speaker=seg.speaker,
                )
                for seg in segments
            ]

            active_speakers = set(seg.speaker for seg in segments)

            if verbose:
                chunk_idx += 1
                t0 = chunk_time_offset + trim_offset_sec
                t1 = t0 + chunk_preds.shape[0] * frame_duration
                print(
                    f"  Chunk {chunk_idx}: {t0:.2f}s-{t1:.2f}s  "
                    f"{len(segments)} segments, "
                    f"context={state.spkcache_len}+{state.fifo_len} frames"
                )

            yield DiarizationOutput(
                segments=segments,
                speaker_probs=chunk_preds,
                num_speakers=len(active_speakers),
            )

            # Compress state if buffers exceed limits
            state = self._maybe_compress_state(state, spkcache_max, fifo_max)

            offset_mel = end_mel

    def _stream_from_chunks(
        self,
        audio_chunks: Iterable[np.ndarray],
        *,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        min_duration: float = 0.0,
        merge_gap: float = 0.0,
        spkcache_max: int = 188,
        fifo_max: int = 188,
        verbose: bool = False,
    ) -> Generator[DiarizationOutput, None, None]:
        """Yield diarization results from an iterable of raw audio chunks."""
        state = self.init_streaming_state()
        chunk_idx = 0

        for raw_chunk in audio_chunks:
            result, state = self.feed(
                raw_chunk,
                state,
                sample_rate=sample_rate,
                threshold=threshold,
                min_duration=min_duration,
                merge_gap=merge_gap,
                spkcache_max=spkcache_max,
                fifo_max=fifo_max,
            )

            if verbose:
                chunk_idx += 1
                print(
                    f"  Chunk {chunk_idx}: "
                    f"{len(result.segments)} segments, "
                    f"context={state.spkcache_len}+{state.fifo_len} frames"
                )

            yield result

    def feed(
        self,
        chunk: Union[np.ndarray, mx.array],
        state: StreamingState,
        *,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        min_duration: float = 0.0,
        merge_gap: float = 0.0,
        spkcache_max: int = 188,
        fifo_max: int = 188,
    ) -> Tuple[DiarizationOutput, StreamingState]:
        """Feed a single audio chunk and get diarization results.

        Designed for real-time streaming where audio arrives incrementally
        (e.g. from a microphone).  Each chunk is independently
        peak-normalized and feature-extracted, then processed through the
        streaming pipeline.

        Use :meth:`init_streaming_state` to create the initial state, then
        call ``feed()`` repeatedly as audio arrives.

        Args:
            chunk: 1-D audio samples (mono, ``float32``).
            state: Current :class:`StreamingState` (from ``init_streaming_state``
                or a previous ``feed`` call).
            sample_rate: Sample rate of the audio chunk.
            threshold: Speaker activity threshold (0-1).
            min_duration: Minimum segment duration in seconds.
            merge_gap: Maximum gap to merge consecutive segments.
            spkcache_max: Maximum speaker cache size (diarization frames).
            fifo_max: Maximum FIFO size (diarization frames).

        Returns:
            ``(output, new_state)`` — the diarization result for this chunk
            and the updated streaming state.

        Example::

            state = model.init_streaming_state()
            for chunk in mic_stream():          # your audio source
                result, state = model.feed(chunk, state)
                for seg in result.segments:
                    print(f"Speaker {seg.speaker}: {seg.start:.2f}s-{seg.end:.2f}s")
        """
        proc = self._processor_config
        subsampling_factor = self.config.fc_encoder_config.subsampling_factor
        frame_duration = (proc.hop_length * subsampling_factor) / proc.sampling_rate

        raw = np.asarray(chunk, dtype=np.float32)
        if raw.ndim > 1:
            raw = raw.mean(axis=-1)

        # Resample if needed
        if sample_rate != proc.sampling_rate:
            raw = self._resample(raw, sample_rate, proc.sampling_rate)

        chunk_time_offset = state.frames_processed * frame_duration

        # Per-chunk peak normalization
        chunk_mx = mx.array(raw)
        chunk_mx = (1.0 / (mx.max(mx.abs(chunk_mx)) + 1e-3)) * chunk_mx

        # Per-chunk feature extraction
        features = extract_mel_features(
            chunk_mx,
            sample_rate=proc.sampling_rate,
            n_fft=proc.n_fft,
            hop_length=proc.hop_length,
            win_length=proc.win_length,
            n_mels=proc.feature_size,
            preemphasis_coeff=proc.preemphasis,
            pad_to=0,
        )
        feature_lengths = mx.array([features.shape[2]])

        chunk_preds, state = self.streaming_step(features, feature_lengths, state)

        segments = self._preds_to_segments(
            chunk_preds,
            frame_duration=frame_duration,
            threshold=threshold,
            min_duration=min_duration,
            merge_gap=merge_gap,
        )

        # Shift to absolute timeline
        segments = [
            DiarizationSegment(
                start=seg.start + chunk_time_offset,
                end=seg.end + chunk_time_offset,
                speaker=seg.speaker,
            )
            for seg in segments
        ]

        state = self._maybe_compress_state(state, spkcache_max, fifo_max)

        active_speakers = set(seg.speaker for seg in segments)
        output = DiarizationOutput(
            segments=segments,
            speaker_probs=chunk_preds,
            num_speakers=len(active_speakers),
        )
        return output, state

    @staticmethod
    def _update_streaming_state(
        state: StreamingState,
        chunk_embs: mx.array,
        chunk_preds: mx.array,
        updated_cache_preds: mx.array,
        updated_fifo_preds: mx.array,
    ) -> StreamingState:
        """Push chunk into FIFO, updating predictions with re-attended values.

        All inputs are mx.arrays that have been eval'd by the caller to
        materialize data and release the forward-pass computation graph.
        """
        spkcache = state.spkcache
        spkcache_preds = (
            updated_cache_preds if state.spkcache_len > 0 else state.spkcache_preds
        )
        fifo_preds = updated_fifo_preds if state.fifo_len > 0 else state.fifo_preds

        new_fifo = mx.concatenate([state.fifo, chunk_embs], axis=1)
        new_fifo_preds = mx.concatenate([fifo_preds, chunk_preds], axis=1)
        mx.eval(new_fifo, new_fifo_preds)

        return StreamingState(
            spkcache=spkcache,
            spkcache_preds=spkcache_preds,
            fifo=new_fifo,
            fifo_preds=new_fifo_preds,
            frames_processed=state.frames_processed + chunk_preds.shape[1],
        )

    @staticmethod
    def _maybe_compress_state(
        state: StreamingState,
        spkcache_max: int,
        fifo_max: int,
    ) -> StreamingState:
        """Move FIFO overflow into speaker cache, compressing if needed."""
        if state.fifo_len <= fifo_max:
            return state

        pop_len = state.fifo_len - fifo_max
        new_cache = mx.concatenate([state.spkcache, state.fifo[:, :pop_len, :]], axis=1)
        new_cache_preds = mx.concatenate(
            [state.spkcache_preds, state.fifo_preds[:, :pop_len, :]], axis=1
        )

        # Compress speaker cache if it exceeds the limit
        if new_cache.shape[1] > spkcache_max:
            new_cache, new_cache_preds = Model._compress_spkcache(
                new_cache, new_cache_preds, spkcache_max
            )

        new_fifo = state.fifo[:, pop_len:, :]
        new_fifo_preds = state.fifo_preds[:, pop_len:, :]

        # Eval to materialize slices and release references to old arrays
        mx.eval(new_cache, new_cache_preds, new_fifo, new_fifo_preds)

        return StreamingState(
            spkcache=new_cache,
            spkcache_preds=new_cache_preds,
            fifo=new_fifo,
            fifo_preds=new_fifo_preds,
            frames_processed=state.frames_processed,
        )

    @staticmethod
    def _compress_spkcache(
        embs: mx.array,
        preds: mx.array,
        target_len: int,
    ) -> Tuple[mx.array, mx.array]:
        """Compress speaker cache by keeping the most informative frames.

        Frames are scored by their total speaker activity (sum of
        log-predictions across speakers).  The top ``target_len`` frames
        are kept in temporal order.

        Args:
            embs: ``(1, N, emb_dim)`` pre-encoded embeddings.
            preds: ``(1, N, n_spk)`` speaker predictions.
            target_len: Desired number of frames after compression.

        Returns:
            ``(compressed_embs, compressed_preds)`` each with
            ``target_len`` frames.
        """
        # Score frames by speaker activity importance
        log_preds = mx.log(mx.clip(preds[0], 1e-7, 1.0))
        frame_scores = mx.sum(log_preds, axis=-1)  # (N,)

        # Select top-k frames preserving temporal order
        top_indices = mx.argsort(-frame_scores)[:target_len]
        top_indices = mx.sort(top_indices)

        compressed_embs = embs[:, top_indices, :]
        compressed_preds = preds[:, top_indices, :]
        mx.eval(compressed_embs, compressed_preds)

        return compressed_embs, compressed_preds

    @staticmethod
    def _preds_to_segments(
        preds: Union[mx.array, np.ndarray],
        frame_duration: float,
        threshold: float = 0.5,
        min_duration: float = 0.0,
        merge_gap: float = 0.0,
    ) -> List[DiarizationSegment]:
        """Convert frame-level predictions to time segments.

        Args:
            preds: (num_frames, num_speakers) - speaker probabilities
            frame_duration: Duration of each frame in seconds
            threshold: Activity threshold
            min_duration: Minimum segment duration
            merge_gap: Maximum gap to merge segments

        Returns:
            List of DiarizationSegment
        """
        if isinstance(preds, mx.array):
            preds = np.array(preds)
        _, num_speakers = preds.shape
        segments = []

        for spk in range(num_speakers):
            activity = preds[:, spk] > threshold
            if not activity.any():
                continue

            # Find contiguous active regions
            changes = np.diff(activity.astype(int), prepend=0, append=0)
            starts = np.where(changes == 1)[0]
            ends = np.where(changes == -1)[0]

            spk_segments = []
            for s, e in zip(starts, ends):
                start_time = s * frame_duration
                end_time = e * frame_duration
                duration = end_time - start_time

                if duration >= min_duration:
                    spk_segments.append(
                        DiarizationSegment(
                            start=start_time,
                            end=end_time,
                            speaker=spk,
                        )
                    )

            # Merge close segments
            if merge_gap > 0 and len(spk_segments) > 1:
                merged = [spk_segments[0]]
                for seg in spk_segments[1:]:
                    if seg.start - merged[-1].end <= merge_gap:
                        merged[-1] = DiarizationSegment(
                            start=merged[-1].start,
                            end=seg.end,
                            speaker=seg.speaker,
                        )
                    else:
                        merged.append(seg)
                spk_segments = merged

            segments.extend(spk_segments)

        # Sort by start time
        segments.sort(key=lambda s: s.start)
        return segments

    @staticmethod
    def _trim_silence(
        waveform: np.ndarray,
        sample_rate: int,
        frame_ms: int = 30,
        energy_ratio: float = 0.01,
        min_speech_sec: float = 0.5,
    ) -> Tuple[np.ndarray, int]:
        """Trim leading/trailing silence from audio using frame energy.

        Per-feature normalization is distorted when silence dominates the audio.
        NeMo's pipeline uses a neural VAD; this is a lightweight energy-based
        alternative that handles the common case of leading/trailing silence.

        Uses an adaptive threshold (fraction of peak energy) so it works across
        different recording levels. Requires min_speech_sec of consecutive speech
        to avoid triggering on brief noise bursts.

        Args:
            waveform: (num_samples,) audio samples
            sample_rate: sample rate in Hz
            frame_ms: frame length in milliseconds for energy computation
            energy_ratio: fraction of peak RMS energy below which a frame is silence
            min_speech_sec: require this many seconds of consecutive speech

        Returns:
            (trimmed_waveform, trim_offset_samples)
        """
        frame_len = int(sample_rate * frame_ms / 1000)
        min_speech_frames = max(3, int(min_speech_sec * 1000 / frame_ms))
        num_frames = len(waveform) // frame_len

        if num_frames < min_speech_frames * 2:
            return waveform, 0

        # Compute per-frame RMS energy
        frames = waveform[: num_frames * frame_len].reshape(num_frames, frame_len)
        energy = np.sqrt(np.mean(frames**2, axis=1))

        # Adaptive threshold: fraction of peak energy
        threshold = energy.max() * energy_ratio
        speech = energy > threshold

        # Find first and last runs of min_speech_frames consecutive speech
        start_frame = 0
        for i in range(num_frames - min_speech_frames + 1):
            if all(speech[i : i + min_speech_frames]):
                start_frame = i
                break

        end_frame = num_frames
        for i in range(num_frames - 1, min_speech_frames - 2, -1):
            if all(speech[i - min_speech_frames + 1 : i + 1]):
                end_frame = i + 1
                break

        start_sample = start_frame * frame_len
        end_sample = min(end_frame * frame_len, len(waveform))

        # Only trim if we'd remove a meaningful amount of silence
        if start_sample == 0 and end_sample == len(waveform):
            return waveform, 0

        return waveform[start_sample:end_sample], start_sample

    def _load_audio(
        self,
        audio: Union[str, np.ndarray, mx.array],
        sample_rate: int,
    ) -> Tuple[np.ndarray, int]:
        """Load and prepare audio from any supported input.

        Handles file paths (via ``audio_io.read``), numpy arrays, and
        mx arrays.  Converts to mono float32 and resamples to the model's
        expected sample rate.

        Returns:
            ``(waveform, sample_rate)`` where waveform is 1-D float32 numpy.
        """
        if isinstance(audio, str):
            waveform, sr = audio_read(audio, dtype="float32")
            sample_rate = sr
        elif isinstance(audio, np.ndarray):
            waveform = audio.astype(np.float32)
        else:
            waveform = np.array(audio, dtype=np.float32)

        if waveform.ndim > 1:
            waveform = waveform.mean(axis=-1)

        proc = self._processor_config
        if sample_rate != proc.sampling_rate:
            waveform = self._resample(waveform, sample_rate, proc.sampling_rate)

        return waveform, proc.sampling_rate

    @staticmethod
    def _resample(waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio using scipy (no librosa dependency)."""
        if orig_sr == target_sr:
            return waveform
        gcd = math.gcd(orig_sr, target_sr)
        return scipy_signal.resample_poly(
            waveform, target_sr // gcd, orig_sr // gcd
        ).astype(np.float32)

    @staticmethod
    def sanitize(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Transform HuggingFace weights to match MLX model structure.

        The HF safetensors has keys like:
            fc_encoder.layers.N.conv.depthwise_conv.weight  (Conv1d: out, 1, K)
            fc_encoder.layers.N.conv.pointwise_conv1.weight (Conv1d: out, in, 1)
            fc_encoder.subsampling.layers.N.weight           (Conv2d: out, in, H, W)
            tf_encoder.layers.N.self_attn.q_proj.weight
            sortformer_modules.encoder_proj.weight
            etc.
        """
        sanitized = {}

        # Keys to skip (batch norm running stats that aren't needed for the
        # nn.Module parameters, but we DO need running_mean/running_var for inference)
        skip_keys = {"num_batches_tracked"}

        for k, v in weights.items():
            # Skip certain keys
            if any(sk in k for sk in skip_keys):
                continue

            new_k = k

            # Remap subsampling conv layer indices
            # HF: fc_encoder.subsampling.layers.0.weight -> layers_0.weight
            # HF: fc_encoder.subsampling.layers.2.weight -> layers_2.weight
            # HF: fc_encoder.subsampling.layers.4.weight -> layers_4.weight
            if "fc_encoder.subsampling.layers." in new_k:
                # fc_encoder.subsampling.layers.{idx}.weight ->
                # fc_encoder.subsampling.layers_{idx}.weight
                new_k = new_k.replace("subsampling.layers.", "subsampling.layers_")

            # Handle Conv2d weights: PyTorch [out, in, H, W] -> MLX [out, H, W, in]
            if "subsampling" in new_k and "weight" in new_k and "linear" not in new_k:
                if v.ndim == 4:
                    v = mx.transpose(v, axes=(0, 2, 3, 1))

            # Handle Conv1d weights: PyTorch [out, in, K] -> MLX [out, K, in]
            # For pointwise_conv1, pointwise_conv2, depthwise_conv
            if (
                any(
                    conv_name in new_k
                    for conv_name in [
                        "pointwise_conv1",
                        "pointwise_conv2",
                        "depthwise_conv",
                    ]
                )
                and "weight" in new_k
            ):
                if v.ndim == 3:
                    v = mx.transpose(v, axes=(0, 2, 1))

            # Remap batch norm keys
            # HF: fc_encoder.layers.N.conv.norm.weight -> BatchNorm1d weight
            # The NeMo model uses nn.BatchNorm1d which has weight, bias, running_mean, running_var

            # Remap self_attn bias_u and bias_v
            # These are stored as parameters in the attention layers
            # HF key: fc_encoder.layers.N.self_attn.bias_u
            # MLX key: fc_encoder.layers.N.self_attn.bias_u (same)

            # Remap relative_k_proj -> relative_k_proj
            # HF: fc_encoder.layers.N.self_attn.relative_k_proj.weight
            # (This is the linear_pos in NeMo)

            sanitized[new_k] = v

        return sanitized
