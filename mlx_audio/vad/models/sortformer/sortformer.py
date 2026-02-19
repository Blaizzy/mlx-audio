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

from mlx_audio.audio_io import read as audio_read
from mlx_audio.dsp import hanning, mel_filters, stft

from .config import FCEncoderConfig, ModelConfig, ModulesConfig, TFEncoderConfig

_LOG_GUARD = 2**-24
_NORM_CONSTANT = 1e-5


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
        waveform = waveform[None, :]

    waveform = preemphasis_filter(waveform, preemphasis_coeff)
    batch_size = waveform.shape[0]

    mel_fb = mel_filters(
        sample_rate=sample_rate,
        n_fft=n_fft,
        n_mels=n_mels,
        f_min=0,
        f_max=None,
        norm="slaney",
        mel_scale="slaney",
    )

    # Center-pad window when win_length < n_fft (matching torch.stft behavior)
    window = hanning(win_length)
    if win_length < n_fft:
        left = (n_fft - win_length) // 2
        right = n_fft - win_length - left
        window = mx.concatenate([mx.zeros((left,)), window, mx.zeros((right,))])

    all_features = []
    for b in range(batch_size):
        spec = stft(
            waveform[b],
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=True,
            pad_mode="constant",
        )
        power = mx.abs(spec) ** 2
        mel_spec = power @ mel_fb.T
        mel_spec = mx.log(mel_spec + _LOG_GUARD)
        all_features.append(mel_spec.T)

    features = mx.stack(all_features)

    if normalize == "per_feature":
        mean = mx.mean(features, axis=2, keepdims=True)
        # Bessel's correction: divide by (N-1)
        var = mx.sum((features - mean) ** 2, axis=2, keepdims=True) / (
            features.shape[2] - 1
        )
        std = mx.sqrt(var)
        features = (features - mean) / (std + _NORM_CONSTANT)

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

        self.layers_0 = nn.Conv2d(
            1, conv_channels, kernel_size=ks, stride=stride, padding=pad
        )
        self.layers_2 = nn.Conv2d(
            conv_channels,
            conv_channels,
            kernel_size=ks,
            stride=stride,
            padding=pad,
            groups=conv_channels,
        )
        self.layers_3 = nn.Conv2d(conv_channels, conv_channels, kernel_size=1)
        self.layers_5 = nn.Conv2d(
            conv_channels,
            conv_channels,
            kernel_size=ks,
            stride=stride,
            padding=pad,
            groups=conv_channels,
        )
        self.layers_6 = nn.Conv2d(conv_channels, conv_channels, kernel_size=1)

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
        # (batch, feat_dim, time) -> NHWC for MLX Conv2d
        x = mx.transpose(x, axes=(0, 2, 1))
        x = mx.expand_dims(x, axis=-1)

        x = nn.relu(self.layers_0(x))
        x = nn.relu(self.layers_3(self.layers_2(x)))
        x = nn.relu(self.layers_6(self.layers_5(x)))

        # NHWC -> (b, t, c, f) for flatten to match NeMo's NCHW ordering
        b, t, f, c = x.shape
        x = mx.transpose(x, axes=(0, 1, 3, 2))
        x = x.reshape(b, t, c * f)
        x = self.linear(x)

        # floor((L - 1) / 2) + 1 per stride-2 stage
        for _ in range(3):
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

        dim = mx.arange(0, self.d_model, 2, dtype=mx.float32)
        div_term = mx.exp(dim * -(math.log(10000.0) / self.d_model))

        angles = positions[:, None] * div_term[None, :]
        pe = mx.zeros((positions.shape[0], self.d_model))
        pe = pe.at[:, 0::2].add(mx.sin(angles))
        pe = pe.at[:, 1::2].add(mx.cos(angles))
        return pe[None, :, :].astype(x.dtype)


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

        q = mx.transpose(q, axes=(0, 2, 1, 3))
        k = mx.transpose(k, axes=(0, 2, 1, 3))
        v = mx.transpose(v, axes=(0, 2, 1, 3))

        q_t = mx.transpose(q, axes=(0, 2, 1, 3))

        p = self.relative_k_proj(pos_emb).reshape(1, -1, self.h, self.d_k)
        p = mx.transpose(p, axes=(0, 2, 1, 3))

        q_with_bias_u = mx.transpose(q_t + self.bias_u, axes=(0, 2, 1, 3))
        q_with_bias_v = mx.transpose(q_t + self.bias_v, axes=(0, 2, 1, 3))

        matrix_ac = q_with_bias_u @ mx.transpose(k, axes=(0, 1, 3, 2))
        matrix_bd = q_with_bias_v @ mx.transpose(p, axes=(0, 1, 3, 2))
        matrix_bd = self.rel_shift(matrix_bd)
        matrix_bd = matrix_bd[:, :, :, : matrix_ac.shape[-1]]

        scores = (matrix_ac + matrix_bd) / self.s_d_k

        if mask is not None:
            scores = mx.where(mask, mx.array(-1e4, dtype=scores.dtype), scores)

        attn = mx.softmax(scores, axis=-1)
        if mask is not None:
            attn = mx.where(mask, mx.array(0.0, dtype=scores.dtype), attn)

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

        self.pointwise_conv1 = nn.Conv1d(d_model, d_model * 2, kernel_size=1, bias=True)
        self.depthwise_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=d_model,
            bias=True,
        )
        self.norm = BatchNorm1d(d_model)
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: (batch, time, d_model)
        Returns:
            x: (batch, time, d_model)
        """
        x = self.pointwise_conv1(x)

        # GLU
        x1, x2 = mx.split(x, 2, axis=-1)
        x = x1 * mx.sigmoid(x2)

        x = self.depthwise_conv(x)
        x = self.norm(x)
        x = nn.silu(x)
        x = self.pointwise_conv2(x)
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

        self.norm_feed_forward1 = nn.LayerNorm(d_model)
        self.feed_forward1 = ConformerFeedForward(d_model, d_ff)
        self.norm_self_att = nn.LayerNorm(d_model)
        self.self_attn = RelPositionMultiHeadAttention(config)
        self.norm_conv = nn.LayerNorm(d_model)
        self.conv = ConformerConvolution(config)
        self.norm_feed_forward2 = nn.LayerNorm(d_model)
        self.feed_forward2 = ConformerFeedForward(d_model, d_ff)
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
        residual = x
        x = self.norm_feed_forward1(x)
        x = self.feed_forward1(x)
        residual = residual + x * self.fc_factor

        x = self.norm_self_att(residual)
        x = self.self_attn(x, x, x, mask=mask, pos_emb=pos_emb)
        residual = residual + x

        x = self.norm_conv(residual)
        x = self.conv(x)
        residual = residual + x

        x = self.norm_feed_forward2(residual)
        x = self.feed_forward2(x)
        residual = residual + x * self.fc_factor

        return self.norm_out(residual)


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
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.k_proj_bias)
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
        positions = mx.arange(seq_len)
        x = encoder_states + self.embed_positions(positions)

        attn_mask = None
        if encoder_mask is not None:
            attn_mask = (~encoder_mask)[:, None, None, :].astype(
                encoder_states.dtype
            ) * -1e4

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
    # AOSC silence profile (v2.1)
    mean_sil_emb: mx.array  # (1, emb_dim) running mean silence embedding
    n_sil_frames: mx.array  # (1,) count of silence frames seen

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
        self.fc_encoder = FastConformerEncoder(config.fc_encoder_config)
        self.tf_encoder = TransformerEncoder(config.tf_encoder_config)
        self.sortformer_modules = SortformerModules(config.modules_config)
        self._processor_config = config.processor_config

    @property
    def dtype(self) -> mx.Dtype:
        return self.sortformer_modules.encoder_proj.weight.dtype

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
        audio_signal = audio_signal.astype(self.dtype)
        emb_seq, emb_seq_length = self.fc_encoder(audio_signal, audio_signal_length)
        emb_seq = mx.transpose(emb_seq, axes=(0, 2, 1))

        if self.sortformer_modules.encoder_proj is not None:
            emb_seq = self.sortformer_modules.encoder_proj(emb_seq)

        encoder_mask = SortformerModules.length_to_mask(
            emb_seq_length, emb_seq.shape[1]
        )
        trans_emb_seq = self.tf_encoder(
            encoder_states=emb_seq, encoder_mask=encoder_mask
        )
        preds = self.sortformer_modules.forward_speaker_sigmoids(trans_emb_seq)
        return preds * encoder_mask[:, :, None]

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

        waveform, trim_offset = self._trim_silence(waveform, proc.sampling_rate)
        trim_offset_sec = trim_offset / proc.sampling_rate

        waveform = (1.0 / (mx.max(mx.abs(waveform)) + 1e-3)) * waveform

        features = extract_mel_features(
            waveform,
            sample_rate=proc.sampling_rate,
            n_fft=proc.n_fft,
            hop_length=proc.hop_length,
            win_length=proc.win_length,
            n_mels=proc.feature_size,
            preemphasis_coeff=proc.preemphasis,
        )
        feature_lengths = mx.array([features.shape[2]])

        if verbose:
            print(f"Audio: {waveform.shape[-1] / proc.sampling_rate:.2f}s")
            if trim_offset > 0:
                print(f"Trimmed {trim_offset_sec:.2f}s leading silence")
            print(f"Features: {features.shape}")

        preds = self(features, feature_lengths)
        mx.eval(preds)

        subsampling_factor = self.config.fc_encoder_config.subsampling_factor
        frame_duration = (proc.hop_length * subsampling_factor) / proc.sampling_rate

        segments = self._preds_to_segments(
            preds[0],
            frame_duration=frame_duration,
            threshold=threshold,
            min_duration=min_duration,
            merge_gap=merge_gap,
        )

        if trim_offset > 0:
            segments = [
                DiarizationSegment(
                    start=seg.start + trim_offset_sec,
                    end=seg.end + trim_offset_sec,
                    speaker=seg.speaker,
                )
                for seg in segments
            ]

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
            mean_sil_emb=mx.zeros((1, emb_dim)),
            n_sil_frames=mx.zeros((1,)),
        )

    def streaming_step(
        self,
        chunk_features: mx.array,
        chunk_length: mx.array,
        state: StreamingState,
        right_context_embs: Optional[mx.array] = None,
    ) -> Tuple[mx.array, StreamingState]:
        """Process one chunk of mel features through the streaming pipeline.

        Each call pre-encodes the chunk, concatenates it with the cached
        context ``[spkcache + fifo + left_ctx + chunk + right_ctx]``, runs the
        full encoder, and returns predictions for the *new chunk only*.

        Args:
            chunk_features: ``(1, n_mels, chunk_mel_frames)`` mel features.
            chunk_length: ``(1,)`` valid length in mel frames.
            state: Current :class:`StreamingState`.
            right_context_embs: Optional ``(1, rc_frames, emb_dim)`` pre-encoded
                right context embeddings (file mode only).

        Returns:
            ``(chunk_preds, new_state)`` where ``chunk_preds`` is an mx.array
            of shape ``(chunk_diar_frames, n_spk)`` with speaker
            probabilities for this chunk.
        """
        mc = self.config.modules_config
        use_context = mc.use_aosc  # left/right context is v2.1 only
        lc = mc.chunk_left_context if use_context else 0
        rc = mc.chunk_right_context if use_context else 0

        # Pre-encode chunk through ConvSubsampling
        chunk_features = chunk_features.astype(self.dtype)
        chunk_embs, chunk_emb_lengths = self.fc_encoder.pre_encode(
            chunk_features, chunk_length
        )
        chunk_diar_len = int(chunk_emb_lengths[0].item())
        chunk_embs = chunk_embs[:, :chunk_diar_len, :]

        # Build left context from end of FIFO (v2.1 only)
        left_ctx = None
        left_ctx_len = 0
        if lc > 0 and state.fifo_len > 0:
            take = min(lc, state.fifo_len)
            left_ctx = state.fifo[:, -take:, :]
            left_ctx_len = take

        # Right context (v2.1 file mode only, pre-encoded by caller)
        right_ctx_len = 0
        if right_context_embs is not None and rc > 0:
            right_ctx_len = right_context_embs.shape[1]

        # Concatenate [spkcache, fifo, left_ctx, chunk, right_ctx]
        parts = []
        if state.spkcache_len > 0:
            parts.append(state.spkcache)
        if state.fifo_len > 0:
            parts.append(state.fifo)
        if left_ctx is not None:
            parts.append(left_ctx)
        parts.append(chunk_embs)
        if right_context_embs is not None and right_ctx_len > 0:
            parts.append(right_context_embs)

        all_embs = mx.concatenate(parts, axis=1)
        total_len = all_embs.shape[1]
        all_lengths = mx.array([total_len])

        # Full encoder pass over assembled sequence
        fc_out, _ = self.fc_encoder.encode(all_embs, all_lengths)
        fc_out = mx.transpose(fc_out, axes=(0, 2, 1))

        if self.sortformer_modules.encoder_proj is not None:
            fc_out = self.sortformer_modules.encoder_proj(fc_out)

        encoder_mask = SortformerModules.length_to_mask(all_lengths, total_len)
        trans_out = self.tf_encoder(fc_out, encoder_mask)
        all_preds = self.sortformer_modules.forward_speaker_sigmoids(trans_out)
        all_preds = all_preds * encoder_mask[:, :, None]

        # Extract predictions for the new chunk only (skip context regions)
        chunk_start = state.spkcache_len + state.fifo_len + left_ctx_len
        chunk_preds = all_preds[:, chunk_start : chunk_start + chunk_diar_len, :]
        updated_cache_preds = all_preds[:, : state.spkcache_len, :]
        updated_fifo_preds = all_preds[
            :, state.spkcache_len : state.spkcache_len + state.fifo_len, :
        ]

        # Eval to materialize and release the forward-pass graph
        mx.eval(chunk_preds, chunk_embs, updated_cache_preds, updated_fifo_preds)

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

        mc = self.config.modules_config
        # Use config defaults when caller uses default values
        if mc.use_aosc:
            spkcache_max = mc.spkcache_len
            fifo_max = mc.fifo_len if mc.fifo_len > 0 else fifo_max

        waveform, sample_rate = self._load_audio(audio, sample_rate)
        proc = self._processor_config

        # v2.1 streaming: skip silence trimming, peak norm, and per-feature norm
        use_v2_feats = mc.use_aosc
        if use_v2_feats:
            trim_offset_sec = 0.0
            features = extract_mel_features(
                waveform,
                sample_rate=proc.sampling_rate,
                n_fft=proc.n_fft,
                hop_length=proc.hop_length,
                win_length=proc.win_length,
                n_mels=proc.feature_size,
                preemphasis_coeff=proc.preemphasis,
                normalize=None,
                pad_to=0,
            )
        else:
            waveform, trim_offset = self._trim_silence(waveform, proc.sampling_rate)
            trim_offset_sec = trim_offset / proc.sampling_rate
            waveform = (1.0 / (mx.max(mx.abs(waveform)) + 1e-3)) * waveform
            features = extract_mel_features(
                waveform,
                sample_rate=proc.sampling_rate,
                n_fft=proc.n_fft,
                hop_length=proc.hop_length,
                win_length=proc.win_length,
                n_mels=proc.feature_size,
                preemphasis_coeff=proc.preemphasis,
            )

        total_mel_frames = features.shape[2]

        subsampling_factor = self.config.fc_encoder_config.subsampling_factor
        frame_duration = (proc.hop_length * subsampling_factor) / proc.sampling_rate

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

        # For v2.1 file mode: pre-encode all mel features so we can provide
        # right context embeddings to each chunk
        rc = mc.chunk_right_context
        all_pre_embs = None
        if use_v2_feats and rc > 0:
            all_pre_embs, _ = self.fc_encoder.pre_encode(
                features, mx.array([total_mel_frames])
            )
            mx.eval(all_pre_embs)

        if verbose:
            audio_dur = waveform.shape[-1] / proc.sampling_rate
            n_chunks = math.ceil(total_mel_frames / chunk_mel)
            print(
                f"Streaming: {audio_dur:.2f}s audio in {n_chunks} chunks "
                f"({chunk_duration:.1f}s each)"
            )

        state = self.init_streaming_state()
        offset_mel = 0
        chunk_idx = 0
        emb_offset = 0  # track position in pre-encoded embeddings

        while offset_mel < total_mel_frames:
            end_mel = min(offset_mel + chunk_mel, total_mel_frames)
            chunk_feat = features[:, :, offset_mel:end_mel]
            chunk_len = mx.array([chunk_feat.shape[2]])

            # Compute right context embeddings for file mode
            right_ctx = None
            if all_pre_embs is not None and rc > 0:
                # Figure out how many diar frames this chunk produces
                chunk_emb_len = int(
                    (
                        mx.floor(
                            (
                                mx.floor(
                                    (
                                        mx.floor(
                                            (
                                                mx.array(
                                                    [chunk_feat.shape[2]],
                                                    dtype=mx.float32,
                                                )
                                                - 1
                                            )
                                            / 2
                                        )
                                        + 1
                                        - 1
                                    )
                                    / 2
                                )
                                + 1
                                - 1
                            )
                            / 2
                        )
                        + 1
                    )[0].item()
                )
                rc_start = emb_offset + chunk_emb_len
                rc_end = min(rc_start + rc, all_pre_embs.shape[1])
                if rc_end > rc_start:
                    right_ctx = all_pre_embs[:, rc_start:rc_end, :]
                emb_offset += chunk_emb_len

            chunk_preds, state = self.streaming_step(
                chunk_feat, chunk_len, state, right_context_embs=right_ctx
            )

            chunk_time_offset = (offset_mel * proc.hop_length) / proc.sampling_rate

            segments = self._preds_to_segments(
                chunk_preds,
                frame_duration=frame_duration,
                threshold=threshold,
                min_duration=min_duration,
                merge_gap=merge_gap,
            )

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

            state = self._maybe_compress_state(
                state, spkcache_max, fifo_max, self.config.modules_config
            )

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

        if not isinstance(chunk, mx.array):
            chunk_mx = mx.array(chunk).astype(mx.float32)
        else:
            chunk_mx = chunk.astype(mx.float32)
        if chunk_mx.ndim > 1:
            chunk_mx = mx.mean(chunk_mx, axis=-1)

        if sample_rate != proc.sampling_rate:
            chunk_mx = self._resample(chunk_mx, sample_rate, proc.sampling_rate)

        chunk_time_offset = state.frames_processed * frame_duration

        use_v2_feats = self.config.modules_config.use_aosc
        if not use_v2_feats:
            chunk_mx = (1.0 / (mx.max(mx.abs(chunk_mx)) + 1e-3)) * chunk_mx

        features = extract_mel_features(
            chunk_mx,
            sample_rate=proc.sampling_rate,
            n_fft=proc.n_fft,
            hop_length=proc.hop_length,
            win_length=proc.win_length,
            n_mels=proc.feature_size,
            preemphasis_coeff=proc.preemphasis,
            normalize=None if use_v2_feats else "per_feature",
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

        segments = [
            DiarizationSegment(
                start=seg.start + chunk_time_offset,
                end=seg.end + chunk_time_offset,
                speaker=seg.speaker,
            )
            for seg in segments
        ]
        state = self._maybe_compress_state(
            state, spkcache_max, fifo_max, self.config.modules_config
        )

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
            mean_sil_emb=state.mean_sil_emb,
            n_sil_frames=state.n_sil_frames,
        )

    @staticmethod
    def _maybe_compress_state(
        state: StreamingState,
        spkcache_max: int,
        fifo_max: int,
        modules_cfg: Optional["ModulesConfig"] = None,
    ) -> StreamingState:
        """Move FIFO overflow into speaker cache, compressing if needed.

        When ``modules_cfg`` is provided and ``use_aosc`` is True, uses AOSC
        compression and transfers frames in ``spkcache_update_period``-sized
        batches with silence profile updates.  Otherwise uses the simple v1
        compression.
        """
        if state.fifo_len <= fifo_max:
            return state

        use_aosc = modules_cfg is not None and modules_cfg.use_aosc

        pop_len = state.fifo_len - fifo_max
        if use_aosc:
            # Transfer in update-period-sized batches
            pop_len = min(pop_len, modules_cfg.spkcache_update_period)

        popped_embs = state.fifo[:, :pop_len, :]
        popped_preds = state.fifo_preds[:, :pop_len, :]

        # Update silence profile from popped frames
        mean_sil_emb = state.mean_sil_emb
        n_sil_frames = state.n_sil_frames
        if use_aosc:
            mean_sil_emb, n_sil_frames = Model._get_silence_profile(
                mean_sil_emb,
                n_sil_frames,
                popped_embs,
                popped_preds,
                modules_cfg.sil_threshold,
            )

        new_cache = mx.concatenate([state.spkcache, popped_embs], axis=1)
        new_cache_preds = mx.concatenate([state.spkcache_preds, popped_preds], axis=1)

        if new_cache.shape[1] > spkcache_max:
            if use_aosc:
                new_cache, new_cache_preds = Model._compress_spkcache_aosc(
                    new_cache, new_cache_preds, mean_sil_emb, modules_cfg
                )
            else:
                new_cache, new_cache_preds = Model._compress_spkcache_simple(
                    new_cache, new_cache_preds, spkcache_max
                )

        new_fifo = state.fifo[:, pop_len:, :]
        new_fifo_preds = state.fifo_preds[:, pop_len:, :]

        mx.eval(
            new_cache,
            new_cache_preds,
            new_fifo,
            new_fifo_preds,
            mean_sil_emb,
            n_sil_frames,
        )

        return StreamingState(
            spkcache=new_cache,
            spkcache_preds=new_cache_preds,
            fifo=new_fifo,
            fifo_preds=new_fifo_preds,
            frames_processed=state.frames_processed,
            mean_sil_emb=mean_sil_emb,
            n_sil_frames=n_sil_frames,
        )

    # =================================================================
    # AOSC (Arrival-Order Speaker Cache) Compression — v2.1
    # =================================================================

    @staticmethod
    def _get_log_pred_scores(preds: mx.array, threshold: float) -> mx.array:
        """Per-frame per-speaker log-likelihood ratio scores.

        High when speaker k is confidently active alone (non-overlapped).

        Args:
            preds: (batch, n_frames, n_spk)
            threshold: min clamp for log computation
        Returns:
            scores: (batch, n_frames, n_spk)
        """
        log_probs = mx.log(mx.clip(preds, a_min=threshold, a_max=None))
        log_1_probs = mx.log(mx.clip(1.0 - preds, a_min=threshold, a_max=None))
        # sum log(1-p_j) across all speakers
        log_1_probs_sum = mx.sum(log_1_probs, axis=2, keepdims=True)
        # broadcast to (batch, n_frames, n_spk)
        log_1_probs_sum = mx.broadcast_to(log_1_probs_sum, preds.shape)
        scores = log_probs - log_1_probs + log_1_probs_sum - math.log(0.5)
        return scores

    @staticmethod
    def _disable_low_scores(
        preds: mx.array, scores: mx.array, min_pos_scores_per_spk: int
    ) -> mx.array:
        """Set scores to -inf for non-speech and overlapped-speech frames.

        Args:
            preds: (batch, n_frames, n_spk)
            scores: (batch, n_frames, n_spk)
            min_pos_scores_per_spk: minimum positive scores before filtering overlap
        Returns:
            scores: (batch, n_frames, n_spk)
        """
        neg_inf = mx.array(float("-inf"))
        # Non-speech → -inf
        is_speech = preds > 0.5
        scores = mx.where(is_speech, scores, neg_inf)

        # Overlapped speech → -inf (only if speaker has enough clean frames)
        is_pos = scores > 0
        # Count positive scores per speaker: (batch, n_spk)
        pos_count = mx.sum(is_pos.astype(mx.float32), axis=1, keepdims=True)
        has_enough = pos_count >= min_pos_scores_per_spk
        is_nonpos_replace = (~is_pos) & is_speech & has_enough
        scores = mx.where(is_nonpos_replace, neg_inf, scores)
        return scores

    @staticmethod
    def _boost_topk_scores(
        scores: mx.array,
        n_boost_per_spk: int,
        scale_factor: float = 1.0,
    ) -> mx.array:
        """Boost the top-K scores per speaker to ensure minimum representation.

        Args:
            scores: (batch, n_frames, n_spk)
            n_boost_per_spk: number of frames to boost per speaker
            scale_factor: multiplier for the boost amount
        Returns:
            scores: (batch, n_frames, n_spk) with boosted values
        """
        if n_boost_per_spk <= 0:
            return scores
        _, n_frames, n_spk = scores.shape
        k = min(n_boost_per_spk, n_frames)
        boost_val = -scale_factor * math.log(0.5)  # positive value

        # Process each speaker: find top-k, create mask, apply boost
        result_slices = []
        for spk in range(n_spk):
            spk_scores = scores[:, :, spk : spk + 1]  # (batch, n_frames, 1)
            flat = spk_scores[:, :, 0]  # (batch, n_frames)

            # Get top-k indices
            topk_idx = mx.argpartition(-flat, kth=k - 1, axis=1)[:, :k]

            # Build one-hot mask via scatter: (batch, n_frames)
            is_finite = flat > float("-inf")
            mask = mx.zeros_like(flat)
            # Set top-k positions to 1
            ones = mx.ones(topk_idx.shape, dtype=mx.float32)
            # Scatter ones into mask at topk positions
            mask = mask.at[mx.arange(mask.shape[0])[:, None], topk_idx].add(ones)

            boost_amount = mask * boost_val * is_finite.astype(mx.float32)
            result_slices.append(flat + boost_amount)

        return mx.stack(result_slices, axis=-1)

    @staticmethod
    def _get_topk_indices(
        scores: mx.array,
        spkcache_len: int,
        spkcache_sil_frames_per_spk: int,
        max_index: int,
    ) -> Tuple[mx.array, mx.array]:
        """Select top spkcache_len frames globally across speakers.

        Args:
            scores: (batch, n_frames, n_spk) — may include silence padding
            spkcache_len: target number of frames
            spkcache_sil_frames_per_spk: silence frames appended per speaker
            max_index: placeholder index for disabled frames
        Returns:
            topk_indices_sorted: (batch, spkcache_len) frame indices
            is_disabled: (batch, spkcache_len) True for disabled positions
        """
        batch_size, n_frames, _ = scores.shape
        n_frames_no_sil = n_frames - spkcache_sil_frames_per_spk

        # Flatten: (batch, n_spk, n_frames) → (batch, n_spk * n_frames)
        scores_flat = mx.transpose(scores, axes=(0, 2, 1)).reshape(batch_size, -1)

        # Top-k
        k = min(spkcache_len, scores_flat.shape[1])
        topk_indices = mx.argpartition(-scores_flat, kth=k - 1, axis=1)[:, :k]
        topk_values = mx.take_along_axis(scores_flat, topk_indices, axis=1)

        # Replace -inf indices with max_index placeholder
        valid_mask = topk_values > float("-inf")
        topk_indices = mx.where(valid_mask, topk_indices, mx.array(max_index))

        # Sort to preserve temporal order
        topk_indices_sorted = mx.sort(topk_indices, axis=1)

        # Determine disabled positions
        is_disabled = topk_indices_sorted == max_index

        # Convert flattened speaker-indices back to frame indices
        topk_indices_sorted = topk_indices_sorted % n_frames

        # Mark silence-pad region as disabled
        is_disabled = is_disabled | (topk_indices_sorted >= n_frames_no_sil)

        # Set disabled indices to 0 as placeholder for gather
        topk_indices_sorted = mx.where(is_disabled, mx.array(0), topk_indices_sorted)

        return topk_indices_sorted, is_disabled

    @staticmethod
    def _gather_spkcache_and_preds(
        embs: mx.array,
        preds: mx.array,
        topk_indices: mx.array,
        is_disabled: mx.array,
        mean_sil_emb: mx.array,
        spkcache_len: int,
    ) -> Tuple[mx.array, mx.array]:
        """Gather selected frames, replacing disabled positions with silence.

        Args:
            embs: (batch, n_frames, emb_dim)
            preds: (batch, n_frames, n_spk)
            topk_indices: (batch, spkcache_len)
            is_disabled: (batch, spkcache_len)
            mean_sil_emb: (batch, emb_dim)
            spkcache_len: target cache length
        Returns:
            gathered_embs: (batch, spkcache_len, emb_dim)
            gathered_preds: (batch, spkcache_len, n_spk)
        """
        emb_dim = embs.shape[2]
        n_spk = preds.shape[2]

        # Expand indices for gather: (batch, spkcache_len, emb_dim)
        idx_emb = mx.expand_dims(topk_indices, axis=-1)
        idx_emb = mx.broadcast_to(
            idx_emb, (topk_indices.shape[0], topk_indices.shape[1], emb_dim)
        )
        gathered_embs = mx.take_along_axis(embs, idx_emb, axis=1)

        # Replace disabled with mean silence embedding
        sil_expanded = mx.expand_dims(mean_sil_emb, axis=1)
        sil_expanded = mx.broadcast_to(
            sil_expanded, (topk_indices.shape[0], spkcache_len, emb_dim)
        )
        disabled_mask = mx.expand_dims(is_disabled, axis=-1)
        gathered_embs = mx.where(disabled_mask, sil_expanded, gathered_embs)

        # Gather preds
        idx_spk = mx.expand_dims(topk_indices, axis=-1)
        idx_spk = mx.broadcast_to(
            idx_spk, (topk_indices.shape[0], topk_indices.shape[1], n_spk)
        )
        gathered_preds = mx.take_along_axis(preds, idx_spk, axis=1)
        gathered_preds = mx.where(disabled_mask, mx.array(0.0), gathered_preds)

        return gathered_embs, gathered_preds

    @staticmethod
    def _get_silence_profile(
        mean_sil_emb: mx.array,
        n_sil_frames: mx.array,
        embs: mx.array,
        preds: mx.array,
        sil_threshold: float,
    ) -> Tuple[mx.array, mx.array]:
        """Update running mean silence embedding from new frames.

        A frame is silence if sum of speaker preds < sil_threshold.

        Args:
            mean_sil_emb: (batch, emb_dim) current mean
            n_sil_frames: (batch,) current count
            embs: (batch, n_frames, emb_dim) new embeddings
            preds: (batch, n_frames, n_spk) new predictions
            sil_threshold: threshold for silence detection
        Returns:
            updated_mean_sil_emb: (batch, emb_dim)
            updated_n_sil_frames: (batch,)
        """
        # Detect silence frames
        is_sil = mx.sum(preds, axis=2) < sil_threshold  # (batch, n_frames)
        sil_count = mx.sum(is_sil.astype(mx.float32), axis=1)  # (batch,)

        # Sum silence embeddings
        sil_emb_sum = mx.sum(
            embs * mx.expand_dims(is_sil.astype(mx.float32), axis=-1), axis=1
        )  # (batch, emb_dim)

        # Incremental mean update
        upd_n_sil = n_sil_frames + sil_count
        old_sil_sum = mean_sil_emb * mx.expand_dims(n_sil_frames, axis=-1)
        total_sil_sum = old_sil_sum + sil_emb_sum
        upd_mean = total_sil_sum / mx.clip(
            mx.expand_dims(upd_n_sil, axis=-1), a_min=1, a_max=None
        )

        return upd_mean, upd_n_sil

    @staticmethod
    def _compress_spkcache_aosc(
        embs: mx.array,
        preds: mx.array,
        mean_sil_emb: mx.array,
        modules_cfg: "ModulesConfig",
    ) -> Tuple[mx.array, mx.array]:
        """AOSC compression: keep the most informative frames per speaker.

        Args:
            embs: (1, N, emb_dim) pre-encoded embeddings
            preds: (1, N, n_spk) speaker predictions
            mean_sil_emb: (1, emb_dim) mean silence embedding
            modules_cfg: module config with AOSC parameters
        Returns:
            (compressed_embs, compressed_preds) each (1, spkcache_len, *)
        """
        n_spk = modules_cfg.num_speakers
        spkcache_len = modules_cfg.spkcache_len
        sil_per_spk = modules_cfg.spkcache_sil_frames_per_spk
        spkcache_len_per_spk = spkcache_len // n_spk - sil_per_spk
        strong_boost = math.floor(spkcache_len_per_spk * modules_cfg.strong_boost_rate)
        weak_boost = math.floor(spkcache_len_per_spk * modules_cfg.weak_boost_rate)
        min_pos = math.floor(spkcache_len_per_spk * modules_cfg.min_pos_scores_rate)

        # 1. Score
        scores = Model._get_log_pred_scores(preds, modules_cfg.pred_score_threshold)
        # 2. Disable non-speech and overlapped speech
        scores = Model._disable_low_scores(preds, scores, min_pos)
        # 3. Boost newly added frames (frames beyond current cache length)
        if modules_cfg.scores_boost_latest > 0 and scores.shape[1] > spkcache_len:
            boost_mask = mx.concatenate(
                [
                    mx.zeros((scores.shape[0], spkcache_len, n_spk)),
                    mx.full(
                        (scores.shape[0], scores.shape[1] - spkcache_len, n_spk),
                        modules_cfg.scores_boost_latest,
                    ),
                ],
                axis=1,
            )
            scores = scores + boost_mask
        # 4. Strong boost (ensure min per-speaker representation)
        scores = Model._boost_topk_scores(scores, strong_boost, scale_factor=2.0)
        # 5. Weak boost (prevent single-speaker dominance)
        scores = Model._boost_topk_scores(scores, weak_boost, scale_factor=1.0)
        # 6. Append silence padding with +inf scores
        if sil_per_spk > 0:
            batch_size = scores.shape[0]
            pad = mx.full((batch_size, sil_per_spk, n_spk), float("inf"))
            scores = mx.concatenate([scores, pad], axis=1)
        # 7. Select top frames
        topk_indices, is_disabled = Model._get_topk_indices(
            scores, spkcache_len, sil_per_spk, modules_cfg.max_index
        )
        # 8. Gather
        compressed_embs, compressed_preds = Model._gather_spkcache_and_preds(
            embs, preds, topk_indices, is_disabled, mean_sil_emb, spkcache_len
        )
        mx.eval(compressed_embs, compressed_preds)
        return compressed_embs, compressed_preds

    @staticmethod
    def _compress_spkcache_simple(
        embs: mx.array,
        preds: mx.array,
        target_len: int,
    ) -> Tuple[mx.array, mx.array]:
        """Simple compression: keep frames with highest total speaker activity.

        This is the v1 compression strategy.

        Args:
            embs: ``(1, N, emb_dim)`` pre-encoded embeddings.
            preds: ``(1, N, n_spk)`` speaker predictions.
            target_len: Desired number of frames after compression.

        Returns:
            ``(compressed_embs, compressed_preds)`` each with
            ``target_len`` frames.
        """
        log_preds = mx.log(mx.clip(preds[0], 1e-7, 1.0))
        frame_scores = mx.sum(log_preds, axis=-1)

        top_indices = mx.argsort(-frame_scores)[:target_len]
        top_indices = mx.sort(top_indices)

        compressed_embs = embs[:, top_indices, :]
        compressed_preds = preds[:, top_indices, :]
        mx.eval(compressed_embs, compressed_preds)

        return compressed_embs, compressed_preds

    @staticmethod
    def _preds_to_segments(
        preds: mx.array,
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
        _, num_speakers = preds.shape
        segments = []

        for spk in range(num_speakers):
            activity = preds[:, spk] > threshold
            if not mx.any(activity).item():
                continue

            # Pad with False and diff to find transitions
            padded = mx.concatenate(
                [
                    mx.zeros((1,), dtype=mx.bool_),
                    activity,
                    mx.zeros((1,), dtype=mx.bool_),
                ]
            )
            changes = padded[1:].astype(mx.int32) - padded[:-1].astype(mx.int32)
            mx.eval(changes)
            changes_list = changes.tolist()

            starts = [i for i, v in enumerate(changes_list) if v == 1]
            ends = [i for i, v in enumerate(changes_list) if v == -1]

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

        segments.sort(key=lambda s: s.start)
        return segments

    @staticmethod
    def _trim_silence(
        waveform: mx.array,
        sample_rate: int,
        frame_ms: int = 30,
        energy_ratio: float = 0.01,
        min_speech_sec: float = 0.5,
    ) -> Tuple[mx.array, int]:
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
        num_frames = waveform.shape[0] // frame_len

        if num_frames < min_speech_frames * 2:
            return waveform, 0

        frames = waveform[: num_frames * frame_len].reshape(num_frames, frame_len)
        energy = mx.sqrt(mx.mean(frames**2, axis=1))
        threshold_val = mx.max(energy).item() * energy_ratio
        speech = energy > threshold_val
        mx.eval(speech)
        speech_list = speech.tolist()

        start_frame = 0
        for i in range(num_frames - min_speech_frames + 1):
            if all(speech_list[i : i + min_speech_frames]):
                start_frame = i
                break

        end_frame = num_frames
        for i in range(num_frames - 1, min_speech_frames - 2, -1):
            if all(speech_list[i - min_speech_frames + 1 : i + 1]):
                end_frame = i + 1
                break

        start_sample = start_frame * frame_len
        end_sample = min(end_frame * frame_len, waveform.shape[0])

        if start_sample == 0 and end_sample == waveform.shape[0]:
            return waveform, 0

        return waveform[start_sample:end_sample], start_sample

    def _load_audio(
        self,
        audio: Union[str, "np.ndarray", mx.array],
        sample_rate: int,
    ) -> Tuple[mx.array, int]:
        """Load and prepare audio from any supported input.

        Handles file paths (via ``audio_io.read``), numpy arrays, and
        mx arrays.  Converts to mono float32 and resamples to the model's
        expected sample rate.

        Returns:
            ``(waveform, sample_rate)`` where waveform is 1-D mx.array.
        """
        if isinstance(audio, str):
            waveform_np, sr = audio_read(audio, dtype="float32")
            waveform = mx.array(waveform_np)
            sample_rate = sr
        elif isinstance(audio, mx.array):
            waveform = audio.astype(mx.float32)
        else:
            # numpy array or other array-like
            waveform = mx.array(audio).astype(mx.float32)

        if waveform.ndim > 1:
            waveform = mx.mean(waveform, axis=-1)

        proc = self._processor_config
        if sample_rate != proc.sampling_rate:
            waveform = self._resample(waveform, sample_rate, proc.sampling_rate)

        return waveform, proc.sampling_rate

    @staticmethod
    def _resample(waveform: mx.array, orig_sr: int, target_sr: int) -> mx.array:
        """Resample audio using scipy (no librosa dependency)."""
        if orig_sr == target_sr:
            return waveform
        import numpy as np
        from scipy import signal as scipy_signal

        gcd = math.gcd(orig_sr, target_sr)
        resampled = scipy_signal.resample_poly(
            np.array(waveform), target_sr // gcd, orig_sr // gcd
        ).astype(np.float32)
        return mx.array(resampled)

    @staticmethod
    def sanitize(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Transform HuggingFace weights to match MLX model structure.

        Handles two formats:
          1. v1 HuggingFace safetensors (keys like ``fc_encoder.subsampling.layers.N.*``
             with Conv weights in PyTorch layout)
          2. Converted safetensors from ``convert.py`` (keys already use ``layers_N``
             and Conv weights are already in MLX layout) — passed through unchanged.
        """
        sanitized = {}
        skip_keys = {"num_batches_tracked"}

        # Detect if already converted (keys use layers_N, not layers.N)
        already_converted = any("subsampling.layers_" in k for k in weights)

        for k, v in weights.items():
            if any(sk in k for sk in skip_keys):
                continue

            new_k = k

            if not already_converted:
                # Remap subsampling.layers.N -> subsampling.layers_N
                if "fc_encoder.subsampling.layers." in new_k:
                    new_k = new_k.replace("subsampling.layers.", "subsampling.layers_")

                # Conv2d: PyTorch (O,I,H,W) -> MLX (O,H,W,I)
                if (
                    "subsampling" in new_k
                    and "weight" in new_k
                    and "linear" not in new_k
                ):
                    if v.ndim == 4:
                        v = mx.transpose(v, axes=(0, 2, 3, 1))

                # Conv1d: PyTorch (O,I,K) -> MLX (O,K,I)
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

            sanitized[new_k] = v

        return sanitized
