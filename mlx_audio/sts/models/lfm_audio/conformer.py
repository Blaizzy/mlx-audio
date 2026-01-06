# Copyright (c) 2025 Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)
# LFM2.5-Audio: FastConformer encoder implementation

import math
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .config import ConformerEncoderConfig


class RelativePositionalEncoding(nn.Module):
    """Relative positional encoding for attention."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        # Create positional encoding matrix
        pe = mx.zeros((max_len, d_model))
        position = mx.arange(0, max_len, dtype=mx.float32)[:, None]
        div_term = mx.exp(
            mx.arange(0, d_model, 2, dtype=mx.float32) * (-math.log(10000.0) / d_model)
        )
        pe = pe.at[:, 0::2].add(mx.sin(position * div_term))
        pe = pe.at[:, 1::2].add(mx.cos(position * div_term))
        self.pe = pe

    def __call__(self, x: mx.array) -> mx.array:
        """Get positional encodings for sequence length."""
        seq_len = x.shape[1]
        return self.pe[:seq_len]


class ConformerFeedForward(nn.Module):
    """Feed-forward module for Conformer."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.linear1(x)
        x = nn.silu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class ConformerConvolution(nn.Module):
    """Convolution module for Conformer."""

    def __init__(
        self,
        d_model: int,
        kernel_size: int = 31,
        norm_type: str = "batch_norm",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pointwise_conv1 = nn.Linear(d_model, 2 * d_model)
        # Depthwise convolution: each channel is convolved independently
        # groups=d_model means each input channel is convolved with its own set of filters
        # MLX Conv1d expects (B, L, C) format
        self.depthwise_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=d_model,  # Depthwise: each channel processed independently
        )
        # Use LayerNorm for simplicity (works on last dimension in MLX)
        self.norm = nn.LayerNorm(d_model)
        self.pointwise_conv2 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, T, D) - already in correct format for MLX
        x = self.pointwise_conv1(x)

        # GLU activation
        x, gate = mx.split(x, 2, axis=-1)
        x = x * mx.sigmoid(gate)

        # Depthwise conv - MLX expects (B, L, C) which is already our format
        x = self.depthwise_conv(x)

        # LayerNorm on features dimension
        x = self.norm(x)

        x = nn.silu(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        return x


class RelativeMultiHeadAttention(nn.Module):
    """Multi-head attention with relative positional encoding."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        pos_bias: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Position projection
        self.pos_proj = nn.Linear(d_model, d_model, bias=False)

        # Learnable biases for relative position
        if pos_bias:
            self.pos_bias_u = mx.zeros((num_heads, self.head_dim))
            self.pos_bias_v = mx.zeros((num_heads, self.head_dim))
        else:
            self.pos_bias_u = None
            self.pos_bias_v = None

        self.dropout = nn.Dropout(dropout)

    def _relative_shift(self, x: mx.array) -> mx.array:
        """Compute relative position scores."""
        B, H, T, _ = x.shape
        # Pad and reshape for relative position
        x = mx.pad(x, [(0, 0), (0, 0), (0, 0), (1, 0)])
        x = x.reshape(B, H, -1, T)
        x = x[:, :, 1:, :]
        x = x.reshape(B, H, T, -1)
        return x[:, :, :, :T]

    def __call__(
        self,
        x: mx.array,
        pos_emb: mx.array,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        B, T, _ = x.shape

        # Projections
        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(B, T, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(B, T, self.num_heads, self.head_dim)

        # Position projection
        pos = self.pos_proj(pos_emb).reshape(1, -1, self.num_heads, self.head_dim)

        # Transpose to (B, H, T, D)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        pos = pos.transpose(0, 2, 1, 3)

        # Compute content and position scores
        if self.pos_bias_u is not None:
            q_with_bias_u = q + self.pos_bias_u[None, :, None, :]
            q_with_bias_v = q + self.pos_bias_v[None, :, None, :]
        else:
            q_with_bias_u = q
            q_with_bias_v = q

        # Content-to-content
        content_score = q_with_bias_u @ k.transpose(0, 1, 3, 2)

        # Content-to-position
        pos_score = q_with_bias_v @ pos.transpose(0, 1, 3, 2)
        pos_score = self._relative_shift(pos_score)

        scores = (content_score + pos_score) * self.scale

        if mask is not None:
            scores = scores + mask

        attn = mx.softmax(scores, axis=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, -1)
        return self.out_proj(out)


class ConformerLayer(nn.Module):
    """Single Conformer layer."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ff_expansion_factor: int = 4,
        conv_kernel_size: int = 31,
        conv_norm_type: str = "batch_norm",
        dropout: float = 0.1,
        dropout_att: float = 0.1,
    ):
        super().__init__()
        d_ff = d_model * ff_expansion_factor

        # Pre-norm for each sub-layer
        self.ff1_norm = nn.LayerNorm(d_model)
        self.ff1 = ConformerFeedForward(d_model, d_ff, dropout)

        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = RelativeMultiHeadAttention(d_model, num_heads, dropout_att)

        self.conv_norm = nn.LayerNorm(d_model)
        self.conv = ConformerConvolution(d_model, conv_kernel_size, conv_norm_type, dropout)

        self.ff2_norm = nn.LayerNorm(d_model)
        self.ff2 = ConformerFeedForward(d_model, d_ff, dropout)

        self.final_norm = nn.LayerNorm(d_model)

    def __call__(
        self,
        x: mx.array,
        pos_emb: mx.array,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        # First FF (half residual)
        x = x + 0.5 * self.ff1(self.ff1_norm(x))

        # Attention
        x = x + self.attn(self.attn_norm(x), pos_emb, mask)

        # Conv
        x = x + self.conv(self.conv_norm(x))

        # Second FF (half residual)
        x = x + 0.5 * self.ff2(self.ff2_norm(x))

        # Final norm
        x = self.final_norm(x)

        return x


class ConvSubsampling(nn.Module):
    """Convolutional subsampling for audio features using 2D depthwise separable convs."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        subsampling_factor: int = 8,
        conv_channels: int = 256,
        subsampling_type: str = "dw_striding",
    ):
        super().__init__()
        self.subsampling_factor = subsampling_factor
        self.subsampling_type = subsampling_type
        self.conv_channels = conv_channels
        self.in_channels = in_channels

        # The checkpoint uses 2D convolutions in depthwise-separable structure:
        # conv.0: strided 3x3 (1 -> 256 channels)
        # conv.2: strided 3x3 depthwise (groups=256)
        # conv.3: pointwise 1x1
        # conv.5: strided 3x3 depthwise
        # conv.6: pointwise 1x1
        # Indices 1, 4 are ReLU (not stored)

        # Using a list-based structure that MLX can load weights into
        # The conv list will have entries at indices 0, 2, 3, 5, 6
        # We use placeholder Nones for ReLU positions
        self.conv = [
            nn.Conv2d(1, conv_channels, kernel_size=3, stride=2, padding=1),  # 0
            None,  # 1 (ReLU)
            nn.Conv2d(1, conv_channels, kernel_size=3, stride=2, padding=1),  # 2
            nn.Conv2d(conv_channels, conv_channels, kernel_size=1, stride=1, padding=0),  # 3
            None,  # 4 (ReLU)
            nn.Conv2d(1, conv_channels, kernel_size=3, stride=2, padding=1),  # 5
            nn.Conv2d(conv_channels, conv_channels, kernel_size=1, stride=1, padding=0),  # 6
        ]

        # Output projection - checkpoint shows (512, 4096)
        # 4096 = 256 * 16, where 16 = 128 / 8 (mel features / subsampling factor)
        self.out_proj = nn.Linear(conv_channels * (in_channels // subsampling_factor), out_channels)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: Input features (B, T, D) where D is mel features

        Returns:
            Subsampled features (B, T', D')
        """
        B, T, D = x.shape

        # Reshape for 2D conv: (B, T, D) -> (B, T, D, 1) - MLX uses NHWC format
        x = x[:, :, :, None]

        # Block 1: strided 3x3 conv
        x = nn.relu(self.conv[0](x))  # (B, T/2, D/2, 256)

        # Block 2: depthwise 3x3 + pointwise 1x1
        # For depthwise, we apply conv[2] per channel (it has 1 input channel)
        B, T2, D2, C = x.shape
        # Simplified: apply strided conv that reduces spatial dims
        x_out = nn.relu(self.conv[2](x[:, :, :, 0:1]))  # First channel
        for c in range(1, C):
            x_c = nn.relu(self.conv[2](x[:, :, :, c:c+1]))
            x_out = x_out + x_c
        x = x_out / C  # Average
        x = mx.broadcast_to(x, (B, x.shape[1], x.shape[2], C))
        x = nn.relu(self.conv[3](x))  # pointwise

        # Block 3: depthwise 3x3 + pointwise 1x1
        B, T3, D3, C = x.shape
        x_out = nn.relu(self.conv[5](x[:, :, :, 0:1]))
        for c in range(1, C):
            x_c = nn.relu(self.conv[5](x[:, :, :, c:c+1]))
            x_out = x_out + x_c
        x = x_out / C
        x = mx.broadcast_to(x, (B, x.shape[1], x.shape[2], C))
        x = nn.relu(self.conv[6](x))  # pointwise

        # Flatten and project: (B, T_out, D_out, C) -> (B, T_out, C*D_out)
        B, T_out, D_out, C = x.shape
        x = x.reshape(B, T_out, -1)  # (B, T_out, D_out*C)
        x = self.out_proj(x)

        return x


class ConformerEncoder(nn.Module):
    """FastConformer encoder for audio processing."""

    def __init__(self, config: ConformerEncoderConfig):
        super().__init__()
        self.config = config

        # Subsampling
        self.subsampling = ConvSubsampling(
            in_channels=config.feat_in,
            out_channels=config.d_model,
            subsampling_factor=config.subsampling_factor,
            conv_channels=config.subsampling_conv_channels,
            subsampling_type=config.subsampling,
        )

        # Positional encoding
        self.pos_enc = RelativePositionalEncoding(config.d_model, config.pos_emb_max_len)

        # Pre-encoder dropout
        self.pre_dropout = nn.Dropout(config.dropout_pre_encoder)

        # Conformer layers
        self.layers = [
            ConformerLayer(
                d_model=config.d_model,
                num_heads=config.n_heads,
                ff_expansion_factor=config.ff_expansion_factor,
                conv_kernel_size=config.conv_kernel_size,
                conv_norm_type=config.conv_norm_type,
                dropout=config.dropout,
                dropout_att=config.dropout_att,
            )
            for _ in range(config.n_layers)
        ]

    def __call__(
        self,
        x: mx.array,
        lengths: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array]:
        """
        Args:
            x: Audio features (B, T, D)
            lengths: Original lengths (B,)

        Returns:
            Encoded features (B, T', D) and new lengths
        """
        # Subsampling
        x = self.subsampling(x)

        # Update lengths
        if lengths is not None:
            lengths = lengths // self.config.subsampling_factor
        else:
            lengths = mx.array([x.shape[1]] * x.shape[0])

        # Get positional embeddings
        pos_emb = self.pos_enc(x)

        # Pre-encoder dropout
        x = self.pre_dropout(x)

        # Create attention mask if needed
        mask = None
        if lengths is not None:
            max_len = x.shape[1]
            # Create padding mask
            idx = mx.arange(max_len)[None, :]
            mask = idx >= lengths[:, None]
            mask = mx.where(mask[:, None, None, :], float("-inf"), 0.0)

        # Apply conformer layers
        for layer in self.layers:
            x = layer(x, pos_emb, mask)

        return x, lengths


class MLP(nn.Module):
    """MLP adapter for conformer to LFM."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_dims: List[int],
        use_layer_norm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        channels = [in_channels, *hidden_dims, out_channels]

        layers = []
        if use_layer_norm:
            layers.append(nn.LayerNorm(channels[0]))

        for i in range(len(channels) - 1):
            layers.append(nn.Linear(channels[i], channels[i + 1]))
            if i != len(channels) - 2:
                layers.append(nn.GELU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        self.layers = layers

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = layer(x)
        return x
