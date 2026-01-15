# Copyright (c) 2025 Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)
# LFM2.5-Audio: FastConformer encoder implementation

import math
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .config import ConformerEncoderConfig


class RelativePositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 5000, xscale: bool = True):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.xscale = math.sqrt(d_model) if xscale else None
        self._pe_cache_len = 0
        self.pe = None

        # div_term for sinusoidal encoding: 10000^(-2i/d_model)
        self._div_term = mx.exp(
            mx.arange(0, d_model, 2, dtype=mx.float32) * (-math.log(10000.0) / d_model)
        )

    def _extend_pe(self, length: int):
        """Extend positional encodings if needed."""
        needed_size = 2 * length - 1
        if self.pe is not None and self.pe.shape[0] >= needed_size:
            return

        # Generate positions from (length-1) to -(length-1) in descending order
        # E.g., for length=39: [38, 37, ..., 1, 0, -1, ..., -37, -38]
        positions = mx.arange(length - 1, -length, -1, dtype=mx.float32)[:, None]

        # Create positional encoding
        pe = mx.zeros((needed_size, self.d_model))
        pe = pe.at[:, 0::2].add(mx.sin(positions * self._div_term))
        pe = pe.at[:, 1::2].add(mx.cos(positions * self._div_term))

        self.pe = pe
        self._pe_cache_len = length

    def __call__(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        """Get positional encodings for sequence.

        Args:
            x: Input tensor (B, T, D)

        Returns:
            Tuple of (scaled_x, pos_emb) where pos_emb has shape (2*T-1, D)
        """
        seq_len = x.shape[1]
        self._extend_pe(seq_len)

        # Scale input if needed
        if self.xscale is not None:
            x = x * self.xscale

        # Return positions for this sequence length
        # For input of length L, we need 2*L-1 positions
        center = self.pe.shape[0] // 2
        start = center - seq_len + 1
        end = center + seq_len
        pos_emb = self.pe[start:end]

        return x, pos_emb


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

        self.depthwise_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=d_model,  # Depthwise: each channel processed independently
        )
        if norm_type == "batch_norm":
            self.norm = nn.BatchNorm(d_model)
        else:
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
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.pos_proj = nn.Linear(d_model, d_model, bias=False)

        if pos_bias:
            self.pos_bias_u = mx.zeros((num_heads, self.head_dim))
            self.pos_bias_v = mx.zeros((num_heads, self.head_dim))
        else:
            self.pos_bias_u = None
            self.pos_bias_v = None

        self.dropout = nn.Dropout(dropout)

    def _rel_shift(self, x: mx.array) -> mx.array:
        """Compute relative positional encoding shift.

        This aligns the position scores so that position i attends to the
        correct relative positions in the positional embedding.

        Args:
            x: Position scores (B, H, T, 2T-1)

        Returns:
            Shifted scores (B, H, T, T) where score[i,j] corresponds to
            relative position i-j.
        """
        B, H, T, pos_len = x.shape
        # Pad on the left: (B, H, T, 2T-1) -> (B, H, T, 2T)
        x = mx.pad(x, [(0, 0), (0, 0), (0, 0), (1, 0)])
        # Reshape: (B, H, T, 2T) -> (B, H, 2T, T)
        x = x.reshape(B, H, pos_len + 1, T)
        # Remove first row: (B, H, 2T, T) -> (B, H, 2T-1, T)
        x = x[:, :, 1:, :]
        # Reshape back: (B, H, 2T-1, T) -> (B, H, T, 2T-1)
        x = x.reshape(B, H, T, pos_len)
        # Take only the first T columns (the valid relative positions)
        return x[:, :, :, :T]

    def __call__(
        self,
        x: mx.array,
        pos_emb: mx.array,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        """Compute attention with relative positional encoding.

        Args:
            x: Input tensor (B, T, D)
            pos_emb: Positional embeddings (2T-1, D) or (1, 2T-1, D)
            mask: Attention mask

        Returns:
            Output tensor (B, T, D)
        """
        B, T, _ = x.shape

        # Projections: (B, T, D) -> (B, T, H, d_k)
        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(B, T, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(B, T, self.num_heads, self.head_dim)

        # Position projection: (2T-1, D) -> (1, 2T-1, H, d_k)
        if pos_emb.ndim == 2:
            pos_emb = pos_emb[None, :, :]
        p = self.pos_proj(pos_emb).reshape(1, -1, self.num_heads, self.head_dim)

        if self.pos_bias_u is not None:
            q_with_bias_u = (q + self.pos_bias_u[None, None, :, :]).transpose(
                0, 2, 1, 3
            )
            q_with_bias_v = (q + self.pos_bias_v[None, None, :, :]).transpose(
                0, 2, 1, 3
            )
        else:
            q_with_bias_u = q.transpose(0, 2, 1, 3)
            q_with_bias_v = q.transpose(0, 2, 1, 3)

        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        p = p.transpose(0, 2, 1, 3)

        matrix_ac = q_with_bias_u @ k.transpose(0, 1, 3, 2)
        matrix_bd = q_with_bias_v @ p.transpose(0, 1, 3, 2)

        matrix_bd = self._rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) * self.scale

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
        self.conv = ConformerConvolution(
            d_model, conv_kernel_size, conv_norm_type, dropout
        )

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

        self.conv = [
            nn.Conv2d(
                1, conv_channels, kernel_size=3, stride=2, padding=1
            ),  # 0: standard conv
            None,  # 1 (ReLU)
            nn.Conv2d(
                conv_channels,
                conv_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=conv_channels,
            ),  # 2
            nn.Conv2d(
                conv_channels, conv_channels, kernel_size=1, stride=1, padding=0
            ),  # 3
            None,  # 4 (ReLU)
            nn.Conv2d(
                conv_channels,
                conv_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=conv_channels,
            ),  # 5
            nn.Conv2d(
                conv_channels, conv_channels, kernel_size=1, stride=1, padding=0
            ),  # 6
        ]

        self.out = nn.Linear(
            conv_channels * (in_channels // subsampling_factor), out_channels
        )

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

        x = nn.relu(self.conv[0](x))  # (B, T/2, D/2, 256)

        x = self.conv[2](x)  # (B, T/4, D/4, 256) - depthwise
        x = nn.relu(self.conv[3](x))  # pointwise 1x1 + relu

        x = self.conv[5](x)  # (B, T/8, D/8, 256) - depthwise
        x = nn.relu(self.conv[6](x))  # pointwise 1x1 + relu

        B, T_out, D_out, C = x.shape
        x = x.transpose(0, 1, 3, 2)  # (B, T_out, C, D_out)
        x = x.reshape(B, T_out, -1)  # (B, T_out, C*D_out)
        x = self.out(x)

        return x


class ConformerEncoder(nn.Module):
    """FastConformer encoder for audio processing."""

    def __init__(self, config: ConformerEncoderConfig):
        super().__init__()
        self.config = config

        # Subsampling
        self.pre_encode = ConvSubsampling(
            in_channels=config.feat_in,
            out_channels=config.d_model,
            subsampling_factor=config.subsampling_factor,
            conv_channels=config.subsampling_conv_channels,
            subsampling_type=config.subsampling,
        )

        self.pos_enc = RelativePositionalEncoding(
            config.d_model, config.pos_emb_max_len, xscale=False
        )

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
        x = self.pre_encode(x)

        # Update lengths
        if lengths is not None:
            lengths = lengths // self.config.subsampling_factor
        else:
            lengths = mx.array([x.shape[1]] * x.shape[0])

        # Get positional embeddings and scale input
        # pos_enc returns (scaled_x, pos_emb) where pos_emb has shape (2T-1, D)
        x, pos_emb = self.pos_enc(x)

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
