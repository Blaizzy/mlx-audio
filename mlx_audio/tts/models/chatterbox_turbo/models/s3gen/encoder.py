# Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


class EspnetRelPositionalEncoding(nn.Module):
    """Relative positional encoding module (ESPnet style).

    Creates positional encodings for both positive and negative relative positions.
    The output has length 2*T-1 for input of length T.

    See: https://arxiv.org/abs/1901.02860
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.xscale = float(d_model) ** 0.5
        self.max_len = max_len
        self.pe = None
        # Initialize with a reasonable size
        self._extend_pe(max_len)

    def _extend_pe(self, size: int):
        """Create or extend positional encoding matrix."""
        if self.pe is not None and self.pe.shape[1] >= size * 2 - 1:
            return

        # Create positive and negative position encodings
        position = mx.arange(0, size, dtype=mx.float32)[:, None]
        div_term = mx.exp(
            mx.arange(0, self.d_model, 2, dtype=mx.float32)
            * (-float(mx.log(mx.array(10000.0))) / self.d_model)
        )

        # Positive positions
        pe_positive = mx.zeros((size, self.d_model))
        pe_positive_sin = mx.sin(position * div_term)
        pe_positive_cos = mx.cos(position * div_term)
        # Interleave: pe[:, 0::2] = sin, pe[:, 1::2] = cos
        pe_positive = mx.concatenate(
            [pe_positive_sin[:, :, None], pe_positive_cos[:, :, None]], axis=-1
        ).reshape(size, self.d_model)

        # Negative positions
        pe_negative_sin = mx.sin(-position * div_term)
        pe_negative_cos = mx.cos(-position * div_term)
        pe_negative = mx.concatenate(
            [pe_negative_sin[:, :, None], pe_negative_cos[:, :, None]], axis=-1
        ).reshape(size, self.d_model)

        # Flip positive and concatenate: [pos_reversed, neg[1:]]
        pe_positive_flipped = pe_positive[::-1, :]  # Reverse order
        pe_negative_tail = pe_negative[1:, :]  # Skip position 0

        pe = mx.concatenate([pe_positive_flipped, pe_negative_tail], axis=0)
        self.pe = pe[None, :, :]  # (1, 2*size-1, d_model)

    def __call__(self, x: mx.array, offset: int = 0) -> Tuple[mx.array, mx.array]:
        """
        Args:
            x: Input tensor (B, T, D)
            offset: Position offset (unused, for compatibility)

        Returns:
            x: Scaled input (B, T, D)
            pos_emb: Positional encoding (1, 2*T-1, D)
        """
        T = x.shape[1]
        self._extend_pe(T)

        # Scale input by sqrt(d_model)
        x = x * self.xscale

        # Get positional embedding centered around current sequence
        # pe has shape (1, 2*max_len-1, d_model)
        # We want positions from (max_len - T) to (max_len + T - 1), which is 2*T-1 positions
        center = self.pe.shape[1] // 2
        pos_emb = self.pe[:, center - T + 1 : center + T, :]

        return x, pos_emb


class LinearInput(nn.Module):
    """Linear input projection with LayerNorm and positional encoding.

    Matches PyTorch LinearNoSubsampling with EspnetRelPositionalEncoding.
    """

    def __init__(self, input_size: int, output_size: int, dropout: float = 0.1):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.norm = nn.LayerNorm(output_size, eps=1e-5)
        self.pos_enc = EspnetRelPositionalEncoding(output_size, dropout=dropout)

    def __call__(
        self, x: mx.array, mask: mx.array
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """
        Args:
            x: Input (B, T, D)
            mask: Mask (B, 1, T)

        Returns:
            x: Projected and scaled input (B, T, D)
            pos_emb: Positional encoding (1, 2*T-1, D)
            mask: Unchanged mask
        """
        x = self.linear(x)
        x = self.norm(x)
        x, pos_emb = self.pos_enc(x)  # pos_emb has shape (1, 2*T-1, D)
        return x, pos_emb, mask


class RelPositionMultiHeadedAttention(nn.Module):
    """Multi-head self-attention with relative positional encoding.

    Matches PyTorch RelPositionMultiHeadedAttention from ESPnet.
    See: https://arxiv.org/abs/1901.02860
    """

    def __init__(
        self,
        n_head: int,
        n_feat: int,
        dropout_rate: float = 0.0,
        key_bias: bool = True,
    ):
        super().__init__()
        self.n_head = n_head
        self.d_k = n_feat // n_head
        self.scale = self.d_k**-0.5

        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat, bias=key_bias)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)

        # Relative positional encoding components
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        # These are learnable parameters (initialized from weights)
        self.pos_bias_u = mx.zeros((n_head, self.d_k))
        self.pos_bias_v = mx.zeros((n_head, self.d_k))

    def _rel_shift(self, x: mx.array) -> mx.array:
        """Compute relative positional encoding shift.

        Args:
            x: Input tensor (batch, head, time1, 2*time1-1)

        Returns:
            Output tensor (batch, head, time1, time1)
        """
        B, n_head, T1, T2 = x.shape
        # Pad with zeros on the left
        zero_pad = mx.zeros((B, n_head, T1, 1))
        x_padded = mx.concatenate([zero_pad, x], axis=-1)

        # Reshape and extract the valid part
        x_padded = x_padded.reshape(B, n_head, T2 + 1, T1)
        x = x_padded[:, :, 1:, :]  # Remove first row
        x = x.reshape(B, n_head, T1, T2)

        # Keep only positions 0 to time1 (the valid relative positions)
        x = x[:, :, :, : T2 // 2 + 1]
        return x

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        pos_emb: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Args:
            x: Input (B, T, D)
            mask: Attention mask (B, T) or (B, T, T)
            pos_emb: Positional embedding (1, 2*T-1, D)

        Returns:
            Output (B, T, D)
        """
        B, T, D = x.shape

        # Compute Q, K, V
        q = self.linear_q(x).reshape(B, T, self.n_head, self.d_k)
        k = self.linear_k(x).reshape(B, T, self.n_head, self.d_k).transpose(0, 2, 1, 3)
        v = self.linear_v(x).reshape(B, T, self.n_head, self.d_k).transpose(0, 2, 1, 3)

        # q stays as (B, T, n_head, d_k) for adding bias, then transpose
        # Content attention: (q + pos_bias_u) @ k^T
        q_with_bias_u = (q + self.pos_bias_u).transpose(
            0, 2, 1, 3
        )  # (B, n_head, T, d_k)
        matrix_ac = q_with_bias_u @ k.transpose(0, 1, 3, 2)  # (B, n_head, T, T)

        # Position attention: (q + pos_bias_v) @ p^T
        if pos_emb is not None:
            T_pos = pos_emb.shape[1]  # 2*T-1
            p = self.linear_pos(pos_emb)  # (1, 2*T-1, D)
            p = p.reshape(1, T_pos, self.n_head, self.d_k).transpose(
                0, 2, 1, 3
            )  # (1, n_head, 2*T-1, d_k)

            q_with_bias_v = (q + self.pos_bias_v).transpose(
                0, 2, 1, 3
            )  # (B, n_head, T, d_k)
            matrix_bd = q_with_bias_v @ p.transpose(0, 1, 3, 2)  # (B, n_head, T, 2*T-1)

            # Apply relative shift when shapes don't match
            if matrix_ac.shape != matrix_bd.shape:
                matrix_bd = self._rel_shift(matrix_bd)

            scores = (matrix_ac + matrix_bd) * self.scale
        else:
            scores = matrix_ac * self.scale

        # Apply mask if provided
        if mask is not None:
            if mask.ndim == 2:
                # (B, T) -> (B, 1, 1, T)
                mask_expanded = mask[:, None, None, :]
            else:
                # (B, T, T) -> (B, 1, T, T)
                mask_expanded = mask[:, None, :, :]
            # PyTorch uses mask.eq(0) and fills with -inf, we use inverted mask
            scores = mx.where(mask_expanded > 0, scores, mx.array(-float("inf")))

        attn = mx.softmax(scores, axis=-1)
        # Replace NaN from softmax(-inf) with 0
        attn = mx.where(mx.isnan(attn), mx.array(0.0), attn)

        out = attn @ v  # (B, n_head, T, d_k)

        # Reshape back
        out = out.transpose(0, 2, 1, 3).reshape(B, T, D)
        return self.linear_out(out)


class PositionwiseFeedForward(nn.Module):
    """Position-wise feed-forward network.

    Matches PyTorch PositionwiseFeedForward from ESPnet/CosyVoice.
    Uses Swish (SiLU) activation as configured in the original model.
    """

    def __init__(self, d_model: int, d_inner: int, dropout: float = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_inner)
        self.w_2 = nn.Linear(d_inner, d_model)

    def __call__(self, x: mx.array) -> mx.array:
        # Use SiLU (Swish) activation to match PyTorch
        return self.w_2(nn.silu(self.w_1(x)))


class ConformerEncoderLayer(nn.Module):
    """Single Conformer encoder layer.

    Matches PyTorch ConformerEncoderLayer from ESPnet/CosyVoice.
    Uses pre-norm (normalize_before=True) style.
    """

    def __init__(
        self,
        size: int,
        n_head: int,
        d_inner: int,
        dropout_rate: float = 0.1,
        key_bias: bool = True,
    ):
        super().__init__()
        # Use eps=1e-12 to match PyTorch
        self.norm_mha = nn.LayerNorm(size, eps=1e-12)
        self.self_attn = RelPositionMultiHeadedAttention(
            n_head, size, dropout_rate, key_bias
        )
        self.norm_ff = nn.LayerNorm(size, eps=1e-12)
        self.feed_forward = PositionwiseFeedForward(size, d_inner, dropout_rate)
        self.size = size

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        pos_emb: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Args:
            x: Input (B, T, D)
            mask: Attention mask (B, T) or (B, T, T)
            pos_emb: Positional embedding (1, 2*T-1, D)

        Returns:
            Output (B, T, D)
        """
        # Multi-head self-attention with pre-norm and residual
        residual = x
        x = self.norm_mha(x)
        x = residual + self.self_attn(x, mask, pos_emb)

        # Feed-forward with pre-norm and residual
        residual = x
        x = self.norm_ff(x)
        x = residual + self.feed_forward(x)

        return x


class PreLookaheadLayer(nn.Module):
    """Pre-lookahead convolution layer."""

    def __init__(self, channels: int, pre_lookahead_len: int = 3):
        super().__init__()
        self.channels = channels
        self.pre_lookahead_len = pre_lookahead_len

        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=pre_lookahead_len + 1,
            stride=1,
            padding=0,
        )
        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=0,
        )

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: Input (B, T, C)

        Returns:
            Output (B, T, C)
        """
        # MLX Conv1d expects (B, T, C) format - no transpose needed
        # Look ahead padding on time dimension (axis 1)
        out = mx.pad(x, [(0, 0), (0, self.pre_lookahead_len), (0, 0)])
        out = nn.leaky_relu(self.conv1(out))

        # Causal padding for second conv
        out = mx.pad(out, [(0, 0), (2, 0), (0, 0)])
        out = self.conv2(out)

        # Residual connection
        return out + x


class Upsample1DEncoder(nn.Module):
    """1D upsampling layer for encoder."""

    def __init__(self, channels: int, stride: int = 2):
        super().__init__()
        self.channels = channels
        self.stride = stride
        self.conv = nn.Conv1d(
            channels, channels, kernel_size=stride * 2 + 1, stride=1, padding=0
        )

    def __call__(self, x: mx.array, x_lens: mx.array) -> Tuple[mx.array, mx.array]:
        """
        Args:
            x: Input (B, T, C) - MLX format
            x_lens: Lengths (B,)

        Returns:
            x: Upsampled (B, T*stride, C)
            x_lens: Updated lengths
        """
        B, T, C = x.shape

        # Nearest neighbor upsampling on time dimension
        x = mx.repeat(x, self.stride, axis=1)

        # Causal padding on time dimension and conv
        x = mx.pad(x, [(0, 0), (self.stride * 2, 0), (0, 0)])
        x = self.conv(x)

        return x, x_lens * self.stride


class UpsampleConformerEncoder(nn.Module):
    """
    Upsampling Conformer encoder for S3Gen.
    Converts speech tokens to mel-spectrogram features.
    """

    def __init__(
        self,
        input_size: int = 512,
        output_size: int = 512,
        attention_heads: int = 8,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self._output_size = output_size

        # Input embedding
        self.embed = LinearInput(input_size, output_size, dropout_rate)

        # Pre-lookahead layer
        self.pre_lookahead_layer = PreLookaheadLayer(output_size, pre_lookahead_len=3)

        # Encoder layers
        self.encoders = [
            ConformerEncoderLayer(
                output_size, attention_heads, linear_units, dropout_rate
            )
            for _ in range(num_blocks)
        ]

        # Upsampling
        self.up_layer = Upsample1DEncoder(output_size, stride=2)

        # Post-upsample embedding
        self.up_embed = LinearInput(input_size, output_size, dropout_rate)

        # Post-upsample encoder layers
        self.up_encoders = [
            ConformerEncoderLayer(
                output_size, attention_heads, linear_units, dropout_rate
            )
            for _ in range(4)
        ]

        # Final norm
        self.after_norm = nn.LayerNorm(output_size)

    def output_size(self) -> int:
        return self._output_size

    def __call__(
        self,
        xs: mx.array,
        xs_lens: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        """
        Args:
            xs: Input (B, T, D)
            xs_lens: Lengths (B,)

        Returns:
            xs: Encoded features (B, T*2, D)
            masks: Output masks (B, 1, T*2)
        """
        B, T, D = xs.shape

        # Create mask
        mask = mx.arange(T)[None, :] < xs_lens[:, None]  # (B, T)
        mask = mask[:, None, :]  # (B, 1, T)

        # Input projection
        xs, pos_emb, mask = self.embed(xs, mask)

        # Pre-lookahead
        xs = self.pre_lookahead_layer(xs)

        # Encoder layers
        mask_1d = mask[:, 0, :]  # (B, T)
        for layer in self.encoders:
            xs = layer(xs, mask_1d, pos_emb)

        # Upsampling - up_layer now works with (B, T, D) directly
        xs, xs_lens = self.up_layer(xs, xs_lens)

        # Update mask
        T2 = xs.shape[1]
        mask = mx.arange(T2)[None, :] < xs_lens[:, None]
        mask = mask[:, None, :]

        # Post-upsample embedding
        xs, pos_emb, mask = self.up_embed(xs, mask)

        # Post-upsample encoder
        mask_1d = mask[:, 0, :]
        for layer in self.up_encoders:
            xs = layer(xs, mask_1d, pos_emb)

        # Final norm
        xs = self.after_norm(xs)

        return xs, mask
