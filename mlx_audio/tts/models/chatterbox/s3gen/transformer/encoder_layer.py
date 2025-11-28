# Ported from https://github.com/resemble-ai/chatterbox

from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


class ConformerEncoderLayer(nn.Module):
    """
    Conformer encoder layer module.

    Combines self-attention, convolution, and feed-forward modules
    with residual connections and layer normalization.
    """

    def __init__(
        self,
        size: int,
        self_attn: nn.Module,
        feed_forward: Optional[nn.Module] = None,
        feed_forward_macaron: Optional[nn.Module] = None,
        conv_module: Optional[nn.Module] = None,
        dropout_rate: float = 0.1,
        normalize_before: bool = True,
    ):
        """
        Args:
            size: Input/output dimension
            self_attn: Self-attention module
            feed_forward: Feed-forward module
            feed_forward_macaron: Optional macaron-style feed-forward
            conv_module: Optional convolution module
            dropout_rate: Dropout rate
            normalize_before: Whether to apply layer norm before or after sub-blocks
        """
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.conv_module = conv_module

        self.norm_ff = nn.LayerNorm(size, eps=1e-12)
        self.norm_mha = nn.LayerNorm(size, eps=1e-12)

        if feed_forward_macaron is not None:
            self.norm_ff_macaron = nn.LayerNorm(size, eps=1e-12)
            self.ff_scale = 0.5
        else:
            self.ff_scale = 1.0

        if self.conv_module is not None:
            self.norm_conv = nn.LayerNorm(size, eps=1e-12)
            self.norm_final = nn.LayerNorm(size, eps=1e-12)

        self.dropout_rate = dropout_rate
        self.size = size
        self.normalize_before = normalize_before

    def __call__(
        self,
        x: mx.array,
        mask: mx.array,
        pos_emb: mx.array,
        mask_pad: mx.array = None,
        att_cache: mx.array = None,
        cnn_cache: mx.array = None,
    ) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
        """
        Compute encoded features.

        Args:
            x: Input tensor (B, T, D)
            mask: Attention mask (B, T, T)
            pos_emb: Positional embeddings
            mask_pad: Padding mask (B, 1, T)
            att_cache: Attention KV cache (1, head, cache_t, d_k * 2)
            cnn_cache: Convolution cache (1, D, cache_t)

        Returns:
            x: Output tensor (B, T, D)
            mask: Mask tensor (B, T, T)
            new_att_cache: Updated attention cache
            new_cnn_cache: Updated convolution cache
        """
        # Macaron-style feed-forward (optional, half-step)
        if self.feed_forward_macaron is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_ff_macaron(x)
            ff_out = self.feed_forward_macaron(x)
            if self.training and self.dropout_rate > 0:
                ff_out = nn.Dropout(self.dropout_rate)(ff_out)
            x = residual + self.ff_scale * ff_out
            if not self.normalize_before:
                x = self.norm_ff_macaron(x)

        # Multi-headed self-attention
        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)

        x_att, new_att_cache = self.self_attn(
            x, x, x, mask, pos_emb=pos_emb, cache=att_cache
        )

        if self.training and self.dropout_rate > 0:
            x_att = nn.Dropout(self.dropout_rate)(x_att)
        x = residual + x_att

        if not self.normalize_before:
            x = self.norm_mha(x)

        # Convolution module (optional)
        new_cnn_cache = mx.zeros((0, 0, 0))
        if self.conv_module is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_conv(x)

            x, new_cnn_cache = self.conv_module(x, mask_pad, cnn_cache)

            if self.training and self.dropout_rate > 0:
                x = nn.Dropout(self.dropout_rate)(x)
            x = residual + x

            if not self.normalize_before:
                x = self.norm_conv(x)

        # Feed-forward module
        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)

        ff_out = self.feed_forward(x)
        if self.training and self.dropout_rate > 0:
            ff_out = nn.Dropout(self.dropout_rate)(ff_out)
        x = residual + self.ff_scale * ff_out

        if not self.normalize_before:
            x = self.norm_ff(x)

        # Final normalization (if using conv module)
        if self.conv_module is not None:
            x = self.norm_final(x)

        return x, mask, new_att_cache, new_cnn_cache
