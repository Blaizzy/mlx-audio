# Original: ClearerVoice-Studio (github.com/modelscope/ClearerVoice-Studio)
# Copyright (c) Speech Lab, Alibaba Group
# Licensed under Apache License 2.0
# MLX port by Dmitry Starkov

from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .ffconvm import FFConvM
from .flash_attention_kernels import FlashAttentionImplementations
from .offsetscale import OffsetScale


class FLASH_ShareA_FFConvM(nn.Module):
    """
    MLX implementation of FLASH_ShareA_FFConvM with 1:1 mathematical equivalence to PyTorch.

    Fast Shared Dual Attention Mechanism with feed-forward convolutional blocks.
    Published in paper: "MossFormer: Pushing the Performance Limit of Monaural Speech Separation
    using Gated Single-Head Transformer with Convolution-Augmented Joint Self-Attentions", ICASSP 2023.
    (https://arxiv.org/abs/2302.11824)

    Args:
        dim (int): Input dimension
        group_size (int): Size of groups for processing (default: 256)
        query_key_dim (int): Dimension of the query and key (default: 128)
        expansion_factor (float): Factor to expand the hidden dimension (default: 4.0)
            Note: The architecture is designed for expansion_factor=4.0. Other values
            will cause dimension mismatches because to_out expects dim*2 inputs, and
            the gated outputs produce hidden_dim/2 = dim*expansion_factor/2 dimensions
        causal (bool): Whether to use causal masking (default: False)
        dropout (float): Dropout rate (default: 0.1)
        rotary_pos_emb: Rotary positional embeddings for attention (default: None)
        norm_klass: Normalization class to use (default: LayerNorm)
        shift_tokens (bool): Whether to shift tokens for attention calculation (default: True)

    Inputs:
        - **x** (batch, seq_len, dim): Input tensor
        - **mask** (batch, seq_len): Optional attention mask

    Returns:
        - **output** (batch, seq_len, dim): Output tensor after applying attention and projections
    """

    def __init__(
        self,
        *,
        dim: int,
        group_size: int = 256,
        query_key_dim: int = 128,
        expansion_factor: float = 4.0,
        causal: bool = False,
        dropout: float = 0.1,
        rotary_pos_emb=None,
        norm_klass=nn.LayerNorm,
        shift_tokens: bool = True,
    ):
        super().__init__()

        # Store configuration
        self.dim = dim
        self.group_size = group_size
        self.query_key_dim = query_key_dim
        self.expansion_factor = expansion_factor
        self.causal = causal
        self.dropout_p = dropout
        self.shift_tokens = shift_tokens

        # Calculate hidden dimension
        hidden_dim = int(dim * expansion_factor)

        # Store for debugging
        self.hidden_dim = hidden_dim

        # Initialize positional embeddings and dropout
        self.rotary_pos_emb = rotary_pos_emb
        self.dropout = nn.Dropout(dropout)

        # Feed-forward layers
        self.to_hidden = FFConvM(
            dim_in=dim,
            dim_out=hidden_dim,
            norm_klass=norm_klass,
            dropout=dropout,
        )
        self.to_qk = FFConvM(
            dim_in=dim,
            dim_out=query_key_dim,
            norm_klass=norm_klass,
            dropout=dropout,
        )

        # Offset and scale for query and key (4 heads: quad_q, lin_q, quad_k, lin_k)
        self.qk_offset_scale = OffsetScale(query_key_dim, heads=4)

        self.to_out = FFConvM(
            dim_in=dim * 2,
            dim_out=dim,
            norm_klass=norm_klass,
            dropout=dropout,
        )

        # Track training state for dropout consistency
        self._training = True

    def eval(self):
        """Set model to evaluation mode."""
        self._training = False
        self.to_hidden.eval()
        self.to_qk.eval()
        self.to_out.eval()
        return self

    def train(self):
        """Set model to training mode."""
        self._training = True
        self.to_hidden.train()
        self.to_qk.train()
        self.to_out.train()
        return self

    def __call__(self, x: mx.array, *, mask: Optional[mx.array] = None) -> mx.array:
        """
        Forward pass for FLASH layer.

        Args:
            x (mx.array): Input tensor of shape (batch, seq_len, features)
            mask (mx.array, optional): Mask for attention

        Returns:
            mx.array: Output tensor after applying attention and projections
        """
        # Pre-normalization step
        normed_x = x
        residual = x  # Save residual for skip connection

        # Token shifting if enabled
        if self.shift_tokens:
            # Split input into two halves
            x_shift, x_pass = mx.split(normed_x, 2, axis=-1)

            # Pad x_shift: shift tokens by 1 position forward
            # PyTorch: F.pad(x_shift, (0, 0, 1, -1), value=0.)
            # This pads the sequence dimension: adds 1 at the beginning, removes 1 at the end
            seq_len = x_shift.shape[1]
            if seq_len > 1:
                # Create padding: (batch, 1, channels)
                pad_shape = list(x_shift.shape)
                pad_shape[1] = 1
                padding = mx.zeros(pad_shape)

                # Concatenate padding at the beginning and remove last timestep
                x_shift = mx.concatenate([padding, x_shift[:, :-1, :]], axis=1)
            else:
                # For single timestep, just zero it out
                x_shift = mx.zeros_like(x_shift)

            # Concatenate shifted and unshifted parts
            normed_x = mx.concatenate([x_shift, x_pass], axis=-1)

        # Initial projections

        hidden_output = self.to_hidden(normed_x)

        v, u = mx.split(hidden_output, 2, axis=-1)

        qk = self.to_qk(normed_x)

        # Offset and scale - returns list of 4 tensors
        qk_outputs = self.qk_offset_scale(qk)
        quad_q, lin_q, quad_k, lin_k = qk_outputs

        # Calculate attention
        att_v, att_u = self.cal_attention(x, quad_q, lin_q, quad_k, lin_k, v, u, mask)

        # Output calculation with gating
        # PyTorch: out = (att_u * v) * self.gateActivate(att_v * u)
        # gateActivate is nn.Sigmoid()
        att_u_v = att_u * v
        att_v_u = att_v * u
        gate = mx.sigmoid(att_v_u)

        out = att_u_v * gate

        # Final projection and residual connection
        # PyTorch: x = x + self.to_out(out)
        final_out = self.to_out(out)

        result = x + final_out  # Residual connection

        return result

    # custom metal kernel based attention (!)
    def cal_attention(
        self,
        x: mx.array,
        quad_q: mx.array,
        lin_q: mx.array,
        quad_k: mx.array,
        lin_k: mx.array,
        v: mx.array,
        u: mx.array,
        mask: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array]:
        """
        Calculate attention output using quadratic and linear attention mechanisms.

        Args:
            x (mx.array): Input tensor of shape (batch, seq_len, features)
            quad_q (mx.array): Quadratic query representation
            lin_q (mx.array): Linear query representation
            quad_k (mx.array): Quadratic key representation
            lin_k (mx.array): Linear key representation
            v (mx.array): Value representation
            u (mx.array): Additional value representation
            mask (mx.array, optional): Mask for attention

        Returns:
            Tuple[mx.array, mx.array]: Attention outputs for v and u
        """
        b, n, device, g = x.shape[0], x.shape[1], None, self.group_size

        # Apply mask to linear keys if provided
        if mask is not None:
            # Expand mask for broadcasting: (batch, seq_len) -> (batch, seq_len, 1)
            lin_mask = mx.expand_dims(mask, axis=-1)
            # Apply mask: set masked positions to 0
            lin_k = lin_k * lin_mask.astype(lin_k.dtype)

        # Apply rotary positional embeddings if available
        if self.rotary_pos_emb is not None:
            # Apply rotary embeddings to queries and keys
            # Check if it's mlx.nn.RoPE or custom implementation
            if hasattr(self.rotary_pos_emb, "rotate_queries_or_keys"):
                # Custom implementation (SimplifiedRotaryEmbedding)
                quad_q = self.rotary_pos_emb.rotate_queries_or_keys(quad_q)
                lin_q = self.rotary_pos_emb.rotate_queries_or_keys(lin_q)
                quad_k = self.rotary_pos_emb.rotate_queries_or_keys(quad_k)
                lin_k = self.rotary_pos_emb.rotate_queries_or_keys(lin_k)
            else:
                # mlx.nn.RoPE - use direct call
                quad_q = self.rotary_pos_emb(quad_q)
                lin_q = self.rotary_pos_emb(lin_q)
                quad_k = self.rotary_pos_emb(quad_k)
                lin_k = self.rotary_pos_emb(lin_k)

        # Padding for group processing
        def padding_to_multiple_of(n: int, mult: int) -> int:
            remainder = n % mult
            if remainder == 0:
                return 0
            return mult - remainder

        padding = padding_to_multiple_of(n, g)

        if padding > 0:
            # Pad all tensors along sequence dimension
            # PyTorch: F.pad(t, (0, 0, 0, padding), value=0.)
            pad_shape = lambda tensor: [(0, 0), (0, padding)] + [(0, 0)] * (
                len(tensor.shape) - 2
            )

            quad_q = mx.pad(quad_q, pad_shape(quad_q))
            quad_k = mx.pad(quad_k, pad_shape(quad_k))
            lin_q = mx.pad(lin_q, pad_shape(lin_q))
            lin_k = mx.pad(lin_k, pad_shape(lin_k))
            v = mx.pad(v, pad_shape(v))
            u = mx.pad(u, pad_shape(u))

            if mask is not None:
                # Pad mask
                mask = mx.pad(mask, [(0, 0), (0, padding)], constant_values=False)
            else:
                # Create default mask: True for original positions, False for padding
                mask = mx.ones((b, n), dtype=mx.bool_)
                mask = mx.pad(mask, [(0, 0), (0, padding)], constant_values=False)

        # Group along sequence for attention
        # PyTorch: rearrange(t, 'b (g n) d -> b g n d', n=self.group_size)
        # MLX equivalent: reshape to group format
        new_seq_len = quad_q.shape[1]
        num_groups = new_seq_len // g

        def group_reshape(tensor):
            # (batch, seq_len, dim) -> (batch, num_groups, group_size, dim)
            batch_size, _, dim = tensor.shape
            return tensor.reshape(batch_size, num_groups, g, dim)

        quad_q = group_reshape(quad_q)
        quad_k = group_reshape(quad_k)
        lin_q = group_reshape(lin_q)
        lin_k = group_reshape(lin_k)
        v = group_reshape(v)
        u = group_reshape(u)

        if mask is not None:
            # Reshape mask: (batch, seq_len) -> (batch, num_groups, 1, group_size)
            mask = mask.reshape(b, num_groups, g)
            mask = mx.expand_dims(mask, axis=2)  # Add dimension for broadcasting

        # Calculate quadratic attention output using optimized kernel
        # Use the simple kernel for ReLU² optimization
        quad_out_v = FlashAttentionImplementations.simple_kernel(quad_q, quad_k, v, g)
        quad_out_u = FlashAttentionImplementations.simple_kernel(quad_q, quad_k, u, g)

        # Calculate linear attention output
        if self.causal:
            # Causal linear attention with cumulative sum
            # PyTorch: lin_kv = einsum('b g n d, b g n e -> b g d e', lin_k, v) / g
            lin_kv = mx.matmul(mx.transpose(lin_k, [0, 1, 3, 2]), v) / g
            # Cumulative sum over groups
            lin_kv = mx.cumsum(lin_kv, axis=1)
            # Pad and shift
            lin_kv = mx.pad(lin_kv, [(0, 0), (1, 0), (0, 0), (0, 0)])[:, :-1, :, :]
            # PyTorch: lin_out_v = einsum('b g d e, b g n d -> b g n e', lin_kv, lin_q)
            lin_out_v = mx.matmul(lin_q, lin_kv)

            # Same for u
            lin_ku = mx.matmul(mx.transpose(lin_k, [0, 1, 3, 2]), u) / g
            lin_ku = mx.cumsum(lin_ku, axis=1)
            lin_ku = mx.pad(lin_ku, [(0, 0), (1, 0), (0, 0), (0, 0)])[:, :-1, :, :]
            lin_out_u = mx.matmul(lin_q, lin_ku)
        else:
            # Non-causal linear attention
            # PyTorch: lin_kv = einsum('b g n d, b g n e -> b d e', lin_k, v) / n
            # We need to:
            # 1. For each batch, compute k^T @ v for all groups and sum
            # 2. Result should be (batch, query_key_dim, value_dim)

            # lin_k shape: (batch, num_groups, group_size, query_key_dim)
            # v shape: (batch, num_groups, group_size, value_dim)

            # Reshape to combine groups and group_size
            batch_size = lin_k.shape[0]
            total_seq = lin_k.shape[1] * lin_k.shape[2]
            query_key_dim = lin_k.shape[3]
            value_dim = v.shape[3]

            # Reshape lin_k: (batch, total_seq, query_key_dim)
            lin_k_reshaped = lin_k.reshape(batch_size, total_seq, query_key_dim)
            # Reshape v: (batch, total_seq, value_dim)
            v_reshaped = v.reshape(batch_size, total_seq, value_dim)

            # Compute k^T @ v: (batch, query_key_dim, value_dim)
            # Note: Division should be by original sequence length n, not padded length
            lin_kv = mx.matmul(mx.transpose(lin_k_reshaped, [0, 2, 1]), v_reshaped) / n

            # lin_q shape: (batch, num_groups, group_size, query_key_dim)
            # lin_kv shape: (batch, query_key_dim, value_dim)
            # We need output shape: (batch, num_groups, group_size, value_dim)

            # PyTorch: lin_out_v = einsum('b g n d, b d e -> b g n e', lin_q, lin_kv)
            # To handle the einsum broadcasting, we need to:
            # 1. Reshape lin_q to combine groups and group_size dimensions
            # 2. Do the matmul
            # 3. Reshape back
            lin_q_reshaped = lin_q.reshape(batch_size, num_groups * g, query_key_dim)
            lin_out_v_reshaped = mx.matmul(lin_q_reshaped, lin_kv)
            lin_out_v = lin_out_v_reshaped.reshape(batch_size, num_groups, g, value_dim)

            # Same for u
            u_reshaped = u.reshape(batch_size, total_seq, value_dim)

            lin_ku = mx.matmul(mx.transpose(lin_k_reshaped, [0, 2, 1]), u_reshaped) / n
            lin_out_u_reshaped = mx.matmul(lin_q_reshaped, lin_ku)
            lin_out_u = lin_out_u_reshaped.reshape(batch_size, num_groups, g, value_dim)

        # Combine quadratic and linear attention outputs
        combined_out_v = quad_out_v + lin_out_v
        combined_out_u = quad_out_u + lin_out_u

        # Reshape back to original format and remove padding
        # (batch, num_groups, group_size, dim) -> (batch, seq_len, dim)
        def ungroup_reshape(tensor):
            batch_size, num_groups, group_size, dim = tensor.shape
            return tensor.reshape(batch_size, num_groups * group_size, dim)

        final_out_v = ungroup_reshape(combined_out_v)[:, :n]  # Remove padding
        final_out_u = ungroup_reshape(combined_out_u)[:, :n]  # Remove padding

        return final_out_v, final_out_u
