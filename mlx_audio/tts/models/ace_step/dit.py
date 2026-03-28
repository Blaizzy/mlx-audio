# Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)


import math
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn

from .config import ModelConfig
from .modules import (
    DiTLayer,
    KVCache,
    RMSNorm,
    RotaryEmbedding,
    TimestepEmbedding,
    create_4d_mask,
)


class DiTModel(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.hidden_size = config.hidden_size
        self.patch_size = config.patch_size
        in_channels = config.in_channels

        # Rotary position embeddings
        self.rotary_emb = RotaryEmbedding(
            config.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
        )

        # Input projection (patch embedding using 1D conv)
        # PyTorch: Conv1d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        # In MLX we use Linear after reshaping
        self.proj_in_weight = mx.zeros(
            (config.hidden_size, config.patch_size, in_channels)
        )
        self.proj_in_bias = mx.zeros((config.hidden_size,))

        # Timestep embeddings
        self.time_embed = TimestepEmbedding(
            in_channels=256,
            time_embed_dim=config.hidden_size,
        )
        self.time_embed_r = TimestepEmbedding(
            in_channels=256,
            time_embed_dim=config.hidden_size,
        )

        # Condition embedder
        self.condition_embedder = nn.Linear(
            config.hidden_size, config.hidden_size, bias=True
        )

        # DiT layers
        self.layers = []
        for layer_idx in range(config.num_hidden_layers):
            layer_type = config.layer_types[layer_idx]
            sliding_window = (
                config.sliding_window if layer_type == "sliding_attention" else None
            )
            self.layers.append(
                DiTLayer(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.intermediate_size,
                    num_attention_heads=config.num_attention_heads,
                    num_key_value_heads=config.num_key_value_heads,
                    head_dim=config.head_dim,
                    rms_norm_eps=config.rms_norm_eps,
                    attention_bias=config.attention_bias,
                    use_cross_attention=True,
                    sliding_window=sliding_window,
                )
            )

        # Output normalization and projection
        self.norm_out = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Output projection (ConvTranspose1d equivalent)
        # PyTorch: ConvTranspose1d(hidden_size, audio_acoustic_hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.proj_out_weight = mx.zeros(
            (config.audio_acoustic_hidden_dim, config.patch_size, config.hidden_size)
        )
        self.proj_out_bias = mx.zeros((config.audio_acoustic_hidden_dim,))

        # Scale-shift table for adaptive output normalization
        self.scale_shift_table = mx.random.normal(
            (1, 2, config.hidden_size)
        ) / math.sqrt(config.hidden_size)

    def _conv1d_forward(
        self, x: mx.array, weight: mx.array, bias: mx.array, stride: int
    ) -> mx.array:
        """Apply 1D convolution.

        Args:
            x: Input [batch, seq_len, in_channels]
            weight: Conv weights [out_channels, kernel_size, in_channels]
            bias: Conv bias [out_channels]
            stride: Stride value

        Returns:
            Output [batch, seq_len // stride, out_channels]
        """
        # MLX conv1d expects: input [N, L, C_in], weight [C_out, K, C_in]
        # x is already [batch, seq_len, in_channels]
        # weight is [out_channels, kernel_size, in_channels]
        output = mx.conv1d(x, weight, stride=stride)
        output = output + bias
        return output

    def _conv_transpose1d_forward(
        self, x: mx.array, weight: mx.array, bias: mx.array, stride: int
    ) -> mx.array:
        """Apply 1D transposed convolution.

        Args:
            x: Input [batch, seq_len, in_channels]
            weight: Conv weights [out_channels, kernel_size, in_channels]
            bias: Conv bias [out_channels]
            stride: Stride value

        Returns:
            Output [batch, seq_len * stride, out_channels]
        """
        batch_size, seq_len, in_channels = x.shape
        out_channels, kernel_size, _ = weight.shape

        # For transposed conv with stride=kernel_size (non-overlapping)
        # Each input position produces kernel_size outputs

        # x: [batch, seq_len, in_channels]
        # weight: [out_channels, kernel_size, in_channels]

        # Compute all outputs at once
        # [batch, seq_len, in_channels] @ [in_channels, kernel_size * out_channels]
        # weight is [out_channels, kernel_size, in_channels]
        # transpose(2, 1, 0) -> [in_channels, kernel_size, out_channels]
        # reshape -> [in_channels, kernel_size * out_channels]
        # This ordering ensures output channels are grouped by kernel position
        weight_reshaped = weight.transpose(2, 1, 0).reshape(
            in_channels, kernel_size * out_channels
        )
        output = x @ weight_reshaped  # [batch, seq_len, out_channels * kernel_size]

        # Reshape to [batch, seq_len, kernel_size, out_channels]
        output = output.reshape(batch_size, seq_len, kernel_size, out_channels)

        # Reshape to [batch, seq_len * kernel_size, out_channels]
        output = output.reshape(batch_size, seq_len * kernel_size, out_channels)

        # Add bias
        output = output + bias

        return output

    def __call__(
        self,
        hidden_states: mx.array,
        timestep: mx.array,
        timestep_r: mx.array,
        attention_mask: Optional[mx.array],
        encoder_hidden_states: mx.array,
        encoder_attention_mask: Optional[mx.array],
        context_latents: mx.array,
        cache: Optional[List[KVCache]] = None,
    ) -> mx.array:
        """Forward pass.

        Args:
            hidden_states: Noisy latents [batch, seq_len, audio_dim]
            timestep: Diffusion timestep [batch]
            timestep_r: Second timestep for flow matching [batch]
            attention_mask: Self-attention mask
            encoder_hidden_states: Encoder outputs [batch, enc_len, hidden_size]
            encoder_attention_mask: Cross-attention mask
            context_latents: Context (source latents + chunk masks) [batch, seq_len, context_dim]
            cache: Optional list of KVCache for cross-attention (one per layer)

        Returns:
            Predicted flow/velocity [batch, seq_len, audio_dim]
        """
        batch_size = hidden_states.shape[0]

        # Compute timestep embeddings
        # Both time_embed and time_embed_r are used and combined
        # time_embed_r receives (timestep - timestep_r) which is 0 when they're equal
        temb_t, timestep_proj_t = self.time_embed(timestep)
        temb_r, timestep_proj_r = self.time_embed_r(timestep - timestep_r)
        # Combine embeddings
        temb = temb_t + temb_r
        timestep_proj = timestep_proj_t + timestep_proj_r

        # Concatenate context latents with hidden states
        hidden_states = mx.concatenate([context_latents, hidden_states], axis=-1)

        # Record original length for later cropping
        original_seq_len = hidden_states.shape[1]

        # Pad if not divisible by patch_size
        pad_length = 0
        if hidden_states.shape[1] % self.patch_size != 0:
            pad_length = self.patch_size - (hidden_states.shape[1] % self.patch_size)
            padding = mx.zeros(
                (batch_size, pad_length, hidden_states.shape[-1]),
                dtype=hidden_states.dtype,
            )
            hidden_states = mx.concatenate([hidden_states, padding], axis=1)

        # Project to patches
        hidden_states = self._conv1d_forward(
            hidden_states, self.proj_in_weight, self.proj_in_bias, self.patch_size
        )

        # Project encoder states
        encoder_hidden_states = self.condition_embedder(encoder_hidden_states)

        seq_len = hidden_states.shape[1]
        encoder_seq_len = encoder_hidden_states.shape[1]

        # Position IDs
        position_ids = mx.arange(seq_len)[None, :]
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Create attention masks (bidirectional for DiT)
        # Note: Don't use the original attention_mask here since it has the pre-patched length
        # After patching, all positions are valid
        self_attn_mask = create_4d_mask(
            seq_len=seq_len,
            dtype=hidden_states.dtype,
            attention_mask=None,  # All positions valid after patching
            is_causal=False,
        )

        # Cross-attention mask
        # Note: The PyTorch implementation ignores the encoder_attention_mask input
        # and creates a bidirectional mask allowing all positions to attend.
        # We match this behavior by setting cross_attn_mask to None (no masking).
        cross_attn_mask = None

        # Pass through DiT layers
        for layer_idx, layer in enumerate(self.layers):
            layer_type = self.config.layer_types[layer_idx]

            # Use sliding window mask for sliding attention layers
            if layer_type == "sliding_attention" and self.config.use_sliding_window:
                layer_mask = create_4d_mask(
                    seq_len=seq_len,
                    dtype=hidden_states.dtype,
                    attention_mask=None,  # All positions valid after patching
                    sliding_window=self.config.sliding_window,
                    is_sliding_window=True,
                    is_causal=False,
                )
            else:
                layer_mask = self_attn_mask

            # Get cache for this layer (KVCache auto-populates on first use)
            layer_cache = cache[layer_idx] if cache is not None else None

            hidden_states = layer(
                hidden_states,
                position_embeddings,
                timestep_proj,
                attention_mask=layer_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=cross_attn_mask,
                cache=layer_cache,
            )

        # Output projection with adaptive layer norm
        shift, scale = mx.split(self.scale_shift_table + temb[:, None, :], 2, axis=1)

        hidden_states = self.norm_out(hidden_states)
        hidden_states = hidden_states * (1 + scale) + shift

        # Project output (de-patchify)
        hidden_states = self._conv_transpose1d_forward(
            hidden_states, self.proj_out_weight, self.proj_out_bias, self.patch_size
        )

        # Crop back to original length
        hidden_states = hidden_states[:, :original_seq_len, :]

        return hidden_states
