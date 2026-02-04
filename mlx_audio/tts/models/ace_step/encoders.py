# Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)


import math
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .config import ModelConfig
from .modules import (
    EncoderLayer,
    RMSNorm,
    RotaryEmbedding,
    create_4d_mask,
    pack_sequences,
)


class LyricEncoder(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Project text embeddings to model hidden size
        self.embed_tokens = nn.Linear(config.text_hidden_dim, config.hidden_size)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(
            config.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
        )

        # Stack of encoder layers
        self.layers = []
        for layer_idx in range(config.num_lyric_encoder_hidden_layers):
            layer_type = config.layer_types[layer_idx % len(config.layer_types)]
            sliding_window = (
                config.sliding_window if layer_type == "sliding_attention" else None
            )
            self.layers.append(
                EncoderLayer(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.intermediate_size,
                    num_attention_heads=config.num_attention_heads,
                    num_key_value_heads=config.num_key_value_heads,
                    head_dim=config.head_dim,
                    rms_norm_eps=config.rms_norm_eps,
                    attention_bias=config.attention_bias,
                    sliding_window=sliding_window,
                )
            )

    def __call__(
        self,
        inputs_embeds: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """Forward pass.

        Args:
            inputs_embeds: Text embeddings [batch, seq_len, text_hidden_dim]
            attention_mask: Optional attention mask [batch, seq_len]

        Returns:
            Encoded hidden states [batch, seq_len, hidden_size]
        """
        # Project input embeddings
        hidden_states = self.embed_tokens(inputs_embeds)

        batch_size, seq_len, _ = hidden_states.shape

        # Position IDs
        position_ids = mx.arange(seq_len)[None, :]

        # Create position embeddings
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Create attention mask (bidirectional)
        if attention_mask is not None:
            attn_mask = create_4d_mask(
                seq_len=seq_len,
                dtype=hidden_states.dtype,
                attention_mask=attention_mask,
                is_causal=False,  # Bidirectional
            )
        else:
            attn_mask = None

        # Pass through encoder layers
        for layer_idx, layer in enumerate(self.layers):
            layer_type = self.config.layer_types[
                layer_idx % len(self.config.layer_types)
            ]

            # Use sliding window mask for sliding attention layers
            if layer_type == "sliding_attention" and self.config.use_sliding_window:
                layer_mask = create_4d_mask(
                    seq_len=seq_len,
                    dtype=hidden_states.dtype,
                    attention_mask=attention_mask,
                    sliding_window=self.config.sliding_window,
                    is_sliding_window=True,
                    is_causal=False,
                )
            else:
                layer_mask = attn_mask

            hidden_states = layer(hidden_states, position_embeddings, layer_mask)

        hidden_states = self.norm(hidden_states)
        return hidden_states


class AttentionPooler(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Linear(config.hidden_size, config.hidden_size)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(
            config.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
        )

        # Special token for pooling (CLS-like token)
        self.special_token = mx.random.normal((1, 1, config.hidden_size)) * 0.02

        # Encoder layers
        self.layers = []
        for layer_idx in range(config.num_attention_pooler_hidden_layers):
            layer_type = config.layer_types[layer_idx % len(config.layer_types)]
            sliding_window = (
                config.sliding_window if layer_type == "sliding_attention" else None
            )
            self.layers.append(
                EncoderLayer(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.intermediate_size,
                    num_attention_heads=config.num_attention_heads,
                    num_key_value_heads=config.num_key_value_heads,
                    head_dim=config.head_dim,
                    rms_norm_eps=config.rms_norm_eps,
                    attention_bias=config.attention_bias,
                    sliding_window=sliding_window,
                )
            )

    def __call__(self, x: mx.array) -> mx.array:

        batch_size, time, patches, dim = x.shape

        x = self.embed_tokens(x)

        # Add special token
        special_tokens = mx.broadcast_to(self.special_token, (batch_size, time, 1, dim))
        x = mx.concatenate([special_tokens, x], axis=2)

        # Reshape: [batch * time, patches + 1, dim]
        x = x.reshape(batch_size * time, patches + 1, dim)

        # Position embeddings
        position_ids = mx.arange(x.shape[1])[None, :]
        position_embeddings = self.rotary_emb(x, position_ids)

        hidden_states = x

        # Pass through layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings, None)

        hidden_states = self.norm(hidden_states)

        # Extract CLS token output
        cls_output = hidden_states[:, 0, :]  # [batch * time, dim]
        cls_output = cls_output.reshape(batch_size, time, dim)

        return cls_output


class AudioTokenDetokenizer(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Linear(config.hidden_size, config.hidden_size)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(
            config.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
        )

        # Special tokens for expanding each token into patches
        self.special_tokens = (
            mx.random.normal((1, config.pool_window_size, config.hidden_size)) * 0.02
        )

        # Encoder layers
        self.layers = []
        for layer_idx in range(config.num_attention_pooler_hidden_layers):
            layer_type = config.layer_types[layer_idx % len(config.layer_types)]
            sliding_window = (
                config.sliding_window if layer_type == "sliding_attention" else None
            )
            self.layers.append(
                EncoderLayer(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.intermediate_size,
                    num_attention_heads=config.num_attention_heads,
                    num_key_value_heads=config.num_key_value_heads,
                    head_dim=config.head_dim,
                    rms_norm_eps=config.rms_norm_eps,
                    attention_bias=config.attention_bias,
                    sliding_window=sliding_window,
                )
            )

        # Output projection
        self.proj_out = nn.Linear(config.hidden_size, config.audio_acoustic_hidden_dim)

    def __call__(self, x: mx.array) -> mx.array:

        batch_size, time, dim = x.shape
        pool_window_size = self.config.pool_window_size

        x = self.embed_tokens(x)

        # Expand and add special tokens: [batch, time, 1, dim] -> [batch, time, P, dim]
        x = x[:, :, None, :]  # [batch, time, 1, dim]
        x = mx.broadcast_to(x, (batch_size, time, pool_window_size, dim))

        # Add learnable special tokens
        special_tokens = mx.broadcast_to(
            self.special_tokens, (batch_size, time, pool_window_size, dim)
        )
        x = x + special_tokens

        # Reshape: [batch * time, P, dim]
        x = x.reshape(batch_size * time, pool_window_size, dim)

        # Position embeddings
        position_ids = mx.arange(pool_window_size)[None, :]
        position_embeddings = self.rotary_emb(x, position_ids)

        hidden_states = x

        # Pass through layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings, None)

        hidden_states = self.norm(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        # Reshape back: [batch, time * P, audio_dim]
        hidden_states = hidden_states.reshape(batch_size, time * pool_window_size, -1)

        return hidden_states


class TimbreEncoder(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Project acoustic features to model hidden size
        self.embed_tokens = nn.Linear(config.timbre_hidden_dim, config.hidden_size)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(
            config.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
        )

        # Special token for aggregating timbre information
        self.special_token = mx.random.normal((1, 1, config.hidden_size))

        # Encoder layers
        self.layers = []
        for layer_idx in range(config.num_timbre_encoder_hidden_layers):
            layer_type = config.layer_types[layer_idx % len(config.layer_types)]
            sliding_window = (
                config.sliding_window if layer_type == "sliding_attention" else None
            )
            self.layers.append(
                EncoderLayer(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.intermediate_size,
                    num_attention_heads=config.num_attention_heads,
                    num_key_value_heads=config.num_key_value_heads,
                    head_dim=config.head_dim,
                    rms_norm_eps=config.rms_norm_eps,
                    attention_bias=config.attention_bias,
                    sliding_window=sliding_window,
                )
            )

    def unpack_timbre_embeddings(
        self,
        timbre_embs_packed: mx.array,
        refer_audio_order_mask: mx.array,
    ) -> Tuple[mx.array, mx.array]:

        n_samples, dim = timbre_embs_packed.shape

        # Get batch size from order mask
        batch_size = int(mx.max(refer_audio_order_mask).item() + 1)

        # Simple case: all samples in one batch
        # This is the common inference case
        timbre_embs_unpack = timbre_embs_packed[None, :, :]  # [1, N, dim]
        new_mask = mx.ones((batch_size, n_samples))

        return timbre_embs_unpack, new_mask

    def __call__(
        self,
        refer_audio_acoustic_hidden_states_packed: mx.array,
        refer_audio_order_mask: mx.array,
    ) -> Tuple[mx.array, mx.array]:

        inputs_embeds = refer_audio_acoustic_hidden_states_packed

        # Project embeddings
        inputs_embeds = self.embed_tokens(inputs_embeds)

        batch_size, seq_len, _ = inputs_embeds.shape

        # Position embeddings
        position_ids = mx.arange(seq_len)[None, :]
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        hidden_states = inputs_embeds

        # Pass through encoder layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings, None)

        hidden_states = self.norm(hidden_states)

        # Extract first token as timbre embedding
        hidden_states = hidden_states[:, 0, :]  # [N, dim]

        # Unpack to batch format
        timbre_embs_unpack, timbre_embs_mask = self.unpack_timbre_embeddings(
            hidden_states, refer_audio_order_mask
        )

        return timbre_embs_unpack, timbre_embs_mask


def round_ste(z: mx.array) -> mx.array:
    """Round with straight-through estimator (for inference, just round)."""
    return mx.round(z)


def floor_ste(z: mx.array) -> mx.array:
    """Floor with straight-through estimator (for inference, just floor)."""
    return mx.floor(z)


class FSQ(nn.Module):
    """Finite Scalar Quantization.

    Based on "Finite Scalar Quantization: VQ-VAE Made Simple"
    https://arxiv.org/abs/2309.15505
    """

    def __init__(
        self,
        levels: list,
        dim: Optional[int] = None,
        preserve_symmetry: bool = False,
        bound_hard_clamp: bool = False,
    ):
        super().__init__()
        self.levels = levels
        self.codebook_dim = len(levels)
        self.preserve_symmetry = preserve_symmetry
        self.bound_hard_clamp = bound_hard_clamp

        # Store levels as array
        self._levels = mx.array(levels, dtype=mx.int32)

        # Compute basis for index calculation
        basis_list = [1]
        for l in levels[:-1]:
            basis_list.append(basis_list[-1] * l)
        self._basis = mx.array(basis_list, dtype=mx.int32)

        # Codebook size
        self.codebook_size = math.prod(levels)

        # Projections if dim != codebook_dim
        dim = dim if dim is not None else self.codebook_dim
        self.dim = dim
        self.has_projections = dim != self.codebook_dim

        if self.has_projections:
            self.project_in = nn.Linear(dim, self.codebook_dim)
            self.project_out = nn.Linear(self.codebook_dim, dim)

    def bound(self, z: mx.array, eps: float = 1e-3) -> mx.array:
        """Bound z using tanh or hard clamp."""
        levels_f = self._levels.astype(z.dtype)
        half_l = (levels_f - 1) * (1 + eps) / 2
        offset = mx.where(self._levels % 2 == 0, 0.5, 0.0)

        if self.bound_hard_clamp:
            # Hard clamp version
            shift = offset / half_l
            bounded_z = mx.clip(z + shift, -1.0, 1.0) * half_l - offset
        else:
            # Tanh version
            shift = mx.arctanh(offset / half_l)
            bounded_z = mx.tanh(z + shift) * half_l - offset

        half_width = self._levels // 2
        return round_ste(bounded_z) / half_width.astype(z.dtype)

    def symmetry_preserving_bound(self, z: mx.array) -> mx.array:
        """Symmetry-preserving quantization."""
        levels_f = self._levels.astype(z.dtype)
        levels_minus_1 = levels_f - 1
        scale = 2.0 / levels_minus_1

        if self.bound_hard_clamp:
            z_bounded = mx.clip(z, -1.0, 1.0)
        else:
            z_bounded = mx.tanh(z)

        bracket = (levels_minus_1 * (z_bounded + 1) / 2.0) + 0.5
        bracket = floor_ste(bracket)
        return scale * bracket - 1.0

    def quantize(self, z: mx.array) -> mx.array:
        """Quantize z, returns quantized zhat."""
        if self.preserve_symmetry:
            return self.symmetry_preserving_bound(z)
        else:
            return self.bound(z)

    def _scale_and_shift(self, zhat_normalized: mx.array) -> mx.array:
        """Convert normalized codes to indices."""
        if self.preserve_symmetry:
            levels_f = self._levels.astype(zhat_normalized.dtype)
            return (zhat_normalized + 1.0) / (2.0 / (levels_f - 1))
        else:
            half_width = (self._levels // 2).astype(zhat_normalized.dtype)
            return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat: mx.array) -> mx.array:
        """Convert indices to normalized codes."""
        if self.preserve_symmetry:
            levels_f = self._levels.astype(zhat.dtype)
            return zhat * (2.0 / (levels_f - 1)) - 1.0
        else:
            half_width = (self._levels // 2).astype(zhat.dtype)
            return (zhat - half_width) / half_width

    def codes_to_indices(self, zhat: mx.array) -> mx.array:
        """Convert quantized codes to codebook indices."""
        zhat_shifted = self._scale_and_shift(zhat)
        basis_f = self._basis.astype(zhat_shifted.dtype)
        indices = mx.sum(zhat_shifted * basis_f, axis=-1)
        return mx.round(indices).astype(mx.int32)

    def indices_to_codes(self, indices: mx.array) -> mx.array:
        """Convert codebook indices back to codes."""
        indices = indices[..., None]
        codes_non_centered = (indices // self._basis) % self._levels
        codes = self._scale_and_shift_inverse(codes_non_centered.astype(mx.float32))
        return codes

    def __call__(self, z: mx.array) -> Tuple[mx.array, mx.array]:
        """Forward pass.

        Args:
            z: Input tensor [batch, seq_len, dim]

        Returns:
            Tuple of (quantized output, indices)
        """
        orig_dtype = z.dtype

        # Project in if needed
        if self.has_projections:
            z = self.project_in(z)

        # Quantize
        codes = self.quantize(z.astype(mx.float32))

        # Get indices
        indices = self.codes_to_indices(codes)

        # Project out if needed
        if self.has_projections:
            codes = self.project_out(codes)

        return codes.astype(orig_dtype), indices


class ResidualFSQ(nn.Module):
    """Residual Finite Scalar Quantization.

    Follows Algorithm 1 in https://arxiv.org/pdf/2107.03312.pdf
    Uses multiple FSQ layers to progressively quantize residuals.
    """

    def __init__(
        self,
        dim: int,
        levels: list,
        num_quantizers: int = 1,
    ):
        super().__init__()
        self.dim = dim
        self.levels = levels
        self.num_quantizers = num_quantizers
        self.codebook_dim = len(levels)
        self.codebook_size = math.prod(levels)

        # Input/output projections if dim != codebook_dim
        requires_projection = self.codebook_dim != dim
        if requires_projection:
            self.project_in = nn.Linear(dim, self.codebook_dim)
            self.project_out = nn.Linear(self.codebook_dim, dim)
        self.has_projections = requires_projection

        # Create FSQ layers
        self.layers = [
            FSQ(
                levels=levels,
                dim=self.codebook_dim,
                preserve_symmetry=True,
                bound_hard_clamp=True,
            )
            for _ in range(num_quantizers)
        ]

        # Compute scales for each quantizer: levels^(-ind)
        levels_arr = mx.array(levels, dtype=mx.float32)
        self._scales = []
        for ind in range(num_quantizers):
            scale = mx.power(levels_arr, -ind)
            self._scales.append(scale)

        # Soft clamp value for input
        self._soft_clamp_value = 1 + (1 / (levels_arr - 1))

    def get_codes_from_indices(self, indices: mx.array) -> mx.array:
        """Get codes from indices for all quantizer layers.

        Args:
            indices: Indices of shape [..., num_quantizers]

        Returns:
            Codes from all layers, shape [num_quantizers, ..., codebook_dim]
        """
        all_codes = []
        for q_idx, layer in enumerate(self.layers):
            layer_indices = indices[..., q_idx]
            codes = layer.indices_to_codes(layer_indices)
            # Apply scale
            codes = codes * self._scales[q_idx]
            all_codes.append(codes)
        return mx.stack(all_codes, axis=0)

    def get_output_from_indices(self, indices: mx.array) -> mx.array:
        """Decode indices back to continuous representations.

        Args:
            indices: Token indices of shape [..., num_quantizers]

        Returns:
            Decoded continuous representations
        """
        codes = self.get_codes_from_indices(indices)
        # Sum across quantizers
        codes_summed = mx.sum(codes, axis=0)

        # Project out if needed
        if self.has_projections:
            codes_summed = self.project_out(codes_summed)

        return codes_summed

    def __call__(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        """Forward pass.

        Args:
            x: Input tensor [batch, seq_len, dim]

        Returns:
            Tuple of (quantized, indices)
        """
        orig_dtype = x.dtype

        # Project in if needed
        if self.has_projections:
            x = self.project_in(x)

        # Soft clamp input
        x_f32 = x.astype(mx.float32)
        clamp_value = self._soft_clamp_value
        x_clamped = mx.tanh(x_f32 / clamp_value) * clamp_value

        # Residual quantization
        quantized_out = mx.zeros_like(x_clamped)
        residual = x_clamped
        all_indices = []

        for layer, scale in zip(self.layers, self._scales):
            # Quantize residual / scale
            quantized, indices = layer(residual / scale)

            # Scale back
            quantized = quantized * scale

            # Update residual and accumulate
            residual = residual - quantized
            quantized_out = quantized_out + quantized

            all_indices.append(indices)

        # Project out if needed
        if self.has_projections:
            quantized_out = self.project_out(quantized_out)

        # Stack indices: [batch, seq_len, num_quantizers]
        all_indices = mx.stack(all_indices, axis=-1)

        return quantized_out.astype(orig_dtype), all_indices


class AudioTokenizer(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Project acoustic features to hidden size
        self.audio_acoustic_proj = nn.Linear(
            config.audio_acoustic_hidden_dim, config.hidden_size
        )

        # Attention pooler
        self.attention_pooler = AttentionPooler(config)

        # Quantizer
        self.quantizer = ResidualFSQ(
            dim=config.fsq_dim,
            levels=config.fsq_input_levels,
            num_quantizers=config.fsq_input_num_quantizers,
        )

        self.pool_window_size = config.pool_window_size

    def __call__(self, hidden_states: mx.array) -> Tuple[mx.array, mx.array]:
        """Forward pass.

        Args:
            hidden_states: Input [batch, time, patches, audio_dim]

        Returns:
            Tuple of (quantized, indices)
        """
        # Project acoustic features
        hidden_states = self.audio_acoustic_proj(hidden_states)

        # Pool sequences
        hidden_states = self.attention_pooler(hidden_states)

        # Quantize
        quantized, indices = self.quantizer(hidden_states)

        return quantized, indices

    def tokenize(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        # Reshape to patches
        batch_size, seq_len, dim = x.shape
        num_patches = seq_len // self.pool_window_size
        x = x.reshape(batch_size, num_patches, self.pool_window_size, dim)

        return self(x)


class ConditionEncoder(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Text projection
        self.text_projector = nn.Linear(
            config.text_hidden_dim, config.hidden_size, bias=False
        )

        # Lyric encoder
        self.lyric_encoder = LyricEncoder(config)

        # Timbre encoder
        self.timbre_encoder = TimbreEncoder(config)

    def __call__(
        self,
        text_hidden_states: mx.array,
        text_attention_mask: mx.array,
        lyric_hidden_states: mx.array,
        lyric_attention_mask: mx.array,
        refer_audio_acoustic_hidden_states_packed: mx.array,
        refer_audio_order_mask: mx.array,
    ) -> Tuple[mx.array, mx.array]:

        # Project text
        text_hidden_states = self.text_projector(text_hidden_states)

        # Encode lyrics
        lyric_hidden_states = self.lyric_encoder(
            lyric_hidden_states, lyric_attention_mask
        )

        # Encode timbre
        timbre_embs_unpack, timbre_embs_mask = self.timbre_encoder(
            refer_audio_acoustic_hidden_states_packed, refer_audio_order_mask
        )

        # Pack sequences: lyrics + timbre + text
        encoder_hidden_states, encoder_attention_mask = pack_sequences(
            lyric_hidden_states,
            timbre_embs_unpack,
            lyric_attention_mask,
            timbre_embs_mask,
        )
        encoder_hidden_states, encoder_attention_mask = pack_sequences(
            encoder_hidden_states,
            text_hidden_states,
            encoder_attention_mask,
            text_attention_mask,
        )

        return encoder_hidden_states, encoder_attention_mask
