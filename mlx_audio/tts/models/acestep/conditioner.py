from typing import Any, Dict, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .dit import MLXAttention, MLXRotaryEmbedding, MLXSwiGLUMLP


class MLXAceStepEncoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        # Self-attention sub-layer
        self.self_attn = MLXAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            rms_norm_eps=config.rms_norm_eps,
            attention_bias=config.attention_bias,
            layer_idx=layer_idx,
            is_cross_attention=False,
            sliding_window=(
                config.sliding_window
                if config.layer_types[layer_idx] == "sliding_attention"
                else None
            ),
        )
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # MLP (feed-forward) sub-layer
        self.mlp = MLXSwiGLUMLP(config.hidden_size, config.intermediate_size)
        self.attention_type = config.layer_types[layer_idx]

    def __call__(
        self,
        hidden_states: mx.array,
        position_cos_sin: Tuple[mx.array, mx.array],
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_cos_sin=position_cos_sin,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class MLXAceStepLyricEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Lyric encoder config usually overrides these but we can fallback to main config
        self.num_layers = getattr(
            config, "num_lyric_encoder_hidden_layers", config.num_hidden_layers
        )

        self.embed_tokens = nn.Linear(config.text_hidden_dim, config.hidden_size)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = MLXRotaryEmbedding(
            config.head_dim, max_len=config.max_position_embeddings
        )

        self.layers = [
            MLXAceStepEncoderLayer(config, layer_idx)
            for layer_idx in range(self.num_layers)
        ]

    def __call__(
        self,
        inputs_embeds: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        hidden_states = self.embed_tokens(inputs_embeds)
        seq_len = hidden_states.shape[1]

        # Rotary embeddings
        cos, sin = self.rotary_emb(seq_len)

        # We need a 4D attention mask [B, 1, L, L] for bidirectional attention
        # Since it's an encoder, it's not causal.
        if attention_mask is not None:
            # Assuming attention_mask is [B, L]
            mask = attention_mask[:, None, None, :]
            # Replace 0s with -inf, 1s with 0
            mask = mx.where(
                mask == 0,
                mx.array(-1e9, dtype=hidden_states.dtype),
                mx.array(0.0, dtype=hidden_states.dtype),
            )
        else:
            mask = None

        for layer in self.layers:
            # Note: AceStep uses self_attn_mask_mapping based on layer type
            # (sliding vs full). MLXAttention handles sliding_window internally if supported.
            hidden_states = layer(
                hidden_states,
                position_cos_sin=(cos, sin),
                attention_mask=mask,
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states


def pack_sequences(
    hidden1: mx.array, hidden2: mx.array, mask1: mx.array, mask2: mx.array
):
    """
    Pack two sequences by concatenating and sorting them based on mask values.
    Valid tokens (mask=1) are shifted to the front.
    """
    # Concatenate hidden states and masks
    hidden_cat = mx.concatenate([hidden1, hidden2], axis=1)  # [B, L, D]
    mask_cat = mx.concatenate([mask1, mask2], axis=1)  # [B, L]

    B, L, D = hidden_cat.shape

    # Sort indices so that mask values of 1 come before 0
    # In mx, argsort is ascending. We want descending, so we sort -mask_cat.
    sort_idx = mx.argsort(-mask_cat, axis=1)  # [B, L]

    # Reorder hidden states using sorted indices
    # mx.take_along_axis is equivalent to torch.gather
    sort_idx_expand = mx.broadcast_to(sort_idx[..., None], (B, L, D))
    hidden_left = mx.take_along_axis(hidden_cat, sort_idx_expand, axis=1)

    # Create new mask based on valid sequence lengths
    lengths = mask_cat.sum(axis=1)  # [B]
    positions = mx.arange(L)[None, :]
    new_mask = positions < lengths[:, None]

    return hidden_left, new_mask.astype(mx.int32)


class MLXAceStepConditionEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.text_projector = nn.Linear(
            config.text_hidden_dim, config.hidden_size, bias=False
        )
        self.lyric_encoder = MLXAceStepLyricEncoder(config)
        # Note: Timbre Encoder is omitted for pure Text-to-Audio (we supply dummy vectors)

    def __call__(
        self,
        text_hidden_states: mx.array,
        text_attention_mask: mx.array,
        lyric_hidden_states: mx.array,
        lyric_attention_mask: mx.array,
    ) -> Tuple[mx.array, mx.array]:

        text_hidden_states = self.text_projector(text_hidden_states)

        lyric_hidden_states = self.lyric_encoder(
            inputs_embeds=lyric_hidden_states,
            attention_mask=lyric_attention_mask,
        )

        # We don't have timbre embs in pure TTA, so we just pack lyric and text
        encoder_hidden_states, encoder_attention_mask = pack_sequences(
            lyric_hidden_states,
            text_hidden_states,
            lyric_attention_mask,
            text_attention_mask,
        )
        return encoder_hidden_states, encoder_attention_mask
