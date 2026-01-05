# Copyright (c) 2025 Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


from .config import FunAudioChatAudioEncoderConfig


class FunAudioChatDiscreteEncoder(nn.Module):
    """Discrete audio encoder that processes quantized audio tokens.

    This encoder:
    1. Embeds discrete audio tokens using a codebook
    2. Groups tokens and applies mean pooling
    3. Projects through output_matching
    4. Optionally fuses continuous audio features (through continual_output_matching)
    """

    def __init__(self, config: FunAudioChatAudioEncoderConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.embed_tokens = nn.Embedding(config.codebook_size, config.output_dim)

        # Output projections (matching HuggingFace)
        self.output_matching = nn.Linear(config.output_dim, config.output_dim, bias=False)
        self.continual_output_matching = nn.Linear(config.output_dim, config.output_dim, bias=False)

        # Group size for temporal pooling (5Hz representation from 25Hz)
        self.group_size = config.group_size

        # Mode for fusing continuous features: "add" or "replace"
        self.continuous_features_mode = config.continuous_features_mode

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        continuous_features: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
    ) -> Tuple[mx.array, Optional[mx.array]]:
        """
        Encode discrete audio tokens and/or continuous features.

        Args:
            input_ids: Optional discrete audio token IDs (batch, seq_len)
            continuous_features: Optional continuous audio features from audio encoder
            attention_mask: Optional attention mask

        Returns:
            Tuple of (encoded features, updated attention mask)
        """
        hidden_states = None

        # Process discrete tokens if provided
        if input_ids is not None:
            # Embed discrete tokens
            inputs_embeds = self.embed_tokens(input_ids)  # (batch, seq_len, output_dim)

            batch_size, seq_len, hidden_dim = inputs_embeds.shape

            # Group tokens and apply mean pooling
            # This reduces from 25Hz to 5Hz representation
            if self.group_size > 1:
                new_seq_len = seq_len // self.group_size
                truncated_len = new_seq_len * self.group_size
                inputs_embeds = inputs_embeds[:, :truncated_len, :]
                inputs_embeds = inputs_embeds.reshape(batch_size, new_seq_len, self.group_size, hidden_dim)
                inputs_embeds_mean = mx.mean(inputs_embeds, axis=2)

                # Update attention mask if provided
                if attention_mask is not None:
                    attention_mask = attention_mask[:, :truncated_len]
                    attention_mask = attention_mask.reshape(batch_size, new_seq_len, self.group_size)
                    attention_mask = mx.min(attention_mask, axis=2)
            else:
                inputs_embeds_mean = inputs_embeds

            # Apply output_matching projection to discrete embeddings
            hidden_states = self.output_matching(inputs_embeds_mean)

        # Process and fuse continuous features if provided
        if continuous_features is not None:
            batch_size = continuous_features.shape[0]
            hidden_dim = continuous_features.shape[-1]

            # Group continuous features and apply mean pooling (like discrete)
            if self.group_size > 1:
                cont_seq_len = continuous_features.shape[1]
                new_cont_seq_len = cont_seq_len // self.group_size
                truncated_len = new_cont_seq_len * self.group_size
                continuous_features = continuous_features[:, :truncated_len, :]
                continuous_features = continuous_features.reshape(batch_size, new_cont_seq_len, self.group_size, hidden_dim)
                continuous_features = mx.mean(continuous_features, axis=2)

            # Apply continual_output_matching projection to continuous features
            continuous_hidden_states = self.continual_output_matching(continuous_features)

            if hidden_states is not None:
                # Ensure shapes match
                cont_seq_len = continuous_hidden_states.shape[1]
                disc_seq_len = hidden_states.shape[1]

                if cont_seq_len != disc_seq_len:
                    if cont_seq_len < disc_seq_len:
                        padding = mx.zeros((batch_size, disc_seq_len - cont_seq_len, hidden_dim))
                        continuous_hidden_states = mx.concatenate([continuous_hidden_states, padding], axis=1)
                    else:
                        continuous_hidden_states = continuous_hidden_states[:, :disc_seq_len, :]

                # Fuse continuous with discrete
                if self.continuous_features_mode == "add":
                    hidden_states = hidden_states + continuous_hidden_states
                elif self.continuous_features_mode == "replace":
                    hidden_states = continuous_hidden_states
                else:
                    raise ValueError(f"Unknown continuous_features_mode: {self.continuous_features_mode}")
            else:
                # No discrete tokens, use continuous only
                hidden_states = continuous_hidden_states

        if hidden_states is None:
            raise ValueError("Either input_ids or continuous_features must be provided")

        return hidden_states, attention_mask
