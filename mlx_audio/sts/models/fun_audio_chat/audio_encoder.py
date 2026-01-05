# Copyright (c) 2025 Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


from .config import FunAudioChatAudioEncoderConfig

def sinusoids(length: int, channels: int, max_timescale: float = 10000) -> mx.array:
    """Returns sinusoids for positional embedding."""
    assert channels % 2 == 0
    log_timescale_increment = math.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = mx.exp(-log_timescale_increment * mx.arange(channels // 2))
    scaled_time = mx.arange(length)[:, None] * inv_timescales[None, :]
    return mx.concatenate([mx.sin(scaled_time), mx.cos(scaled_time)], axis=1)


class FunAudioChatAudioAttention(nn.Module):
    """Multi-head attention for audio encoder."""

    def __init__(self, config: FunAudioChatAudioEncoderConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = config.encoder_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        batch_size, seq_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape to (batch, num_heads, seq_len, head_dim)
        query_states = query_states.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        query_states = query_states.transpose(0, 2, 1, 3)

        key_states = key_states.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        key_states = key_states.transpose(0, 2, 1, 3)

        value_states = value_states.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        value_states = value_states.transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        attn_output = mx.fast.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            scale=self.scaling,
            mask=attention_mask,
        )

        # Reshape back to (batch, seq_len, embed_dim)
        attn_output = attn_output.transpose(0, 2, 1, 3)
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)

        return self.out_proj(attn_output)


class FunAudioChatAudioEncoderLayer(nn.Module):
    """Single transformer encoder layer for audio processing."""

    def __init__(self, config: FunAudioChatAudioEncoderConfig):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = FunAudioChatAudioAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

        self.activation_fn = nn.gelu

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        # Self-attention with pre-norm
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask=attention_mask)
        hidden_states = residual + hidden_states

        # Feed-forward with pre-norm
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class FunAudioChatAudioEncoder(nn.Module):

    def __init__(self, config: FunAudioChatAudioEncoderConfig):
        super().__init__()
        self.config = config

        # Chunking window size
        # Each chunk is n_window * 2 = 200 mel frames
        self.n_window = 100

        # Convolutional frontend
        self.conv1 = nn.Conv1d(config.num_mel_bins, config.d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(config.d_model, config.d_model, kernel_size=3, stride=2, padding=1)

        # Positional embeddings (sinusoidal)
        self._positional_embedding = sinusoids(config.max_source_positions, config.d_model)

        # Transformer encoder layers
        self.layers = [
            FunAudioChatAudioEncoderLayer(config)
            for _ in range(config.encoder_layers)
        ]

        self.layer_norm = nn.LayerNorm(config.d_model)

        self.proj = nn.Linear(config.d_model, config.output_dim, bias=True)

        self.audio_bos_eos_token = nn.Embedding(2, config.output_dim)

    def _create_block_diagonal_mask(
        self, chunk_lengths_after_cnn: list, total_seq_len: int
    ) -> mx.array:
        """Create block-diagonal attention mask for chunked processing.

        Each chunk can only attend to tokens within itself (not across chunks).

        Args:
            chunk_lengths_after_cnn: List of sequence lengths for each chunk after CNN
            total_seq_len: Total sequence length (sum of chunk_lengths_after_cnn)

        Returns:
            Attention mask of shape (1, 1, total_seq_len, total_seq_len)
            0 = attend, -inf = don't attend
        """
        # Create mask filled with -inf (don't attend)
        mask = mx.full((1, 1, total_seq_len, total_seq_len), float('-inf'))

        # For each chunk, set the corresponding block to 0 (attend)
        start = 0
        for length in chunk_lengths_after_cnn:
            end = start + length
            # Create indices for this block
            block_mask = mx.zeros((1, 1, total_seq_len, total_seq_len))
            rows = mx.arange(total_seq_len)
            cols = mx.arange(total_seq_len)
            row_mask = (rows >= start) & (rows < end)
            col_mask = (cols >= start) & (cols < end)
            block_indicator = row_mask[:, None] & col_mask[None, :]
            block_indicator = block_indicator[None, None, :, :]  # (1, 1, seq, seq)

            # Update mask: where block_indicator is True, set to 0
            mask = mx.where(block_indicator, mx.zeros_like(mask), mask)
            start = end

        return mask

    def __call__(
        self,
        input_features: mx.array,
        attention_mask: Optional[mx.array] = None,
        target_length: Optional[int] = None,
    ) -> mx.array:
        """
        Encode mel-spectrogram features with chunking for long audio.

        Args:
            input_features: Mel-spectrogram (batch, seq_len, num_mel_bins) or (batch, num_mel_bins, seq_len)
            attention_mask: Optional attention mask
            target_length: Optional target output length (for alignment with discrete tokens)

        Returns:
            Encoded audio features (batch, output_seq_len, output_dim)
        """
        # Ensure input is (batch, seq_len, num_mel_bins)
        if input_features.shape[-1] != self.config.num_mel_bins:
            # Input is (batch, num_mel_bins, seq_len), need to transpose
            input_features = input_features.transpose(0, 2, 1)

        batch_size = input_features.shape[0]
        input_seq_len = input_features.shape[1]

        # Chunk size in mel frames
        chunk_size = self.n_window * 2  # 200 frames

        # For short audio (less than chunk_size), process without chunking
        if input_seq_len <= chunk_size:
            # Conv frontend expects (batch, seq_len, num_mel_bins)
            hidden_states = nn.gelu(self.conv1(input_features))
            hidden_states = nn.gelu(self.conv2(hidden_states))

            # Add positional embeddings
            seq_len = hidden_states.shape[1]
            pos_embed = self._positional_embedding[:seq_len]
            if self.config.scale_embedding:
                hidden_states = hidden_states * math.sqrt(self.config.d_model)
            hidden_states = hidden_states + pos_embed

            # Transformer encoder layers
            for layer in self.layers:
                hidden_states = layer(hidden_states, attention_mask=attention_mask)
        else:
            # Long audio: process in chunks with batched transformer
            # Split into chunks of chunk_size (200 frames)
            num_full_chunks = input_seq_len // chunk_size
            remainder = input_seq_len % chunk_size

            # Collect chunk info
            chunk_lengths = []  # Mel frame lengths
            chunks = []

            for i in range(num_full_chunks):
                start_idx = i * chunk_size
                end_idx = start_idx + chunk_size
                chunks.append(input_features[:, start_idx:end_idx, :])
                chunk_lengths.append(chunk_size)

            if remainder > 0:
                start_idx = num_full_chunks * chunk_size
                chunks.append(input_features[:, start_idx:, :])
                chunk_lengths.append(remainder)

            # Process each chunk through conv layers
            conv_outputs = []
            chunk_lengths_after_cnn = []

            for chunk in chunks:
                conv_out = nn.gelu(self.conv1(chunk))
                conv_out = nn.gelu(self.conv2(conv_out))
                conv_outputs.append(conv_out[0])  # Remove batch dim: (seq, d_model)
                chunk_lengths_after_cnn.append(conv_out.shape[1])

            # Concatenate all chunks: (total_seq_len, d_model)
            hidden_states = mx.concatenate(conv_outputs, axis=0)
            total_seq_len = hidden_states.shape[0]

            # Add positional embeddings
            # Each chunk gets pos embeddings from 0 to chunk_len
            pos_embeds = []
            for length in chunk_lengths_after_cnn:
                pos_embeds.append(self._positional_embedding[:length])
            pos_embed = mx.concatenate(pos_embeds, axis=0)

            if self.config.scale_embedding:
                hidden_states = hidden_states * math.sqrt(self.config.d_model)
            hidden_states = hidden_states + pos_embed

            # Add batch dimension: (1, total_seq_len, d_model)
            hidden_states = hidden_states[None, :, :]

            # Create block-diagonal attention mask
            block_mask = self._create_block_diagonal_mask(chunk_lengths_after_cnn, total_seq_len)

            # Transformer encoder layers with block-diagonal attention
            for layer in self.layers:
                hidden_states = layer(hidden_states, attention_mask=block_mask)

        # Average pooling with kernel=2, stride=2 (reduces sequence by 2x)
        hidden_states = hidden_states.transpose(0, 2, 1)
        seq_len = hidden_states.shape[2]
        new_seq_len = seq_len // 2
        hidden_states = hidden_states[:, :, :new_seq_len * 2]
        hidden_states = hidden_states.reshape(batch_size, self.config.d_model, new_seq_len, 2)
        hidden_states = mx.mean(hidden_states, axis=3)
        hidden_states = hidden_states.transpose(0, 2, 1)

        # Final layer norm
        hidden_states = self.layer_norm(hidden_states)

        # Project to output dimension
        hidden_states = self.proj(hidden_states)

        return hidden_states
