# Copyright 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from tqdm import tqdm

from mlx_audio.stt.models.base import STTOutput

from .config import AudioEncoderConfig, ModelConfig, TextConfig


@dataclass
class StreamingResult:
    """Result object for streaming transcription.

    Attributes:
        text: Decoded text for this emission.
        is_final: True if this is a final (committed) result, False if partial.
        start_time: Start timestamp in seconds.
        end_time: End timestamp in seconds.
        language: Language of the transcription.
    """

    text: str
    is_final: bool
    start_time: float
    end_time: float
    language: str = "en"


def split_audio_into_chunks(
    wav: np.ndarray,
    sr: int,
    max_chunk_sec: float = 1200.0,
    min_chunk_sec: float = 1.0,
    search_expand_sec: float = 5.0,
    min_window_ms: float = 100.0,
) -> List[Tuple[np.ndarray, float]]:
    """Split long audio into chunks at low-energy boundaries.

    Args:
        wav: Audio waveform (1D numpy array).
        sr: Sample rate.
        max_chunk_sec: Maximum chunk duration in seconds (default: 1200 = 20 min).
        min_chunk_sec: Minimum chunk duration in seconds (default: 1.0).
        search_expand_sec: Window to search for silence around cut point.
        min_window_ms: Minimum window size for energy calculation.

    Returns:
        List of (chunk_waveform, offset_seconds) tuples.
    """
    # Ensure mono
    if wav.ndim > 1:
        wav = wav.mean(axis=-1) if wav.shape[-1] <= 2 else wav.mean(axis=0)

    total_samples = len(wav)
    total_sec = total_samples / sr

    # If short enough, return as-is
    if total_sec <= max_chunk_sec:
        # Pad if too short
        if total_sec < min_chunk_sec:
            min_samples = int(min_chunk_sec * sr)
            wav = np.pad(wav, (0, min_samples - len(wav)))
        return [(wav, 0.0)]

    chunks = []
    start_sample = 0
    max_chunk_samples = int(max_chunk_sec * sr)
    search_samples = int(search_expand_sec * sr)
    min_window_samples = int(min_window_ms * sr / 1000)

    while start_sample < total_samples:
        end_sample = min(start_sample + max_chunk_samples, total_samples)

        # If this is the last chunk, take the rest
        if end_sample >= total_samples:
            chunk = wav[start_sample:total_samples]
            offset_sec = start_sample / sr
            # Pad if too short
            if len(chunk) < min_chunk_sec * sr:
                min_samples = int(min_chunk_sec * sr)
                chunk = np.pad(chunk, (0, min_samples - len(chunk)))
            chunks.append((chunk, offset_sec))
            break

        # Search for low-energy point around the cut
        search_start = max(start_sample, end_sample - search_samples)
        search_end = min(total_samples, end_sample + search_samples)
        search_region = wav[search_start:search_end]

        # Calculate energy using sliding window
        if len(search_region) > min_window_samples:
            energy = np.convolve(
                search_region**2,
                np.ones(min_window_samples) / min_window_samples,
                mode="valid",
            )
            # Find minimum energy point
            min_idx = np.argmin(energy) + min_window_samples // 2
            cut_sample = search_start + min_idx
        else:
            cut_sample = end_sample

        # Ensure we make progress
        cut_sample = max(cut_sample, start_sample + sr)  # At least 1 second

        chunk = wav[start_sample:cut_sample]
        offset_sec = start_sample / sr

        # Pad if too short
        if len(chunk) < min_chunk_sec * sr:
            min_samples = int(min_chunk_sec * sr)
            chunk = np.pad(chunk, (0, min_samples - len(chunk)))

        chunks.append((chunk, offset_sec))
        start_sample = cut_sample

    return chunks


def create_additive_causal_mask(N: int, offset: int = 0) -> mx.array:
    """Create an additive causal attention mask."""
    rinds = mx.arange(offset + N)
    linds = mx.arange(offset, offset + N) if offset else rinds
    mask = linds[:, None] < rinds[None]
    return mask * -1e9


def _floor_div(a: mx.array, b: int) -> mx.array:
    """Floor division matching Python semantics."""
    return mx.floor(a.astype(mx.float32) / b).astype(mx.int32)


def _get_feat_extract_output_lengths(input_lengths: mx.array) -> mx.array:
    """Compute output length of the convolutional layers."""
    input_lengths_leave = input_lengths % 100
    feat_lengths = _floor_div(input_lengths_leave - 1, 2) + 1
    output_lengths = (
        _floor_div(_floor_div(feat_lengths - 1, 2) + 1 - 1, 2)
        + 1
        + (input_lengths // 100) * 13
    )
    return output_lengths


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal position embeddings for the audio encoder."""

    def __init__(self, length: int, channels: int, max_timescale: float = 10000.0):
        super().__init__()
        if channels % 2 != 0:
            raise ValueError("SinusoidalPositionEmbedding needs even channels input")

        log_timescale_increment = math.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = mx.exp(
            -log_timescale_increment * mx.arange(channels // 2, dtype=mx.float32)
        )
        positions = mx.arange(length, dtype=mx.float32)[:, None]
        scaled_time = positions * inv_timescales[None, :]
        self._positional_embedding = mx.concatenate(
            [mx.sin(scaled_time), mx.cos(scaled_time)], axis=1
        )

    def __call__(self, seqlen: int) -> mx.array:
        return self._positional_embedding[:seqlen, :]


class AudioAttention(nn.Module):
    """Multi-headed attention for audio encoder."""

    def __init__(self, config: AudioEncoderConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = config.encoder_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = self.head_dim**-0.5

        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def __call__(
        self,
        hidden_states: mx.array,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        bsz, seq_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states) * self.scaling
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.reshape(
            bsz, seq_len, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        key_states = key_states.reshape(
            bsz, seq_len, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        value_states = value_states.reshape(
            bsz, seq_len, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)

        attn_output = mx.fast.scaled_dot_product_attention(
            query_states, key_states, value_states, scale=1.0, mask=mask
        )

        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(
            bsz, seq_len, self.embed_dim
        )
        return self.out_proj(attn_output)


class AudioEncoderLayer(nn.Module):
    """A single transformer encoder layer for audio."""

    def __init__(self, config: AudioEncoderConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = AudioAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def __call__(
        self,
        hidden_states: mx.array,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states, mask=mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = nn.gelu(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class AudioEncoder(nn.Module):
    """Qwen3-ASR Audio Encoder with Conv2d frontend and transformer layers."""

    def __init__(self, config: AudioEncoderConfig):
        super().__init__()
        self.config = config
        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.n_window = config.n_window
        self.n_window_infer = config.n_window_infer
        self.conv_chunksize = config.conv_chunksize

        self.conv2d1 = nn.Conv2d(
            1, config.downsample_hidden_size, kernel_size=3, stride=2, padding=1
        )
        self.conv2d2 = nn.Conv2d(
            config.downsample_hidden_size,
            config.downsample_hidden_size,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.conv2d3 = nn.Conv2d(
            config.downsample_hidden_size,
            config.downsample_hidden_size,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        freq_after_conv = ((((config.num_mel_bins + 1) // 2) + 1) // 2 + 1) // 2
        self.conv_out = nn.Linear(
            config.downsample_hidden_size * freq_after_conv, embed_dim, bias=False
        )
        self.positional_embedding = SinusoidalPositionEmbedding(
            self.max_source_positions, embed_dim
        )
        self.layers = [AudioEncoderLayer(config) for _ in range(config.encoder_layers)]
        self.ln_post = nn.LayerNorm(embed_dim)
        self.proj1 = nn.Linear(embed_dim, embed_dim)
        self.proj2 = nn.Linear(embed_dim, config.output_dim)

    def _create_block_attention_mask(
        self, seq_len: int, cu_seqlens: List[int], dtype: mx.Dtype
    ) -> mx.array:
        """Create attention mask for ragged/block attention."""
        mask = mx.full((seq_len, seq_len), -1e9, dtype=dtype)
        for i in range(len(cu_seqlens) - 1):
            start = cu_seqlens[i]
            end = cu_seqlens[i + 1]
            mask[start:end, start:end] = 0.0
        return mask

    def __call__(
        self,
        input_features: mx.array,
        feature_attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        if feature_attention_mask is not None:
            feature_lens = feature_attention_mask.sum(axis=-1).astype(mx.int32)
        else:
            feature_lens = mx.array(
                [input_features.shape[-1]] * input_features.shape[0], dtype=mx.int32
            )

        feature_lens_np = np.array(feature_lens)
        aftercnn_lens = _get_feat_extract_output_lengths(feature_lens)
        chunk_size = self.n_window * 2
        chunk_num = np.ceil(feature_lens_np / chunk_size).astype(np.int32)

        chunk_lengths = []
        for i in range(len(feature_lens_np)):
            num_chunks = int(chunk_num[i])
            feat_len = int(feature_lens_np[i])
            for j in range(num_chunks):
                if j == num_chunks - 1:
                    remainder = feat_len % chunk_size
                    chunk_lengths.append(chunk_size if remainder == 0 else remainder)
                else:
                    chunk_lengths.append(chunk_size)

        chunk_lengths = np.array(chunk_lengths, dtype=np.int32)

        chunks = []
        for i in range(len(feature_lens_np)):
            feat = input_features[i]
            feat_len = int(feature_lens_np[i])
            num_chunks = int(chunk_num[i])
            pos = 0
            for j in range(num_chunks):
                if j == num_chunks - 1:
                    remainder = feat_len % chunk_size
                    clen = chunk_size if remainder == 0 else remainder
                else:
                    clen = chunk_size
                chunk = feat[:, pos : pos + clen]
                chunks.append(chunk)
                pos += clen

        max_chunk_len = int(max(chunk_lengths))
        padded_chunks = []
        for i, chunk in enumerate(chunks):
            clen = int(chunk_lengths[i])
            if clen < max_chunk_len:
                pad_width = max_chunk_len - clen
                chunk = mx.pad(chunk, [(0, 0), (0, pad_width)])
            padded_chunks.append(chunk)

        padded_feature = mx.stack(padded_chunks, axis=0)

        feature_lens_after_cnn = _get_feat_extract_output_lengths(
            mx.array(chunk_lengths)
        )
        feature_lens_after_cnn_np = np.array(feature_lens_after_cnn)
        max_len_after_cnn = int(feature_lens_after_cnn_np.max())

        padded_mask_after_cnn = np.zeros(
            (len(chunk_lengths), max_len_after_cnn), dtype=bool
        )
        for i, length in enumerate(feature_lens_after_cnn_np):
            padded_mask_after_cnn[i, : int(length)] = True

        x = padded_feature[:, :, :, None]
        x = nn.gelu(self.conv2d1(x))
        x = nn.gelu(self.conv2d2(x))
        x = nn.gelu(self.conv2d3(x))

        b, f, t, c = x.shape
        x = x.transpose(0, 2, 3, 1).reshape(b, t, c * f)
        x = self.conv_out(x)

        pos_emb = self.positional_embedding(x.shape[1])
        x = x + pos_emb[None, :, :]

        hidden_list = []
        for i in range(x.shape[0]):
            valid_len = int(feature_lens_after_cnn_np[i])
            hidden_list.append(x[i, :valid_len])

        hidden_states = mx.concatenate(hidden_list, axis=0)

        aftercnn_lens_np = np.array(aftercnn_lens)
        window_aftercnn = max_len_after_cnn * (
            self.n_window_infer // (self.n_window * 2)
        )

        cu_chunk_lens = [0]
        for cnn_len in aftercnn_lens_np:
            cnn_len = int(cnn_len)
            num_full_windows = cnn_len // window_aftercnn
            for _ in range(num_full_windows):
                cu_chunk_lens.append(window_aftercnn)
            remainder = cnn_len % window_aftercnn
            if remainder != 0:
                cu_chunk_lens.append(remainder)

        cu_seqlens = np.cumsum(cu_chunk_lens).tolist()

        seq_len = hidden_states.shape[0]
        attention_mask = self._create_block_attention_mask(
            seq_len, cu_seqlens, hidden_states.dtype
        )
        attention_mask = attention_mask[None, None, :, :]

        hidden_states = hidden_states[None, :, :]

        for layer in self.layers:
            hidden_states = layer(hidden_states, mask=attention_mask)

        hidden_states = hidden_states[0]
        hidden_states = self.ln_post(hidden_states)
        hidden_states = nn.gelu(self.proj1(hidden_states))
        hidden_states = self.proj2(hidden_states)

        return hidden_states


class TextAttention(nn.Module):
    """Multi-headed attention for text decoder with Q/K norms."""

    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=False
        )

        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=config.rope_theta)

    def __call__(
        self,
        hidden_states: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = hidden_states.shape

        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        queries = queries.reshape(B, L, self.num_heads, self.head_dim)
        keys = keys.reshape(B, L, self.num_kv_heads, self.head_dim)
        values = values.reshape(B, L, self.num_kv_heads, self.head_dim)

        queries = self.q_norm(queries)
        keys = self.k_norm(keys)

        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        if cache is not None:
            offset = cache.offset
            queries = self.rope(queries, offset=offset)
            keys = self.rope(keys, offset=offset)
        else:
            offset = 0
            queries = self.rope(queries)
            keys = self.rope(keys)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        query_len = queries.shape[2]
        mask = create_additive_causal_mask(query_len, offset=offset).astype(
            queries.dtype
        )

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )

        output = output.transpose(0, 2, 1, 3).reshape(B, query_len, -1)
        return self.o_proj(output)


class TextMLP(nn.Module):
    """MLP for text decoder with SwiGLU activation."""

    def __init__(self, config: TextConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TextDecoderLayer(nn.Module):
    """A single transformer decoder layer."""

    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = TextAttention(config, layer_idx)
        self.mlp = TextMLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(
        self,
        hidden_states: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, cache=cache)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class TextModel(nn.Module):
    """Text decoder model (Qwen3-based)."""

    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            TextDecoderLayer(config, i) for i in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        cache: Optional[List[Any]] = None,
    ) -> mx.array:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        if cache is None:
            cache = [None] * len(self.layers)

        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, cache=cache[i])

        return self.norm(hidden_states)


class Qwen3ASRModel(nn.Module):
    """Qwen3-ASR Model for speech recognition."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.text_config.vocab_size
        self.audio_tower = AudioEncoder(config.audio_config)
        self.model = TextModel(config.text_config)

        if config.text_config.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(
                config.text_config.hidden_size,
                config.text_config.vocab_size,
                bias=False,
            )

    def get_audio_features(
        self,
        input_features: mx.array,
        feature_attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """Encode audio features."""
        return self.audio_tower(input_features, feature_attention_mask)

    def _build_inputs_embeds(
        self,
        input_ids: mx.array,
        audio_features: mx.array,
    ) -> mx.array:
        """Build input embeddings with audio features merged in."""
        inputs_embeds = self.model.embed_tokens(input_ids)
        audio_features = audio_features.astype(inputs_embeds.dtype)

        audio_token_mask = input_ids == self.config.audio_token_id

        if audio_token_mask.any():
            batch_size, seq_len, hidden_dim = inputs_embeds.shape
            flat_mask = audio_token_mask.flatten()
            flat_mask_np = np.array(flat_mask)
            audio_indices = np.where(flat_mask_np)[0]

            if len(audio_indices) > 0 and audio_features.shape[0] > 0:
                num_to_replace = min(len(audio_indices), audio_features.shape[0])
                flat_embeds = inputs_embeds.reshape(-1, hidden_dim)

                result_list = []
                audio_idx = 0
                for i in range(flat_embeds.shape[0]):
                    if audio_idx < num_to_replace and i == audio_indices[audio_idx]:
                        result_list.append(audio_features[audio_idx])
                        audio_idx += 1
                    else:
                        result_list.append(flat_embeds[i])

                inputs_embeds = mx.stack(result_list, axis=0).reshape(
                    batch_size, seq_len, hidden_dim
                )

        return inputs_embeds

    def _forward_with_embeds(
        self,
        inputs_embeds: mx.array,
        cache: Optional[List[Any]] = None,
    ) -> mx.array:
        """Forward pass using pre-computed embeddings."""
        hidden_states = self.model(inputs_embeds=inputs_embeds, cache=cache)

        if self.lm_head is not None:
            logits = self.lm_head(hidden_states)
        else:
            logits = self.model.embed_tokens.as_linear(hidden_states)

        return logits

    def __call__(
        self,
        input_ids: mx.array,
        input_embeddings: Optional[mx.array] = None,
        input_features: Optional[mx.array] = None,
        feature_attention_mask: Optional[mx.array] = None,
        cache: Optional[List[Any]] = None,
    ) -> mx.array:
        if input_embeddings is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
        else:
            inputs_embeds = input_embeddings

        if input_features is not None and (
            cache is None or cache[0] is None or cache[0].offset == 0
        ):
            audio_features = self.get_audio_features(
                input_features, feature_attention_mask
            )
            audio_features = audio_features.astype(inputs_embeds.dtype)

            audio_token_mask = input_ids == self.config.audio_token_id

            if audio_token_mask.any():
                batch_size, seq_len, hidden_dim = inputs_embeds.shape

                flat_mask = audio_token_mask.flatten()
                flat_mask_np = np.array(flat_mask)
                audio_indices = np.where(flat_mask_np)[0]

                if len(audio_indices) > 0 and audio_features.shape[0] > 0:
                    num_to_replace = min(len(audio_indices), audio_features.shape[0])
                    flat_embeds = inputs_embeds.reshape(-1, hidden_dim)

                    result_list = []
                    audio_idx = 0
                    for i in range(flat_embeds.shape[0]):
                        if audio_idx < num_to_replace and i == audio_indices[audio_idx]:
                            result_list.append(audio_features[audio_idx])
                            audio_idx += 1
                        else:
                            result_list.append(flat_embeds[i])

                    inputs_embeds = mx.stack(result_list, axis=0).reshape(
                        batch_size, seq_len, hidden_dim
                    )

        hidden_states = self.model(inputs_embeds=inputs_embeds, cache=cache)

        if self.lm_head is not None:
            logits = self.lm_head(hidden_states)
        else:
            logits = self.model.embed_tokens.as_linear(hidden_states)

        return logits

    @property
    def layers(self):
        return self.model.layers

    @property
    def sample_rate(self) -> int:
        """Sample rate for audio input."""
        return 16000

    def make_cache(self) -> List[Any]:
        """Create KV cache for generation."""
        from mlx_lm.models.cache import KVCache

        return [KVCache() for _ in range(self.config.text_config.num_hidden_layers)]

    @staticmethod
    def sanitize(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Sanitize weights from HuggingFace/PyTorch format to MLX format."""
        sanitized = {}
        is_formatted = not any(k.startswith("thinker.") for k in weights.keys())

        for k, v in weights.items():
            if k.startswith("thinker."):
                k = k[len("thinker.") :]

            if k == "lm_head.weight":
                continue

            if (
                not is_formatted
                and "conv2d" in k
                and "weight" in k
                and len(v.shape) == 4
            ):
                v = v.transpose(0, 2, 3, 1)

            sanitized[k] = v

        return sanitized

    def model_quant_predicate(self, p: str, m: nn.Module) -> bool:
        """Determine which layers to quantize."""
        return not p.startswith("audio_tower")

    @classmethod
    def post_load_hook(
        cls, model: "Qwen3ASRModel", model_path: Path
    ) -> "Qwen3ASRModel":
        """Hook called after model weights are loaded."""
        import transformers
        from transformers import AutoTokenizer, WhisperFeatureExtractor

        # Suppress the harmless warning about model_type mismatch when loading
        # tokenizer for custom model types not registered in transformers
        # (warning is emitted via logging, not warnings module)
        prev_verbosity = transformers.logging.get_verbosity()
        transformers.logging.set_verbosity_error()
        try:
            model._tokenizer = AutoTokenizer.from_pretrained(
                str(model_path), trust_remote_code=True
            )
            model._feature_extractor = WhisperFeatureExtractor.from_pretrained(
                str(model_path)
            )
        finally:
            transformers.logging.set_verbosity(prev_verbosity)

        if not hasattr(model.config, "model_repo") or model.config.model_repo is None:
            try:
                index = model_path.parts.index("hub")
                model.config.model_repo = (
                    model_path.parts[index + 1]
                    .replace("models--", "")
                    .replace("--", "/")
                )
            except (ValueError, IndexError):
                model.config.model_repo = str(model_path)

        return model

    def _preprocess_audio(
        self,
        audio: Union[str, mx.array, np.ndarray, List[Union[str, mx.array, np.ndarray]]],
    ) -> Tuple[mx.array, mx.array, int]:
        """Preprocess audio for the model."""
        from mlx_audio.stt.utils import load_audio

        audio_input = audio[0] if isinstance(audio, list) else audio

        if isinstance(audio_input, str):
            audio_input = load_audio(audio_input)

        audio_np = (
            np.array(audio_input) if isinstance(audio_input, mx.array) else audio_input
        )

        audio_inputs = self._feature_extractor(
            audio_np,
            sampling_rate=16000,
            return_attention_mask=True,
            truncation=False,
            padding=True,
            return_tensors="np",
        )
        input_features = mx.array(audio_inputs["input_features"])
        feature_attention_mask = mx.array(audio_inputs["attention_mask"])

        audio_lengths = feature_attention_mask.sum(axis=-1)
        aftercnn_lens = _get_feat_extract_output_lengths(audio_lengths)
        num_audio_tokens = int(aftercnn_lens[0].item())

        return input_features, feature_attention_mask, num_audio_tokens

    def _build_prompt(
        self, num_audio_tokens: int, language: str = "English"
    ) -> mx.array:
        """Build prompt with audio tokens."""
        supported = self.config.support_languages or []
        supported_lower = {lang.lower(): lang for lang in supported}

        # Match language (case-insensitive) against supported languages
        lang_name = supported_lower.get(language.lower(), language)

        prompt = (
            f"<|im_start|>system\n<|im_end|>\n"
            f"<|im_start|>user\n<|audio_start|>{'<|audio_pad|>' * num_audio_tokens}<|audio_end|><|im_end|>\n"
            f"<|im_start|>assistant\nlanguage {lang_name}<asr_text>"
        )

        input_ids = self._tokenizer.encode(prompt, return_tensors="np")
        return mx.array(input_ids)

    def stream_generate(
        self,
        audio: Union[str, mx.array, np.ndarray, List[Union[str, mx.array, np.ndarray]]],
        *,
        max_tokens: int = 8192,
        sampler: Optional[Callable[[mx.array], mx.array]] = None,
        logits_processors: Optional[List[Callable]] = None,
        language: str = "English",
        prefill_step_size: int = 2048,
        verbose: bool = False,
    ) -> Generator[Tuple[mx.array, mx.array], None, None]:
        """Stream generate tokens from audio using mlx_lm generate_step."""
        from mlx_lm.generate import generate_step

        if not hasattr(self, "_tokenizer") or not hasattr(self, "_feature_extractor"):
            raise RuntimeError(
                "Tokenizer/FeatureExtractor not initialized. Call post_load_hook first."
            )

        input_features, feature_attention_mask, num_audio_tokens = (
            self._preprocess_audio(audio)
        )
        input_ids = self._build_prompt(num_audio_tokens, language)
        eos_token_ids = [151645, 151643]

        # Step 1: Encode audio features
        with tqdm(
            total=1, desc="Encoding audio", disable=not verbose, leave=False
        ) as pbar:
            audio_features = self.get_audio_features(
                input_features, feature_attention_mask
            )
            mx.eval(audio_features)
            pbar.update(1)

        # Free input features
        del input_features, feature_attention_mask
        mx.clear_cache()

        # Step 2: Build input embeddings with audio merged
        with tqdm(
            total=1, desc="Building embeddings", disable=not verbose, leave=False
        ) as pbar:
            inputs_embeds = self._build_inputs_embeds(input_ids, audio_features)
            mx.eval(inputs_embeds)
            pbar.update(1)

        # Free audio features
        del audio_features
        mx.clear_cache()

        # Remove batch dim for generate_step
        input_embeddings = inputs_embeds[0]
        prompt = input_ids[0] if input_ids.ndim > 1 else input_ids

        # Free the batched version
        del inputs_embeds
        mx.clear_cache()

        # Progress bars
        prefill_pbar = None
        gen_pbar = None

        if verbose:
            prefill_pbar = tqdm(total=1, desc="Prefilling", unit="tok", leave=False)

        def prefill_progress(processed: int, total: int):
            nonlocal gen_pbar
            if prefill_pbar is not None:
                if prefill_pbar.total != total:
                    prefill_pbar.total = total
                    prefill_pbar.refresh()
                prefill_pbar.n = processed
                prefill_pbar.refresh()
                if processed >= total:
                    prefill_pbar.close()
                    if gen_pbar is None and verbose:
                        gen_pbar = tqdm(
                            total=max_tokens, desc="Generating", unit="tok", leave=False
                        )

        # Use generate_step from mlx_lm (handles chunked prefill automatically)
        for token, logprobs in generate_step(
            prompt=prompt,
            input_embeddings=input_embeddings,
            model=self,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
            prefill_step_size=prefill_step_size,
            prompt_progress_callback=prefill_progress if verbose else None,
        ):
            if gen_pbar is not None:
                gen_pbar.update(1)

            if token in eos_token_ids:
                break

            yield token, logprobs

        if gen_pbar is not None:
            gen_pbar.close()

    def _generate_single_chunk(
        self,
        audio_chunk: np.ndarray,
        *,
        max_tokens: int = 8192,
        sampler: Optional[Callable] = None,
        logits_processors: Optional[List[Callable]] = None,
        language: str = "English",
        prefill_step_size: int = 2048,
        verbose: bool = False,
    ) -> Tuple[str, int, int]:
        """Generate transcription for a single audio chunk.

        Returns:
            Tuple of (text, prompt_tokens, generation_tokens).
        """
        generated_tokens = []
        prompt_tokens = 0

        for token, _ in self.stream_generate(
            audio_chunk,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
            language=language,
            prefill_step_size=prefill_step_size,
            verbose=verbose,
        ):
            if prompt_tokens == 0:
                # Get prompt tokens from first iteration
                _, _, num_audio_tokens = self._preprocess_audio(audio_chunk)
                input_ids = self._build_prompt(num_audio_tokens, language)
                prompt_tokens = input_ids.shape[1]
            generated_tokens.append(int(token))

        text = self._tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return text, prompt_tokens, len(generated_tokens)

    def generate(
        self,
        audio: Union[str, mx.array, np.ndarray, List[Union[str, mx.array, np.ndarray]]],
        *,
        max_tokens: int = 8192,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        min_tokens_to_keep: int = 1,
        repetition_penalty: Optional[float] = None,
        repetition_context_size: int = 100,
        language: str = "English",
        prefill_step_size: int = 2048,
        max_chunk_sec: float = 1200.0,
        min_chunk_sec: float = 1.0,
        verbose: bool = False,
        stream: bool = False,
        **kwargs,
    ) -> Union[STTOutput, Generator[str, None, None]]:
        """Generate transcription from audio.

        Automatically chunks long audio and processes sequentially.

        Args:
            max_chunk_sec: Maximum chunk duration in seconds (default: 1200 = 20 min).
            min_chunk_sec: Minimum chunk duration in seconds (default: 1.0).
            stream: If True, return a generator that yields tokens as they are generated.
        """
        # If streaming requested, delegate to stream_transcribe
        if stream:
            return self.stream_transcribe(
                audio,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                min_tokens_to_keep=min_tokens_to_keep,
                repetition_penalty=repetition_penalty,
                repetition_context_size=repetition_context_size,
                language=language,
                prefill_step_size=prefill_step_size,
                max_chunk_sec=max_chunk_sec,
                min_chunk_sec=min_chunk_sec,
                verbose=verbose,
            )

        from mlx_lm.sample_utils import make_logits_processors, make_sampler

        from mlx_audio.stt.utils import load_audio

        del kwargs

        start_time = time.time()

        if not hasattr(self, "_tokenizer") or not hasattr(self, "_feature_extractor"):
            raise RuntimeError(
                "Tokenizer/FeatureExtractor not initialized. Call post_load_hook first."
            )

        # Load audio
        audio_input = audio[0] if isinstance(audio, list) else audio
        if isinstance(audio_input, str):
            audio_input = load_audio(audio_input)
        audio_np = (
            np.array(audio_input) if isinstance(audio_input, mx.array) else audio_input
        )

        # Split into chunks if needed
        chunks = split_audio_into_chunks(
            audio_np,
            sr=self.sample_rate,
            max_chunk_sec=max_chunk_sec,
            min_chunk_sec=min_chunk_sec,
        )

        sampler = make_sampler(
            temperature,
            top_p,
            min_p,
            min_tokens_to_keep=min_tokens_to_keep,
            top_k=top_k,
        )

        logits_processors = (
            make_logits_processors(
                repetition_penalty=repetition_penalty,
                repetition_context_size=repetition_context_size,
            )
            if repetition_penalty
            else None
        )

        # Process chunks
        all_texts = []
        segments = []
        total_prompt_tokens = 0
        total_generation_tokens = 0

        chunk_iter = tqdm(
            chunks, desc="Processing chunks", disable=not verbose or len(chunks) == 1
        )
        for chunk_audio, offset_sec in chunk_iter:
            chunk_duration = len(chunk_audio) / self.sample_rate

            text, prompt_toks, gen_toks = self._generate_single_chunk(
                chunk_audio,
                max_tokens=max_tokens,
                sampler=sampler,
                logits_processors=logits_processors,
                language=language,
                prefill_step_size=prefill_step_size,
                verbose=verbose
                and len(chunks) == 1,  # Only show inner progress for single chunk
            )
            all_texts.append(text)
            total_prompt_tokens += prompt_toks
            total_generation_tokens += gen_toks

            # Create segment for this chunk
            segments.append(
                {
                    "text": text,
                    "start": offset_sec,
                    "end": offset_sec + chunk_duration,
                }
            )

            # Clear cache between chunks
            mx.clear_cache()

        end_time = time.time()

        # Combine transcriptions
        full_text = " ".join(all_texts)

        return STTOutput(
            text=full_text,
            segments=segments,
            prompt_tokens=total_prompt_tokens,
            generation_tokens=total_generation_tokens,
            total_tokens=total_prompt_tokens + total_generation_tokens,
            total_time=end_time - start_time,
            prompt_tps=(
                total_prompt_tokens / (end_time - start_time)
                if end_time > start_time
                else 0
            ),
            generation_tps=(
                total_generation_tokens / (end_time - start_time)
                if end_time > start_time
                else 0
            ),
        )

    def stream_transcribe(
        self,
        audio: Union[str, mx.array, np.ndarray, List[Union[str, mx.array, np.ndarray]]],
        *,
        max_tokens: int = 8192,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        min_tokens_to_keep: int = 1,
        repetition_penalty: Optional[float] = None,
        repetition_context_size: int = 100,
        language: str = "English",
        prefill_step_size: int = 2048,
        max_chunk_sec: float = 1200.0,
        min_chunk_sec: float = 1.0,
        verbose: bool = False,
    ) -> Generator[StreamingResult, None, None]:
        """Stream transcription token-by-token from audio.

        Automatically chunks long audio and streams tokens from each chunk sequentially.

        Yields:
            StreamingResult objects with text, timing, and status information.
        """
        from mlx_lm.sample_utils import make_logits_processors, make_sampler

        from mlx_audio.stt.utils import load_audio

        if not hasattr(self, "_tokenizer") or not hasattr(self, "_feature_extractor"):
            raise RuntimeError(
                "Tokenizer/FeatureExtractor not initialized. Call post_load_hook first."
            )

        # Load audio
        audio_input = audio[0] if isinstance(audio, list) else audio
        if isinstance(audio_input, str):
            audio_input = load_audio(audio_input)
        audio_np = (
            np.array(audio_input) if isinstance(audio_input, mx.array) else audio_input
        )

        # Calculate total audio duration
        total_duration = len(audio_np) / self.sample_rate

        # Split into chunks if needed
        chunks = split_audio_into_chunks(
            audio_np,
            sr=self.sample_rate,
            max_chunk_sec=max_chunk_sec,
            min_chunk_sec=min_chunk_sec,
        )

        sampler = make_sampler(
            temperature,
            top_p,
            min_p,
            min_tokens_to_keep=min_tokens_to_keep,
            top_k=top_k,
        )

        logits_processors = (
            make_logits_processors(
                repetition_penalty=repetition_penalty,
                repetition_context_size=repetition_context_size,
            )
            if repetition_penalty
            else None
        )

        # Normalize language code for output
        lang_code = language[:2].lower() if language else "en"

        # Process each chunk and stream tokens
        for chunk_idx, (chunk_audio, offset_sec) in enumerate(chunks):
            chunk_duration = len(chunk_audio) / self.sample_rate
            is_last_chunk = chunk_idx == len(chunks) - 1
            token_count = 0

            for token, _ in self.stream_generate(
                chunk_audio,
                max_tokens=max_tokens,
                sampler=sampler,
                logits_processors=logits_processors,
                language=language,
                prefill_step_size=prefill_step_size,
                verbose=verbose and len(chunks) == 1,
            ):
                text = self._tokenizer.decode([int(token)])

                # Estimate timing based on token position within chunk
                # This is approximate since we don't have word-level alignment
                prev_progress = token_count / max(max_tokens, 1)
                token_count += 1
                curr_progress = min(token_count / max(max_tokens, 1), 1.0)

                estimated_start = offset_sec + (chunk_duration * prev_progress)
                estimated_end = offset_sec + (chunk_duration * curr_progress)

                yield StreamingResult(
                    text=text,
                    is_final=False,
                    start_time=estimated_start,
                    end_time=estimated_end,
                    language=lang_code,
                )

            # Yield final result for this chunk
            yield StreamingResult(
                text="",
                is_final=is_last_chunk,
                start_time=offset_sec,
                end_time=offset_sec + chunk_duration,
                language=lang_code,
            )

            # Clear cache between chunks
            mx.clear_cache()


class Model:

    _FORCED_ALIGNER_TYPE = "qwen3_forced_aligner"
    _FORCED_ALIGNER_MAX_CLASSES = 10000

    def __init__(self, config):
        from .qwen3_forced_aligner import ForcedAlignerConfig, ForcedAlignerModel

        is_aligner = (
            isinstance(config, ForcedAlignerConfig)
            or getattr(config, "model_type", "") == self._FORCED_ALIGNER_TYPE
        )

        self._model = (
            ForcedAlignerModel(config) if is_aligner else Qwen3ASRModel(config)
        )
        self.config = self._model.config

    def __getattr__(self, name):
        return getattr(self._model, name)

    def __call__(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    def parameters(self):
        return self._model.parameters()

    def load_weights(self, weights, strict=False):
        return self._model.load_weights(weights, strict=strict)

    def eval(self):
        return self._model.eval()

    @classmethod
    def _is_forced_aligner_weights(cls, weights: Dict[str, mx.array]) -> bool:
        """Check if weights are for ForcedAligner based on lm_head output size."""
        for key, value in weights.items():
            if "lm_head" in key and "weight" in key:
                return value.shape[0] < cls._FORCED_ALIGNER_MAX_CLASSES
        return False

    @classmethod
    def sanitize(cls, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        if cls._is_forced_aligner_weights(weights):
            from .qwen3_forced_aligner import ForcedAlignerModel

            return ForcedAlignerModel.sanitize(weights)
        return Qwen3ASRModel.sanitize(weights)

    @classmethod
    def post_load_hook(cls, model: "Model", model_path: Path) -> "Model":
        internal_cls = type(model._model)
        if hasattr(internal_cls, "post_load_hook"):
            model._model = internal_cls.post_load_hook(model._model, model_path)
            model.config = model._model.config
        return model
