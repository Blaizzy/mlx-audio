# Copyright 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from tqdm import tqdm

from mlx_audio.stt.models.base import STTOutput

from .config import AudioEncoderConfig, ModelConfig, TextConfig


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


class RMSNorm(nn.Module):
    """RMS Layer Normalization."""

    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps)


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

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
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
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
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
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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


class Model(nn.Module):
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
        input_features: Optional[mx.array] = None,
        feature_attention_mask: Optional[mx.array] = None,
        cache: Optional[List[Any]] = None,
    ) -> mx.array:
        inputs_embeds = self.model.embed_tokens(input_ids)

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
    def post_load_hook(cls, model: "Model", model_path: Path) -> "Model":
        """Hook called after model weights are loaded."""
        from transformers import AutoTokenizer, WhisperFeatureExtractor

        model._tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), trust_remote_code=True
        )
        model._feature_extractor = WhisperFeatureExtractor.from_pretrained(
            str(model_path)
        )

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
        """Stream generate tokens from audio."""
        if not hasattr(self, "_tokenizer") or not hasattr(self, "_feature_extractor"):
            raise RuntimeError(
                "Tokenizer/FeatureExtractor not initialized. Call post_load_hook first."
            )

        input_features, feature_attention_mask, num_audio_tokens = (
            self._preprocess_audio(audio)
        )
        input_ids = self._build_prompt(num_audio_tokens, language)
        cache = self.make_cache()
        eos_token_ids = [151645, 151643]

        if sampler is None:

            def sampler(logits):
                return mx.argmax(logits, axis=-1)

        # Step 1: Encode audio features
        with tqdm(total=1, desc="Encoding audio", disable=not verbose, leave=False) as pbar:
            audio_features = self.get_audio_features(input_features, feature_attention_mask)
            mx.eval(audio_features)
            pbar.update(1)

        # Step 2: Build input embeddings with audio merged
        with tqdm(total=1, desc="Building embeddings", disable=not verbose, leave=False) as pbar:
            inputs_embeds = self._build_inputs_embeds(input_ids, audio_features)
            mx.eval(inputs_embeds)
            pbar.update(1)

        # Free audio features memory
        del audio_features, input_features, feature_attention_mask

        # Step 3: Chunked prefill through decoder
        prompt_len = inputs_embeds.shape[1]
        prefill_pbar = tqdm(
            total=prompt_len,
            desc="Prefilling",
            unit="tok",
            disable=not verbose,
            leave=False,
        )

        for i in range(0, prompt_len, prefill_step_size):
            chunk_end = min(i + prefill_step_size, prompt_len)
            chunk_embeds = inputs_embeds[:, i:chunk_end, :]
            logits = self._forward_with_embeds(chunk_embeds, cache=cache)
            mx.eval(logits)
            prefill_pbar.update(chunk_end - i)

        prefill_pbar.close()

        # Free embeddings memory
        del inputs_embeds

        # Generation progress bar
        gen_pbar = tqdm(
            total=max_tokens,
            desc="Generating",
            unit="tok",
            disable=not verbose,
            leave=False,
        )

        if logits_processors:
            for processor in logits_processors:
                logits = processor(input_ids, logits)

        token = sampler(logits[:, -1, :])
        logprobs = mx.log(mx.softmax(logits[:, -1, :], axis=-1))
        mx.eval(token)

        token_val = int(token.item())
        if token_val in eos_token_ids:
            gen_pbar.close()
            return

        gen_pbar.update(1)
        yield token, logprobs

        for _ in range(max_tokens - 1):
            logits = self(mx.array([[token_val]]), cache=cache)

            if logits_processors:
                for processor in logits_processors:
                    logits = processor(input_ids, logits)

            token = sampler(logits[:, -1, :])
            logprobs = mx.log(mx.softmax(logits[:, -1, :], axis=-1))
            mx.eval(token)

            token_val = int(token.item())
            if token_val in eos_token_ids:
                break

            gen_pbar.update(1)
            yield token, logprobs

        gen_pbar.close()

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
        verbose: bool = False,
        **kwargs,
    ) -> STTOutput:
        """Generate transcription from audio."""
        from mlx_lm.sample_utils import make_logits_processors, make_sampler

        del kwargs

        start_time = time.time()

        if not hasattr(self, "_tokenizer") or not hasattr(self, "_feature_extractor"):
            raise RuntimeError(
                "Tokenizer/FeatureExtractor not initialized. Call post_load_hook first."
            )

        _, _, num_audio_tokens = self._preprocess_audio(audio)
        input_ids = self._build_prompt(num_audio_tokens, language)
        prompt_tokens = input_ids.shape[1]

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

        generated_tokens = []
        for token, _ in self.stream_generate(
            audio,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
            language=language,
            prefill_step_size=prefill_step_size,
            verbose=verbose,
        ):
            generated_tokens.append(int(token.item()))

        end_time = time.time()
        mx.clear_cache()

        text = self._tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return STTOutput(
            text=text,
            prompt_tokens=prompt_tokens,
            generation_tokens=len(generated_tokens),
            total_tokens=prompt_tokens + len(generated_tokens),
            total_time=end_time - start_time,
            prompt_tps=(
                prompt_tokens / (end_time - start_time) if end_time > start_time else 0
            ),
            generation_tps=(
                len(generated_tokens) / (end_time - start_time)
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
        verbose: bool = False,
    ) -> Generator[str, None, None]:
        """Stream transcription token-by-token from audio."""
        from mlx_lm.sample_utils import make_logits_processors, make_sampler

        if not hasattr(self, "_tokenizer") or not hasattr(self, "_feature_extractor"):
            raise RuntimeError(
                "Tokenizer/FeatureExtractor not initialized. Call post_load_hook first."
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

        for token, _ in self.stream_generate(
            audio,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
            language=language,
            prefill_step_size=prefill_step_size,
            verbose=verbose,
        ):
            text = self._tokenizer.decode([int(token.item())])
            yield text

        mx.clear_cache()
