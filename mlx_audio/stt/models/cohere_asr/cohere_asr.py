import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from mlx_audio.stt.models.base import STTOutput

from .audio import CohereAudioFrontend
from .config import DecoderInnerConfig, EncoderConfig, ModelConfig
from .tokenizer import CohereAsrTokenizer

NO_SPACE_LANGS = {"ja", "zh"}


class ConvSubsampling(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        feat_out = config.d_model if config.feat_out <= 0 else config.feat_out

        self.conv = [
            nn.Conv2d(
                in_channels=1,
                out_channels=config.subsampling_conv_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=config.subsampling_conv_channels,
                out_channels=config.subsampling_conv_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=config.subsampling_conv_channels,
            ),
            nn.Conv2d(
                in_channels=config.subsampling_conv_channels,
                out_channels=config.subsampling_conv_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=config.subsampling_conv_channels,
                out_channels=config.subsampling_conv_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=config.subsampling_conv_channels,
            ),
            nn.Conv2d(
                in_channels=config.subsampling_conv_channels,
                out_channels=config.subsampling_conv_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
        ]
        self.out = nn.Linear(
            config.subsampling_conv_channels
            * (config.feat_in // config.subsampling_factor),
            feat_out,
        )

    def _mask(self, x: mx.array, lengths: mx.array) -> mx.array:
        valid = mx.arange(x.shape[1])[None, :] < lengths[:, None]
        valid = mx.expand_dims(valid, axis=-1)
        valid = mx.broadcast_to(valid, (x.shape[0], x.shape[1], x.shape[2]))
        return valid.astype(x.dtype)

    @staticmethod
    def _update_lengths(lengths: mx.array) -> mx.array:
        return ((lengths + 2 - 3) // 2) + 1

    def __call__(self, x: mx.array, lengths: mx.array) -> Tuple[mx.array, mx.array]:
        x = mx.expand_dims(x, axis=1).transpose(0, 2, 3, 1)
        stride_indices = {0, 2, 5}

        for idx, layer in enumerate(self.conv):
            mask = self._mask(x, lengths)
            x = x * mx.expand_dims(mask, axis=-1)
            x = layer(x)
            if idx in stride_indices:
                lengths = self._update_lengths(lengths)

        mask = self._mask(x, lengths)
        x = x * mx.expand_dims(mask, axis=-1)
        x = x.transpose(0, 1, 3, 2).reshape(x.shape[0], x.shape[1], -1)
        return self.out(x), lengths.astype(mx.int32)


class RelPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, scale_input: bool = False):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.scale = d_model**0.5 if scale_input else 1.0
        self._pe = None
        self._build_pe()

    def _build_pe(self):
        positions = mx.arange(self.max_len - 1, -self.max_len, -1, dtype=mx.int32)
        positions = mx.expand_dims(positions, axis=1).astype(mx.float32)
        div_term = mx.exp(
            mx.arange(0, self.d_model, 2, dtype=mx.float32)
            * -(np.log(10000.0) / self.d_model)
        )
        pe = mx.zeros((2 * self.max_len - 1, self.d_model), dtype=mx.float32)
        pe[:, 0::2] = mx.sin(positions * div_term)
        pe[:, 1::2] = mx.cos(positions * div_term)
        self._pe = mx.expand_dims(pe, axis=0)

    def __call__(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        if x.shape[1] > self.max_len:
            self.max_len = x.shape[1] + 1
            self._build_pe()

        x = x * self.scale
        buffer_len = self._pe.shape[1]
        start_idx = buffer_len // 2 - (x.shape[1] - 1)
        end_idx = buffer_len // 2 + x.shape[1]
        pos_emb = self._pe[:, start_idx:end_idx].astype(x.dtype)
        return x, pos_emb


class ConformerFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear2(nn.silu(self.linear1(x)))


class RelPositionMultiHeadAttention(nn.Module):
    def __init__(self, n_head: int, n_feat: int):
        super().__init__()
        self.n_head = n_head
        self.n_feat = n_feat
        self.head_dim = n_feat // n_head
        self.scale = self.head_dim**-0.5
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.pos_bias_u = mx.zeros((n_head, self.head_dim))
        self.pos_bias_v = mx.zeros((n_head, self.head_dim))

    def rel_shift(self, x: mx.array) -> mx.array:
        batch, heads, q_len, pos_len = x.shape
        x = mx.pad(x, [(0, 0), (0, 0), (0, 0), (1, 0)])
        x = x.reshape(batch, heads, pos_len + 1, q_len)
        x = x[:, :, 1:, :]
        return x.reshape(batch, heads, q_len, pos_len)

    def __call__(
        self,
        x: mx.array,
        pos_emb: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        batch, q_len, _ = x.shape
        p = self.linear_pos(pos_emb)
        _, pos_len, _ = p.shape

        q = self.linear_q(x).reshape(batch, q_len, self.n_head, self.head_dim)
        k = self.linear_k(x).reshape(batch, q_len, self.n_head, self.head_dim)
        v = self.linear_v(x).reshape(batch, q_len, self.n_head, self.head_dim)
        p = p.reshape(1, pos_len, self.n_head, self.head_dim)

        q_u = (q + self.pos_bias_u).transpose(0, 2, 1, 3)
        q_v = (q + self.pos_bias_v).transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        p = p.transpose(0, 2, 1, 3)

        matrix_bd = mx.matmul(q_v, p.swapaxes(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)
        matrix_bd = matrix_bd[:, :, :, : k.shape[2]] * self.scale

        if attention_mask is not None:
            matrix_bd = matrix_bd + attention_mask.astype(matrix_bd.dtype)

        output = mx.fast.scaled_dot_product_attention(
            q_u, k, v, scale=self.scale, mask=matrix_bd
        )
        output = output.transpose(0, 2, 1, 3).reshape(batch, q_len, self.n_feat)
        return self.linear_out(output)


class ConformerConvolution(nn.Module):
    def __init__(self, d_model: int, kernel_size: int):
        super().__init__()
        self.pointwise_conv1 = nn.Conv1d(
            d_model,
            d_model * 2,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.depthwise_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            groups=d_model,
        )
        self.batch_norm = nn.BatchNorm(d_model)
        self.pointwise_conv2 = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def __call__(self, x: mx.array, pad_mask: Optional[mx.array] = None) -> mx.array:
        x = self.pointwise_conv1(x)
        x = nn.glu(x, axis=2)
        if pad_mask is not None:
            x = mx.where(mx.expand_dims(pad_mask, axis=-1), 0.0, x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = nn.silu(x)
        return self.pointwise_conv2(x)


class ConformerLayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_heads: int, conv_kernel_size: int):
        super().__init__()
        self.norm_feed_forward1 = nn.LayerNorm(d_model)
        self.feed_forward1 = ConformerFeedForward(d_model, d_ff)
        self.norm_self_att = nn.LayerNorm(d_model)
        self.self_attn = RelPositionMultiHeadAttention(n_heads, d_model)
        self.norm_conv = nn.LayerNorm(d_model)
        self.conv = ConformerConvolution(d_model, conv_kernel_size)
        self.norm_feed_forward2 = nn.LayerNorm(d_model)
        self.feed_forward2 = ConformerFeedForward(d_model, d_ff)
        self.norm_out = nn.LayerNorm(d_model)

    def __call__(
        self,
        x: mx.array,
        pos_emb: mx.array,
        attention_mask: Optional[mx.array] = None,
        pad_mask: Optional[mx.array] = None,
    ) -> mx.array:
        x = x + 0.5 * self.feed_forward1(self.norm_feed_forward1(x))
        x = x + self.self_attn(self.norm_self_att(x), pos_emb, attention_mask)
        x = x + self.conv(self.norm_conv(x), pad_mask=pad_mask)
        x = x + 0.5 * self.feed_forward2(self.norm_feed_forward2(x))
        return self.norm_out(x)


class ConformerEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        enc_config = config.encoder
        self.d_model = enc_config.d_model
        self.pre_encode = ConvSubsampling(enc_config)
        self.pos_enc = RelPositionalEncoding(
            enc_config.d_model,
            max_len=enc_config.pos_emb_max_len,
            scale_input=enc_config.xscaling,
        )
        self.layers = [
            ConformerLayer(
                enc_config.d_model,
                enc_config.d_model * enc_config.ff_expansion_factor,
                enc_config.n_heads,
                enc_config.conv_kernel_size,
            )
            for _ in range(enc_config.n_layers)
        ]

    def _create_masks(
        self, lengths: mx.array, max_len: int, dtype: mx.Dtype
    ) -> Tuple[mx.array, mx.array]:
        valid = mx.arange(max_len)[None, :] < lengths[:, None]
        pad_mask = ~valid
        attn_mask = mx.where(
            mx.expand_dims(valid[:, None, :] & valid[:, :, None], axis=1),
            0.0,
            -1e9,
        ).astype(dtype)
        return pad_mask, attn_mask

    def __call__(
        self, input_features: mx.array, lengths: mx.array
    ) -> Tuple[mx.array, mx.array]:
        x, lengths = self.pre_encode(input_features, lengths)
        x, pos_emb = self.pos_enc(x)
        pad_mask, attention_mask = self._create_masks(lengths, x.shape[1], x.dtype)

        for layer in self.layers:
            x = layer(x, pos_emb, attention_mask=attention_mask, pad_mask=pad_mask)

        return x, lengths


class FixedPositionalEncoding(nn.Module):
    def __init__(self, hidden_size: int, max_sequence_length: int = 1024):
        super().__init__()
        position = mx.arange(max_sequence_length, dtype=mx.float32)[:, None]
        div_term = mx.exp(
            -(np.log(10000.0) / hidden_size)
            * mx.arange(0, hidden_size, 2, dtype=mx.float32)
        )
        pos_enc = mx.zeros((max_sequence_length, hidden_size), dtype=mx.float32)
        pos_enc[:, 0::2] = mx.sin(position * div_term)
        pos_enc[:, 1::2] = mx.cos(position * div_term)
        self.pos_enc = pos_enc / np.sqrt(hidden_size)

    def __call__(self, position_ids: mx.array) -> mx.array:
        return self.pos_enc[position_ids]


class DecoderAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.query_net = nn.Linear(hidden_size, hidden_size)
        self.key_net = nn.Linear(hidden_size, hidden_size)
        self.value_net = nn.Linear(hidden_size, hidden_size)
        self.out_projection = nn.Linear(hidden_size, hidden_size)

    def _reshape(self, x: mx.array) -> mx.array:
        batch, seq_len, _ = x.shape
        return x.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )

    def __call__(
        self,
        hidden_states: mx.array,
        context_states: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        query = self._reshape(self.query_net(hidden_states))
        source = hidden_states if context_states is None else context_states

        if cache is not None and context_states is not None:
            key, value = cache
        else:
            key = self._reshape(self.key_net(source))
            value = self._reshape(self.value_net(source))
            if cache is not None and context_states is None:
                key = mx.concatenate([cache[0], key], axis=2)
                value = mx.concatenate([cache[1], value], axis=2)

        output = mx.fast.scaled_dot_product_attention(
            query,
            key,
            value,
            scale=self.scale,
            mask=attention_mask,
        )
        output = output.transpose(0, 2, 1, 3).reshape(
            hidden_states.shape[0], hidden_states.shape[1], self.hidden_size
        )
        return self.out_projection(output), (key, value)


class DecoderFeedForward(nn.Module):
    def __init__(self, hidden_size: int, inner_size: int, hidden_act: str = "relu"):
        super().__init__()
        self.dense_in = nn.Linear(hidden_size, inner_size)
        self.dense_out = nn.Linear(inner_size, hidden_size)
        self.hidden_act = hidden_act

    def __call__(self, x: mx.array) -> mx.array:
        if self.hidden_act.lower() in {"silu", "swish"}:
            x = nn.silu(self.dense_in(x))
        else:
            x = nn.relu(self.dense_in(x))
        return self.dense_out(x)


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self, hidden_size: int, inner_size: int, num_heads: int, hidden_act: str
    ):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(hidden_size)
        self.first_sub_layer = DecoderAttention(hidden_size, num_heads)
        self.layer_norm_2 = nn.LayerNorm(hidden_size)
        self.second_sub_layer = DecoderAttention(hidden_size, num_heads)
        self.layer_norm_3 = nn.LayerNorm(hidden_size)
        self.third_sub_layer = DecoderFeedForward(hidden_size, inner_size, hidden_act)

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        self_attention_mask: Optional[mx.array] = None,
        cross_attention_mask: Optional[mx.array] = None,
        cache: Optional[Dict[str, Tuple[mx.array, mx.array]]] = None,
    ) -> Tuple[mx.array, Dict[str, Tuple[mx.array, mx.array]]]:
        cache = cache or {"self_attn": None, "cross_attn": None}

        residual = hidden_states
        hidden_states_norm = self.layer_norm_1(hidden_states)
        self_out, self_cache = self.first_sub_layer(
            hidden_states_norm,
            attention_mask=self_attention_mask,
            cache=cache["self_attn"],
        )
        hidden_states = residual + self_out

        residual = hidden_states
        hidden_states_norm = self.layer_norm_2(hidden_states)
        cross_out, cross_cache = self.second_sub_layer(
            hidden_states_norm,
            context_states=encoder_hidden_states,
            attention_mask=cross_attention_mask,
            cache=cache["cross_attn"],
        )
        hidden_states = residual + cross_out

        residual = hidden_states
        hidden_states = residual + self.third_sub_layer(
            self.layer_norm_3(hidden_states)
        )

        return hidden_states, {"self_attn": self_cache, "cross_attn": cross_cache}


class TransformerDecoderEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        max_sequence_length: int,
        padding_idx: int = 2,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = FixedPositionalEncoding(
            hidden_size, max_sequence_length=max_sequence_length
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.padding_idx = padding_idx

    def __call__(self, input_ids: mx.array, positions: mx.array) -> mx.array:
        return self.layer_norm(
            self.token_embedding(input_ids) + self.position_embedding(positions)
        )


class TransformerDecoderCore(nn.Module):
    def __init__(self, config: DecoderInnerConfig):
        super().__init__()
        self.layers = [
            TransformerDecoderLayer(
                config.hidden_size,
                config.inner_size,
                config.num_attention_heads,
                config.hidden_act,
            )
            for _ in range(config.num_layers)
        ]
        self.final_layer_norm = nn.LayerNorm(config.hidden_size)

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        self_attention_mask: Optional[mx.array] = None,
        cross_attention_mask: Optional[mx.array] = None,
        cache: Optional[List[Dict[str, Tuple[mx.array, mx.array]]]] = None,
    ) -> Tuple[mx.array, List[Dict[str, Tuple[mx.array, mx.array]]]]:
        if cache is None:
            cache = [None] * len(self.layers)

        new_cache = []
        for layer, layer_cache in zip(self.layers, cache):
            hidden_states, updated_cache = layer(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                self_attention_mask=self_attention_mask,
                cross_attention_mask=cross_attention_mask,
                cache=layer_cache,
            )
            new_cache.append(updated_cache)

        return self.final_layer_norm(hidden_states), new_cache


class TransformerDecoderWrapper(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        dec_config = config.transf_decoder.config_dict
        self.embedding = TransformerDecoderEmbedding(
            vocab_size=config.head.num_classes,
            hidden_size=dec_config.hidden_size,
            max_sequence_length=dec_config.max_sequence_length,
            padding_idx=2,
        )
        self.decoder = TransformerDecoderCore(dec_config)

    def __call__(
        self,
        input_ids: mx.array,
        encoder_hidden_states: mx.array,
        encoder_mask: Optional[mx.array] = None,
        cache: Optional[List[Dict[str, Tuple[mx.array, mx.array]]]] = None,
        start_pos: int = 0,
    ) -> Tuple[mx.array, List[Dict[str, Tuple[mx.array, mx.array]]]]:
        batch, seq_len = input_ids.shape
        positions = mx.arange(start_pos, start_pos + seq_len, dtype=mx.int32)
        positions = mx.broadcast_to(positions[None, :], (batch, seq_len))
        hidden_states = self.embedding(input_ids, positions)

        self_attention_mask = None
        if seq_len > 1:
            self_attention_mask = nn.MultiHeadAttention.create_additive_causal_mask(
                seq_len
            ).astype(hidden_states.dtype)
            if (
                cache is not None
                and cache[0] is not None
                and cache[0]["self_attn"] is not None
            ):
                cached_len = cache[0]["self_attn"][0].shape[2]
                prefix = mx.zeros((seq_len, cached_len), dtype=hidden_states.dtype)
                self_attention_mask = mx.concatenate(
                    [prefix, self_attention_mask], axis=1
                )

        cross_attention_mask = None
        if encoder_mask is not None:
            cross_attention_mask = mx.where(
                encoder_mask[:, None, None, :], 0.0, -1e9
            ).astype(hidden_states.dtype)

        return self.decoder(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            self_attention_mask=self_attention_mask,
            cross_attention_mask=cross_attention_mask,
            cache=cache,
        )


class _MLPWrapper(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int):
        super().__init__()
        self.layer0 = nn.Linear(hidden_size, num_classes)


class TokenClassifierHead(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int, use_log_softmax: bool):
        super().__init__()
        self.mlp = _MLPWrapper(hidden_size, num_classes)
        self.use_log_softmax = use_log_softmax

    def __call__(self, hidden_states: mx.array) -> mx.array:
        logits = self.mlp.layer0(hidden_states)
        if self.use_log_softmax:
            return nn.log_softmax(logits, axis=-1)
        return logits


def split_audio_chunks_energy(
    waveform: np.ndarray,
    sample_rate: int,
    max_audio_clip_s: float,
    overlap_chunk_second: float,
    min_energy_window_samples: int,
) -> List[Tuple[int, int]]:
    chunk_size = max(1, int(round(max_audio_clip_s * sample_rate)))
    boundary_context = max(1, int(round(overlap_chunk_second * sample_rate)))
    total_samples = waveform.shape[0]

    if total_samples <= chunk_size:
        return [(0, total_samples)]

    chunks = []
    start = 0
    while start < total_samples:
        if start + chunk_size >= total_samples:
            chunks.append((start, total_samples))
            break

        search_start = max(start, start + chunk_size - boundary_context)
        search_end = min(start + chunk_size, total_samples)
        split_point = _find_split_point_energy(
            waveform,
            search_start,
            search_end,
            min_energy_window_samples,
        )
        split_point = max(start + 1, min(split_point, total_samples))
        chunks.append((start, split_point))
        start = split_point

    return chunks


def _find_split_point_energy(
    waveform: np.ndarray,
    start_idx: int,
    end_idx: int,
    min_energy_window_samples: int,
) -> int:
    segment = waveform[start_idx:end_idx]
    if segment.shape[0] <= min_energy_window_samples:
        return (start_idx + end_idx) // 2

    quietest_idx = start_idx
    min_energy = float("inf")
    upper = segment.shape[0] - min_energy_window_samples
    for offset in range(0, upper, min_energy_window_samples):
        window = segment[offset : offset + min_energy_window_samples]
        energy = float(np.sqrt(np.mean(window * window)))
        if energy < min_energy:
            min_energy = energy
            quietest_idx = start_idx + offset
    return quietest_idx


def join_chunk_texts(texts: Iterable[str], language: str) -> str:
    parts = [piece.strip() for piece in texts if piece and piece.strip()]
    if not parts:
        return ""
    separator = "" if language in NO_SPACE_LANGS else " "
    return separator.join(parts)


class Model(nn.Module):
    def __init__(self, config: Union[ModelConfig, Dict]):
        super().__init__()
        if isinstance(config, dict):
            config = ModelConfig.from_dict(config)
        self.config = config
        self.encoder = ConformerEncoder(config)
        self.transf_decoder = TransformerDecoderWrapper(config)
        self.decoder_hidden_size = config.transf_decoder.config_dict.hidden_size
        self.encoder_decoder_proj = (
            nn.Linear(config.encoder.d_model, self.decoder_hidden_size)
            if config.encoder.d_model != self.decoder_hidden_size
            else None
        )
        self.log_softmax = TokenClassifierHead(
            hidden_size=config.head.hidden_size,
            num_classes=config.head.num_classes,
            use_log_softmax=bool(config.head.log_softmax),
        )
        self.audio_frontend = CohereAudioFrontend(config.preprocessor)
        self._tokenizer: Optional[CohereAsrTokenizer] = None

    @property
    def sample_rate(self) -> int:
        return self.config.sample_rate

    def _validate_language(self, language: str) -> None:
        if language not in set(self.config.supported_languages):
            raise ValueError(
                f"Unsupported language '{language}'. Supported languages: {sorted(self.config.supported_languages)}"
            )

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        model_weights = dict(tree_flatten(self.parameters()))
        sanitized = {}
        for key, value in weights.items():
            if key.startswith("preprocessor."):
                continue
            if key.endswith("num_batches_tracked"):
                continue

            new_key = key
            if key.startswith("transf_decoder._embedding."):
                new_key = key.replace(
                    "transf_decoder._embedding.", "transf_decoder.embedding."
                )
            elif key.startswith("transf_decoder._decoder."):
                new_key = key.replace(
                    "transf_decoder._decoder.", "transf_decoder.decoder."
                )

            expected = model_weights.get(new_key)
            if expected is not None and hasattr(expected, "shape"):
                if value.shape != expected.shape:
                    if value.ndim == 3:
                        transposed = mx.transpose(value, (0, 2, 1))
                        if transposed.shape == expected.shape:
                            value = transposed
                    elif value.ndim == 4:
                        transposed = mx.transpose(value, (0, 2, 3, 1))
                        if transposed.shape == expected.shape:
                            value = transposed
            elif new_key.endswith("weight"):
                if value.ndim == 3:
                    value = mx.transpose(value, (0, 2, 1))
                elif value.ndim == 4:
                    value = mx.transpose(value, (0, 2, 3, 1))

            sanitized[new_key] = value

        return sanitized

    @classmethod
    def post_load_hook(cls, model: "Model", model_path: Path) -> "Model":
        model_path = Path(model_path)
        tokenizer_path = model_path / "tokenizer.model"
        tokenizer_config_path = model_path / "tokenizer_config.json"
        special_tokens_map_path = model_path / "special_tokens_map.json"

        model._tokenizer = CohereAsrTokenizer(
            str(tokenizer_path),
            str(tokenizer_config_path),
            str(special_tokens_map_path),
        )
        model.audio_frontend.load_buffers_from_checkpoint(model_path)
        return model

    def _to_mono(
        self,
        audio: Union[str, Path, np.ndarray, mx.array],
        sample_rate: Optional[int] = None,
    ) -> np.ndarray:
        from mlx_audio.stt.utils import load_audio, resample_audio

        if isinstance(audio, (str, Path)):
            return np.array(
                load_audio(str(audio), sr=self.sample_rate), dtype=np.float32
            )

        if isinstance(audio, mx.array):
            arr = np.array(audio)
        else:
            arr = np.asarray(audio, dtype=np.float32)

        if arr.ndim == 2:
            if arr.shape[0] <= 8 and arr.shape[1] > arr.shape[0]:
                arr = arr.mean(axis=0)
            else:
                arr = arr.mean(axis=1)

        if sample_rate is not None and sample_rate != self.sample_rate:
            arr = resample_audio(
                arr.astype(np.float32, copy=False), sample_rate, self.sample_rate
            )

        if arr.ndim != 1:
            raise ValueError(f"Expected mono waveform, got shape {arr.shape}.")

        return arr.astype(np.float32, copy=False)

    def _encode_waveforms(
        self, waveforms: List[np.ndarray]
    ) -> Tuple[mx.array, mx.array, mx.array]:
        input_features, lengths = self.audio_frontend(waveforms)
        conv_weight = self.encoder.pre_encode.conv[0].weight
        if input_features.dtype != conv_weight.dtype:
            input_features = input_features.astype(conv_weight.dtype)

        encoder_hidden_states, encoder_lengths = self.encoder(input_features, lengths)
        if self.encoder_decoder_proj is not None:
            encoder_hidden_states = self.encoder_decoder_proj(encoder_hidden_states)

        encoder_mask = (
            mx.arange(encoder_hidden_states.shape[1])[None, :]
            < encoder_lengths[:, None]
        )
        return encoder_hidden_states, encoder_lengths, encoder_mask

    def _generate_batch_tokens(
        self,
        waveforms: List[np.ndarray],
        prompt_tokens: List[int],
        max_tokens: int,
    ) -> Tuple[List[List[int]], int]:
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not initialized. Load the model first.")

        encoder_hidden_states, _, encoder_mask = self._encode_waveforms(waveforms)
        batch_size = len(waveforms)
        prompt_ids = mx.array([prompt_tokens] * batch_size, dtype=mx.int32)
        logits, cache = self.transf_decoder(
            prompt_ids,
            encoder_hidden_states,
            encoder_mask=encoder_mask,
            cache=None,
            start_pos=0,
        )
        logits = self.log_softmax(logits)

        if max_tokens <= 0:
            return [[] for _ in range(batch_size)], len(prompt_tokens)

        next_tokens = mx.argmax(logits[:, -1, :], axis=-1)
        next_tokens_np = np.array(next_tokens)
        eos_id = self._tokenizer.eos_token_id
        finished = next_tokens_np == eos_id
        generated = [[] for _ in range(batch_size)]

        for idx, token_id in enumerate(next_tokens_np.tolist()):
            if token_id != eos_id:
                generated[idx].append(int(token_id))

        prompt_len = len(prompt_tokens)
        current_tokens = next_tokens_np.astype(np.int32)

        for step in range(max_tokens - 1):
            if bool(np.all(finished)):
                break

            feed_tokens = current_tokens.copy()
            feed_tokens[finished] = eos_id
            logits, cache = self.transf_decoder(
                mx.array(feed_tokens[:, None], dtype=mx.int32),
                encoder_hidden_states,
                encoder_mask=encoder_mask,
                cache=cache,
                start_pos=prompt_len + step,
            )
            logits = self.log_softmax(logits)
            current_tokens = np.array(mx.argmax(logits[:, -1, :], axis=-1)).astype(
                np.int32
            )

            for idx, token_id in enumerate(current_tokens.tolist()):
                if not finished[idx] and token_id != eos_id:
                    generated[idx].append(int(token_id))
            finished = finished | (current_tokens == eos_id)

        return generated, prompt_len

    def _transcribe_waveforms_batched(
        self,
        waveforms: List[np.ndarray],
        language: str,
        punctuation: bool,
        batch_size: int,
        max_tokens: int,
    ) -> Tuple[List[str], List[int], int]:
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not initialized. Load the model first.")

        prompt_tokens = self._tokenizer.build_prompt_tokens(language, punctuation)
        order = sorted(
            range(len(waveforms)), key=lambda idx: waveforms[idx].shape[0], reverse=True
        )

        texts = [""] * len(waveforms)
        generation_counts = [0] * len(waveforms)

        for start in range(0, len(order), batch_size):
            batch_indices = order[start : start + batch_size]
            batch_waveforms = [waveforms[idx] for idx in batch_indices]
            generated_ids, prompt_len = self._generate_batch_tokens(
                batch_waveforms,
                prompt_tokens,
                max_tokens,
            )
            batch_texts = self._tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )

            for row_idx, original_idx in enumerate(batch_indices):
                texts[original_idx] = batch_texts[row_idx].strip()
                generation_counts[original_idx] = len(generated_ids[row_idx])

        return texts, generation_counts, len(prompt_tokens)

    def _prepare_segments(
        self, waveforms: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[Dict[str, Union[int, float, None]]]]:
        segment_waveforms = []
        segment_meta = []
        fast_path_threshold_s = max(
            0.0, self.config.max_audio_clip_s - self.config.overlap_chunk_second
        )

        for sample_idx, waveform in enumerate(waveforms):
            duration_s = waveform.shape[0] / self.sample_rate
            if duration_s <= fast_path_threshold_s:
                segment_waveforms.append(waveform)
                segment_meta.append(
                    {
                        "sample_idx": sample_idx,
                        "chunk_idx": None,
                        "start": 0.0,
                        "end": duration_s,
                    }
                )
                continue

            chunks = split_audio_chunks_energy(
                waveform=waveform,
                sample_rate=self.sample_rate,
                max_audio_clip_s=self.config.max_audio_clip_s,
                overlap_chunk_second=self.config.overlap_chunk_second,
                min_energy_window_samples=self.config.min_energy_window_samples,
            )
            for chunk_idx, (start_idx, end_idx) in enumerate(chunks):
                segment_waveforms.append(waveform[start_idx:end_idx].copy())
                segment_meta.append(
                    {
                        "sample_idx": sample_idx,
                        "chunk_idx": chunk_idx,
                        "start": start_idx / self.sample_rate,
                        "end": end_idx / self.sample_rate,
                    }
                )

        return segment_waveforms, segment_meta

    def transcribe(
        self,
        *,
        language: str,
        audio_files: Optional[List[str]] = None,
        audio_arrays: Optional[List[np.ndarray]] = None,
        sample_rates: Optional[List[int]] = None,
        punctuation: bool = True,
        batch_size: Optional[int] = None,
        max_tokens: int = 256,
    ) -> List[str]:
        if (audio_files is None) == (audio_arrays is None):
            raise ValueError("Provide exactly one of audio_files or audio_arrays.")
        if audio_arrays is not None and sample_rates is None:
            raise ValueError(
                "sample_rates are required when audio_arrays are provided."
            )
        if audio_arrays is not None and len(audio_arrays) != len(sample_rates):
            raise ValueError("audio_arrays and sample_rates must have the same length.")

        self._validate_language(language)

        if audio_files is not None:
            waveforms = [self._to_mono(path) for path in audio_files]
            total_inputs = len(audio_files)
        else:
            waveforms = [
                self._to_mono(audio, sample_rate=sample_rate)
                for audio, sample_rate in zip(audio_arrays, sample_rates)
            ]
            total_inputs = len(audio_arrays)

        if total_inputs == 0:
            return []

        segment_waveforms, segment_meta = self._prepare_segments(waveforms)
        texts, _, _ = self._transcribe_waveforms_batched(
            segment_waveforms,
            language=language,
            punctuation=punctuation,
            batch_size=batch_size or self.config.batch_size,
            max_tokens=max_tokens,
        )

        outputs = [""] * total_inputs
        grouped = {}
        for meta, text in zip(segment_meta, texts):
            sample_idx = int(meta["sample_idx"])
            chunk_idx = meta["chunk_idx"]
            if chunk_idx is None:
                outputs[sample_idx] = text
                continue
            grouped.setdefault(sample_idx, []).append((int(chunk_idx), text))

        for sample_idx, chunk_items in grouped.items():
            chunk_items.sort(key=lambda item: item[0])
            outputs[sample_idx] = join_chunk_texts(
                [text for _, text in chunk_items], language=language
            )

        return outputs

    def generate(
        self,
        audio: Union[str, Path, np.ndarray, mx.array],
        *,
        language: str = "en",
        punctuation: bool = True,
        batch_size: Optional[int] = None,
        max_tokens: int = 256,
        verbose: bool = False,
        stream: bool = False,
        sample_rate: Optional[int] = None,
        **kwargs,
    ) -> STTOutput:
        if stream:
            raise NotImplementedError(
                "Streaming generation is not implemented for Cohere ASR."
            )

        start_time = time.time()
        self._validate_language(language)
        waveform = self._to_mono(audio, sample_rate=sample_rate)
        segment_waveforms, segment_meta = self._prepare_segments([waveform])
        texts, generation_counts, prompt_len = self._transcribe_waveforms_batched(
            segment_waveforms,
            language=language,
            punctuation=punctuation,
            batch_size=batch_size or self.config.batch_size,
            max_tokens=max_tokens,
        )

        segments = []
        for meta, text in zip(segment_meta, texts):
            segments.append(
                {
                    "text": text,
                    "start": float(meta["start"]),
                    "end": float(meta["end"]),
                }
            )

        final_text = join_chunk_texts(texts, language=language)
        total_time = time.time() - start_time
        generation_tokens = int(sum(generation_counts))
        prompt_tokens = int(prompt_len * len(segment_waveforms))

        if verbose:
            print(final_text)

        return STTOutput(
            text=final_text,
            segments=segments,
            language=language,
            prompt_tokens=prompt_tokens,
            generation_tokens=generation_tokens,
            total_tokens=prompt_tokens + generation_tokens,
            total_time=total_time,
            prompt_tps=prompt_tokens / total_time if total_time > 0 else 0.0,
            generation_tps=generation_tokens / total_time if total_time > 0 else 0.0,
        )
