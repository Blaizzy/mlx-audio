import argparse
import glob
import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import tqdm
from mlx.utils import tree_map, tree_unflatten
from mlx_lm.models.base import create_causal_mask
from scipy.io.wavfile import write as write_wav
from transformers import BertTokenizer

from ..base import BaseModelArgs, GenerationResult
from .pipeline import Pipeline

mx.random.seed(42)

TEXT_ENCODING_OFFSET = 10_048
SEMANTIC_PAD_TOKEN = 10_000
TEXT_PAD_TOKEN = 129595
SEMANTIC_INFER_TOKEN = 129_599

CONTEXT_WINDOW_SIZE = 1024

SEMANTIC_RATE_HZ = 49.9
SEMANTIC_VOCAB_SIZE = 10_000

CODEBOOK_SIZE = 1024
N_COARSE_CODEBOOKS = 2
N_FINE_CODEBOOKS = 8
COARSE_RATE_HZ = 75
COARSE_SEMANTIC_PAD_TOKEN = 12_048
COARSE_INFER_TOKEN = 12_050
SAMPLE_RATE = 24_000


def filter_dataclass_fields(data_dict, dataclass_type):
    """Filter a dictionary to only include keys that are fields in the dataclass."""
    valid_fields = {f.name for f in dataclass_type.__dataclass_fields__.values()}
    return {k: v for k, v in data_dict.items() if k in valid_fields}


@dataclass
class SemanticConfig(BaseModelArgs):
    bad_words_ids: list[list[int]] = None
    block_size: int = 1024
    input_vocab_size: int = 129600
    output_vocab_size: int = 129600
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    bias: bool = False
    model_type: str = "semantic"
    dropout: float = 0.0
    architectures: list[str] = None


@dataclass
class CoarseAcousticsConfig(BaseModelArgs):
    block_size: int = 1024
    input_vocab_size: int = 12096
    output_vocab_size: int = 12096
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    bias: bool = False
    model_type: str = "coarse_acoustics"
    dropout: float = 0.0


@dataclass
class FineAcousticsConfig(BaseModelArgs):
    block_size: int = 1024
    input_vocab_size: int = 1056
    output_vocab_size: int = 1056
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    bias: bool = False
    model_type: str = "fine_acoustics"
    n_codes_total: int = 8
    n_codes_given: int = 1
    dropout: float = 0.0


@dataclass
class CodecConfig(BaseModelArgs):
    model_type: str = "codec"
    sample_rate: int = 24000
    target_bandwidth: float = 6.0


@dataclass
class ModelConfig(BaseModelArgs):
    semantic_config: SemanticConfig
    coarse_acoustics_config: CoarseAcousticsConfig
    fine_acoustics_config: FineAcousticsConfig
    codec_config: CodecConfig
    block_size: int = 1024
    input_vocab_size: int = 10_048
    output_vocab_size: int = 10_048
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False
    n_codes_total: Optional[int] = None
    n_codes_given: Optional[int] = None
    model_size: str = "base"
    model_type: str = "bark"
    initializer_range: float = 0.02
    codec_path: str = "mlx-community/encodec-24khz-float32"
    sample_rate: int = 24000


class LayerNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5, bias: bool = True):
        super().__init__()
        self.bias = mx.zeros((dims,)) if bias else None
        self.weight = mx.ones((dims,))
        self.dims = dims
        self.eps = eps

    def __call__(self, x):
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        x = (x - mean) * mx.rsqrt(var + self.eps)
        if self.bias is not None:
            x = x * self.weight + self.bias
        else:
            x = x * self.weight
        return x


class CausalSelfAttention(nn.Module):
    def __init__(
        self, args: Union[SemanticConfig, CoarseAcousticsConfig, FineAcousticsConfig]
    ):
        super().__init__()
        self.att_proj = nn.Linear(args.n_embd, 3 * args.n_embd, bias=args.bias)
        self.out_proj = nn.Linear(args.n_embd, args.n_embd, bias=args.bias)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.n_head = args.n_head
        self.n_embd = args.n_embd
        self.dropout = args.dropout
        self.bias = (
            mx.tril(mx.ones([args.block_size, args.block_size]))
            .reshape(1, 1, args.block_size, args.block_size)
            .astype(mx.float32)
        )

    def __call__(self, x, past_kv=None, use_cache=False):
        B, T, C = x.shape
        query, key, value = mx.split(self.att_proj(x), 3, axis=2)
        key = key.reshape(B, T, self.n_head, C // self.n_head).transpose(0, 2, 1, 3)
        query = query.reshape(B, T, self.n_head, C // self.n_head).transpose(0, 2, 1, 3)
        value = value.reshape(B, T, self.n_head, C // self.n_head).transpose(0, 2, 1, 3)
        if past_kv is not None:
            past_key, past_value = past_kv
            key = mx.concatenate([past_key, key], axis=-2)
            value = mx.concatenate([past_value, value], axis=-2)

        FULL_T = key.shape[-2]
        if use_cache is True:
            present = (key, value)
        else:
            present = None

        y = mx.fast.scaled_dot_product_attention(
            query,
            key,
            value,
            scale=1.0 / math.sqrt(key.shape[3]),
            mask=self.bias[:, :, FULL_T - T : FULL_T, :FULL_T],
        )
        y = self.attn_dropout(y)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)
        y = self.resid_dropout(self.out_proj(y))
        return (y, present)


class NonCausalSelfAttention(nn.Module):
    def __init__(
        self, args: Union[SemanticConfig, CoarseAcousticsConfig, FineAcousticsConfig]
    ):
        super().__init__()
        self.att_proj = nn.Linear(args.n_embd, 3 * args.n_embd, bias=args.bias)
        self.out_proj = nn.Linear(args.n_embd, args.n_embd, bias=args.bias)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.n_head = args.n_head
        self.n_embd = args.n_embd
        self.dropout = args.dropout

    def __call__(self, x):
        B, T, C = x.shape
        query, key, value = mx.split(self.att_proj(x), 3, axis=2)
        key = key.reshape(B, T, self.n_head, C // self.n_head).transpose(0, 2, 1, 3)
        query = query.reshape(B, T, self.n_head, C // self.n_head).transpose(0, 2, 1, 3)
        value = value.reshape(B, T, self.n_head, C // self.n_head).transpose(0, 2, 1, 3)

        y = mx.fast.scaled_dot_product_attention(
            query, key, value, scale=1.0 / math.sqrt(key.shape[3])
        )
        y = self.attn_dropout(y)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)
        y = self.resid_dropout(self.out_proj(y))
        return y


class MLP(nn.Module):
    def __init__(
        self, args: Union[SemanticConfig, CoarseAcousticsConfig, FineAcousticsConfig]
    ):
        super().__init__()

        self.in_proj = nn.Linear(args.n_embd, 4 * args.n_embd, bias=False)
        self.out_proj = nn.Linear(4 * args.n_embd, args.n_embd, bias=False)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(args.dropout)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.in_proj(x)
        x = self.gelu(x)
        x = self.out_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(
        self, args: Union[SemanticConfig, CoarseAcousticsConfig], layer_idx: int = 0
    ):
        super().__init__()
        self.args = args
        self.layernorm_1 = LayerNorm(args.n_embd, bias=False)
        self.attn = CausalSelfAttention(args)
        self.layernorm_2 = LayerNorm(args.n_embd, bias=False)
        self.mlp = MLP(args)
        self.layer_idx = layer_idx

    def __call__(self, x: mx.array, past_kv=None, use_cache=False):
        attn_output, prev_kvs = self.attn(
            self.layernorm_1(x), past_kv=past_kv, use_cache=use_cache
        )
        x = x + attn_output
        x = x + self.mlp(self.layernorm_2(x))
        return (x, prev_kvs)


class FineBlock(nn.Module):
    def __init__(self, args: FineAcousticsConfig):
        super().__init__()
        self.args = args
        self.layernorm_1 = nn.LayerNorm(args.n_embd)
        self.attn = NonCausalSelfAttention(args)
        self.layernorm_2 = nn.LayerNorm(args.n_embd)
        self.mlp = MLP(args)

    def __call__(self, x: mx.array):
        x = x + self.attn(self.layernorm_1(x))
        x = x + self.mlp(self.layernorm_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, args: Union[SemanticConfig, CoarseAcousticsConfig]):
        super().__init__()
        self.args = args
        self.input_embeds_layer = nn.Embedding(args.input_vocab_size, args.n_embd)
        self.position_embeds_layer = nn.Embedding(args.block_size, args.n_embd)
        self.drop = nn.Dropout(args.dropout)
        self.layers = [Block(args=args) for _ in range(args.n_layer)]
        self.layernorm_final = LayerNorm(args.n_embd, bias=False)
        self.lm_head = nn.Linear(args.n_embd, args.output_vocab_size, bias=False)

    def __call__(
        self,
        x: mx.array,
        merge_context: bool = False,
        past_kv: mx.array = None,
        position_ids: mx.array = None,
        use_cache: bool = False,
    ) -> mx.array:
        b, t = x.shape

        if past_kv is not None:
            assert t == 1
            tok_emb = self.input_embeds_layer(x)
        else:
            if merge_context:
                assert x.shape[1] >= 256 + 256 + 1
                t = x.shape[1] - 256
                tok_emb = mx.concatenate(
                    [
                        self.input_embeds_layer(x[:, :256])
                        + self.input_embeds_layer(x[:, 256 : 256 + 256]),
                        self.input_embeds_layer(x[:, 256 + 256 :]),
                    ],
                    axis=1,
                )
            else:
                tok_emb = self.input_embeds_layer(x)

        # past length
        if past_kv is None:
            past_length = 0
            past_kv = tuple([None] * len(self.layers))
        else:
            past_length = past_kv[0][0].shape[-2]

        if position_ids is None:
            position_ids = mx.arange(past_length, t + past_length)
            position_ids = position_ids.reshape(1, -1)  # shape (1, t)

        pos_emb = self.position_embeds_layer(
            position_ids
        )  # position embeddings of shape (1, t, n_embd)
        x = self.drop(tok_emb + pos_emb)

        new_kv = () if use_cache else None

        for i, (block, past_layer_kv) in enumerate(zip(self.layers, past_kv)):
            x, kv = block(x, past_kv=past_layer_kv, use_cache=use_cache)

            if use_cache:
                new_kv = new_kv + (kv,)

        x = self.layernorm_final(x)

        logits = self.lm_head(
            x[:, -1:, :]
        )  # note: using list [-1] to preserve the time dim

        return (logits, new_kv)


class FineGPT(nn.Module):
    def __init__(self, args: FineAcousticsConfig):
        super().__init__()
        self.args = args
        self.n_codes_total = args.n_codes_total
        self.input_embeds_layers = [
            nn.Embedding(args.input_vocab_size, args.n_embd)
            for _ in range(args.n_codes_total)
        ]
        self.position_embeds_layer = nn.Embedding(args.block_size, args.n_embd)
        self.drop = nn.Dropout(args.dropout)
        self.layers = [FineBlock(args=args) for _ in range(args.n_layer)]
        self.layernorm_final = nn.LayerNorm(args.n_embd)

        self.lm_heads = [
            nn.Linear(args.n_embd, args.output_vocab_size, bias=False)
            for _ in range(args.n_codes_given, args.n_codes_total)
        ]
        for i in range(self.n_codes_total - args.n_codes_given):
            self.input_embeds_layers[i + 1].weight = self.lm_heads[i].weight

    def __call__(self, pred_idx: mx.array, idx: mx.array) -> mx.array:
        b, t, codes = idx.shape
        assert (
            t <= self.args.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        assert pred_idx > 0, "cannot predict 0th codebook"
        assert codes == self.n_codes_total, (b, t, codes)
        pos = mx.arange(0, t).astype(mx.int64).reshape(1, t)  # shape (1, t)
        tok_embs = [
            self.input_embeds_layers[i](idx[:, :, i].astype(mx.int32)).reshape(
                b, t, -1, 1
            )
            for i in range(self.n_codes_total)
        ]  # token embeddings of shape (b, t, n_embd)
        tok_emb = mx.concatenate(tok_embs, axis=-1)
        pos_emb = self.position_embeds_layer(
            pos
        )  # position embeddings of shape (1, t, n_embd)
        x = tok_emb[:, :, :, : pred_idx + 1].sum(axis=-1)
        x = self.drop(x + pos_emb)
        for block in self.layers:
            x = block(x)
        x = self.layernorm_final(x)

        logits = self.lm_heads[pred_idx - self.args.n_codes_given](x)
        return logits


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Convert config dictionaries to proper configuration objects if needed
        if isinstance(config.semantic_config, dict):
            filtered_config = filter_dataclass_fields(
                config.semantic_config, SemanticConfig
            )
            semantic_config = SemanticConfig(**filtered_config)
        else:
            semantic_config = config.semantic_config

        if isinstance(config.coarse_acoustics_config, dict):
            filtered_config = filter_dataclass_fields(
                config.coarse_acoustics_config, CoarseAcousticsConfig
            )
            coarse_config = CoarseAcousticsConfig(**filtered_config)
        else:
            coarse_config = config.coarse_acoustics_config

        if isinstance(config.fine_acoustics_config, dict):
            filtered_config = filter_dataclass_fields(
                config.fine_acoustics_config, FineAcousticsConfig
            )
            fine_config = FineAcousticsConfig(**filtered_config)
        else:
            fine_config = config.fine_acoustics_config

        self.semantic = GPT(semantic_config)
        self.fine_acoustics = FineGPT(fine_config)
        self.coarse_acoustics = GPT(coarse_config)

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

    def sanitize(self, weights):

        sanitized_weights = {}
        for key, value in weights.items():
            # there's no _orig_mod.transformer
            if "_orig_mod.transformer." in key:
                key = key.replace("_orig_mod.transformer.", "")
            # transformer block mapping
            if "h" in key:
                layer_count = 24 if self.config.model_size == "large" else 12
                for i in range(layer_count):
                    prefix = f"h.{i}."
                    key = key.replace(prefix, f"layers.{i}.")

            # lm_head
            if "lm_head" in key:
                key = key.replace("_orig_mod.", "")

            if "codec" in key:
                pass
            else:
                sanitized_weights[key] = value

        return sanitized_weights

    @property
    def sample_rate(self):
        return self.config.sample_rate

    def generate(self, text: str, voice: str = None, **kwargs):
        pipeline = Pipeline(
            model=self,
            tokenizer=self.tokenizer,
            config=self.config,
        )

        # Track overall generation time
        start_time = time.time()

        for segment_idx, (audio, tokens) in enumerate(
            pipeline(text, voice=voice, use_kv_caching=True, **kwargs)
        ):
            # Track per-segment generation time
            segment_time = time.time() - start_time

            samples = audio.shape[0] if audio is not None else 0
            assert samples > 0, "No audio generated"

            # Calculate token count
            token_count = len(tokens) if tokens is not None else 0

            # Calculate audio duration in seconds
            sample_rate = self.config.sample_rate
            audio_duration_seconds = samples / sample_rate * audio.shape[1]

            # Calculate milliseconds per sample
            ms_per_sample = (
                1000 / sample_rate
            )  # This gives 0.0417 ms per sample at 24kHz

            # Calculate real-time factor (RTF)
            rtf = (
                segment_time / audio_duration_seconds
                if audio_duration_seconds > 0
                else 0
            )

            # Format duration as HH:MM:SS.mmm
            duration_mins = int(audio_duration_seconds // 60)
            duration_secs = int(audio_duration_seconds % 60)
            duration_ms = int((audio_duration_seconds % 1) * 1000)
            duration_hours = int(audio_duration_seconds // 3600)
            duration_str = f"{duration_hours:02d}:{duration_mins:02d}:{duration_secs:02d}.{duration_ms:03d}"

            yield GenerationResult(
                audio=audio[0],
                samples=samples,
                sample_rate=sample_rate,
                segment_idx=segment_idx,
                token_count=token_count,
                audio_duration=duration_str,
                real_time_factor=round(rtf, 2),
                prompt={
                    "tokens": token_count,
                    "tokens-per-sec": (
                        round(token_count / segment_time, 2) if segment_time > 0 else 0
                    ),
                },
                audio_samples={
                    "samples": samples,
                    "samples-per-sec": (
                        round(samples / segment_time, 2) if segment_time > 0 else 0
                    ),
                },
                processing_time_seconds=segment_time,
                peak_memory_usage=mx.get_peak_memory() / 1e9,
            )

            # Clear cache after each segment to avoid memory leaks
            mx.clear_cache()
