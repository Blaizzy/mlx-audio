
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Dict
from mlx.utils import tree_unflatten, tree_map

from enum import Enum
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import math
from transformers import BertTokenizer
import glob
import time
import tqdm
import math
from scipy.io.wavfile import write as write_wav
from encodec import EncodecModel
from .pipeline import Pipeline
from ..base import GenerationResult, BaseModelArgs

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


@dataclass
class ModelConfig(BaseModelArgs):
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



model_args = {
    "bark-coarse": ModelConfig(input_vocab_size=12096, output_vocab_size=12096),
    "bark-fine": ModelConfig(
        n_codes_given=1, n_codes_total=8, input_vocab_size=1056, output_vocab_size=1056
    ),
    "bark-text": ModelConfig(
        input_vocab_size=129600,
        output_vocab_size=10048,
    ),
    "bark-coarse-large": ModelConfig(
        n_layer=24,
        n_head=16,
        n_embd=1024,
        input_vocab_size=12096,
        output_vocab_size=12096,
    ),
    "bark-fine-large": ModelConfig(
        n_codes_given=1,
        n_codes_total=8,
        n_layer=24,
        n_head=16,
        n_embd=1024,
        input_vocab_size=1056,
        output_vocab_size=1056,
    ),
    "bark-text-large": ModelConfig(
        n_layer=24,
        n_head=16,
        n_embd=1024,
        input_vocab_size=129600,
        output_vocab_size=10048,
    ),
}


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
    def __init__(self, args: ModelConfig):
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
        att = (query @ key.transpose(0, 1, 3, 2)) * (1.0 / math.sqrt(key.shape[3]))

        att = mx.where(
            self.bias[:, :, FULL_T - T : FULL_T, :FULL_T] == 0, float("-1e9"), att
        )
        att = mx.softmax(att.astype(mx.float32), axis=-1).astype(att.dtype)
        att = self.attn_dropout(att)
        y = (att @ value).transpose(0, 2, 1, 3).reshape(B, T, C)
        y = self.resid_dropout(self.out_proj(y))
        return (y, present)


class NonCausalSelfAttention(nn.Module):
    def __init__(self, args: ModelConfig):
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

        att = (query @ key.transpose(0, 1, 3, 2)) * (1.0 / math.sqrt(key.shape[3]))
        att = mx.softmax(att.astype(mx.float32), axis=-1).astype(att.dtype)
        att = self.attn_dropout(att)
        y = (att @ value).transpose(0, 2, 1, 3).reshape(B, T, C)
        y = self.resid_dropout(self.out_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, args: ModelConfig):
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
    def __init__(self, args: ModelConfig, layer_idx: int = 0):
        super().__init__()
        self.args = args
        self.layernorm_1 = LayerNorm(args.n_embd, bias=True)
        self.attn = CausalSelfAttention(args)
        self.layernorm_2 = LayerNorm(args.n_embd, bias=True)
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
    def __init__(self, args: ModelConfig):
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
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.args = args
        self.wte = nn.Embedding(args.input_vocab_size, args.n_embd)
        self.wpe = nn.Embedding(args.block_size, args.n_embd)
        self.drop = nn.Dropout(args.dropout)
        self.layers = [Block(args=args) for _ in range(args.n_layer)]
        self.ln_f = LayerNorm(args.n_embd, bias=True)
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
            tok_emb = self.wte(x)
        else:
            if merge_context:
                assert x.shape[1] >= 256 + 256 + 1
                t = x.shape[1] - 256
                tok_emb = mx.concatenate(
                    [
                        self.wte(x[:, :256]) + self.wte(x[:, 256 : 256 + 256]),
                        self.wte(x[:, 256 + 256 :]),
                    ],
                    axis=1,
                )
            else:
                tok_emb = self.wte(x)

        # past length
        if past_kv is None:
            past_length = 0
            past_kv = tuple([None] * len(self.layers))
        else:
            past_length = past_kv[0][0].shape[-2]

        if position_ids is None:
            position_ids = mx.arange(past_length, t + past_length)
            position_ids = position_ids.reshape(1, -1)  # shape (1, t)

        pos_emb = self.wpe(position_ids)  # position embeddings of shape (1, t, n_embd)
        x = self.drop(tok_emb + pos_emb)

        new_kv = () if use_cache else None

        for i, (block, past_layer_kv) in enumerate(zip(self.layers, past_kv)):
            x, kv = block(x, past_kv=past_layer_kv, use_cache=use_cache)

            if use_cache:
                new_kv = new_kv + (kv,)

        x = self.ln_f(x)

        logits = self.lm_head(
            x[:, -1:, :]
        )  # note: using list [-1] to preserve the time dim

        return (logits, new_kv)


class FineGPT(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.args = args
        self.n_codes_total = args.n_codes_total
        self.input_embeds_layers = [
            nn.Embedding(args.input_vocab_size, args.n_embd)
            for _ in range(args.n_codes_total)
        ]
        self.wpe = nn.Embedding(args.block_size, args.n_embd)
        self.drop = nn.Dropout(args.dropout)
        self.layers = [FineBlock(args=args) for _ in range(args.n_layer)]
        self.ln_f = nn.LayerNorm(args.n_embd)
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
            wte(idx[:, :, i].astype(mx.int32)).reshape(b, t, -1, 1)
            for i, wte in enumerate(self.wtes)
        ]  # token embeddings of shape (b, t, n_embd)
        tok_emb = mx.concatenate(tok_embs, axis=-1)
        pos_emb = self.wpe(pos)  # position embeddings of shape (1, t, n_embd)
        x = tok_emb[:, :, :, : pred_idx + 1].sum(axis=-1)
        x = self.drop(x + pos_emb)
        for block in self.layers:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_heads[pred_idx - self.args.n_codes_given](x)
        return logits

class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # if config.model_size == "large":
        #     self.semantic = GPT(model_args["bark-text-large"])
        #     self.fine_acoustics = FineGPT(model_args["bark-fine-large"])
        #     self.bark_coarse = GPT(model_args["bark-coarse-large"])
        # else:
        self.semantic = GPT(model_args["bark-text"])
        self.fine_acoustics = FineGPT(model_args["bark-fine"])
        self.coarse_acoustics = GPT(model_args["bark-coarse"])

        self.codec_model = EncodecModel.encodec_model_24khz()
        self.codec_model.eval()
        self.codec_model.to("cpu")
        self.codec_model.set_target_bandwidth(6.0)

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


    def generate(
        self,
        text: str,
        verbose: bool = False,
        **kwargs
    ):
        pipeline = Pipeline(
            model=self,
            tokenizer=self.tokenizer,
        )

        # Track overall generation time
        start_time = time.time()

        metrics = []

        for segment_idx, (audio, phonemes) in enumerate(
            pipeline(text, **kwargs)
        ):
            # Track per-segment generation time
            segment_time = time.time() - start_time

            samples = audio.shape[0] if audio is not None else 0
            assert samples > 0, "No audio generated"

            # Calculate token count
            token_count = len(phonemes) if phonemes is not None else 0

            # Calculate audio duration in seconds
            sample_rate = 24000  # Assuming 24kHz sample rate, adjust if different
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
                peak_memory_usage=mx.metal.get_peak_memory() / 1e9,
            )


