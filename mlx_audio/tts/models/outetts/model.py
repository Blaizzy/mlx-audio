import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.generate import stream_generate
from mlx_lm.models.base import BaseModelArgs, create_attention_mask
from mlx_lm.models.rope_utils import initialize_rope
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from tqdm import tqdm
from transformers import AutoTokenizer

from ..base import GenerationResult
from .dac_interface import DacInterface
from .prompt_processor import PromptProcessor


@dataclass
class ModelConfig(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    head_dim: Optional[int] = None
    max_position_embeddings: Optional[int] = None
    num_key_value_heads: Optional[int] = None
    attention_bias: bool = False
    mlp_bias: bool = False
    rope_theta: float = 500000.0
    rope_traditional: bool = False
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    tie_word_embeddings: bool = True
    tokenizer_path: str = "OuteAI/Llama-OuteTTS-1.0-1B"


class Attention(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads

        self.head_dim = head_dim = args.head_dim or args.hidden_size // n_heads

        self.scale = head_dim**-0.5
        if hasattr(args, "attention_bias"):
            attention_bias = args.attention_bias
        else:
            attention_bias = False

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=attention_bias)

        self.rope = initialize_rope(
            self.head_dim,
            args.rope_theta,
            args.rope_traditional,
            args.rope_scaling,
            args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()

        dim = args.hidden_size
        hidden_dim = args.intermediate_size
        if hasattr(args, "mlp_bias"):
            mlp_bias = args.mlp_bias
        else:
            mlp_bias = False

        self.gate_proj = nn.Linear(dim, hidden_dim, bias=mlp_bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=mlp_bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=mlp_bias)

    def __call__(self, x) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        self.self_attn = Attention(args)
        self.mlp = MLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.args = args

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out


class LlamaModel(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        assert self.vocab_size > 0
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args=args) for _ in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        mask: mx.array = None,
        cache=None,
    ):
        h = self.embed_tokens(inputs)

        if mask is None:
            mask = create_attention_mask(h, cache)

        if cache is None:
            cache = [None] * len(self.layers)

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, cache=c)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, config: ModelConfig, **kwargs):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
        self.model = LlamaModel(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        mask: mx.array = None,
        cache=None,
    ):
        out = self.model(inputs, mask, cache)
        if self.config.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    def sanitize(self, weights):
        return weights

    @property
    def layers(self):
        return self.model.layers

    def generate(
        self,
        text,
        voice: Optional[str] = None,
        temperature: float = 0.4,
        top_p: float = 0.9,
        split_pattern: str = "\n",
        max_tokens: int = 1200,
        verbose: bool = False,
        **kwargs,
    ):
        prompt = text.replace("\\n", "\n").replace("\\t", "\t")
        prompts = prompt.split(split_pattern)

        self.prompt_processor = PromptProcessor(self.config.tokenizer_path)
        self.audio_codec = DacInterface()

        if voice is None:
            voice = f"{Path(__file__).parent}/default_speaker.json"

        with open(voice, "r") as f:
            speaker = json.load(f)

        sampler = make_sampler(
            temperature,
            top_p,
            min_p=kwargs.get("min_p", 0.05),
            top_k=kwargs.get("top_k", 40),
        )
        logits_processors = make_logits_processors(
            kwargs.get("logit_bias", None),
            kwargs.get("repetition_penalty", 1.1),
            kwargs.get("repetition_context_size", 64),
        )

        all_audio = []

        for prompt in prompts:
            completion_prompt = self.prompt_processor.get_completion_prompt(
                prompt, speaker
            )
            input_ids = self.tokenizer.encode(
                completion_prompt, add_special_tokens=False, return_tensors="mlx"
            )
            input_length = input_ids.shape[1]

            time_start = time.time()

            for i, response in enumerate(
                tqdm(
                    stream_generate(
                        self,
                        tokenizer=self.tokenizer,
                        prompt=input_ids.squeeze(0),
                        max_tokens=max_tokens,
                        sampler=sampler,
                        logits_processors=logits_processors,
                    ),
                    total=max_tokens,
                    disable=not verbose,
                )
            ):
                next_token = mx.array([response.token])
                input_ids = mx.concatenate([input_ids, next_token[None, :]], axis=1)

            output_ids = input_ids[:, input_length:].tolist()[0]
            output = self.prompt_processor.extract_audio_from_tokens(output_ids)
            audio = self.audio_codec.decode(mx.array([output])).squeeze(0)
            all_audio.append(audio)

        time_end = time.time()

        for i in range(len(all_audio)):
            audio = all_audio[i][0]

            samples = audio.shape[0] if audio is not None else 0
            assert samples > 0, "No audio generated"

            token_count = input_ids.shape[1] if input_ids is not None else 0

            sample_rate = 24000
            audio_duration_seconds = samples / sample_rate

            elapsed_time = time_end - time_start
            rtf = audio_duration_seconds / elapsed_time

            duration_mins = int(audio_duration_seconds // 60)
            duration_secs = int(audio_duration_seconds % 60)
            duration_ms = int((audio_duration_seconds % 1) * 1000)
            duration_hours = int(audio_duration_seconds // 3600)
            duration_str = f"{duration_hours:02d}:{duration_mins:02d}:{duration_secs:02d}.{duration_ms:03d}"

            yield GenerationResult(
                audio=audio,
                samples=samples,
                segment_idx=i,
                token_count=token_count,
                audio_duration=duration_str,
                real_time_factor=rtf,
                prompt={
                    "tokens": token_count,
                    "tokens-per-sec": (
                        round(token_count / elapsed_time, 2) if elapsed_time > 0 else 0
                    ),
                },
                audio_samples={
                    "samples": samples,
                    "samples-per-sec": (
                        round(samples / elapsed_time, 2) if elapsed_time > 0 else 0
                    ),
                },
                processing_time_seconds=time_end - time_start,
                peak_memory_usage=mx.get_peak_memory() / 1e9,
            )
