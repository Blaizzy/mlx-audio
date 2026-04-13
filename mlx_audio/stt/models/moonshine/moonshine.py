import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from ..base import STTOutput
from .config import ModelConfig


class MoonshineRotaryEmbedding(nn.Module):
    def __init__(
        self, dim: int, max_position_embeddings: int = 512, base: float = 10000.0
    ):
        super().__init__()
        inv_freq = 1.0 / (base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
        self._inv_freq = inv_freq  # shape: (dim // 2,)
        self._dim = dim
        self._max_seq_len = max_position_embeddings

    def __call__(
        self, x: mx.array, position_ids: mx.array
    ) -> Tuple[mx.array, mx.array]:
        freqs = (
            position_ids[:, :, None].astype(mx.float32) * self._inv_freq[None, None, :]
        )
        emb = mx.concatenate([freqs, freqs], axis=-1)
        cos = mx.cos(emb)
        sin = mx.sin(emb)
        return cos, sin


def rotate_half(x: mx.array) -> mx.array:
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    rotated = mx.stack([-x2, x1], axis=-1)
    return rotated.reshape(x.shape)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = mx.expand_dims(cos, axis=unsqueeze_dim)
    sin = mx.expand_dims(sin, axis=unsqueeze_dim)
    half = cos.shape[-1] // 2
    cos = mx.repeat(cos[..., :half], 2, axis=-1)
    sin = mx.repeat(sin[..., :half], 2, axis=-1)

    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    q_embed = mx.concatenate([q_embed, q_pass], axis=-1)
    k_embed = mx.concatenate([k_embed, k_pass], axis=-1)
    return q_embed, k_embed


class MoonshineAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        bias: bool = False,
        is_causal: bool = False,
        partial_rotary_factor: float = 0.9,
        max_position_embeddings: int = 512,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.num_kv_groups = num_heads // num_kv_heads
        self.is_causal = is_causal
        self.scale = self.head_dim**-0.5

        rotary_ndims = int(self.head_dim * partial_rotary_factor)
        self.rotary_ndims = rotary_ndims - (rotary_ndims % 2)

        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)

        self.rotary_emb = MoonshineRotaryEmbedding(
            self.rotary_ndims,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
        )

    def __call__(
        self,
        x: mx.array,
        encoder_hidden_states: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
        position_ids: Optional[mx.array] = None,
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        B, T, _ = x.shape
        is_cross_attention = encoder_hidden_states is not None

        q = self.q_proj(x)
        if is_cross_attention:
            k = self.k_proj(encoder_hidden_states)
            v = self.v_proj(encoder_hidden_states)
        else:
            k = self.k_proj(x)
            v = self.v_proj(x)

        q = q.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        S = k.shape[1]
        k = k.reshape(B, S, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, S, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        if not is_cross_attention:
            if position_ids is None:
                offset = cache[0].shape[2] if cache is not None else 0
                position_ids = mx.arange(offset, offset + T)[None, :]

            cos, sin = self.rotary_emb(q, position_ids)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if cache is not None:
            if is_cross_attention:
                k, v = cache
            else:
                prev_k, prev_v = cache
                k = mx.concatenate([prev_k, k], axis=2)
                v = mx.concatenate([prev_v, v], axis=2)

        if self.num_kv_groups > 1:
            k = mx.repeat(k, self.num_kv_groups, axis=1)
            v = mx.repeat(v, self.num_kv_groups, axis=1)

        mask = None
        if self.is_causal and T > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(T)
            if k.shape[2] > T:
                prefix = mx.zeros((T, k.shape[2] - T))
                mask = mx.concatenate([prefix, mask], axis=1)

        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        o = o.transpose(0, 2, 1, 3).reshape(B, T, -1)

        return self.o_proj(o), (k, v)


class MoonshineEncoderMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, act: str = "gelu"):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.act = nn.GELU() if act == "gelu" else nn.SiLU()

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(self.act(self.fc1(x)))


class MoonshineDecoderMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, 2 * intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.fc1(x)
        x, gate = mx.split(x, 2, axis=-1)
        return self.fc2(nn.silu(gate) * x)


class MoonshineEncoderLayer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.self_attn = MoonshineAttention(
            hidden_size=config.hidden_size,
            num_heads=config.encoder_num_attention_heads,
            num_kv_heads=config.encoder_num_key_value_heads,
            bias=config.attention_bias,
            is_causal=False,
            partial_rotary_factor=config.partial_rotary_factor,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
        )
        self.mlp = MoonshineEncoderMLP(
            config.hidden_size, config.intermediate_size, config.encoder_hidden_act
        )
        self.input_layernorm = nn.LayerNorm(config.hidden_size, bias=False)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, bias=False)

    def __call__(
        self, x: mx.array, position_ids: Optional[mx.array] = None
    ) -> mx.array:
        residual = x
        x = self.input_layernorm(x)
        x, _ = self.self_attn(x, position_ids=position_ids)
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return x


class MoonshineDecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.self_attn = MoonshineAttention(
            hidden_size=config.hidden_size,
            num_heads=config.decoder_num_attention_heads,
            num_kv_heads=config.decoder_num_key_value_heads,
            bias=config.attention_bias,
            is_causal=True,
            partial_rotary_factor=config.partial_rotary_factor,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
        )
        self.encoder_attn = MoonshineAttention(
            hidden_size=config.hidden_size,
            num_heads=config.decoder_num_attention_heads,
            num_kv_heads=config.decoder_num_key_value_heads,
            bias=config.attention_bias,
            is_causal=False,
            partial_rotary_factor=config.partial_rotary_factor,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
        )
        self.mlp = MoonshineDecoderMLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = nn.LayerNorm(config.hidden_size, bias=False)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, bias=False)
        self.final_layernorm = nn.LayerNorm(config.hidden_size, bias=False)

    def __call__(
        self,
        x: mx.array,
        encoder_hidden_states: mx.array,
        self_attn_cache: Optional[Tuple[mx.array, mx.array]] = None,
        cross_attn_cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array], Tuple[mx.array, mx.array]]:
        residual = x
        x = self.input_layernorm(x)
        x, new_self_cache = self.self_attn(x, cache=self_attn_cache)
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x, new_cross_cache = self.encoder_attn(
            x, encoder_hidden_states=encoder_hidden_states, cache=cross_attn_cache
        )
        x = residual + x

        residual = x
        x = self.final_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return x, new_self_cache, new_cross_cache


class MoonshineEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        dim = config.hidden_size
        self.conv1 = nn.Conv1d(1, dim, kernel_size=127, stride=64, bias=False)
        self.groupnorm = nn.GroupNorm(1, dim)
        self.conv2 = nn.Conv1d(dim, 2 * dim, kernel_size=7, stride=3, bias=True)
        self.conv3 = nn.Conv1d(2 * dim, dim, kernel_size=3, stride=2, bias=True)
        self.layers = [
            MoonshineEncoderLayer(config)
            for _ in range(config.encoder_num_hidden_layers)
        ]
        self.layer_norm = nn.LayerNorm(dim, bias=False)

    def __call__(self, audio: mx.array) -> mx.array:
        if audio.ndim == 1:
            audio = audio[None, :]
        x = audio[:, :, None]
        x = mx.tanh(self.conv1(x))
        x = self.groupnorm(x)
        x = nn.gelu(self.conv2(x))
        x = nn.gelu(self.conv3(x))

        position_ids = mx.arange(x.shape[1])[None, :]
        for layer in self.layers:
            x = layer(x, position_ids=position_ids)

        return self.layer_norm(x)


class MoonshineDecoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            MoonshineDecoderLayer(config)
            for _ in range(config.decoder_num_hidden_layers)
        ]
        self.norm = nn.LayerNorm(config.hidden_size, bias=False)

    def __call__(
        self,
        tokens: mx.array,
        encoder_hidden_states: mx.array,
        cache: Optional[List[dict]] = None,
    ) -> Tuple[mx.array, List[dict]]:
        x = self.embed_tokens(tokens)

        if cache is None:
            cache = [
                {"self_attn": None, "cross_attn": None} for _ in range(len(self.layers))
            ]

        new_cache = []
        for i, layer in enumerate(self.layers):
            x, new_self, new_cross = layer(
                x,
                encoder_hidden_states,
                self_attn_cache=cache[i]["self_attn"],
                cross_attn_cache=cache[i]["cross_attn"],
            )
            new_cache.append({"self_attn": new_self, "cross_attn": new_cross})

        return self.norm(x), new_cache


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if isinstance(config, dict):
            config = ModelConfig.from_dict(config)
        self.config = config
        self.encoder = MoonshineEncoder(config)
        self.decoder = MoonshineDecoder(config)
        if not config.tie_word_embeddings:
            self.proj_out = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self._tokenizer = None

    @property
    def sample_rate(self) -> int:
        return 16000

    def _get_logits(self, hidden_states: mx.array) -> mx.array:
        if self.config.tie_word_embeddings:
            return self.decoder.embed_tokens.as_linear(hidden_states)
        return self.proj_out(hidden_states)

    def generate(
        self,
        audio,
        *,
        max_tokens: int = 200,
        temperature: float = 0.0,
        verbose: bool = False,
        stream: bool = False,
        dtype: mx.Dtype = mx.float32,
        **kwargs,
    ) -> STTOutput:
        kwargs.pop("generation_stream", None)
        kwargs.pop("language", None)
        kwargs.pop("source_lang", None)
        kwargs.pop("target_lang", None)

        start_time = time.time()

        if isinstance(audio, (str, Path)):
            from mlx_audio.stt.utils import load_audio

            audio = load_audio(str(audio), sr=self.sample_rate, dtype=dtype)
        elif not isinstance(audio, mx.array):
            audio = mx.array(audio)

        if audio.dtype != dtype:
            audio = audio.astype(dtype)

        encoder_out = self.encoder(audio)
        mx.eval(encoder_out)

        tokens = [self.config.decoder_start_token_id]
        cache = None

        for _ in range(max_tokens):
            token_ids = mx.array([[tokens[-1]]], dtype=mx.int32)
            hidden, cache = self.decoder(token_ids, encoder_out, cache=cache)
            mx.eval(hidden)

            logits = self._get_logits(hidden[:, -1, :])

            if temperature > 0:
                next_token = int(mx.random.categorical(logits / temperature))
            else:
                next_token = int(logits.argmax())

            if next_token == self.config.eos_token_id:
                break
            tokens.append(next_token)

        generated = tokens[1:]
        text = self._decode_tokens(generated)

        end_time = time.time()
        total_time = end_time - start_time

        if verbose:
            print(f"Generated {len(generated)} tokens in {total_time:.2f}s")
            print(f"Text: {text}")

        return STTOutput(
            text=text.strip(),
            segments=[{"text": text.strip(), "start": 0.0, "end": 0.0}],
            prompt_tokens=1,
            generation_tokens=len(generated),
            total_tokens=1 + len(generated),
            total_time=total_time,
            prompt_tps=1 / total_time if total_time > 0 else 0,
            generation_tps=len(generated) / total_time if total_time > 0 else 0,
        )

    def _decode_tokens(self, tokens: List[int]) -> str:
        if self._tokenizer is not None:
            return self._tokenizer.decode(tokens, skip_special_tokens=True)
        return "".join(chr(t) if t < 128 else f"<{t}>" for t in tokens)

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        sanitized = {}

        for key, value in weights.items():
            new_key = key

            if key.startswith("model.encoder."):
                new_key = key[len("model.") :]

            elif key.startswith("model.decoder."):
                new_key = key[len("model.") :]

            elif key.startswith("proj_out."):
                if self.config.tie_word_embeddings:
                    continue
                new_key = key

            else:
                new_key = key

            if "conv" in new_key and "weight" in new_key and value.ndim == 3:
                value = mx.transpose(value, (0, 2, 1))

            sanitized[new_key] = value

        return sanitized

    @classmethod
    def post_load_hook(cls, model: "Model", model_path: Path) -> "Model":
        model_path = Path(model_path)
        try:
            from transformers import AutoTokenizer

            model._tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        except Exception:
            pass
        return model
