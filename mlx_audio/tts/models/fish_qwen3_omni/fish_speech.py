from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import KVCache

from mlx_audio.tts.models.base import BatchGenerationResult, GenerationResult

from .config import ModelConfig
from .prompt import (
    Conversation,
    Message,
    TextPart,
    VQPart,
    group_turns_into_batches,
    split_text_by_speaker,
)
from .tokenizer import IM_END_TOKEN, FishTokenizer

RAS_WIN_SIZE = 10
RAS_HIGH_TEMP = 1.0
RAS_HIGH_TOP_P = 0.9


@dataclass
class ForwardResult:
    logits: mx.array
    hidden_states: mx.array


class Identity(nn.Module):
    def __call__(self, x):
        return x


class FishRotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        rope_base: float,
        max_position_embeddings: int,
    ):
        super().__init__()
        freqs = 1.0 / (
            rope_base
            ** (mx.arange(0, head_dim, 2, dtype=mx.float32)[: head_dim // 2] / head_dim)
        )
        positions = mx.arange(max_position_embeddings, dtype=mx.float32)
        angles = positions[:, None] * freqs[None, :]

        # Match the reference implementation, which caches RoPE phases in bf16.
        self._cos = mx.cos(angles).astype(mx.bfloat16)
        self._sin = mx.sin(angles).astype(mx.bfloat16)

    def __call__(self, x: mx.array, offset: int = 0) -> mx.array:
        seqlen = x.shape[2]
        cos = self._cos[offset : offset + seqlen][None, None, :, :]
        sin = self._sin[offset : offset + seqlen][None, None, :, :]

        x_float = x.astype(mx.float32).reshape(*x.shape[:-1], -1, 2)
        x_even = x_float[..., 0]
        x_odd = x_float[..., 1]
        rotated = mx.stack(
            [
                x_even * cos - x_odd * sin,
                x_odd * cos + x_even * sin,
            ],
            axis=-1,
        )
        return rotated.reshape(x.shape).astype(x.dtype)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        rope_base: float,
        max_position_embeddings: int,
        attention_qkv_bias: bool,
        attention_o_bias: bool,
        attention_qk_norm: bool,
        norm_eps: float,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        total_head_dim = (n_heads + 2 * n_kv_heads) * head_dim

        self.wqkv = nn.Linear(dim, total_head_dim, bias=attention_qkv_bias)
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=attention_o_bias)
        self.q_norm = (
            nn.RMSNorm(head_dim, eps=norm_eps) if attention_qk_norm else Identity()
        )
        self.k_norm = (
            nn.RMSNorm(head_dim, eps=norm_eps) if attention_qk_norm else Identity()
        )
        self.rope = FishRotaryEmbedding(
            head_dim=head_dim,
            rope_base=rope_base,
            max_position_embeddings=max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Union[None, str, mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        bsz, seqlen, _ = x.shape
        q_size = self.n_heads * self.head_dim
        kv_size = self.n_kv_heads * self.head_dim

        qkv = self.wqkv(x)
        q, k, v = mx.split(qkv, [q_size, q_size + kv_size], axis=-1)

        q = q.reshape(bsz, seqlen, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(bsz, seqlen, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(bsz, seqlen, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if cache is not None:
            q = self.rope(q, offset=cache.offset)
            k = self.rope(k, offset=cache.offset)
            k, v = cache.update_and_fetch(k, v)
        else:
            q = self.rope(q)
            k = self.rope(k)

        output = mx.fast.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=self.scale,
            mask=mask,
        )
        output = output.transpose(0, 2, 1, 3).reshape(bsz, seqlen, -1)
        return self.wo(output)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        rope_base: float,
        max_position_embeddings: int,
        attention_qkv_bias: bool,
        attention_o_bias: bool,
        attention_qk_norm: bool,
        norm_eps: float,
    ):
        super().__init__()
        self.attention = Attention(
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            rope_base=rope_base,
            max_position_embeddings=max_position_embeddings,
            attention_qkv_bias=attention_qkv_bias,
            attention_o_bias=attention_o_bias,
            attention_qk_norm=attention_qk_norm,
            norm_eps=norm_eps,
        )
        self.feed_forward = FeedForward(dim, intermediate_size)
        self.attention_norm = nn.RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = nn.RMSNorm(dim, eps=norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        h = x + self.attention(self.attention_norm(x), mask=mask, cache=cache)
        return h + self.feed_forward(self.ffn_norm(h))


class DualARTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        tc = config.text_config
        ac = config.audio_decoder_config

        self.embeddings = nn.Embedding(tc.vocab_size, tc.dim)
        self.codebook_embeddings = nn.Embedding(
            ac.vocab_size * ac.num_codebooks, tc.dim
        )
        self.layers = [
            TransformerBlock(
                dim=tc.dim,
                n_heads=tc.n_head,
                n_kv_heads=tc.n_local_heads,
                head_dim=tc.head_dim,
                intermediate_size=tc.intermediate_size,
                rope_base=tc.rope_base,
                max_position_embeddings=tc.max_seq_len,
                attention_qkv_bias=tc.attention_qkv_bias,
                attention_o_bias=tc.attention_o_bias,
                attention_qk_norm=tc.attention_qk_norm,
                norm_eps=tc.norm_eps,
            )
            for _ in range(tc.n_layer)
        ]
        self.norm = nn.RMSNorm(tc.dim, eps=tc.norm_eps)

        self.fast_project_in = (
            nn.Linear(tc.dim, ac.dim, bias=False) if tc.dim != ac.dim else Identity()
        )
        self.fast_embeddings = nn.Embedding(ac.vocab_size, ac.dim)
        self.fast_layers = [
            TransformerBlock(
                dim=ac.dim,
                n_heads=ac.n_head,
                n_kv_heads=ac.n_local_heads,
                head_dim=ac.head_dim,
                intermediate_size=ac.intermediate_size,
                rope_base=ac.rope_base,
                max_position_embeddings=ac.num_codebooks,
                attention_qkv_bias=ac.attention_qkv_bias,
                attention_o_bias=ac.attention_o_bias,
                attention_qk_norm=ac.attention_qk_norm,
                norm_eps=ac.norm_eps,
            )
            for _ in range(ac.n_layer)
        ]
        self.fast_norm = nn.RMSNorm(ac.dim, eps=ac.norm_eps)
        self.fast_output = nn.Linear(ac.dim, ac.vocab_size, bias=False)

    @property
    def num_codebooks(self) -> int:
        return self.config.audio_decoder_config.num_codebooks

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.layers]

    def make_fast_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.fast_layers]

    def _embed(self, inp: mx.array) -> mx.array:
        semantic_ids = inp[:, 0]
        codebook_rows = inp[:, 1:]
        vq_embeds = []
        for i in range(self.num_codebooks):
            vq_embeds.append(
                self.codebook_embeddings(
                    codebook_rows[:, i]
                    + i * self.config.audio_decoder_config.vocab_size
                )
            )
        vq_sum = mx.stack(vq_embeds, axis=0).sum(axis=0)
        semantic_mask = (semantic_ids >= self.config.semantic_start_token_id) & (
            semantic_ids <= self.config.semantic_end_token_id
        )
        vq_sum = mx.where(semantic_mask[:, :, None], vq_sum, 0)
        x = self.embeddings(semantic_ids) + vq_sum
        scale = math.sqrt(self.num_codebooks + 1)
        return mx.where(semantic_mask[:, :, None], x / scale, x)

    def __call__(
        self,
        inp: mx.array,
        cache: Optional[list[KVCache]] = None,
        attention_mask: Optional[mx.array] = None,
    ) -> ForwardResult:
        x = self._embed(inp)
        if cache is None:
            cache = [None] * len(self.layers)

        if attention_mask is not None:
            seq_len = x.shape[1]
            total_len = attention_mask.shape[-1]
            if seq_len > 1:
                if total_len == seq_len:
                    causal = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
                else:
                    query_pos = mx.arange(total_len - seq_len, total_len)[:, None]
                    key_pos = mx.arange(total_len)[None, :]
                    causal = mx.where(key_pos > query_pos, -1e9, 0.0)
                causal = causal.astype(x.dtype)[None, None, :, :]
                pad_mask = (1 - attention_mask[:, None, None, :].astype(x.dtype)) * -1e9
                mask = causal + pad_mask
            else:
                mask = (1 - attention_mask[:, None, None, :].astype(x.dtype)) * -1e9
        else:
            mask = create_attention_mask(x, cache[0])

        for layer, layer_cache in zip(self.layers, cache):
            x = layer(x, mask=mask, cache=layer_cache)
        slow_out = self.norm(x)
        logits = self.embeddings.as_linear(slow_out)
        return ForwardResult(
            logits=logits, hidden_states=self.fast_project_in(slow_out)
        )

    def fast_forward(
        self, hidden_state: mx.array, previous_codebooks: mx.array
    ) -> mx.array:
        if hidden_state.ndim == 3:
            hidden_state = hidden_state[:, -1]
        hidden_state = hidden_state[:, None, :]
        if previous_codebooks.size > 0:
            codebook_embeddings = self.fast_embeddings(previous_codebooks)
            x = mx.concatenate([hidden_state, codebook_embeddings], axis=1)
        else:
            x = hidden_state
        mask = create_attention_mask(x, None)
        for layer in self.fast_layers:
            x = layer(x, mask=mask, cache=None)
        x = self.fast_norm(x)
        logits = self.fast_output(x[:, -1])
        return logits

    def fast_forward_cached(
        self, x: mx.array, cache: Optional[list[KVCache]] = None
    ) -> mx.array:
        if x.ndim == 2:
            x = x[:, None, :]
        elif x.ndim == 3:
            x = x[:, -1:]

        if cache is None:
            cache = [None] * len(self.fast_layers)
        mask = create_attention_mask(x, cache[0])
        for layer, layer_cache in zip(self.fast_layers, cache):
            x = layer(x, mask=mask, cache=layer_cache)
        x = self.fast_norm(x)
        return self.fast_output(x[:, -1])


@mx.compile
def _sample_logits(
    logits: mx.array, temperature: float, top_p: float, top_k: int
) -> mx.array:
    if temperature <= 0:
        return mx.argmax(logits, axis=-1).astype(mx.int32)

    vocab_size = logits.shape[-1]
    if top_k <= 0 or top_k > vocab_size:
        top_k = vocab_size

    sorted_indices = mx.argsort(-logits, axis=-1)
    sorted_logits = mx.take_along_axis(logits, sorted_indices, axis=-1)
    cum_probs = mx.cumsum(mx.softmax(sorted_logits, axis=-1), axis=-1)

    rank_indices = mx.arange(vocab_size, dtype=sorted_indices.dtype)
    if sorted_logits.ndim > 1:
        rank_indices = mx.broadcast_to(rank_indices, sorted_logits.shape)
    tokens_to_remove = (cum_probs > top_p) | (rank_indices >= top_k)
    tokens_to_remove[..., 0] = False

    inverse_indices = mx.put_along_axis(
        mx.zeros_like(sorted_indices),
        sorted_indices,
        mx.arange(vocab_size, dtype=sorted_indices.dtype),
        axis=-1,
    )
    tokens_to_remove = mx.take_along_axis(tokens_to_remove, inverse_indices, axis=-1)
    filtered_logits = mx.where(tokens_to_remove, -mx.inf, logits).astype(mx.float32)
    probs = mx.softmax(filtered_logits * (1.0 / max(temperature, 1e-5)), axis=-1)
    noise = -mx.log(mx.random.uniform(shape=probs.shape, low=1e-6, high=1.0))
    return mx.argmax(probs / noise, axis=-1).astype(mx.int32)


def _format_duration(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def _adjust_speed(audio: mx.array, speed: float) -> mx.array:
    if abs(speed - 1.0) < 1e-6:
        return audio
    old_length = int(audio.shape[0])
    new_length = max(1, int(old_length / speed))
    positions = mx.linspace(0, old_length - 1, new_length)
    left = mx.floor(positions).astype(mx.int32)
    right = mx.minimum(left + 1, old_length - 1)
    right_weight = positions - left
    left_weight = 1.0 - right_weight
    return left_weight * audio[left] + right_weight * audio[right]


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model = DualARTransformer(config)
        self.tokenizer: Optional[FishTokenizer] = None
        self.codec = None
        self.semantic_logit_bias: Optional[mx.array] = None

    @property
    def sample_rate(self) -> int:
        return self.config.sample_rate

    @property
    def model_type(self) -> str:
        return "fish_speech"

    def load_weights(self, weights, strict: bool = True):
        remapped = []
        for key, value in weights:
            if key.startswith("model."):
                key = key[len("model.") :]
            remapped.append((key, value))
        return self.model.load_weights(remapped, strict=strict)

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        remapped = {}
        for key, value in weights.items():
            if key.startswith("model."):
                # Already in MLX format (e.g., previously converted/quantized model)
                remapped[key] = value
            elif key.startswith("text_model.model."):
                new_key = key[len("text_model.model.") :]
                remapped[f"model.{new_key}"] = value
            elif key.startswith("audio_decoder."):
                suffix = key[len("audio_decoder.") :]
                if suffix.startswith("codebook_embeddings."):
                    new_key = suffix
                else:
                    new_key = f"fast_{suffix}"
                remapped[f"model.{new_key}"] = value
        return remapped

    def _build_conversation(
        self, prompt_texts: list[str], prompt_tokens: list[mx.array]
    ) -> Conversation:
        conversation = Conversation()
        if prompt_texts and prompt_tokens:
            tagged_prompt_texts = []
            for idx, text in enumerate(prompt_texts):
                if "<|speaker:" in text:
                    tagged_prompt_texts.append(text)
                else:
                    tagged_prompt_texts.append(f"<|speaker:{idx}|>{text}")
            system_parts = [
                TextPart(
                    "convert the provided text to speech reference to the following:\n\nText:\n"
                ),
                TextPart("\n".join(tagged_prompt_texts)),
                TextPart("\n\nSpeech:\n"),
                VQPart(mx.concatenate(prompt_tokens, axis=1)),
            ]
        else:
            system_parts = [TextPart("convert the provided text to speech")]

        conversation.append(
            Message(
                role="system",
                parts=system_parts,
                add_im_start=True,
                add_im_end=True,
            )
        )
        return conversation

    def _prepare_reference_prompt(
        self, ref_audio: Optional[mx.array], ref_text: Optional[str]
    ) -> tuple[list[str], list[mx.array]]:
        prompt_tokens = []
        prompt_texts = []
        if ref_audio is not None:
            if self.codec is None:
                raise ValueError("Codec not loaded. Call post_load_hook first.")

            audio = ref_audio
            if audio.ndim == 1:
                audio = audio[None, None, :]
            elif audio.ndim == 2:
                audio = audio[None, :, :]
            if audio.shape[1] != 1:
                audio = mx.mean(audio, axis=1, keepdims=True)
            indices, feature_lengths = self.codec.encode(audio)
            prompt_length = int(feature_lengths[0].item())
            prompt_tokens.append(indices[0, :, :prompt_length])
            prompt_texts.append(ref_text or "")

        return prompt_texts, prompt_tokens

    def _split_generation_text(self, text: str, chunk_length: int) -> list[str]:
        turns = split_text_by_speaker(text)
        return (
            group_turns_into_batches(turns, max_speakers=5, max_bytes=chunk_length)
            if turns
            else [text]
        )

    def _sample_semantic(
        self,
        logits: mx.array,
        previous_semantic_tokens: list[int],
        top_p: float,
        top_k: int,
        temperature: float,
    ) -> mx.array:
        if self.semantic_logit_bias is None:
            raise ValueError("Semantic logits bias is not initialized.")

        biased_logits = logits + self.semantic_logit_bias.astype(logits.dtype)
        normal = _sample_logits(
            biased_logits, temperature=temperature, top_p=top_p, top_k=top_k
        )

        token_value = int(normal[0].item())
        should_use_high = (
            token_value in previous_semantic_tokens
            and self.config.semantic_start_token_id
            <= token_value
            <= self.config.semantic_end_token_id
        )
        if not should_use_high:
            return normal

        high_temp = _sample_logits(
            biased_logits,
            temperature=RAS_HIGH_TEMP,
            top_p=RAS_HIGH_TOP_P,
            top_k=top_k,
        )
        return high_temp

    def _sample_semantic_batch(
        self,
        logits: mx.array,
        previous_semantic_tokens: list[list[int]],
        top_p: float,
        top_k: int,
        temperature: float,
    ) -> mx.array:
        if self.semantic_logit_bias is None:
            raise ValueError("Semantic logits bias is not initialized.")

        biased_logits = logits + self.semantic_logit_bias.astype(logits.dtype)
        normal = _sample_logits(
            biased_logits, temperature=temperature, top_p=top_p, top_k=top_k
        )

        normal_tokens = normal.tolist()
        if not isinstance(normal_tokens, list):
            normal_tokens = [normal_tokens]
        high_temp_indices = []
        for idx, token_value in enumerate(normal_tokens):
            should_use_high = (
                token_value in previous_semantic_tokens[idx]
                and self.config.semantic_start_token_id
                <= token_value
                <= self.config.semantic_end_token_id
            )
            if should_use_high:
                high_temp_indices.append(idx)

        if not high_temp_indices:
            return normal

        high_logits = mx.take(
            biased_logits, mx.array(high_temp_indices, dtype=mx.int32), axis=0
        )
        high_temp = _sample_logits(
            high_logits,
            temperature=RAS_HIGH_TEMP,
            top_p=RAS_HIGH_TOP_P,
            top_k=top_k,
        )

        high_tokens = high_temp.tolist()
        if not isinstance(high_tokens, list):
            high_tokens = [high_tokens]
        sampled_tokens = list(normal_tokens)
        for idx, token_value in zip(high_temp_indices, high_tokens):
            sampled_tokens[idx] = token_value
        return mx.array(sampled_tokens, dtype=normal.dtype)

    def _prepare_batched_prompt_inputs(
        self, conversations: list[Conversation]
    ) -> tuple[mx.array, mx.array]:
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Call post_load_hook first.")

        encoded_prompts = []
        prompt_lengths = []
        for conversation in conversations:
            prompt_conversation = Conversation(list(conversation.messages))
            prompt_conversation.append(
                Message(
                    role="assistant",
                    parts=[],
                    modality="voice",
                    add_im_start=True,
                    add_im_end=False,
                )
            )
            prompt = prompt_conversation.encode_for_inference(
                self.tokenizer, num_codebooks=self.model.num_codebooks
            )
            encoded_prompts.append(prompt.astype(mx.int32))
            prompt_lengths.append(prompt.shape[1])

        max_prompt_len = max(prompt_lengths)
        padded_prompts = []
        mask_rows = []
        for prompt, prompt_len in zip(encoded_prompts, prompt_lengths):
            pad_len = max_prompt_len - prompt_len
            if pad_len > 0:
                padding = mx.zeros((prompt.shape[0], pad_len), dtype=mx.int32)
                prompt = mx.concatenate([padding, prompt], axis=1)
                mask = mx.concatenate(
                    [
                        mx.zeros((1, pad_len), dtype=mx.float32),
                        mx.ones((1, prompt_len), dtype=mx.float32),
                    ],
                    axis=1,
                )
            else:
                mask = mx.ones((1, prompt_len), dtype=mx.float32)
            padded_prompts.append(prompt[None, :, :])
            mask_rows.append(mask)

        return mx.concatenate(padded_prompts, axis=0), mx.concatenate(mask_rows, axis=0)

    def _generate_codes_for_batch(
        self,
        conversation: Conversation,
        batch_text: str,
        max_new_tokens: int,
        top_p: float,
        top_k: int,
        temperature: float,
    ) -> mx.array:
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Call post_load_hook first.")

        prompt_conversation = Conversation(list(conversation.messages))
        prompt_conversation.append(
            Message(
                role="assistant",
                parts=[],
                modality="voice",
                add_im_start=True,
                add_im_end=False,
            )
        )
        prompt = prompt_conversation.encode_for_inference(
            self.tokenizer, num_codebooks=self.model.num_codebooks
        )
        prompt = prompt[None, :, :]

        cache = self.model.make_cache()
        result = self.model(prompt, cache=cache)
        logits = result.logits[:, -1]
        hidden_state = result.hidden_states[:, -1]

        previous_semantic_tokens: list[int] = []
        generated_steps = []
        im_end_id = self.tokenizer.get_token_id(IM_END_TOKEN)
        text_token_count = len(self.tokenizer.encode(batch_text))
        semantic_token_budget = min(
            max_new_tokens,
            max(32, text_token_count * 12),
        )

        for _ in range(semantic_token_budget):
            semantic_token = self._sample_semantic(
                logits=logits,
                previous_semantic_tokens=previous_semantic_tokens,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
            )
            semantic_token_id = int(semantic_token[0].item())
            if semantic_token_id == im_end_id:
                break

            previous_semantic_tokens.append(semantic_token_id)
            previous_semantic_tokens = previous_semantic_tokens[-RAS_WIN_SIZE:]

            semantic_code = (
                semantic_token - self.config.semantic_start_token_id
            ).astype(mx.int32)
            semantic_code = mx.clip(
                semantic_code, 0, self.config.audio_decoder_config.vocab_size - 1
            )
            previous_codebooks = semantic_code[:, None]
            fast_cache = self.model.make_fast_cache()
            fast_prefill = self.model.fast_forward_cached(hidden_state, fast_cache)
            mx.async_eval(fast_prefill)
            fast_hidden = self.model.fast_embeddings(semantic_code)

            for _ in range(self.model.num_codebooks - 1):
                residual_logits = self.model.fast_forward_cached(
                    fast_hidden, fast_cache
                )
                residual_token = _sample_logits(
                    residual_logits,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )
                previous_codebooks = mx.concatenate(
                    [previous_codebooks, residual_token[:, None]], axis=1
                )
                fast_hidden = self.model.fast_embeddings(residual_token)

            generated_steps.append(previous_codebooks[0])

            next_input = mx.concatenate(
                [semantic_token[:, None].astype(mx.int32), previous_codebooks], axis=1
            )
            next_result = self.model(next_input[:, :, None], cache=cache)
            logits = next_result.logits[:, -1]

        if not generated_steps:
            raise RuntimeError(
                f"No audio tokens were generated for batch text: {batch_text!r}"
            )

        return mx.stack(generated_steps, axis=1).astype(mx.int32)

    def _generate_codes_for_text_batch(
        self,
        conversations: list[Conversation],
        batch_texts: list[str],
        max_new_tokens: int,
        top_p: float,
        top_k: int,
        temperature: float,
    ) -> list[mx.array]:
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Call post_load_hook first.")

        batch_size = len(conversations)
        if batch_size == 0:
            return []

        prompt, attention_mask = self._prepare_batched_prompt_inputs(conversations)
        cache = self.model.make_cache()
        result = self.model(prompt, cache=cache, attention_mask=attention_mask)
        logits = result.logits[:, -1]
        hidden_state = result.hidden_states[:, -1]

        previous_semantic_tokens: list[list[int]] = [[] for _ in range(batch_size)]
        generated_steps: list[list[mx.array]] = [[] for _ in range(batch_size)]
        finished = [False] * batch_size
        im_end_id = self.tokenizer.get_token_id(IM_END_TOKEN)
        token_budgets = [
            min(max_new_tokens, max(32, len(self.tokenizer.encode(text)) * 12))
            for text in batch_texts
        ]
        max_budget = max(token_budgets)
        im_end_tokens = mx.full((batch_size,), im_end_id, dtype=mx.int32)

        for step in range(max_budget):
            active = [
                (not finished[idx]) and step < token_budgets[idx]
                for idx in range(batch_size)
            ]
            if not any(active):
                break

            sampled_semantic = self._sample_semantic_batch(
                logits=logits,
                previous_semantic_tokens=previous_semantic_tokens,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
            )
            active_mask = mx.array(active, dtype=mx.bool_)
            semantic_token = mx.where(active_mask, sampled_semantic, im_end_tokens)

            semantic_token_ids = semantic_token.tolist()
            if not isinstance(semantic_token_ids, list):
                semantic_token_ids = [semantic_token_ids]
            should_continue = []
            for idx, token_id in enumerate(semantic_token_ids):
                keep_generating = active[idx] and token_id != im_end_id
                should_continue.append(keep_generating)
                if active[idx] and not keep_generating:
                    finished[idx] = True

            if not any(should_continue):
                break

            continue_mask = mx.array(should_continue, dtype=mx.bool_)
            semantic_code = (
                semantic_token - self.config.semantic_start_token_id
            ).astype(mx.int32)
            semantic_code = mx.clip(
                semantic_code, 0, self.config.audio_decoder_config.vocab_size - 1
            )
            semantic_code = mx.where(
                continue_mask, semantic_code, mx.zeros_like(semantic_code)
            )
            previous_codebooks = semantic_code[:, None]

            fast_cache = self.model.make_fast_cache()
            fast_prefill = self.model.fast_forward_cached(hidden_state, fast_cache)
            mx.async_eval(fast_prefill)
            fast_hidden = self.model.fast_embeddings(semantic_code)

            for _ in range(self.model.num_codebooks - 1):
                residual_logits = self.model.fast_forward_cached(
                    fast_hidden, fast_cache
                )
                residual_token = _sample_logits(
                    residual_logits,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )
                residual_token = mx.where(
                    continue_mask, residual_token, mx.zeros_like(residual_token)
                )
                previous_codebooks = mx.concatenate(
                    [previous_codebooks, residual_token[:, None]], axis=1
                )
                fast_hidden = self.model.fast_embeddings(residual_token)

            for idx, keep_generating in enumerate(should_continue):
                if not keep_generating:
                    continue
                token_id = semantic_token_ids[idx]
                previous_semantic_tokens[idx].append(token_id)
                previous_semantic_tokens[idx] = previous_semantic_tokens[idx][
                    -RAS_WIN_SIZE:
                ]
                generated_steps[idx].append(previous_codebooks[idx])
                if step + 1 >= token_budgets[idx]:
                    finished[idx] = True

            next_input = mx.concatenate(
                [semantic_token[:, None].astype(mx.int32), previous_codebooks], axis=1
            )
            attention_mask = mx.concatenate(
                [attention_mask, mx.ones((batch_size, 1), dtype=attention_mask.dtype)],
                axis=1,
            )
            next_result = self.model(
                next_input[:, :, None],
                cache=cache,
                attention_mask=attention_mask,
            )
            logits = next_result.logits[:, -1]
            hidden_state = next_result.hidden_states[:, -1]

            if all(finished):
                break
            if step > 0 and step % 50 == 0:
                mx.clear_cache()

        empty_indices = [
            idx
            for idx, sequence_steps in enumerate(generated_steps)
            if not sequence_steps
        ]
        if empty_indices:
            raise RuntimeError(
                "No audio tokens were generated for batch sequence(s): "
                + ", ".join(str(idx) for idx in empty_indices)
            )

        return [
            mx.stack(sequence_steps, axis=1).astype(mx.int32)
            for sequence_steps in generated_steps
        ]

    def _decode_codes(self, codes: mx.array) -> mx.array:
        if self.codec is None:
            raise ValueError("Codec not loaded. Call post_load_hook first.")
        feature_lengths = mx.array([codes.shape[1]], dtype=mx.int32)
        audio, audio_lengths = self.codec.decode(codes[None, :, :], feature_lengths)
        length = int(audio_lengths[0].item())
        return audio[0, 0, :length]

    def _decode_codes_batch(self, codes_list: list[mx.array]) -> list[mx.array]:
        if self.codec is None:
            raise ValueError("Codec not loaded. Call post_load_hook first.")
        if not codes_list:
            return []

        max_len = max(codes.shape[1] for codes in codes_list)
        padded_codes = []
        feature_lengths = []
        for codes in codes_list:
            feature_lengths.append(codes.shape[1])
            pad_len = max_len - codes.shape[1]
            if pad_len > 0:
                padding = mx.zeros((codes.shape[0], pad_len), dtype=codes.dtype)
                codes = mx.concatenate([codes, padding], axis=1)
            padded_codes.append(codes[None, :, :])

        batch_codes = mx.concatenate(padded_codes, axis=0)
        lengths = mx.array(feature_lengths, dtype=mx.int32)
        audio, audio_lengths = self.codec.decode(batch_codes, lengths)
        mx.async_eval(audio)

        decoded = []
        for idx in range(len(codes_list)):
            length = int(audio_lengths[idx].item())
            decoded.append(audio[idx, 0, :length])
        return decoded

    def generate(
        self,
        text: str,
        voice: Optional[str] = None,
        ref_audio: Optional[mx.array] = None,
        ref_text: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.7,
        top_k: int = 30,
        repetition_penalty: float = 1.2,
        stream: bool = False,
        speed: float = 1.0,
        chunk_length: int = 300,
        verbose: bool = True,
        **kwargs,
    ):
        del voice, repetition_penalty, verbose, kwargs

        if stream:
            raise NotImplementedError("Fish Speech streaming is not implemented yet.")
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Call post_load_hook first.")
        if self.codec is None:
            raise ValueError("Codec not loaded. Call post_load_hook first.")

        prompt_texts, prompt_tokens = self._prepare_reference_prompt(
            ref_audio, ref_text
        )

        base_conversation = self._build_conversation(prompt_texts, prompt_tokens)
        batches = self._split_generation_text(text, chunk_length)

        conversation = Conversation(list(base_conversation.messages))
        segment_idx = 0
        for batch_text in batches:
            conversation.append(
                Message(
                    role="user",
                    parts=[TextPart(batch_text)],
                    add_im_start=True,
                    add_im_end=True,
                )
            )
            start_time = time.perf_counter()
            codes = self._generate_codes_for_batch(
                conversation=conversation,
                batch_text=batch_text,
                max_new_tokens=max_tokens,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
            )
            audio = self._decode_codes(codes)
            if abs(speed - 1.0) > 1e-6:
                audio = _adjust_speed(audio, speed)
            mx.async_eval(audio)

            conversation.append(
                Message(
                    role="assistant",
                    parts=[VQPart(codes)],
                    modality="voice",
                    add_im_start=True,
                    add_im_end=True,
                )
            )

            elapsed = max(time.perf_counter() - start_time, 1e-6)
            audio_duration = float(audio.shape[0]) / float(self.sample_rate)
            prompt_tokens_count = len(self.tokenizer.encode(batch_text))
            yield GenerationResult(
                audio=audio,
                samples=int(audio.shape[0]),
                sample_rate=self.sample_rate,
                segment_idx=segment_idx,
                token_count=int(codes.shape[1]),
                audio_duration=_format_duration(audio_duration),
                real_time_factor=audio_duration / elapsed if elapsed > 0 else 0.0,
                prompt={
                    "tokens": prompt_tokens_count,
                    "tokens-per-sec": (
                        prompt_tokens_count / elapsed if elapsed > 0 else 0.0
                    ),
                },
                audio_samples={
                    "samples": int(audio.shape[0]),
                    "samples-per-sec": (
                        float(audio.shape[0]) / elapsed if elapsed > 0 else 0.0
                    ),
                },
                processing_time_seconds=elapsed,
                peak_memory_usage=float(mx.get_peak_memory() / 1e9),
            )
            segment_idx += 1

    @staticmethod
    def _normalize_batch_arg(name: str, value, batch_size: int) -> list:
        if value is None:
            return [None] * batch_size
        if isinstance(value, (list, tuple)):
            if len(value) != batch_size:
                raise ValueError(
                    f"{name} length ({len(value)}) must match texts length ({batch_size})"
                )
            return list(value)
        return [value] * batch_size

    def batch_generate(
        self,
        texts: list[str],
        voices: Optional[list[Optional[str]]] = None,
        ref_audios: Optional[list[Optional[mx.array]]] = None,
        ref_texts: Optional[list[Optional[str]]] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.7,
        top_k: int = 30,
        repetition_penalty: float = 1.2,
        stream: bool = False,
        speed: float = 1.0,
        chunk_length: int = 300,
        verbose: bool = True,
        **kwargs,
    ):
        del repetition_penalty, verbose

        if stream:
            raise NotImplementedError("Fish Speech streaming is not implemented yet.")
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Call post_load_hook first.")
        if self.codec is None:
            raise ValueError("Codec not loaded. Call post_load_hook first.")

        if ref_audios is None and "ref_audio" in kwargs:
            ref_audios = kwargs.pop("ref_audio")
        if ref_texts is None and "ref_text" in kwargs:
            ref_texts = kwargs.pop("ref_text")
        kwargs.clear()

        batch_size = len(texts)
        if batch_size == 0:
            return

        if voices is not None and len(voices) != batch_size:
            raise ValueError(
                f"voices length ({len(voices)}) must match texts length ({batch_size})"
            )

        ref_audio_list = self._normalize_batch_arg("ref_audios", ref_audios, batch_size)
        ref_text_list = self._normalize_batch_arg("ref_texts", ref_texts, batch_size)

        states = []
        for idx, text in enumerate(texts):
            prompt_texts, prompt_tokens = self._prepare_reference_prompt(
                ref_audio_list[idx], ref_text_list[idx]
            )
            base_conversation = self._build_conversation(prompt_texts, prompt_tokens)
            states.append(
                {
                    "sequence_idx": idx,
                    "conversation": Conversation(list(base_conversation.messages)),
                    "batches": self._split_generation_text(text, chunk_length),
                    "next_batch": 0,
                }
            )

        while True:
            active_states = [
                state for state in states if state["next_batch"] < len(state["batches"])
            ]
            if not active_states:
                break

            active_conversations = []
            active_texts = []
            for state in active_states:
                batch_text = state["batches"][state["next_batch"]]
                state["conversation"].append(
                    Message(
                        role="user",
                        parts=[TextPart(batch_text)],
                        add_im_start=True,
                        add_im_end=True,
                    )
                )
                active_conversations.append(state["conversation"])
                active_texts.append(batch_text)

            start_time = time.perf_counter()
            codes_list = self._generate_codes_for_text_batch(
                conversations=active_conversations,
                batch_texts=active_texts,
                max_new_tokens=max_tokens,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
            )
            audio_list = self._decode_codes_batch(codes_list)

            for state, codes, audio in zip(active_states, codes_list, audio_list):
                if abs(speed - 1.0) > 1e-6:
                    audio = _adjust_speed(audio, speed)
                mx.async_eval(audio)

                state["conversation"].append(
                    Message(
                        role="assistant",
                        parts=[VQPart(codes)],
                        modality="voice",
                        add_im_start=True,
                        add_im_end=True,
                    )
                )
                state["next_batch"] += 1

                elapsed = max(time.perf_counter() - start_time, 1e-6)
                samples = int(audio.shape[0])
                duration_seconds = samples / float(self.sample_rate)
                yield BatchGenerationResult(
                    audio=audio,
                    sequence_idx=state["sequence_idx"],
                    samples=samples,
                    sample_rate=self.sample_rate,
                    token_count=int(codes.shape[1]),
                    audio_duration=_format_duration(duration_seconds),
                    processing_time_seconds=elapsed,
                    peak_memory_usage=float(mx.get_peak_memory() / 1e9),
                )

            mx.clear_cache()

    @classmethod
    def post_load_hook(cls, model: "Model", model_path: Path) -> "Model":
        from mlx_audio.codec.models.fish_s1_dac import DAC as FishS1DAC

        model.tokenizer = FishTokenizer(str(model_path))
        model.codec = FishS1DAC.from_pretrained(str(model_path))
        vocab_size = max(
            model.tokenizer.vocab_size,
            model.config.text_config.vocab_size,
        )
        semantic_bias = mx.full((1, vocab_size), -1e9, dtype=mx.float32)
        semantic_bias[
            :,
            model.config.semantic_start_token_id : model.config.semantic_end_token_id
            + 1,
        ] = 0.0
        semantic_bias[:, model.tokenizer.get_token_id(IM_END_TOKEN)] = 0.0
        model.semantic_logit_bias = semantic_bias
        return model
