from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Generator, Sequence

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.qwen3 import Qwen3Model

from mlx_audio.tts.models.base import GenerationResult

from .config import DEFAULT_AUDIO_TOKENIZER_REPO, ModelConfig
from .processor import (
    MossTTSDelayProcessor,
    MossTTSLocalProcessor,
    apply_de_delay_pattern,
)
from .sampling import sample_token

_INT64_MAX = 9223372036854775807


def _as_reference_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _collapse_reference_list(values: list):
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return values


class MossTTSRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((int(dim),))
        self.eps = float(eps)

    def __call__(self, x: mx.array) -> mx.array:
        return (
            self.weight
            * mx.rsqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + self.eps)
            * x
        )


class MossTTSMLP(nn.Module):
    def __init__(self, input_size: int, ffn_hidden_size: int, output_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(input_size, ffn_hidden_size, bias=False)
        self.up_proj = nn.Linear(input_size, ffn_hidden_size, bias=False)
        self.down_proj = nn.Linear(ffn_hidden_size, output_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


def _causal_mask(length: int, dtype: mx.Dtype) -> mx.array:
    positions = mx.arange(int(length), dtype=mx.int32)
    allowed = positions[None, None, :, None] >= positions[None, None, None, :]
    return mx.where(allowed, 0.0, mx.finfo(dtype).min).astype(dtype)


class MossTTSLocalAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        dim = int(args.hidden_size)
        self.n_heads = int(args.num_attention_heads)
        self.n_kv_heads = int(args.num_key_value_heads)
        self.head_dim = int(args.head_dim)
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        batch, length, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = self.q_norm(q.reshape(batch, length, self.n_heads, self.head_dim))
        k = self.k_norm(k.reshape(batch, length, self.n_kv_heads, self.head_dim))
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.reshape(batch, length, self.n_kv_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        out = mx.fast.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=self.scale,
            mask=mask,
        )
        out = out.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self.o_proj(out)


class MossTTSLocalTransformerBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.self_attn = MossTTSLocalAttention(args)
        self.mlp = MossTTSMLP(
            int(args.hidden_size),
            int(args.intermediate_size),
            int(args.hidden_size),
        )
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        h = x + self.self_attn(self.input_layernorm(x), mask=mask)
        return h + self.mlp(self.post_attention_layernorm(h))


class MossTTSLocalTransformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.layers = [
            MossTTSLocalTransformerBlock(args)
            for _ in range(int(args.num_hidden_layers))
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, inputs_embeds: mx.array) -> mx.array:
        mask = _causal_mask(inputs_embeds.shape[1], inputs_embeds.dtype)
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states, mask=mask)
        return self.norm(hidden_states)


class MosiTTSModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding_list = [
            nn.Embedding(config.vocab_size, config.hidden_size),
        ]
        self.embedding_list.extend(
            [
                nn.Embedding(config.audio_vocab_size + 1, config.hidden_size)
                for _ in range(config.n_vq)
            ]
        )
        self.language_model = Qwen3Model(config.language_config)

    def _prepare_multi_modal_inputs(
        self,
        input_ids: mx.array,
        *,
        n_vq_for_inference: int | None = None,
    ) -> mx.array:
        if input_ids.ndim != 3 or input_ids.shape[-1] != self.config.n_vq + 1:
            raise ValueError(
                "Expected input_ids shape "
                f"[batch, seq, {self.config.n_vq + 1}], got {input_ids.shape}"
            )
        channels = min(
            input_ids.shape[-1],
            1 + int(n_vq_for_inference or self.config.n_vq),
        )
        inputs_embeds = mx.zeros(
            (*input_ids.shape[:2], self.config.hidden_size),
            dtype=self.embedding_list[0].weight.dtype,
        )
        for channel_index in range(channels):
            inputs_embeds = inputs_embeds + self.embedding_list[channel_index](
                input_ids[..., channel_index]
            )
        return inputs_embeds

    def __call__(
        self,
        input_ids: mx.array | None = None,
        *,
        inputs_embeds: mx.array | None = None,
        cache=None,
        n_vq_for_inference: int | None = None,
    ) -> mx.array:
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds are required")
            inputs_embeds = self._prepare_multi_modal_inputs(
                input_ids,
                n_vq_for_inference=n_vq_for_inference,
            )
        dummy_inputs = mx.zeros(inputs_embeds.shape[:2], dtype=mx.int32)
        return self.language_model(
            dummy_inputs,
            cache=cache,
            input_embeddings=inputs_embeds,
        )


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if isinstance(config, dict):
            config = ModelConfig.from_dict(config)
        if config.language_config is None:
            raise ValueError("MOSS-TTS requires language_config")
        self.config = config
        if config.is_local_transformer:
            self.model = MosiTTSModel(config)
            self.channels = 1 + config.n_vq
            self.local_transformer_config = config.local_transformer_config()
            self.local_transformer = MossTTSLocalTransformer(
                self.local_transformer_config
            )
            self.speech_embedding_to_local_mlp = MossTTSMLP(
                input_size=config.hidden_size,
                ffn_hidden_size=int(config.additional_mlp_ffn_hidden_size),
                output_size=int(config.local_hidden_size),
            )
            self.local_to_speech_embedding_mlps = [
                MossTTSMLP(
                    input_size=int(config.local_hidden_size),
                    ffn_hidden_size=int(config.additional_mlp_ffn_hidden_size),
                    output_size=config.hidden_size,
                )
                for _ in range(self.channels)
            ]
            self.layer_norm_before_lm_heads = [
                MossTTSRMSNorm(config.hidden_size) for _ in range(self.channels)
            ]
            self.lm_heads = [
                nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            ]
            self.lm_heads.extend(
                [
                    nn.Linear(
                        config.hidden_size,
                        config.audio_vocab_size + 1,
                        bias=False,
                    )
                    for _ in range(config.n_vq)
                ]
            )
        else:
            self.language_model = Qwen3Model(config.language_config)
            self.emb_ext = [
                nn.Embedding(config.audio_vocab_size + 1, config.hidden_size)
                for _ in range(config.n_vq)
            ]
            self.lm_heads = [
                nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            ]
            self.lm_heads.extend(
                [
                    nn.Linear(
                        config.hidden_size,
                        config.audio_vocab_size + 1,
                        bias=False,
                    )
                    for _ in range(config.n_vq)
                ]
            )
        self.tokenizer = None
        self.audio_tokenizer = None
        self.generation_config: dict[str, object] = {}

    @property
    def sample_rate(self) -> int:
        return int(self.config.sampling_rate)

    @property
    def model_type(self) -> str:
        return self.config.model_type

    @property
    def layers(self):
        if self.config.is_local_transformer:
            return self.model.language_model.layers
        return self.language_model.layers

    @classmethod
    def post_load_hook(cls, model: "Model", model_path: Path) -> "Model":
        try:
            from transformers import AutoTokenizer

            model.tokenizer = AutoTokenizer.from_pretrained(
                str(model_path),
                trust_remote_code=False,
            )
        except Exception:
            model.tokenizer = None
        model.generation_config = cls._load_generation_config(model_path)
        return model

    @staticmethod
    def _load_generation_config(model_path: Path) -> dict[str, object]:
        generation_config_path = model_path / "generation_config.json"
        if not generation_config_path.exists():
            return {}
        try:
            with open(generation_config_path, encoding="utf-8") as f:
                config = json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}
        return config if isinstance(config, dict) else {}

    def _generation_config_value(self, key: str, default):
        value = self.generation_config.get(key, default)
        return default if value is None else value

    def _processor(self):
        if self.config.is_local_transformer:
            return MossTTSLocalProcessor(self.tokenizer, self.config)
        return MossTTSDelayProcessor(self.tokenizer, self.config)

    def model_quant_predicate(self, path: str, module) -> bool:
        del module
        skip_prefixes = ("emb_ext", "model.embedding_list", "audio_tokenizer")
        return not any(path.startswith(prefix) for prefix in skip_prefixes)

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        sanitized = {}
        for key, value in weights.items():
            if key.startswith("model.") and not self.config.is_local_transformer:
                key = key[len("model.") :]
            sanitized[key] = value
        return sanitized

    def _ensure_audio_tokenizer(
        self,
        *,
        source: str | None = None,
    ):
        if self.audio_tokenizer is None:
            from mlx_audio.codec import MossAudioTokenizer

            self.audio_tokenizer = MossAudioTokenizer.from_model_dir(
                self.config.model_path,
                fallback_source=source
                or self.config.audio_tokenizer_pretrained_name_or_path
                or DEFAULT_AUDIO_TOKENIZER_REPO,
            )
        return self.audio_tokenizer

    def encode_reference_audio(
        self,
        ref_audio,
        *,
        sample_rate: int | None = None,
        num_quantizers: int | None = None,
        source: str | None = None,
    ) -> mx.array:
        tokenizer = self._ensure_audio_tokenizer(source=source)
        return tokenizer.encode_audio(
            ref_audio,
            sample_rate=sample_rate,
            num_quantizers=num_quantizers or self.config.n_vq,
        )

    def decode_audio_token_ids(
        self,
        audio_token_ids: mx.array,
        *,
        num_quantizers: int | None = None,
        source: str | None = None,
    ) -> mx.array:
        tokenizer = self._ensure_audio_tokenizer(source=source)
        return tokenizer.decode_audio_codes(
            audio_token_ids,
            num_quantizers=num_quantizers or self.config.n_vq,
        )

    def _build_inputs_embeds(self, input_ids: mx.array) -> mx.array:
        if self.config.is_local_transformer:
            return self.model._prepare_multi_modal_inputs(input_ids)
        if input_ids.ndim != 3 or input_ids.shape[-1] != self.config.n_vq + 1:
            raise ValueError(
                "Expected input_ids shape "
                f"[batch, seq, {self.config.n_vq + 1}], got {input_ids.shape}"
            )
        inputs_embeds = self.language_model.embed_tokens(input_ids[..., 0])
        for index, embed_layer in enumerate(self.emb_ext):
            inputs_embeds = inputs_embeds + embed_layer(input_ids[..., index + 1])
        return inputs_embeds

    def _head_logits(self, hidden_states: mx.array, head_index: int) -> mx.array:
        logits = self.lm_heads[head_index](hidden_states)
        if head_index == 0:
            return logits
        invalid_pad = mx.full(logits[..., -1:].shape, -mx.inf, dtype=logits.dtype)
        return mx.concatenate([logits[..., :-1], invalid_pad], axis=-1)

    def __call__(
        self,
        input_ids: mx.array | None = None,
        *,
        inputs_embeds: mx.array | None = None,
        cache=None,
        head_indices: Sequence[int] | None = None,
        labels: mx.array | None = None,
        n_vq_for_inference: int | None = None,
    ) -> list[mx.array]:
        if self.config.is_local_transformer:
            return self._local_forward(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                cache=cache,
                labels=labels,
                head_indices=head_indices,
                n_vq_for_inference=n_vq_for_inference,
            )
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds are required")
            inputs_embeds = self._build_inputs_embeds(input_ids)
        dummy_inputs = mx.zeros(inputs_embeds.shape[:2], dtype=mx.int32)
        hidden_states = self.language_model(
            dummy_inputs,
            cache=cache,
            input_embeddings=inputs_embeds,
        )
        if head_indices is None:
            head_indices = range(self.config.n_vq + 1)
        return [self._head_logits(hidden_states, int(index)) for index in head_indices]

    def make_cache(self):
        if self.config.is_local_transformer:
            return make_prompt_cache(self.model.language_model)
        return make_prompt_cache(self.language_model)

    def _masked_embedding(
        self, embedding: nn.Embedding, input_ids: mx.array
    ) -> mx.array:
        mask = input_ids != -100
        safe_ids = mx.where(mask, input_ids, 0).astype(mx.int32)
        return mx.where(mask[..., None], embedding(safe_ids), 0.0)

    def _local_forward(
        self,
        input_ids: mx.array | None = None,
        *,
        inputs_embeds: mx.array | None = None,
        cache=None,
        labels: mx.array | None = None,
        head_indices: Sequence[int] | None = None,
        n_vq_for_inference: int | None = None,
    ) -> list[mx.array]:
        if inputs_embeds is None and input_ids is None:
            raise ValueError("input_ids or inputs_embeds are required")
        global_hidden_states = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            cache=cache,
            n_vq_for_inference=n_vq_for_inference,
        )
        if labels is None:
            if input_ids is None:
                raise ValueError("labels are required when inputs_embeds are provided")
            labels = input_ids
        if labels.ndim != 3 or labels.shape[-1] != self.channels:
            raise ValueError(
                f"Expected labels shape [batch, seq, {self.channels}], got {labels.shape}"
            )

        local_inputs = [global_hidden_states]
        for channel_index in range(self.channels - 1):
            local_inputs.append(
                self._masked_embedding(
                    self.model.embedding_list[channel_index],
                    labels[..., channel_index],
                )
            )
        local_inputs = mx.stack(local_inputs, axis=0)
        local_inputs = self.speech_embedding_to_local_mlp(local_inputs)
        channels, batch, seq_len, local_dim = local_inputs.shape
        local_inputs = local_inputs.transpose(1, 2, 0, 3).reshape(
            batch * seq_len,
            channels,
            local_dim,
        )
        local_outputs = self.local_transformer(local_inputs)

        if head_indices is None:
            head_indices = range(self.channels)
        logits = []
        for head_index in head_indices:
            head_index = int(head_index)
            head_hidden = local_outputs[:, head_index, :]
            head_hidden = self.local_to_speech_embedding_mlps[head_index](head_hidden)
            head_hidden = self.layer_norm_before_lm_heads[head_index](head_hidden)
            head_hidden = head_hidden.reshape(batch, seq_len, self.config.hidden_size)
            logits.append(self.lm_heads[head_index](head_hidden))
        return logits

    @staticmethod
    def _find_last_equal(values: mx.array, target: int) -> int:
        matches = [
            idx for idx, value in enumerate(values.tolist()) if int(value) == target
        ]
        return matches[-1] if matches else -1

    @staticmethod
    def _set_logits_to_neg_inf(logits: mx.array, token_ids: Sequence[int]) -> mx.array:
        if len(token_ids) == 0:
            return logits
        logits = logits + 0.0
        logits[:, [int(token_id) for token_id in token_ids]] = -mx.inf
        return logits

    @staticmethod
    def _keep_only_logits(logits: mx.array, token_ids: Sequence[int]) -> mx.array:
        keep = mx.zeros((logits.shape[-1],), dtype=mx.bool_)
        keep[[int(token_id) for token_id in token_ids]] = True
        return mx.where(keep[None, :], logits, -mx.inf)

    def generate_delay_pattern_ids(
        self,
        input_ids: mx.array,
        *,
        max_new_tokens: int = 4096,
        text_temperature: float = 1.5,
        text_top_p: float = 1.0,
        text_top_k: int = 50,
        audio_temperature: float = 1.7,
        audio_top_p: float = 0.8,
        audio_top_k: int = 25,
        audio_repetition_penalty: float = 1.0,
    ) -> list[tuple[int, mx.array]]:
        if input_ids.ndim != 3:
            raise ValueError(f"Expected input_ids rank 3, got {input_ids.shape}")
        if input_ids.shape[0] != 1:
            raise NotImplementedError("MOSS-TTS batch generation is not implemented.")

        if text_temperature > 0:
            text_do_sample = True
        else:
            text_temperature = 1.0
            text_do_sample = False
        if audio_temperature > 0:
            audio_do_sample = True
        else:
            audio_temperature = 1.0
            audio_do_sample = False

        batch_size, seq_len, width = input_ids.shape
        n_vq = width - 1
        if n_vq != self.config.n_vq:
            raise ValueError(f"Expected {self.config.n_vq} VQ channels, got {n_vq}")

        cache = self.make_cache()
        current_input_ids = input_ids
        generation_ids = input_ids

        is_stopping = False
        audio_lengths = 0
        delayed_lengths = _INT64_MAX

        last_text_token = int(input_ids[0, -1, 0].item())
        is_continuation = last_text_token in {
            self.config.audio_start_token_id,
            self.config.audio_assistant_gen_slot_token_id,
        }
        audio_start_idx = self._find_last_equal(
            input_ids[0, :, 0], self.config.audio_start_token_id
        )
        is_audio = bool(is_continuation and audio_start_idx != -1)
        if is_audio:
            audio_lengths = int(seq_len - audio_start_idx)

        text_exclude_outside_audio = [
            self.config.pad_token_id,
            self.config.audio_assistant_gen_slot_token_id,
            self.config.audio_assistant_delay_slot_token_id,
            self.config.audio_end_token_id,
        ]
        text_keep_inside_audio = [
            self.config.audio_assistant_gen_slot_token_id,
            self.config.audio_assistant_delay_slot_token_id,
        ]

        for time_step in range(int(max_new_tokens)):
            outputs = self(current_input_ids, cache=cache)
            next_token_logits = [
                logits[:, -1, :]
                / float(text_temperature if idx == 0 else audio_temperature)
                for idx, logits in enumerate(outputs)
            ]

            next_text_token_value = self.config.pad_token_id
            if not is_stopping and delayed_lengths < n_vq:
                next_text_token_value = self.config.audio_assistant_delay_slot_token_id
            elif not is_stopping and delayed_lengths == n_vq:
                next_text_token_value = self.config.audio_end_token_id
                is_audio = False
            elif not is_stopping and delayed_lengths > n_vq:
                text_logits = next_token_logits[0]
                if is_audio:
                    text_logits = self._keep_only_logits(
                        text_logits, text_keep_inside_audio
                    )
                else:
                    text_logits = self._set_logits_to_neg_inf(
                        text_logits, text_exclude_outside_audio
                    )
                if time_step == 0:
                    text_logits = self._set_logits_to_neg_inf(
                        text_logits, [self.config.audio_assistant_delay_slot_token_id]
                    )
                if time_step <= n_vq:
                    text_logits = self._set_logits_to_neg_inf(
                        text_logits, [self.config.im_end_token_id]
                    )
                next_text_token = sample_token(
                    text_logits,
                    top_p=text_top_p,
                    top_k=text_top_k,
                    do_sample=text_do_sample,
                )
                mx.eval(next_text_token)
                next_text_token_value = int(next_text_token.item())

            if next_text_token_value == self.config.audio_start_token_id:
                is_audio = True
            if next_text_token_value == self.config.im_end_token_id:
                is_stopping = True

            next_audio_tokens = mx.full(
                (batch_size, n_vq),
                self.config.audio_pad_code,
                dtype=mx.int32,
            )

            for codebook_index in range(n_vq):
                pre_audio = audio_lengths > codebook_index
                post_audio = (
                    True
                    if delayed_lengths == _INT64_MAX
                    else codebook_index > delayed_lengths - 1
                )
                if not (pre_audio and post_audio):
                    continue

                channel_logits = next_token_logits[codebook_index + 1]
                channel_logits = self._set_logits_to_neg_inf(
                    channel_logits, [self.config.audio_pad_code]
                )
                channel_token = sample_token(
                    channel_logits,
                    prev_tokens=generation_ids[:, :, codebook_index + 1],
                    repetition_penalty=audio_repetition_penalty,
                    top_p=audio_top_p,
                    top_k=audio_top_k,
                    do_sample=audio_do_sample,
                )
                next_audio_tokens[:, codebook_index] = channel_token

            if next_text_token_value in {
                self.config.audio_start_token_id,
                self.config.audio_assistant_gen_slot_token_id,
                self.config.audio_assistant_delay_slot_token_id,
            }:
                audio_lengths += 1
            if next_text_token_value == self.config.audio_end_token_id:
                audio_lengths = 0
            if (
                delayed_lengths == _INT64_MAX
                and next_text_token_value
                == self.config.audio_assistant_delay_slot_token_id
            ):
                delayed_lengths = 0
            if delayed_lengths != _INT64_MAX:
                delayed_lengths += 1
            if delayed_lengths > n_vq:
                delayed_lengths = _INT64_MAX

            next_text_token = mx.array([[next_text_token_value]], dtype=mx.int32)
            current_input_ids = mx.concatenate(
                [next_text_token[:, :, None], next_audio_tokens[:, None, :]],
                axis=2,
            )
            generation_ids = mx.concatenate([generation_ids, current_input_ids], axis=1)
            mx.eval(current_input_ids)

            if is_stopping:
                break

        start_idx = self._find_last_equal(
            input_ids[0, :, 0], self.config.im_start_token_id
        )
        start_idx = start_idx + 3 if start_idx != -1 else int(seq_len)
        start_length = int(seq_len - start_idx)
        return [(start_length, generation_ids[0, start_idx:])]

    def generate_local_ids(
        self,
        input_ids: mx.array,
        *,
        max_new_tokens: int = 4096,
        text_temperature: float = 1.5,
        text_top_p: float = 1.0,
        text_top_k: int = 50,
        text_repetition_penalty: float = 1.0,
        audio_temperature: float = 1.0,
        audio_top_p: float = 0.95,
        audio_top_k: int = 50,
        audio_repetition_penalty: float = 1.1,
        n_vq_for_inference: int | None = None,
    ) -> list[tuple[int, mx.array]]:
        if input_ids.ndim != 3:
            raise ValueError(f"Expected input_ids rank 3, got {input_ids.shape}")
        if input_ids.shape[0] != 1:
            raise NotImplementedError("MOSS-TTS batch generation is not implemented.")

        if text_temperature > 0:
            text_do_sample = True
        else:
            text_temperature = 1.0
            text_do_sample = False
        if audio_temperature > 0:
            audio_do_sample = True
        else:
            audio_temperature = 1.0
            audio_do_sample = False

        batch_size, seq_len, channels = input_ids.shape
        if channels != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {channels}")
        n_vq = max(
            1,
            min(
                self.channels - 1,
                int(n_vq_for_inference or self.channels - 1),
            ),
        )
        active_channels = 1 + n_vq

        cache = self.make_cache()
        current_input_ids = input_ids
        generation_ids = input_ids

        for _ in range(int(max_new_tokens)):
            global_hidden_states = self.model(
                current_input_ids,
                cache=cache,
                n_vq_for_inference=n_vq,
            )
            current_local_input = self.speech_embedding_to_local_mlp(
                global_hidden_states[:, -1, :]
            )
            local_inputs = mx.zeros(
                (batch_size, 0, int(self.config.local_hidden_size)),
                dtype=current_local_input.dtype,
            )
            next_tokens = []

            for channel_index in range(active_channels):
                local_inputs = mx.concatenate(
                    [local_inputs, current_local_input[:, None, :]],
                    axis=1,
                )
                local_outputs = self.local_transformer(local_inputs)
                head_hidden = local_outputs[:, -1, :]
                head_hidden = self.local_to_speech_embedding_mlps[channel_index](
                    head_hidden
                )
                head_hidden = self.layer_norm_before_lm_heads[channel_index](
                    head_hidden
                )
                logits = self.lm_heads[channel_index](head_hidden)
                if channel_index != 0:
                    logits = self._set_logits_to_neg_inf(
                        logits,
                        [self.config.audio_pad_code],
                    )

                is_text = channel_index == 0
                repetition_penalty = (
                    text_repetition_penalty if is_text else audio_repetition_penalty
                )
                if not (text_do_sample if is_text else audio_do_sample):
                    repetition_penalty = 1.0
                token = sample_token(
                    logits / float(text_temperature if is_text else audio_temperature),
                    prev_tokens=generation_ids[:, :, channel_index],
                    repetition_penalty=repetition_penalty,
                    top_p=text_top_p if is_text else audio_top_p,
                    top_k=text_top_k if is_text else audio_top_k,
                    do_sample=text_do_sample if is_text else audio_do_sample,
                )
                mx.eval(token)
                next_tokens.append(token)

                current_local_input = self.model.embedding_list[channel_index](token)
                current_local_input = self.speech_embedding_to_local_mlp(
                    current_local_input
                )

            for _channel_index in range(active_channels, channels):
                next_tokens.append(mx.zeros((batch_size,), dtype=mx.int32))
            next_tokens = mx.stack(next_tokens, axis=-1).astype(mx.int32)

            current_input_ids = next_tokens[:, None, :]
            generation_ids = mx.concatenate([generation_ids, current_input_ids], axis=1)
            mx.eval(current_input_ids)

            if int(next_tokens[0, 0].item()) == self.config.audio_end_token_id:
                break

        audio_start_idx = self._find_last_equal(
            input_ids[0, :, 0], self.config.audio_start_token_id
        )
        start_idx = audio_start_idx if audio_start_idx != -1 else int(seq_len)
        start_length = int(seq_len - start_idx - 1) if audio_start_idx != -1 else 0
        return [(start_length, generation_ids[0, start_idx:])]

    def _decode_generated_audio(
        self,
        outputs: list[tuple[int, mx.array]],
        *,
        source: str | None = None,
    ) -> tuple[mx.array, int]:
        audio_segments = []
        token_count = 0
        for start_length, generation_ids in outputs:
            audio_codes = generation_ids[:, 1:].astype(mx.int32)
            if not self.config.is_local_transformer:
                audio_codes = apply_de_delay_pattern(audio_codes)
            is_pad = mx.all(audio_codes == self.config.audio_pad_code, axis=1)
            non_pad_indices = [
                idx for idx, value in enumerate(is_pad.tolist()) if not bool(value)
            ]
            if not non_pad_indices:
                continue
            breaks = [0]
            for idx in range(1, len(non_pad_indices)):
                if non_pad_indices[idx] != non_pad_indices[idx - 1] + 1:
                    breaks.append(idx)
            breaks.append(len(non_pad_indices))

            for break_start, break_end in zip(breaks[:-1], breaks[1:]):
                segment_indices = non_pad_indices[break_start:break_end]
                codes = audio_codes[segment_indices[0] : segment_indices[-1] + 1]
                token_count += int(codes.shape[0])
                audio = self.decode_audio_token_ids(
                    codes,
                    num_quantizers=self.config.n_vq,
                    source=source,
                )
                if start_length > 0 and not audio_segments:
                    first_codes_length = int(codes.shape[0])
                    if first_codes_length > 0:
                        trim_ratio = max(
                            0.0,
                            min(float(start_length) / float(first_codes_length), 1.0),
                        )
                        trim_samples = int(audio.shape[0] * trim_ratio)
                        audio = audio[trim_samples:]
                audio_segments.append(audio)

        if not audio_segments:
            return mx.zeros((0, 1), dtype=mx.float32), 0
        return mx.concatenate(audio_segments, axis=0), token_count

    def _build_generation_result(
        self,
        *,
        audio: mx.array,
        started_at: float,
        token_count: int,
        prompt_token_count: int,
    ) -> GenerationResult:
        elapsed = max(time.perf_counter() - started_at, 1e-6)
        samples = int(audio.shape[0]) if audio.ndim > 0 else 0
        audio_duration_seconds = samples / float(self.sample_rate)
        duration_mins = int(audio_duration_seconds // 60)
        duration_secs = int(audio_duration_seconds % 60)
        duration_ms = int((audio_duration_seconds % 1) * 1000)
        duration_str = (
            f"{int(audio_duration_seconds // 3600):02d}:"
            f"{duration_mins:02d}:{duration_secs:02d}.{duration_ms:03d}"
        )
        return GenerationResult(
            audio=audio,
            samples=samples,
            sample_rate=self.sample_rate,
            segment_idx=0,
            token_count=token_count,
            audio_duration=duration_str,
            real_time_factor=(audio_duration_seconds / elapsed if elapsed > 0 else 0.0),
            prompt={
                "tokens": prompt_token_count,
                "tokens-per-sec": round(prompt_token_count / elapsed, 2),
            },
            audio_samples={
                "samples": samples,
                "samples-per-sec": round(samples / elapsed, 2),
            },
            processing_time_seconds=elapsed,
            peak_memory_usage=mx.get_peak_memory() / 1e9,
        )

    def generate(
        self,
        text: str,
        ref_audio=None,
        ref_text: str | list[str] | None = None,
        prompt_audio_codes: mx.array | list[mx.array] | None = None,
        mode: str = "generation",
        stream: bool = False,
        max_tokens: int | None = None,
        **kwargs,
    ) -> Generator[GenerationResult, None, None]:
        if stream:
            raise NotImplementedError("MOSS-TTS streaming is not implemented yet.")
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not initialized.")

        started_at = time.perf_counter()
        processor = self._processor()

        if prompt_audio_codes is None and ref_audio is not None:
            encoded_audio_codes = [
                self.encode_reference_audio(
                    ref_audio_item,
                    sample_rate=kwargs.get("ref_audio_sample_rate"),
                    num_quantizers=self.config.n_vq,
                    source=kwargs.get("audio_tokenizer_source"),
                )
                for ref_audio_item in _as_reference_list(ref_audio)
            ]
            prompt_audio_codes = _collapse_reference_list(encoded_audio_codes)
        elif prompt_audio_codes is not None:
            if isinstance(prompt_audio_codes, (list, tuple)):
                prompt_audio_codes = [
                    (
                        item
                        if isinstance(item, mx.array)
                        else mx.array(item, dtype=mx.int32)
                    )
                    for item in prompt_audio_codes
                ]
            elif not isinstance(prompt_audio_codes, mx.array):
                prompt_audio_codes = mx.array(prompt_audio_codes, dtype=mx.int32)

        normalized_mode = str(mode or "generation").strip().lower()
        if normalized_mode in {"voice_clone", "direct"}:
            normalized_mode = "generation"
        if normalized_mode not in {"generation", "continuation"}:
            raise ValueError("mode must be generation or continuation")
        ref_text_values = _as_reference_list(ref_text)
        if normalized_mode == "continuation" and len(ref_text_values) > 1:
            raise ValueError("MOSS-TTS continuation mode accepts one ref_text value.")
        prompt_audio_codes_list = _as_reference_list(prompt_audio_codes)
        if normalized_mode == "continuation" and len(prompt_audio_codes_list) > 1:
            raise ValueError(
                "MOSS-TTS continuation mode accepts one reference audio segment."
            )
        ref_text_value = ref_text_values[0] if ref_text_values else ""

        user_kwargs = {
            "text": (
                text if normalized_mode == "generation" else ref_text_value + text
            ),
            "tokens": kwargs.get("tokens"),
            "instruction": kwargs.get("instruction"),
            "quality": kwargs.get("quality"),
            "sound_event": kwargs.get("sound_event"),
            "ambient_sound": kwargs.get("ambient_sound"),
            "language": kwargs.get("language"),
            "scene": kwargs.get("scene"),
        }
        if normalized_mode == "generation" and prompt_audio_codes is not None:
            user_kwargs["reference"] = prompt_audio_codes_list

        if normalized_mode == "generation":
            conversations = [processor.build_user_message(**user_kwargs)]
        else:
            if prompt_audio_codes is None:
                raise ValueError(
                    "continuation mode requires ref_audio or prompt_audio_codes"
                )
            conversations = [
                processor.build_user_message(**user_kwargs),
                processor.build_assistant_message(
                    audio_codes_list=prompt_audio_codes_list
                ),
            ]

        batch = processor([conversations], mode=normalized_mode)
        if self.config.is_local_transformer:
            outputs = self.generate_local_ids(
                batch["input_ids"],
                max_new_tokens=int(max_tokens if max_tokens is not None else 4096),
                text_temperature=float(kwargs.get("text_temperature", 1.5)),
                text_top_p=float(kwargs.get("text_top_p", 1.0)),
                text_top_k=int(kwargs.get("text_top_k", 50)),
                text_repetition_penalty=float(
                    kwargs.get("text_repetition_penalty", 1.0)
                ),
                audio_temperature=float(kwargs.get("audio_temperature", 1.0)),
                audio_top_p=float(kwargs.get("audio_top_p", 0.95)),
                audio_top_k=int(kwargs.get("audio_top_k", 50)),
                audio_repetition_penalty=float(
                    kwargs.get("audio_repetition_penalty", 1.1)
                ),
                n_vq_for_inference=kwargs.get("n_vq_for_inference"),
            )
        else:
            max_new_tokens = int(
                max_tokens
                if max_tokens is not None
                else self._generation_config_value("max_new_tokens", 4096)
            )
            text_temperature = float(
                kwargs.get(
                    "text_temperature",
                    self._generation_config_value("temperature", 1.5),
                )
            )
            text_top_p = float(
                kwargs.get("text_top_p", self._generation_config_value("top_p", 1.0))
            )
            audio_temperature = float(
                kwargs.get(
                    "audio_temperature",
                    self._generation_config_value("temperature", 1.7),
                )
            )
            audio_top_p = float(
                kwargs.get("audio_top_p", self._generation_config_value("top_p", 0.8))
            )
            audio_top_k = int(
                kwargs.get("audio_top_k", self._generation_config_value("top_k", 25))
            )
            audio_repetition_penalty = float(
                kwargs.get(
                    "audio_repetition_penalty",
                    self._generation_config_value("repetition_penalty", 1.0),
                )
            )
            outputs = self.generate_delay_pattern_ids(
                batch["input_ids"],
                max_new_tokens=max_new_tokens,
                text_temperature=text_temperature,
                text_top_p=text_top_p,
                text_top_k=int(kwargs.get("text_top_k", 50)),
                audio_temperature=audio_temperature,
                audio_top_p=audio_top_p,
                audio_top_k=audio_top_k,
                audio_repetition_penalty=audio_repetition_penalty,
            )
        audio, token_count = self._decode_generated_audio(
            outputs,
            source=kwargs.get("audio_tokenizer_source"),
        )
        yield self._build_generation_result(
            audio=audio,
            started_at=started_at,
            token_count=token_count,
            prompt_token_count=int(batch["input_ids"].shape[1]),
        )
