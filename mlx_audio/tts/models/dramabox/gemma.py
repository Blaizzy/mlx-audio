from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import mlx.core as mx
from mlx_lm import load as mlx_lm_load
from mlx_lm.models.cache import create_causal_mask


@dataclass
class EncodedPrompt:
    hidden_states: list[mx.array]
    attention_mask: mx.array


def load_text_encoder(model_id: str):
    return mlx_lm_load(model_id)


def _language_core(model):
    if hasattr(model, "language_model"):
        model = model.language_model
    return model.model if hasattr(model, "model") else model


def _tokenize(tokenizer, text: str, max_length: int) -> tuple[mx.array, mx.array]:
    tokenizer = getattr(tokenizer, "_tokenizer", tokenizer)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = tokenizer(
        text.strip(),
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="np",
    )
    return (
        mx.array(encoded["input_ids"], dtype=mx.int32),
        mx.array(encoded["attention_mask"], dtype=mx.int32),
    )


def _left_padding(attention_mask: mx.array) -> mx.array:
    return mx.sum(attention_mask == 0, axis=-1)


def _attention_masks(h: mx.array, core, attention_mask: mx.array):
    left_padding = _left_padding(attention_mask)
    global_mask = create_causal_mask(h.shape[1], left_padding=left_padding)
    if core.sliding_window_pattern > 1:
        sliding_window_mask = create_causal_mask(
            h.shape[1],
            window_size=core.window_size,
            left_padding=left_padding,
        )
    else:
        sliding_window_mask = None
    return global_mask, sliding_window_mask


def encode_prompt_hidden_states(
    model,
    tokenizer,
    text: str,
    max_length: int = 1024,
) -> EncodedPrompt:
    input_ids, attention_mask = _tokenize(tokenizer, text, max_length)
    core = _language_core(model)
    h = core.embed_tokens(input_ids)
    h *= mx.array(math.sqrt(core.args.hidden_size), mx.bfloat16).astype(h.dtype)
    hidden_states = []
    global_mask, sliding_window_mask = _attention_masks(h, core, attention_mask)
    for i, layer in enumerate(core.layers):
        hidden_states.append(h)
        is_global = i % core.sliding_window_pattern == core.sliding_window_pattern - 1
        mask = global_mask if is_global else sliding_window_mask
        h = layer(h, mask, None)
    hidden_states.append(core.norm(h))
    return EncodedPrompt(hidden_states=hidden_states, attention_mask=attention_mask)


def encode_prompts_hidden_states(
    model,
    tokenizer,
    prompts: Sequence[str],
    max_length: int = 1024,
) -> list[EncodedPrompt]:
    return [
        encode_prompt_hidden_states(model, tokenizer, prompt, max_length=max_length)
        for prompt in prompts
    ]
