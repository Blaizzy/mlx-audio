# Copyright © 2023-2024 Apple Inc.
# Derived from mlx-lm v0.31.3 (ed1fca4cef15a824c5f1702c80f70b4cffc8e4dd),
# mlx_lm/generate.py. Trimmed to mlx-audio's single-stream generation APIs.

import contextlib
import inspect
import time
from dataclasses import dataclass
from typing import Any, Callable, Generator, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_reduce

from .models.cache import make_prompt_cache

generation_stream = mx.new_thread_local_stream(mx.default_device())


@contextlib.contextmanager
def wired_limit(model: nn.Module, streams: Optional[List[mx.Stream]] = None):
    if not mx.metal.is_available():
        yield
        return

    model_bytes = tree_reduce(
        lambda total, value: (
            total + value.nbytes if isinstance(value, mx.array) else total
        ),
        model,
        0,
    )
    old_limit = mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    try:
        yield
    finally:
        if streams is None:
            mx.synchronize()
        else:
            for stream in streams:
                mx.synchronize(stream)
        mx.set_wired_limit(old_limit)


@dataclass
class GenerationResponse:
    text: str
    token: int
    logprobs: mx.array
    from_draft: bool
    prompt_tokens: int
    prompt_tps: float
    generation_tokens: int
    generation_tps: float
    peak_memory: float
    finish_reason: Optional[str] = None


class _NaiveDetokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.reset()

    def reset(self):
        self.tokens = []
        self.text = ""
        self.offset = 0

    def add_token(self, token):
        self.tokens.append(token)
        self.text = self.tokenizer.decode(self.tokens)

    def finalize(self):
        self.text = self.tokenizer.decode(self.tokens)

    @property
    def last_segment(self):
        segment = self.text[self.offset :]
        self.offset = len(self.text)
        return segment


def _supports_input_embeddings(model: nn.Module) -> bool:
    try:
        return "input_embeddings" in inspect.signature(model.__call__).parameters
    except (TypeError, ValueError):
        return False


def _eos_ids(tokenizer) -> set[int]:
    if hasattr(tokenizer, "eos_token_ids"):
        return set(tokenizer.eos_token_ids)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    return set() if eos_token_id is None else {eos_token_id}


def _encode(tokenizer, prompt: Union[str, mx.array, List[int]]) -> mx.array:
    if isinstance(prompt, mx.array):
        return prompt
    if isinstance(prompt, str):
        bos_token = getattr(tokenizer, "bos_token", None)
        prompt = tokenizer.encode(
            prompt,
            add_special_tokens=bos_token is None or not prompt.startswith(bos_token),
        )
    return mx.array(prompt)


def generate_step(
    prompt: mx.array,
    model: nn.Module,
    *,
    max_tokens: int = 256,
    sampler: Optional[Callable[[mx.array], mx.array]] = None,
    logits_processors: Optional[List[Callable[[mx.array, mx.array], mx.array]]] = None,
    max_kv_size: Optional[int] = None,
    prompt_cache: Optional[Any] = None,
    prefill_step_size: int = 2048,
    prompt_progress_callback: Optional[Callable[[int, int], None]] = None,
    input_embeddings: Optional[mx.array] = None,
) -> Generator[Tuple[mx.array, mx.array], None, None]:
    if input_embeddings is not None:
        if not _supports_input_embeddings(model):
            raise ValueError("Model does not support input embeddings.")
        if len(prompt) and len(prompt) != len(input_embeddings):
            raise ValueError("prompt and input_embeddings must have the same length.")
    elif not len(prompt):
        raise ValueError("Either input_embeddings or prompt must be provided.")

    if prompt_cache is None:
        prompt_cache = make_prompt_cache(model, max_kv_size=max_kv_size)
    prompt_progress_callback = prompt_progress_callback or (lambda *_: None)
    sampler = sampler or (lambda logits: mx.argmax(logits, axis=-1))
    tokens = None

    def model_call(input_tokens, embeddings=None):
        if embeddings is None:
            return model(input_tokens, cache=prompt_cache)
        return model(input_tokens, cache=prompt_cache, input_embeddings=embeddings)

    def step(input_tokens, embeddings=None):
        nonlocal tokens
        with mx.stream(generation_stream):
            logits = model_call(
                input_tokens[None],
                embeddings[None] if embeddings is not None else None,
            )[:, -1, :]
            if logits_processors and len(input_tokens):
                tokens = (
                    mx.concatenate([tokens, input_tokens])
                    if tokens is not None
                    else input_tokens
                )
                for processor in logits_processors:
                    logits = processor(tokens, logits)
            logprobs = logits - mx.logsumexp(logits, keepdims=True)
            return sampler(logprobs), logprobs.squeeze(0)

    with mx.stream(generation_stream):
        total = len(input_embeddings) if input_embeddings is not None else len(prompt)
        processed = 0
        prompt_progress_callback(processed, total)
        while total - processed > 1:
            count = min(prefill_step_size, total - processed - 1)
            model_call(
                prompt[:count][None],
                (
                    input_embeddings[:count][None]
                    if input_embeddings is not None
                    else None
                ),
            )
            mx.eval([cache.state for cache in prompt_cache])
            processed += count
            prompt_progress_callback(processed, total)
            prompt = prompt[count:]
            if input_embeddings is not None:
                input_embeddings = input_embeddings[count:]
            mx.clear_cache()
        token, logprobs = step(prompt, input_embeddings)

    mx.async_eval(token, logprobs)
    count = 0
    while count < max_tokens:
        next_token, next_logprobs = step(token)
        mx.async_eval(next_token, next_logprobs)
        if count == 0:
            mx.eval(token)
            prompt_progress_callback(total, total)
        yield token.item(), logprobs
        if count % 256 == 0:
            mx.clear_cache()
        token, logprobs = next_token, next_logprobs
        count += 1


def stream_generate(
    model: nn.Module,
    tokenizer,
    prompt: Union[str, mx.array, List[int]],
    max_tokens: int = 256,
    **kwargs,
) -> Generator[GenerationResponse, None, None]:
    if kwargs.pop("draft_model", None) is not None:
        raise ValueError("Speculative decoding is not implemented in mlx-audio.lm.")

    prompt = _encode(tokenizer, prompt)
    detokenizer = _NaiveDetokenizer(tokenizer)
    eos_ids = _eos_ids(tokenizer)
    last_token = last_logprobs = None
    prompt_tps = 0.0
    index = -1

    with wired_limit(model, [generation_stream]):
        started = time.perf_counter()
        for index, (token, logprobs) in enumerate(
            generate_step(prompt, model, max_tokens=max_tokens, **kwargs)
        ):
            if index == 0:
                prompt_tps = prompt.size / (time.perf_counter() - started)
                started = time.perf_counter()
            if token in eos_ids:
                break
            detokenizer.add_token(token)
            last_token, last_logprobs = token, logprobs
            if index + 1 == max_tokens:
                break
            yield GenerationResponse(
                text=detokenizer.last_segment,
                token=token,
                logprobs=logprobs,
                from_draft=False,
                prompt_tokens=prompt.size,
                prompt_tps=prompt_tps,
                generation_tokens=index + 1,
                generation_tps=(index + 1) / (time.perf_counter() - started),
                peak_memory=mx.get_peak_memory() / 1e9,
            )

    if last_token is None:
        return
    detokenizer.finalize()
    yield GenerationResponse(
        text=detokenizer.last_segment,
        token=last_token,
        logprobs=last_logprobs,
        from_draft=False,
        prompt_tokens=prompt.size,
        prompt_tps=prompt_tps,
        generation_tokens=index + 1,
        generation_tps=(index + 1) / (time.perf_counter() - started),
        peak_memory=mx.get_peak_memory() / 1e9,
        finish_reason="stop" if last_token in eos_ids else "length",
    )


def generate(
    model: nn.Module, tokenizer, prompt: Union[str, List[int]], verbose=False, **kwargs
) -> str:
    text = ""
    for response in stream_generate(model, tokenizer, prompt, **kwargs):
        if verbose:
            print(response.text, end="", flush=True)
        text += response.text
    if verbose:
        print()
    return text
