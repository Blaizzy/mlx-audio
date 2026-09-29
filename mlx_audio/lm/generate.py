# Copyright © 2023-2024 Apple Inc.
# Vendored from mlx-lm.

import contextlib
import inspect
import time
import warnings
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
    max_rec_size = mx.device_info()["max_recommended_working_set_size"]
    if model_bytes > 0.9 * max_rec_size:
        warnings.warn(
            "Generating with a model that requires "
            f"{model_bytes / 1 << 30:.2f} GB, close to the maximum recommended "
            f"size of {max_rec_size / 1 << 30:.2f} GB. This can be slow; see the "
            "MLX documentation on tuning the wired limit.",
            stacklevel=2,
        )
    old_limit = mx.set_wired_limit(max_rec_size)
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


class _BoundedStreamingDetokenizer:
    """Compatibility detokenizer for tokenizers without a Rust backend."""

    _REDECODE_WINDOW = 64
    _BOUNDARY_OVERLAP = 32

    def __init__(self, tokenizer, **decode_kwargs):
        self.tokenizer = tokenizer
        self.decode_kwargs = decode_kwargs
        self.clean_spaces = bool(
            getattr(tokenizer, "clean_up_tokenization_spaces", False)
        )
        self.reset()

    def reset(self):
        self.tokens = []
        self._text = ""
        self._current_tokens = []
        self._current_text = ""
        self._next_compaction = self._REDECODE_WINDOW
        self.offset = 0

    def add_token(self, token):
        self.tokens.append(token)
        self._current_tokens.append(token)
        self._compact_pending_tokens()

    def _decode(self, token_ids):
        return self.tokenizer.decode(token_ids, **self.decode_kwargs)

    def _compact_pending_tokens(self):
        if len(self._current_tokens) <= self._next_compaction:
            return

        full_text = self._decode(self._current_tokens)
        preferred_cut = len(self._current_tokens) - self._BOUNDARY_OVERLAP
        for cut in range(preferred_cut, 0, -1):
            prefix = self._decode(self._current_tokens[:cut])
            suffix = self._decode(self._current_tokens[cut:])
            if (
                prefix + suffix == full_text
                and not prefix.endswith("\ufffd")
                and not suffix.startswith("\ufffd")
            ):
                self._text += prefix
                self._current_tokens = self._current_tokens[cut:]
                self._current_text = ""
                self._next_compaction = self._REDECODE_WINDOW
                return

        # No split decodes independently (e.g. a decoder that strips the leading
        # space of every sequence). Keep the tokens pending, which stays correct
        # at the cost of re-decoding them, and back off before searching again.
        self._next_compaction = 2 * len(self._current_tokens)

    def finalize(self):
        if self._current_tokens:
            self._text += self._decode(self._current_tokens)
        self._current_tokens = []
        self._current_text = ""
        self._next_compaction = self._REDECODE_WINDOW

    @property
    def text(self):
        has_incomplete_codepoint = False
        if self._current_tokens:
            current_text = self._decode(self._current_tokens)
            has_incomplete_codepoint = current_text.endswith("\ufffd")
            if has_incomplete_codepoint:
                current_text = current_text[:-1]
            elif self.clean_spaces and current_text.endswith(" "):
                current_text = current_text[:-1]
            self._current_text = current_text

        if not has_incomplete_codepoint and self._current_text.endswith("\n"):
            self._text += self._current_text
            self._current_tokens = []
            self._current_text = ""
            self._next_compaction = self._REDECODE_WINDOW

        return self._text + self._current_text

    @property
    def last_segment(self):
        text = self.text
        segment = text[self.offset :]
        self.offset = len(text)
        return segment


class _BackendStreamingDetokenizer:
    """Linear incremental decoding through tokenizers.DecodeStream."""

    def __init__(self, tokenizer, backend, *, skip_special_tokens=False):
        from tokenizers.decoders import DecodeStream

        self.tokenizer = tokenizer
        self.backend = backend
        self.skip_special_tokens = skip_special_tokens
        self._decode_stream_type = DecodeStream
        self.reset()

    def reset(self):
        self.tokens = []
        self._text = ""
        self.offset = 0
        self._stream = self._decode_stream_type(
            skip_special_tokens=self.skip_special_tokens
        )

    def add_token(self, token):
        token = int(token)
        self.tokens.append(token)
        self._text += self._stream.step(self.backend, token) or ""

    def finalize(self):
        return None

    @property
    def text(self):
        return self._text

    @property
    def last_segment(self):
        segment = self._text[self.offset :]
        self.offset = len(self._text)
        return segment


class StreamingDetokenizer:
    """Backend-aware incremental detokenizer with a bounded fallback."""

    def __init__(self, tokenizer, **decode_kwargs):
        backend = getattr(tokenizer, "backend_tokenizer", None)
        if backend is None:
            backend = getattr(tokenizer, "_tokenizer", None)

        skip_special_tokens = bool(decode_kwargs.get("skip_special_tokens", False))
        extra_decode_kwargs = set(decode_kwargs) - {"skip_special_tokens"}
        try:
            from tokenizers import Tokenizer
            from tokenizers.decoders import DecodeStream  # noqa: F401
        except ImportError:
            Tokenizer = ()

        if isinstance(backend, Tokenizer) and not extra_decode_kwargs:
            self._impl = _BackendStreamingDetokenizer(
                tokenizer,
                backend,
                skip_special_tokens=skip_special_tokens,
            )
        else:
            self._impl = _BoundedStreamingDetokenizer(tokenizer, **decode_kwargs)

    def reset(self):
        self._impl.reset()

    def add_token(self, token):
        self._impl.add_token(token)

    def finalize(self):
        return self._impl.finalize()

    @property
    def tokens(self):
        return self._impl.tokens

    @property
    def text(self):
        return self._impl.text

    @property
    def last_segment(self):
        return self._impl.last_segment


def _supports_input_embeddings(model: nn.Module) -> bool:
    try:
        return "input_embeddings" in inspect.signature(model.__call__).parameters
    except (TypeError, ValueError):
        return False


def _eos_ids(tokenizer) -> set[int]:
    ids = getattr(tokenizer, "eos_token_ids", None)
    if ids is None:
        ids = getattr(tokenizer, "eos_token_id", None)
    if ids is None:
        return set()
    if isinstance(ids, int):
        return {ids}
    return {int(i) for i in ids}


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
    while count != max_tokens:
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
    detokenizer = StreamingDetokenizer(tokenizer)
    eos_ids = _eos_ids(tokenizer)
    last_token = last_logprobs = None
    prompt_tps = 0.0
    index = -1
    finish_reason = "length"

    with wired_limit(model, [generation_stream]):
        started = time.perf_counter()
        for index, (token, logprobs) in enumerate(
            generate_step(prompt, model, max_tokens=max_tokens, **kwargs)
        ):
            if index == 0:
                prompt_tps = prompt.size / (time.perf_counter() - started)
                started = time.perf_counter()
            last_token, last_logprobs = token, logprobs
            if token in eos_ids:
                finish_reason = "stop"
                break
            detokenizer.add_token(token)
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
        finish_reason=finish_reason,
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
