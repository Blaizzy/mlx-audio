from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx


@dataclass
class TTSSamplingParams:
    temperature: float = 1.15
    top_k: int = 106
    top_p: float = 0.0
    min_p: float = 0.18
    max_tokens: int = 1024
    ignore_eos: bool = False
    repetition_window: int = 50
    repetition_penalty: float = 1.2
    repetition_codebooks: int = 8
    seed: Optional[int] = None


@dataclass
class Zonos2GenerationState:
    n_codebooks: int = 9
    eoa_id: int = 1024
    text_vocab: int = 519
    eos_frame: Optional[int] = None
    eos_countdown: int = 0
    generated: list[list[int]] = None

    def __post_init__(self) -> None:
        if self.generated is None:
            self.generated = []

    @property
    def finished(self) -> bool:
        return self.eos_frame is not None and self.eos_countdown <= 0

    def append(self, frame: list[int], ignore_eos: bool = False) -> None:
        self.generated.append(frame[: self.n_codebooks])
        if ignore_eos:
            return
        if self.eos_frame is None:
            eos_cols = [
                frame[i] == self.eoa_id
                for i in range(min(self.n_codebooks, len(frame)))
            ]
            if any(eos_cols):
                step = len(self.generated) - 1
                max_eos_cb = max(i for i, is_eos in enumerate(eos_cols) if is_eos)
                self.eos_frame = max(0, step - max_eos_cb)
                self.eos_countdown = self.n_codebooks + 1
        if self.eos_frame is not None and self.eos_countdown > 0:
            self.eos_countdown -= 1


def _apply_repetition_penalty(
    logits: mx.array,
    generated: list[list[int]],
    *,
    penalty: float,
    window: int,
    repetition_codebooks: int,
) -> mx.array:
    if penalty <= 1.0 or window <= 0 or not generated:
        return logits
    n_codebooks, vocab_size = logits.shape
    if repetition_codebooks < 0:
        limit = n_codebooks
    else:
        limit = min(n_codebooks, int(repetition_codebooks))
    recent = generated[-int(window) :]
    vocab = mx.arange(vocab_size)
    rows = []
    for cb in range(limit):
        seen = {
            int(row[cb])
            for row in recent
            if cb < len(row) and 0 <= int(row[cb]) < vocab_size
        }
        row = logits[cb]
        if seen:
            token_ids = mx.array(sorted(seen), dtype=mx.int32)
            mask = mx.any(vocab[None, :] == token_ids[:, None], axis=0)
            penalized = mx.where(row > 0, row / penalty, row * penalty)
            row = mx.where(mask, penalized, row)
        rows.append(row)
    rows.extend(logits[cb] for cb in range(limit, n_codebooks))
    return mx.stack(rows, axis=0)


def _apply_top_k(logits: mx.array, top_k: int) -> mx.array:
    if top_k <= 0 or top_k >= logits.shape[-1]:
        return logits
    kth = mx.sort(logits, axis=-1)[:, -int(top_k)][:, None]
    return mx.where(logits < kth, float("-inf"), logits)


def _normalize_probs(probs: mx.array) -> mx.array:
    denom = mx.sum(probs, axis=-1, keepdims=True)
    return probs / mx.maximum(denom, 1e-12)


def _apply_top_p(probs: mx.array, top_p: float) -> mx.array:
    if top_p <= 0.0 or top_p >= 1.0:
        return probs
    order = mx.argsort(-probs, axis=-1)
    sorted_probs = mx.take_along_axis(probs, order, axis=-1)
    cumsum = mx.cumsum(sorted_probs, axis=-1)
    mask = cumsum - sorted_probs > top_p
    sorted_probs = mx.where(mask, 0.0, sorted_probs)
    out = mx.put_along_axis(mx.zeros_like(probs), order, sorted_probs, axis=-1)
    return _normalize_probs(out)


def _apply_min_p(probs: mx.array, min_p: float) -> mx.array:
    if min_p <= 0.0:
        return probs
    threshold = mx.max(probs, axis=-1, keepdims=True) * float(min_p)
    out = mx.where(probs < threshold, 0.0, probs)
    return _normalize_probs(out)


def sample_frame(
    logits: mx.array,
    state: Zonos2GenerationState,
    params: TTSSamplingParams,
    key: Optional[mx.array] = None,
) -> list[int]:
    logits = mx.array(logits, dtype=mx.float32)
    if logits.ndim != 2:
        raise ValueError(f"logits must be [codebooks, vocab], got {logits.shape}")

    logits = _apply_repetition_penalty(
        logits,
        state.generated,
        penalty=float(params.repetition_penalty),
        window=int(params.repetition_window),
        repetition_codebooks=int(params.repetition_codebooks),
    )

    if params.temperature <= 1e-8:
        ids = mx.argmax(logits, axis=-1).astype(mx.int32)
    else:
        filtered = logits / float(params.temperature)
        filtered = _apply_top_k(filtered, int(params.top_k))
        probs = mx.softmax(filtered, axis=-1)
        probs = _apply_top_p(probs, float(params.top_p))
        probs = _apply_min_p(probs, float(params.min_p))
        finite = mx.all(mx.isfinite(probs), axis=-1)
        positive = mx.sum(probs, axis=-1) > 0
        valid = finite & positive
        safe_probs = mx.where(mx.isfinite(probs), probs, 0.0)
        sample_logits = mx.where(
            valid[:, None],
            mx.log(mx.maximum(safe_probs, 1e-20)),
            mx.zeros_like(filtered),
        )
        sampled = mx.random.categorical(sample_logits, axis=-1, key=key).astype(
            mx.int32
        )
        greedy = mx.argmax(filtered, axis=-1).astype(mx.int32)
        ids = mx.where(valid, sampled, greedy)

    return [int(token) for token in ids.tolist()] + [int(state.text_vocab)]


def format_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
