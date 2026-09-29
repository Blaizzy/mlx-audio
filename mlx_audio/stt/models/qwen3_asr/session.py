# Copyright 2026 The NetEase Youdao team and MLX-Audio contributors.
# SPDX-License-Identifier: Apache-2.0
# Stable-prefix algorithm adapted from netease-youdao/Confucius4-R2T2,
# r2t2/r2t2_asr.py and example.py at 80c22e6140bcb9166fb9906798894fc8b18c8309.
"""Cooperative Qwen3-ASR input streaming with an immutable, committed text prefix."""

import math
from collections import deque
from threading import Lock

import numpy as np


class Qwen3ASRStreamingSession:
    input_sample_rate = 16000

    def __init__(
        self,
        model,
        *,
        temperature=0.0,
        language=None,
        context="",
        chunk_size_sec=0.16,
        lookahead_sec=0.16,
        unfixed_token_num=1,
        max_audio_seconds=30.0,
    ):
        if temperature != 0.0:
            raise ValueError("Qwen3-ASR streaming supports greedy temperature=0 only")
        if not math.isfinite(chunk_size_sec) or not 0.08 <= chunk_size_sec <= 2.0:
            raise ValueError("chunk_size_sec must be between 0.08 and 2.0")
        if not math.isfinite(lookahead_sec) or lookahead_sec < 0:
            raise ValueError("lookahead_sec must be finite and nonnegative")
        if not isinstance(unfixed_token_num, int) or unfixed_token_num < 0:
            raise ValueError("unfixed_token_num must be a nonnegative integer")
        if not math.isfinite(max_audio_seconds) or max_audio_seconds <= 0:
            raise ValueError("max_audio_seconds must be finite and positive")
        if not hasattr(model, "_tokenizer") or not hasattr(model, "_feature_extractor"):
            raise RuntimeError("Tokenizer/feature extractor not loaded")
        if language is not None:
            supported = {s.lower(): s for s in model.config.support_languages}
            if language.lower() not in supported:
                raise ValueError(f"Unsupported language: {language}")
            language = supported[language.lower()]
        self.model = model
        self.language = language
        self.context = context
        self.chunk_samples = round(chunk_size_sec * self.input_sample_rate)
        self.first_chunk_samples = self.chunk_samples + round(
            lookahead_sec * self.input_sample_rate
        )
        self.max_samples = round(max_audio_seconds * self.input_sample_rate)
        if self.first_chunk_samples > self.max_samples:
            raise ValueError("Initial chunk and lookahead exceed max_audio_seconds")
        self.unfixed_token_num = unfixed_token_num
        self._tokenizer = model._tokenizer
        self._eos = model._eos_token_ids()
        self._lock = Lock()
        self._queue = deque()
        self._queued = 0
        self._received = 0
        self._closed = False
        self._done = False
        self._audio = np.empty(0, dtype=np.float32)
        self._decoder = None
        self._tokens = []
        self._prefix = ""
        self.text = ""
        self._final = False
        self._chunk_limit = 0
        self._next_token_limit = max(1, self.first_chunk_samples // 1280)

    @property
    def done(self):
        return self._done

    def feed(self, samples):
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim != 1 or not np.isfinite(samples).all():
            raise ValueError("expected finite mono PCM samples")
        with self._lock:
            if self._closed:
                raise RuntimeError("streaming input is closed")
            if self._received + samples.size > self.max_samples:
                raise BufferError(
                    "Qwen3-ASR utterance exceeds max_audio_seconds; close at a turn boundary"
                )
            if samples.size:
                self._queue.append(samples.copy())
                self._queued += samples.size
                self._received += samples.size

    def close(self):
        with self._lock:
            self._closed = True

    def cancel(self):
        """Cancel on the decoder thread, discarding queued audio and decoder state."""
        with self._lock:
            self._closed = self._done = True
            self._queue.clear()
            self._queued = 0
        if self._decoder is not None:
            self._decoder.close()
        self._decoder = None
        self._audio = np.empty(0, dtype=np.float32)

    def _start_decode(self):
        size = self.chunk_samples if self._audio.size else self.first_chunk_samples
        parts = []
        with self._lock:
            if self._queued < size and not self._closed:
                return False
            take = min(size, self._queued)
            self._final = self._closed and self._queued <= size
            while take:
                head = self._queue.popleft()
                count = min(take, head.size)
                parts.append(head[:count])
                if count < head.size:
                    self._queue.appendleft(head[count:])
                self._queued -= count
                take -= count
        if parts:
            self._audio = np.concatenate([self._audio, *parts])
        if not self._audio.size:
            self._done = self._final
            return False
        # Revisit even an exact chunk boundary on close: release the held tail.
        self._chunk_limit = (
            max(512, 2 * self.unfixed_token_num)
            if self._final
            else max(self._next_token_limit, 2 * self.unfixed_token_num)
        )
        self._decoder = self.model._streaming_decode(
            self._audio,
            self._prefix,
            language=self.language,
            context=self.context,
            max_tokens=self._chunk_limit,
        )
        self._tokens = []
        return True

    def _finish_decode(self):
        self._decoder.close()
        self._decoder = None
        generated = self._tokenizer.decode(self._tokens, skip_special_tokens=True)
        raw = (self._prefix + generated).split("|", 1)[0]
        ids = self._tokenizer.encode(raw, add_special_tokens=False)
        # The unstable tail is re-generated on the next audio update. Retain
        # complete UTF-8 characters, including when a token ends mid-character.
        end = max(0, len(ids) - (0 if self._final else self.unfixed_token_num))
        candidate = self._tokenizer.decode(ids[:end], skip_special_tokens=True)
        while "\ufffd" in candidate and end:
            end -= 1
            candidate = self._tokenizer.decode(ids[:end], skip_special_tokens=True)
        if len(candidate) < len(self._prefix):
            candidate = self._prefix
        if not candidate.startswith(self._prefix):
            raise RuntimeError("Qwen3-ASR tokenizer changed the committed prefix")
        self._prefix = candidate
        if self.language is not None:
            visible = candidate
        elif "<asr_text>" in candidate:
            visible = candidate.split("<asr_text>", 1)[1]
        else:
            visible = ""
        visible = visible.lstrip()
        if not visible.startswith(self.text):
            raise RuntimeError("Qwen3-ASR decoder changed committed text")
        delta = visible[len(self.text) :]
        self.text = visible
        # Match upstream's adaptive token allowance: give a stalled hypothesis
        # more decoding room; Chinese characters usually need more tokens.
        base = max(1, self.chunk_samples // 1280)
        limit = base if delta else self._next_token_limit + 1
        if self.text and "\u4e00" <= self.text[-1] <= "\u9fff":
            limit *= 2
        self._next_token_limit = min(max(4, 2 * base), limit)
        if self._final:
            self._done = True
            self._audio = np.empty(0, dtype=np.float32)
        return [delta] if delta else []

    def step(self, *, max_decode_tokens=4):
        if not isinstance(max_decode_tokens, int) or max_decode_tokens <= 0:
            raise ValueError("max_decode_tokens must be a positive integer")
        if self.done:
            return []
        if self._decoder is None and not self._start_decode():
            return []
        for _ in range(max_decode_tokens):
            try:
                token = next(self._decoder)
            except StopIteration:
                return self._finish_decode()
            if token in self._eos:
                return self._finish_decode()
            self._tokens.append(token)
            if "|" in self._tokenizer.decode(self._tokens, skip_special_tokens=True):
                return self._finish_decode()
            if len(self._tokens) >= self._chunk_limit:
                return self._finish_decode()
        return []
