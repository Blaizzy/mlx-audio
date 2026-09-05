"""Cooperative live-input adapter for the realtime STT server."""

from collections import deque
from threading import Lock

import mlx.core as mx
import numpy as np

from . import tokenizer
from .audio import StreamingLogMelSpectrogram
from .streaming import ConformerStreamingState


class NemotronStreamingSession:
    """Single-consumer decoder; feed/close may run on another thread.

    Queued input is limited to 30 seconds. Overflow is explicit; audio is never
    silently dropped. step budgets joint evaluations (including blanks), not
    just visible text, so silence also yields cooperatively.
    """

    def __init__(self, model, *, temperature=0.0, language=None):
        if temperature != 0.0:
            raise ValueError("Nemotron streaming supports greedy temperature=0 only")
        self.model = model
        self.language = language or model.default_language
        model._resolve_prompt_index(self.language)
        self.input_sample_rate = model.preprocessor_config.sample_rate
        self._lock = Lock()
        self.reset()

    def reset(self):
        """Reset from the decoder thread, with no concurrent step in progress."""
        with self._lock:
            self._audio = deque()
            self._queued = 0
            self._closed = False
            self._done = False
        self._frontend = StreamingLogMelSpectrogram(self.model.preprocessor_config)
        self._encoder = ConformerStreamingState(
            self.model.encoder, att_context_size=self.model.default_att_context_size
        )
        self._encoded = deque()
        self._frame = 0
        self._symbols = 0
        self._last_token = self.model.blank_id
        self._hidden = None
        self._flushed = False
        self._has_text = False

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
            if self._queued + samples.size > 30 * self.input_sample_rate:
                raise BufferError("Nemotron streaming input queue exceeds 30 seconds")
            if samples.size:
                self._audio.append(samples.copy())
                self._queued += samples.size

    def close(self):
        """Signal end-of-input; step drains and flushes it exactly once."""
        with self._lock:
            self._closed = True

    def cancel(self):
        """Discard a session on the decoder thread without producing a final."""
        self.reset()
        with self._lock:
            self._closed = True
            self._done = True

    def _ingest(self):
        # At most one native encoder chunk of audio per cooperative step.
        limit = self._encoder.chunk_mel * self.model.preprocessor_config.hop_length
        parts = []
        with self._lock:
            while self._audio and limit:
                head = self._audio.popleft()
                count = min(limit, head.size)
                parts.append(head[:count])
                if count < head.size:
                    self._audio.appendleft(head[count:])
                self._queued -= count
                limit -= count
            final = self._closed and not self._audio
        if not parts and not final:
            return
        samples = np.concatenate(parts) if parts else np.empty(0, dtype=np.float32)
        # Empty streams should not manufacture a reflected silence frame.
        if final and not samples.size and not self._frontend.total_samples:
            self._flushed = True
            return
        mel = self._frontend.push(mx.array(samples), final=final)
        for encoded in self._encoder.push(mel, final=final):
            prompted = self.model.apply_prompt(encoded, self.language)
            self._encoder.materialize(prompted)
            self._encoded.append(prompted)
        self._flushed = final

    def step(self, *, max_decode_tokens=4):
        if max_decode_tokens <= 0:
            raise ValueError("max_decode_tokens must be positive")
        if self.done:
            return []
        if not self._encoded and not self._flushed:
            self._ingest()
        deltas = []
        for _ in range(max_decode_tokens):
            if not self._encoded:
                break
            feature = self._encoded[0][:, self._frame : self._frame + 1]
            token = (
                mx.array([[self._last_token]], dtype=mx.int32)
                if self._last_token != self.model.blank_id
                else None
            )
            output, (h, c) = self.model.decoder(token, self._hidden)
            hidden = (h.astype(feature.dtype), c.astype(feature.dtype))
            prediction = int(
                mx.argmax(self.model.joint(feature, output.astype(feature.dtype)))
            )
            if prediction != self.model.blank_id:
                self._last_token = prediction
                self._hidden = hidden
                mx.eval(*hidden)
                text = tokenizer.decode([prediction], self.model.vocabulary)
                if not self._has_text:
                    text = text.lstrip()
                if text:
                    self._has_text = True
                    deltas.append(text)
                self._symbols += 1
            if prediction == self.model.blank_id or (
                self.model.max_symbols is not None
                and self._symbols >= self.model.max_symbols
            ):
                self._frame += 1
                self._symbols = 0
                if self._frame == self._encoded[0].shape[1]:
                    self._encoded.popleft()
                    self._frame = 0
        if self._flushed and not self._encoded:
            self._done = True
        return deltas
