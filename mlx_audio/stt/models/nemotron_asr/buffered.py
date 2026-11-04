"""Buffered streaming for Nemotron Unified (``chunked_limited_with_rc``) checkpoints.

The Unified encoder has non-causal convolutions and a right context, so it cannot
reuse per-layer caches. Each step re-runs the mel frontend and the encoder on a sliding audio buffer of
left context + chunk + right context, keeps the chunk's encoder frames, and lets
greedy RNN-T decoding continue from the previous step's state.

Step ``k`` covers the audio chunk ``[k*C, (k+1)*C)``. It is the last step once
``(k+1)*C + R`` reaches the end of the audio; the last chunk then runs to the end
with no right context. A live session only runs a non-last step once it holds
more than ``(k+1)*C + R`` samples, so it segments a stream exactly as NeMo
segments the finished file.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from .audio import log_mel_spectrogram

# NVIDIA's recommended [chunk, right] encoder frames (80 ms each) per latency
# (chunk + right context). Other latencies take the chunk of the row at or
# below them and the rest as right context.
DELAY_CONTEXTS = {
    160: (1, 1),
    240: (1, 2),
    320: (1, 3),
    560: (2, 5),
    1120: (7, 7),
    2080: (13, 13),
}
DEFAULT_LEFT_CONTEXT = 70


@dataclass(frozen=True)
class BufferedContext:
    """Buffer layout in encoder frames: ``[left, chunk, right]``."""

    left: int
    chunk: int
    right: int

    @classmethod
    def from_att_context_size(cls, att_context_size) -> "BufferedContext":
        if len(att_context_size) != 3:
            raise ValueError(
                "buffered streaming needs att_context_size=[left, chunk, right]"
            )
        left, chunk, right = (int(v) for v in att_context_size)
        if left < 0 or chunk < 1 or right < 0:
            raise ValueError(
                f"invalid buffered context {att_context_size}: need left >= 0, "
                "chunk >= 1, right >= 0 encoder frames"
            )
        return cls(left, chunk, right)

    @classmethod
    def from_delay_ms(
        cls, delay_ms: int, left: int = DEFAULT_LEFT_CONTEXT
    ) -> "BufferedContext":
        """Context for a latency, rounded to whole 80 ms frames within 160-2080 ms."""
        frames = min(max(int(delay_ms / 80 + 0.5), 2), 26)  # round half up
        row = max(ms for ms in DELAY_CONTEXTS if ms <= frames * 80)
        chunk = DELAY_CONTEXTS[row][0]
        return cls(left, chunk, frames - chunk)

    def as_list(self) -> list[int]:
        return [self.left, self.chunk, self.right]


class BufferedEncoderState:
    """Turns PCM pushes into prompted encoder frames, one chunk per step.

    ``push`` returns ``(1, frames, d_model)`` arrays, already prompt-conditioned
    and evaluated, in audio order. Joined, they cover the audio once.
    """

    def __init__(self, model, context: BufferedContext, language: str | None):
        self.model = model
        self.context = context
        self.language = language
        pre = model.preprocessor_config
        self.hop = pre.hop_length
        self.frame_samples = model.encoder_config.subsampling_factor * self.hop
        self.left_samples = context.left * self.frame_samples
        self.chunk_samples = context.chunk * self.frame_samples
        self.right_samples = context.right * self.frame_samples
        self._samples = mx.zeros(0, dtype=mx.float32)
        self._start = 0  # absolute index of self._samples[0]
        self._total = 0
        self._next = 0  # absolute start of the next chunk
        self._prev_start = 0  # buffer start of the previous step
        self._full_window = False
        self._done = False

    @property
    def done(self) -> bool:
        return self._done

    @property
    def total_samples(self) -> int:
        return self._total

    # Most windows encoded in one batched pass.
    MAX_BATCH = 16

    def push(self, samples, *, final: bool = False) -> list[mx.array]:
        """Prompted chunks this push completes, in audio order. Chunks that become
        ready together (a step after a slow one, a whole file) encode their
        equal-length windows as one batch."""
        samples = mx.array(samples, dtype=mx.float32).reshape(-1)
        if self._done:
            if samples.size:
                raise RuntimeError("buffered encoder is finished")
            return []
        if samples.size:
            self._samples = mx.concatenate([self._samples, samples])
            self._total += samples.size

        C, R, L = self.chunk_samples, self.right_samples, self.left_samples
        ready = []
        while self._total > self._next + C + R:
            start = max(0, self._next - L)
            ready.append(
                (start, self._next + C + R, (self._next - start) // self.frame_samples)
            )
            self._prev_start = start
            self._next += C
        outputs = self._encode_batches(ready)
        if not self._full_window and ready:
            # Until the left context fills, every window is a new size and MLX's
            # buffer cache would keep each size's buffers. Drop them after each such
            # step; from the first full window on, the cache reuses one size.
            self._full_window = ready[-1][1] - ready[-1][0] == L + C + R
            mx.clear_cache()
        if final:
            if self._total > self._next:
                # NeMo caps the last buffer at left + chunk + right samples.
                start = max(self._prev_start, self._total - (L + C + R))
                left = (self._next - start) // self.frame_samples
                outputs.extend(
                    self._encode([start], self._total - start, left, last=True)
                )
            self._done = True
        self._trim()
        # A last window under one hop (10 ms) of new audio has no frame to decode.
        return [o for o in outputs if o.shape[1] > 0]

    def flush(self) -> list[mx.array]:
        return self.push(mx.zeros(0, dtype=mx.float32), final=True)

    def _trim(self):
        # The next step starts its buffer at or after the previous step's start.
        keep = max(0, self._next - self.chunk_samples - self.left_samples)
        drop = keep - self._start
        if drop > 0:
            self._samples = self._samples[drop:]
            self._start = keep

    def _encode_batches(self, windows) -> list[mx.array]:
        outputs = []
        i = 0
        while i < len(windows):
            # Windows of equal length and left context share every shape and mask.
            j = i + 1
            while (
                j < len(windows)
                and j - i < self.MAX_BATCH
                and windows[j][1] - windows[j][0] == windows[i][1] - windows[i][0]
                and windows[j][2] == windows[i][2]
            ):
                j += 1
            group = windows[i:j]
            outputs.extend(
                self._encode(
                    [w[0] for w in group],
                    group[0][1] - group[0][0],
                    group[0][2],
                    last=False,
                )
            )
            i = j
        return outputs

    def _encode(
        self, starts: list[int], length: int, left: int, *, last: bool
    ) -> list[mx.array]:
        """Encode windows ``[start, start + length)``, one per start, in one batch."""
        model = self.model
        mels = [
            log_mel_spectrogram(
                self._samples[s - self._start : s - self._start + length],
                model.preprocessor_config,
            )
            for s in starts
        ]
        lengths = mx.array([length // self.hop] * len(starts), dtype=mx.int32)
        encoded, encoded_lengths = model.encoder(
            mx.concatenate(mels, axis=0).astype(model.compute_dtype),
            lengths=lengths,
            att_context_size=self.context.as_list(),
        )
        count = int(encoded_lengths[0]) - left if last else self.context.chunk
        prompted = model.apply_prompt(encoded[:, left : left + count], self.language)
        mx.eval(prompted)
        return [prompted[b : b + 1] for b in range(len(starts))]
