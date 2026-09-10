"""Shared, model-independent live-input STT session contract."""

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class StreamingSession(Protocol):
    """A producer queues audio while one consumer drives incremental decoding.

    Model factories own construction and model-specific options. Feed native-rate
    mono float32 PCM, then close the input and keep stepping until ``done``.
    ``feed`` and ``close`` may run on a producer thread; ``step`` must run on a
    single model executor. Construction may also perform model/MLX work.
    The producer must not mutate submitted arrays while they may be queued.
    Do not feed after close or completion; create a new session for a new turn.

    The decode budget is model-specific (e.g. RNNT joint evaluations, including
    blanks, versus autoregressive tokens), not a wall-clock deadline or a bound
    on encoder work. An empty delta list does not imply completion.
    Buffer limits and early stopping are implementation-specific. Reset and
    cancellation are optional extensions, not requirements of this protocol.
    """

    input_sample_rate: int

    @property
    def done(self) -> bool:
        """Whether decoding has finished; no further steps are needed."""
        ...

    def feed(self, samples: np.ndarray) -> None:
        """Queue native-rate mono PCM without running model inference."""
        ...

    def close(self) -> None:
        """Signal end of input; the consumer must still drain with step()."""
        ...

    def step(self, *, max_decode_tokens: int = 4) -> list[str]:
        """Advance decoding and return append-only text deltas."""
        ...
