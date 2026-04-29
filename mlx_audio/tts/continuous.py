from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class TTSBatchOptions:
    temperature: float = 0.9
    top_p: float = 1.0
    top_k: int = 50
    repetition_penalty: float = 1.05
    max_tokens: int = 4096
    lang_code: str = "auto"
    stream: bool = False
    streaming_interval: float = 2.0
    max_batch_size: int = 8
    verbose: bool = False


@dataclass
class TTSBatchItem:
    sequence_id: int
    text: str
    voice: str | None = None
    instruct: str | None = None
    speed: float | None = None
    gender: str | None = None
    pitch: float | None = None
    ref_audio: Any = None
    ref_text: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class TTSBatchEvent:
    sequence_id: int
    audio: Any = None
    sample_rate: int | None = None
    samples: int = 0
    token_count: int = 0
    done: bool = False
    is_streaming_chunk: bool = False
    is_final_chunk: bool = False
    error: BaseException | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class TTSBatchSession(Protocol):
    @property
    def idle(self) -> bool: ...

    @property
    def available_slots(self) -> int: ...

    def add(self, items: list[TTSBatchItem]) -> None: ...

    def cancel(self, sequence_id: int) -> None: ...

    def step(self) -> list[TTSBatchEvent]: ...
