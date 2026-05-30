"""Server-side turn detection for the OpenAI-compatible ``/v1/realtime`` endpoint.

The realtime WebSocket endpoint transcribes a continuous audio stream, but on
its own it has no notion of *when a speaker's turn ends* — the client has to
send ``input_audio_buffer.commit`` by hand. This module adds OpenAI-compatible
``turn_detection`` so the server itself decides turn boundaries: it emits
``input_audio_buffer.speech_started`` / ``input_audio_buffer.speech_stopped``
and auto-commits, exactly like OpenAI's ``server_vad`` mode.

Only ``server_vad`` is implemented. It runs a streaming VAD model (Silero by
default) frame by frame and applies the same ``threshold`` /
``prefix_padding_ms`` / ``silence_duration_ms`` logic the OpenAI Realtime API
exposes. The endpointing logic (:class:`TurnDetector`) is a pure state machine,
deliberately free of any model dependency so it can be unit-tested with
synthetic probabilities; :class:`StreamingVad` adds the model and the framing.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import numpy as np

# Silero consumes fixed 512-sample windows at 16 kHz (32 ms per frame); the 8 kHz
# branch is not used here because realtime STT models run at 16 kHz.
VAD_SAMPLE_RATE: int = 16000
VAD_FRAME_SIZE: int = 512
VAD_FRAME_MS: float = 1000.0 * VAD_FRAME_SIZE / VAD_SAMPLE_RATE


class TurnDetectionError(ValueError):
    """Raised when a client requests an unsupported ``turn_detection`` config."""


@dataclass(frozen=True)
class ServerVadConfig:
    """Resolved ``server_vad`` parameters, mirroring the OpenAI schema."""

    threshold: float = 0.5
    prefix_padding_ms: int = 300
    silence_duration_ms: int = 500

    def to_dict(self) -> dict:
        return {
            "type": "server_vad",
            "threshold": self.threshold,
            "prefix_padding_ms": self.prefix_padding_ms,
            "silence_duration_ms": self.silence_duration_ms,
        }


def parse_turn_detection(turn_detection: Optional[dict]) -> Optional[ServerVadConfig]:
    """Map an OpenAI ``turn_detection`` object onto a :class:`ServerVadConfig`.

    Returns ``None`` for ``null`` (manual-commit mode). Raises
    :class:`TurnDetectionError` for ``semantic_vad`` (not implemented yet) or an
    unknown ``type``.
    """
    if not turn_detection:
        return None
    td_type: Optional[str] = turn_detection.get("type")
    if td_type == "server_vad":
        defaults: ServerVadConfig = ServerVadConfig()
        return ServerVadConfig(
            threshold=float(turn_detection.get("threshold", defaults.threshold)),
            prefix_padding_ms=int(
                turn_detection.get("prefix_padding_ms", defaults.prefix_padding_ms)
            ),
            silence_duration_ms=int(
                turn_detection.get("silence_duration_ms", defaults.silence_duration_ms)
            ),
        )
    if td_type == "semantic_vad":
        raise TurnDetectionError(
            "semantic_vad is not supported by this server yet; use server_vad"
        )
    raise TurnDetectionError(f"unknown turn_detection type: {td_type!r}")


class TurnEventKind(str, Enum):
    SPEECH_STARTED = "speech_started"
    SPEECH_STOPPED = "speech_stopped"


@dataclass(frozen=True)
class TurnEvent:
    """A detected turn boundary. ``audio_ms`` is the offset from session start."""

    kind: TurnEventKind
    audio_ms: int


class TurnDetector:
    """Pure endpointing state machine over per-frame speech probabilities.

    Feed it one VAD probability per frame via :meth:`push`. It emits a
    ``SPEECH_STARTED`` when the probability first crosses ``threshold`` and a
    ``SPEECH_STOPPED`` once ``silence_duration_ms`` of sub-threshold audio has
    elapsed after speech. ``prefix_padding_ms`` only shifts the reported
    ``audio_start_ms`` earlier — it does not gate transcription.

    The running clock is kept across turns so reported offsets stay monotonic
    for the lifetime of the session.
    """

    def __init__(self, config: ServerVadConfig):
        self._config: ServerVadConfig = config
        self._elapsed_ms: float = 0.0
        self._in_speech: bool = False
        self._silence_ms: float = 0.0

    def push(self, probability: float, frame_ms: float) -> List[TurnEvent]:
        self._elapsed_ms += frame_ms
        events: List[TurnEvent] = []
        is_speech: bool = probability >= self._config.threshold
        if not self._in_speech:
            if is_speech:
                self._in_speech = True
                self._silence_ms = 0.0
                start: float = (
                    self._elapsed_ms - frame_ms - self._config.prefix_padding_ms
                )
                events.append(
                    TurnEvent(TurnEventKind.SPEECH_STARTED, max(0, int(start)))
                )
        else:
            if is_speech:
                self._silence_ms = 0.0
            else:
                self._silence_ms += frame_ms
                if self._silence_ms >= self._config.silence_duration_ms:
                    self._in_speech = False
                    self._silence_ms = 0.0
                    events.append(
                        TurnEvent(TurnEventKind.SPEECH_STOPPED, int(self._elapsed_ms))
                    )
        return events

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    def reset_turn(self) -> None:
        """Clear speech state after a turn is committed; keep the clock."""
        self._in_speech = False
        self._silence_ms = 0.0


class StreamingVad:
    """Drive a :class:`TurnDetector` from a streaming VAD model.

    ``vad_model`` must expose the Silero streaming protocol:
    ``initial_state(sample_rate=...)`` and
    ``feed(chunk, state, sample_rate=...) -> (probability, state)``, consuming
    fixed :data:`VAD_FRAME_SIZE`-sample windows at :data:`VAD_SAMPLE_RATE`.
    Audio that doesn't fill a whole frame is buffered until the next call.
    """

    def __init__(self, vad_model, config: ServerVadConfig):
        self._vad = vad_model
        self._config: ServerVadConfig = config
        self._state = vad_model.initial_state(sample_rate=VAD_SAMPLE_RATE)
        self._detector: TurnDetector = TurnDetector(config)
        self._buffer: np.ndarray = np.zeros(0, dtype=np.float32)

    def process(self, samples: np.ndarray) -> List[TurnEvent]:
        """Feed 16 kHz float32 ``samples``; return any turn events detected.

        Runs MLX work, so call it from a worker thread rather than the event
        loop.
        """
        import mlx.core as mx

        if samples.size:
            self._buffer = np.concatenate([self._buffer, samples.astype(np.float32)])
        events: List[TurnEvent] = []
        while self._buffer.shape[0] >= VAD_FRAME_SIZE:
            frame: np.ndarray = self._buffer[:VAD_FRAME_SIZE]
            self._buffer = self._buffer[VAD_FRAME_SIZE:]
            probability, self._state = self._vad.feed(
                frame, self._state, sample_rate=VAD_SAMPLE_RATE
            )
            mx.eval(probability)
            prob: float = float(np.array(probability).reshape(-1)[0])
            events.extend(self._detector.push(prob, VAD_FRAME_MS))
        return events

    @property
    def in_speech(self) -> bool:
        return self._detector.in_speech

    def reset_turn(self) -> None:
        self._detector.reset_turn()
