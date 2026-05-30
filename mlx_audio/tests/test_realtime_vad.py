"""Tests for server-side turn detection (``mlx_audio.realtime_vad``).

The endpointing state machine (:class:`TurnDetector`) is tested with synthetic
per-frame probabilities — no model is loaded. :class:`StreamingVad` is tested
with a fake VAD model that returns scripted probabilities, exercising the
512-sample framing and buffering.
"""

import mlx.core as mx
import numpy as np
import pytest

from mlx_audio.realtime_vad import (
    VAD_FRAME_MS,
    VAD_FRAME_SIZE,
    ServerVadConfig,
    StreamingVad,
    TurnDetectionError,
    TurnDetector,
    TurnEventKind,
    parse_turn_detection,
)

# --- parse_turn_detection -------------------------------------------------


def test_parse_none_is_manual_mode():
    assert parse_turn_detection(None) is None
    assert parse_turn_detection({}) is None


def test_parse_server_vad_defaults_match_openai():
    cfg = parse_turn_detection({"type": "server_vad"})
    assert cfg == ServerVadConfig(
        threshold=0.5, prefix_padding_ms=300, silence_duration_ms=500
    )


def test_parse_server_vad_custom_values_round_trip():
    cfg = parse_turn_detection(
        {
            "type": "server_vad",
            "threshold": 0.7,
            "prefix_padding_ms": 100,
            "silence_duration_ms": 800,
        }
    )
    assert cfg == ServerVadConfig(0.7, 100, 800)
    assert cfg.to_dict() == {
        "type": "server_vad",
        "threshold": 0.7,
        "prefix_padding_ms": 100,
        "silence_duration_ms": 800,
    }


def test_parse_semantic_vad_is_rejected():
    with pytest.raises(TurnDetectionError):
        parse_turn_detection({"type": "semantic_vad"})


def test_parse_unknown_type_is_rejected():
    with pytest.raises(TurnDetectionError):
        parse_turn_detection({"type": "telepathy"})


# --- TurnDetector (pure state machine) ------------------------------------


def _push_all(detector, probs, frame_ms=20.0):
    events = []
    for p in probs:
        events.extend(detector.push(p, frame_ms))
    return events


def test_silence_only_emits_nothing():
    d = TurnDetector(ServerVadConfig(0.5, prefix_padding_ms=0, silence_duration_ms=100))
    assert _push_all(d, [0.0] * 50) == []
    assert d.in_speech is False


def test_speech_then_silence_emits_start_then_stop():
    d = TurnDetector(ServerVadConfig(0.5, prefix_padding_ms=0, silence_duration_ms=100))
    # 10 speech frames (200 ms), then 6 silence frames (>= 100 ms of silence)
    events = _push_all(d, [0.9] * 10 + [0.1] * 6, frame_ms=20.0)
    assert [e.kind for e in events] == [
        TurnEventKind.SPEECH_STARTED,
        TurnEventKind.SPEECH_STOPPED,
    ]
    start, stop = events
    assert start.audio_ms == 0
    # 200 ms speech + 100 ms silence to confirm the stop.
    assert stop.audio_ms == 300
    assert d.in_speech is False


def test_short_silence_does_not_end_the_turn():
    d = TurnDetector(ServerVadConfig(0.5, prefix_padding_ms=0, silence_duration_ms=200))
    events = _push_all(d, [0.9] * 5 + [0.1] * 4, frame_ms=20.0)  # only 80 ms silence
    assert [e.kind for e in events] == [TurnEventKind.SPEECH_STARTED]
    assert d.in_speech is True


def test_prefix_padding_shifts_reported_start_earlier():
    d = TurnDetector(
        ServerVadConfig(0.5, prefix_padding_ms=60, silence_duration_ms=100)
    )
    events = _push_all(d, [0.1] * 5 + [0.9] * 3, frame_ms=20.0)
    assert events[0].kind == TurnEventKind.SPEECH_STARTED
    # onset at 120 ms; reported start = 120 - 20 (frame) - 60 (padding) = 40 ms.
    assert events[0].audio_ms == 40


def test_prefix_padding_is_clamped_to_zero():
    d = TurnDetector(
        ServerVadConfig(0.5, prefix_padding_ms=500, silence_duration_ms=100)
    )
    events = _push_all(d, [0.9], frame_ms=20.0)
    assert events[0].audio_ms == 0


def test_two_turns_in_one_session_keep_a_monotonic_clock():
    d = TurnDetector(ServerVadConfig(0.5, prefix_padding_ms=0, silence_duration_ms=40))
    first = _push_all(d, [0.9] * 3 + [0.0] * 2, frame_ms=20.0)
    d.reset_turn()
    second = _push_all(d, [0.9] * 3 + [0.0] * 2, frame_ms=20.0)
    assert [e.kind for e in first] == [
        TurnEventKind.SPEECH_STARTED,
        TurnEventKind.SPEECH_STOPPED,
    ]
    assert [e.kind for e in second] == [
        TurnEventKind.SPEECH_STARTED,
        TurnEventKind.SPEECH_STOPPED,
    ]
    # The clock continues across turns: the second turn ends later than the
    # first, and never rewinds before the first turn's end.
    assert second[0].audio_ms >= first[-1].audio_ms
    assert second[-1].audio_ms > first[-1].audio_ms


def test_reset_turn_clears_speech_state():
    d = TurnDetector(ServerVadConfig(0.5, prefix_padding_ms=0, silence_duration_ms=100))
    _push_all(d, [0.9] * 3)
    assert d.in_speech is True
    d.reset_turn()
    assert d.in_speech is False


# --- StreamingVad (framing + model) ---------------------------------------


class _FakeVad:
    """Returns a scripted probability per consumed 512-sample frame."""

    def __init__(self, probs):
        self._probs = probs

    def initial_state(self, sample_rate=16000):
        return {"i": 0}

    def feed(self, chunk, state, sample_rate=16000):
        i = state["i"]
        p = self._probs[i] if i < len(self._probs) else 0.0
        return mx.array([[p]]), {"i": i + 1}


def test_streamingvad_buffers_until_a_full_frame():
    vad = StreamingVad(
        _FakeVad([0.9]),
        ServerVadConfig(0.5, prefix_padding_ms=0, silence_duration_ms=1000),
    )
    # Less than one frame: nothing consumed, no model call, no events.
    assert vad.process(np.zeros(VAD_FRAME_SIZE - 1, dtype=np.float32)) == []
    # The missing sample completes exactly one frame → one speech_started.
    events = vad.process(np.zeros(1, dtype=np.float32))
    assert [e.kind for e in events] == [TurnEventKind.SPEECH_STARTED]


def test_streamingvad_detects_speech_then_silence():
    probs = [0.9] * 4 + [0.0] * 40
    vad = StreamingVad(
        _FakeVad(probs),
        ServerVadConfig(
            0.5, prefix_padding_ms=0, silence_duration_ms=int(5 * VAD_FRAME_MS)
        ),
    )
    events = vad.process(np.zeros(VAD_FRAME_SIZE * 20, dtype=np.float32))
    kinds = [e.kind for e in events]
    assert TurnEventKind.SPEECH_STARTED in kinds
    assert TurnEventKind.SPEECH_STOPPED in kinds
    assert kinds.index(TurnEventKind.SPEECH_STARTED) < kinds.index(
        TurnEventKind.SPEECH_STOPPED
    )
