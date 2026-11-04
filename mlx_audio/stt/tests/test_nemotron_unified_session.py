"""Live-session behaviour of Nemotron Unified (buffered) streaming.

Tiny random models; what is checked is the session contract a realtime server relies
on: no crash on degenerate input, reset and cancel, input from another thread, bounded
buffers on long streams, and batched encoding equal to one window at a time.
"""

import threading

import mlx.core as mx
import numpy as np
import pytest

from mlx_audio.stt.models.nemotron_asr.buffered import (
    BufferedContext,
    BufferedEncoderState,
)
from mlx_audio.stt.tests.test_nemotron_unified import _unified_model


def _audio(samples: int, seed: int = 5) -> np.ndarray:
    return np.random.default_rng(seed).normal(0, 0.05, samples).astype(np.float32)


def _run(session, audio, feed, budget=8):
    parts = []
    for start in range(0, audio.size, feed):
        session.feed(audio[start : start + feed])
        parts.extend(session.step(max_decode_tokens=budget))
    session.close()
    while not session.done:
        parts.extend(session.step(max_decode_tokens=budget))
    return "".join(parts)


@pytest.mark.parametrize("samples", [1, 80, 159, 160, 1279, 1280, 1281])
@pytest.mark.parametrize("delay", [160, 2080])
def test_session_survives_stop_after_a_few_samples(samples, delay):
    # A dictation stopped right after it started: under one hop, one frame, one chunk.
    model = _unified_model()
    session = model.create_streaming_session(
        language="en-US", transcription_delay_ms=delay
    )
    _run(session, _audio(samples), feed=samples)
    assert session.done


def test_session_with_no_audio_finishes_empty():
    model = _unified_model()
    session = model.create_streaming_session(language="en-US")
    session.close()
    out = []
    while not session.done:
        out += session.step()
    assert out == []


def test_reset_gives_a_fresh_session():
    model = _unified_model()
    a, b = _audio(9 * 1280 + 37, seed=1), _audio(12 * 1280, seed=2)
    fresh = _run(model.create_streaming_session(language="en-US"), b, feed=1280)
    reused = model.create_streaming_session(language="en-US")
    _run(reused, a, feed=1280)
    reused.reset()
    assert _run(reused, b, feed=1280) == fresh


def test_cancel_mid_stream_then_reset():
    model = _unified_model()
    audio = _audio(15 * 1280)
    session = model.create_streaming_session(language="en-US")
    session.feed(audio[: 7 * 1280])
    session.step()
    session.cancel()
    assert session.done
    assert session.step() == []
    with pytest.raises(RuntimeError):
        session.feed(audio[:1280])
    session.reset()
    fresh = _run(model.create_streaming_session(language="en-US"), audio, feed=1280)
    assert _run(session, audio, feed=1280) == fresh


def test_feed_from_another_thread_matches_single_thread():
    model = _unified_model()
    audio = _audio(40 * 1280 + 500)
    expected = _run(model.create_streaming_session(language="en-US"), audio, feed=1280)
    session = model.create_streaming_session(language="en-US")

    def producer():
        for start in range(0, audio.size, 997):
            session.feed(audio[start : start + 997])
        session.close()

    t = threading.Thread(target=producer)
    t.start()
    parts = []
    while not session.done:
        parts.extend(session.step(max_decode_tokens=3))
    t.join()
    assert "".join(parts) == expected


def test_two_sessions_interleaved_do_not_share_state():
    model = _unified_model()
    a, b = _audio(20 * 1280, seed=3), _audio(20 * 1280, seed=4)
    alone_a = _run(model.create_streaming_session(language="en-US"), a, feed=1280)
    alone_b = _run(
        model.create_streaming_session(language="en-US", transcription_delay_ms=560),
        b,
        feed=1280,
    )
    sa = model.create_streaming_session(language="en-US")
    sb = model.create_streaming_session(language="en-US", transcription_delay_ms=560)
    pa, pb = [], []
    for start in range(0, a.size, 1280):
        sa.feed(a[start : start + 1280])
        sb.feed(b[start : start + 1280])
        pa += sa.step()
        pb += sb.step()
    for s, p in ((sa, pa), (sb, pb)):
        s.close()
        while not s.done:
            p += s.step()
    assert "".join(pa) == alone_a
    assert "".join(pb) == alone_b


def test_long_stream_keeps_buffers_bounded():
    model = _unified_model()
    context = BufferedContext(70, 1, 3)
    state = BufferedEncoderState(model, context, "en-US")
    state._encode = lambda starts, length, left, *, last: [mx.zeros((1, 1, 1))] * len(
        starts
    )
    bound = (
        context.left + 2 * context.chunk + context.right
    ) * state.frame_samples + 1280
    for _ in range(20_000):  # ~27 minutes of 80 ms pushes
        state.push(np.zeros(1280, np.float32))
        assert state._samples.size <= bound


@pytest.mark.parametrize("delay", [160, 560, 2080])
def test_batched_encode_matches_one_window_at_a_time(delay):
    model = _unified_model()
    context = BufferedContext.from_delay_ms(delay, left=6)
    audio = _audio(40 * 1280 + 700)
    whole = BufferedEncoderState(model, context, "en-US").push(audio, final=True)
    one = BufferedEncoderState(model, context, "en-US")
    one.MAX_BATCH = 1
    single = one.push(audio, final=True)
    assert len(whole) == len(single)
    for x, y in zip(whole, single):
        assert x.shape == y.shape
        assert mx.abs(x - y).max().item() < 1e-4


def test_feed_rejects_stereo_and_non_finite():
    model = _unified_model()
    session = model.create_streaming_session(language="en-US")
    with pytest.raises(ValueError):
        session.feed(np.zeros((1280, 2), np.float32))
    with pytest.raises(ValueError):
        session.feed(np.array([0.0, np.nan], np.float32))


@pytest.mark.parametrize("message_ms", [100, 200, 480, 1000])
def test_server_pattern_keeps_up_with_large_messages(message_ms):
    # The realtime server steps once per received message, with a budget of 8. A
    # session that took one chunk per step overflowed its 30 s queue; one that kept
    # the budget piled up encoded frames (12.5 per 1 s message, plus the tokens; the
    # random model emits far more tokens than a real one).
    model = _unified_model()
    session = model.create_streaming_session(
        language="en-US", transcription_delay_ms=320
    )
    message = message_ms * 16
    audio = _audio(120 * 16000)
    for start in range(0, audio.size, message):
        session.feed(audio[start : start + message])
        session.step(max_decode_tokens=8)
        assert session._pending_frames == 0
    assert session._queued < 2 * message + session._buffered.chunk_samples


def test_undecoded_frames_count_against_the_input_limit():
    model = _unified_model()
    session = model.create_streaming_session(language="en-US")
    session._pending_frames = 30 * 16000 // session._buffered.frame_samples
    with pytest.raises(BufferError):
        session.feed(np.zeros(1280, np.float32))
