"""Speaker streaming session regressions, using tiny weights and deterministic logits."""

import mlx.core as mx
import numpy as np
import pytest

from mlx_audio.stt.tests.test_nemotron_session import model
from mlx_audio.vad.tests.test_nemotron_diarization import tiny_model


def audio(seed, seconds):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 0.1, round(seconds * 16000)).astype(np.float32)


def recording(diar):
    """Wrap ``diar.feed`` to record the speaker probabilities it commits."""
    probs, feed = [], diar.feed

    def recorded(*args, **kwargs):
        result, state = feed(*args, **kwargs)
        probs.append(np.array(result.speaker_probs))
        return result, state

    diar.feed = recorded
    return probs


def session(diar):
    return model().create_speaker_streaming_session(diar, threshold=0.05)


def test_close_streams_ends_every_stream_and_keeps_diarization():
    closing, through = tiny_model(), tiny_model()
    closed_probs, through_probs = recording(closing), recording(through)
    closed, uninterrupted = session(closing), session(through)
    first, second = audio(3, 2.0), audio(4, 2.0)

    closed.feed(first)
    uninterrupted.feed(first)
    ended = closed.close_streams()

    assert ended, "the tiny diarizer marks speakers active at this threshold"
    labels = [transcript.speaker for transcript in ended]
    assert labels == sorted(labels, key=lambda label: int(label.split("_")[1]))
    assert any(
        transcript.tokens for transcript in ended
    ), "held-back tokens are decoded"
    assert closed.close_streams() == [], "no stream is left open"
    assert not closed.done

    closed.feed(second)
    uninterrupted.feed(second)
    # Diarization is untouched: the same speakers, scored the same, either way.
    assert len(closed_probs) == len(through_probs)
    for after_close, without in zip(closed_probs, through_probs):
        np.testing.assert_array_equal(after_close, without)


def test_close_streams_on_a_new_session_ends_nothing_and_a_closed_session_refuses():
    fresh = session(tiny_model())
    assert fresh.close_streams() == []
    fresh.feed(audio(5, 0.5), final=True)
    with pytest.raises(RuntimeError):
        fresh.close_streams()
