"""Live-input session regressions, using tiny weights and deterministic logits."""

import mlx.core as mx
import numpy as np
import pytest

from mlx_audio.stt.models.nemotron_asr import Model, ModelConfig
from mlx_audio.stt.tests.test_nemotron_asr import _tiny_config


def model():
    result = Model(ModelConfig.from_dict(_tiny_config()))
    # One visible token followed by blank, repeated for every encoder frame.
    calls = [0]

    def joint(feature, output):
        token = 2 if calls[0] % 2 == 0 else result.blank_id
        calls[0] += 1
        return mx.array(
            [0.0 if i != token else 1.0 for i in range(result.blank_id + 1)]
        )

    result.joint = joint
    return result


def drain(session):
    text = []
    for _ in range(2000):
        text.extend(session.step(max_decode_tokens=8))
        if session.done:
            return "".join(text)
    pytest.fail("session did not finish")


@pytest.mark.parametrize("length", [1, 159, 512, 17920, 24000])
def test_partials_and_tail_match_whole_waveform(length):
    audio = np.random.default_rng(12).normal(0, 0.01, length).astype(np.float32)
    baseline = list(model().stream_generate(mx.array(audio)))[-1].text.strip()
    session = model().create_streaming_session()
    text = []
    for start in range(0, audio.size, 317):
        session.feed(audio[start : start + 317])
        text.extend(session.step(max_decode_tokens=8))
    if length == 24000:
        assert text, "must produce deltas before close"
    assert not session.done
    session.close()
    session.close()
    actual = "".join(text) + drain(session)
    assert actual == baseline
    assert session.step() == []


def test_empty_reset_cancel_and_validation():
    session = model().create_streaming_session()
    session.close()
    assert drain(session) == ""
    with pytest.raises(RuntimeError):
        session.feed(np.zeros(1))
    session.reset()
    with pytest.raises(ValueError):
        session.feed(np.zeros((2, 2)))
    with pytest.raises(ValueError):
        session.feed(np.array([np.nan]))
    with pytest.raises(BufferError):
        session.feed(np.zeros(30 * session.input_sample_rate + 1))
    session.feed(np.zeros(1000))
    session.cancel()
    assert session.done
    assert session._queued == 0
    assert session.step() == []


def test_cooperative_budget_and_decoder_state():
    session = model().create_streaming_session()
    session.feed(np.zeros(20000))
    # Centered STFT needs lookahead beyond the first native audio chunk.
    assert session.step(max_decode_tokens=1) == []
    assert session.step(max_decode_tokens=1) == ["hello"]
    assert session._frame == 0
    assert session._hidden is not None
    hidden = session._hidden
    assert session.step(max_decode_tokens=1) == []
    assert session._frame == 1
    assert session._hidden is hidden  # blanks do not advance the predictor
    assert len(session._encoded) <= 2
    with pytest.raises(ValueError):
        session.step(max_decode_tokens=0)
    with pytest.raises(ValueError):
        model().create_streaming_session(temperature=1)


def test_silence_has_bounded_state_and_independent_sessions():
    m = model()
    m.joint = lambda feature, output: mx.array([0.0] * m.blank_id + [1.0])
    first = m.create_streaming_session()
    second = m.create_streaming_session()
    for _ in range(1000):
        first.feed(np.zeros(320, dtype=np.float32))
        assert first.step(max_decode_tokens=8) == []
        assert first._frontend.buffered_samples < 1024
        assert len(first._encoded) <= 2
        assert first._queued < first.input_sample_rate * 2
    assert second._frontend.total_samples == 0
    first.close()
    assert drain(first) == ""
    second.close()
    assert drain(second) == ""


def test_real_tiny_decoder_parity():
    mx.random.seed(42)
    m = Model(ModelConfig.from_dict(_tiny_config()))
    audio = np.random.default_rng(42).normal(0, 0.03, 24000).astype(np.float32)
    expected = list(m.stream_generate(mx.array(audio)))[-1].text.strip()
    session = m.create_streaming_session()
    parts = []
    for start in range(0, len(audio), 509):
        session.feed(audio[start : start + 509])
        parts.extend(session.step(max_decode_tokens=8))
    session.close()
    assert "".join(parts) + drain(session) == expected


def test_special_tokens_and_symbol_cap_across_steps():
    m = model()
    m.max_symbols = 2
    predictions = iter([1, 2, m.blank_id])

    def joint(feature, output):
        predicted = next(predictions)
        return mx.array([float(i == predicted) for i in range(m.blank_id + 1)])

    m.joint = joint
    session = m.create_streaming_session()
    session.feed(np.zeros(20000))
    assert session.step(max_decode_tokens=1) == []  # frontend lookahead
    assert session.step(max_decode_tokens=1) == []  # language token
    assert session._symbols == 1
    assert session._frame == 0
    assert session.step(max_decode_tokens=1) == ["hello"]
    assert session._frame == 1  # cap reached without waiting for blank
    assert session.step(max_decode_tokens=1) == []
    assert session._frame == 2
