"""Both real session implementations satisfy the same consumer contract.

Uses tiny random models and deterministic tokens, never downloaded weights.
"""

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from mlx_audio.server import _open_streaming_session
from mlx_audio.stt.models.voxtral_realtime.config import (
    DecoderConfig,
    EncoderConfig,
    ModelConfig,
)
from mlx_audio.stt.models.voxtral_realtime.voxtral_realtime import Model
from mlx_audio.stt.streaming import StreamingSession
from mlx_audio.stt.tests.test_nemotron_session import model as nemotron_model


def voxtral_model():
    config = ModelConfig(
        encoder_args=EncoderConfig(
            dim=32,
            n_layers=1,
            n_heads=2,
            n_kv_heads=2,
            head_dim=16,
            hidden_dim=64,
        ),
        decoder=DecoderConfig(
            dim=32,
            n_layers=1,
            n_heads=2,
            n_kv_heads=2,
            head_dim=16,
            hidden_dim=64,
            vocab_size=64,
        ),
        n_left_pad_tokens=2,
        transcription_delay_ms=80,
    )
    model = Model(config)
    # The production encoder projection targets the full decoder width.
    model.encoder.audio_language_projection_0 = nn.Linear(128, 32, bias=False)
    model.encoder.audio_language_projection_2 = nn.Linear(32, 32, bias=False)
    model._tokenizer = SimpleNamespace(decode=lambda tokens: "a" * len(tokens))
    model._next_token_mx = lambda logits, temperature: mx.array(3)
    return model


@pytest.mark.parametrize(
    "factory", [nemotron_model, voxtral_model], ids=["nemotron", "voxtral"]
)
def test_shared_server_session_contract(factory):
    session: StreamingSession = _open_streaming_session(
        factory(), temperature=0.0, delay_ms=80
    )
    assert isinstance(session, StreamingSession)
    assert session.input_sample_rate == 16000
    assert not session.done
    text = []
    for _ in range(20):
        session.feed(np.zeros(1600, dtype=np.float32))
        deltas = session.step(max_decode_tokens=4)
        assert isinstance(deltas, list)
        assert all(isinstance(delta, str) for delta in deltas)
        text.extend(deltas)
    assert text, "Expected incremental output before close"
    session.close()
    for _ in range(2000):
        deltas = session.step(max_decode_tokens=4)
        assert all(isinstance(delta, str) for delta in deltas)
        if session.done:
            break
    assert session.done
    assert session.step() == []
