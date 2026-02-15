"""Regression tests for MOSS-TTS realtime integration."""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from mlx_audio.tts.models.moss_tts_realtime import model as realtime_model_module
from mlx_audio.tts.models.moss_tts_realtime.processor import (
    _normalize_preencoded_audio_tokens,
)


class _FakeProcessor:
    def encode_prompt_audio(self, audio):
        return audio

    def tokens_from_text(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [101, 102, 103] if text else []

    def make_text_prefix(self, token_ids):
        return list(token_ids)


class _FakeInferencer:
    def __init__(self, *args, **kwargs):
        del args, kwargs


class _FakeSession:
    instances: list["_FakeSession"] = []

    def __init__(self, *args, **kwargs):
        del args, kwargs
        self.calls: list[str] = []
        _FakeSession.instances.append(self)

    def reset_turn(self, *args, **kwargs):
        del args, kwargs
        self.calls.append("reset")

    def push_text_tokens(self, token_ids):
        del token_ids
        self.calls.append("push")
        return [mx.array([0.1], dtype=mx.float32)]

    def end_text(self):
        self.calls.append("end")
        return [mx.array([0.2], dtype=mx.float32)]

    def drain(self, max_steps=None):
        del max_steps
        self.calls.append("drain")
        return [mx.array([0.3], dtype=mx.float32)]

    def close(self):
        self.calls.append("close")


class _ModelShim:
    def __init__(self):
        self.model = object()
        self.tokenizer = object()
        self.config = object()
        self.processor = _FakeProcessor()

    def _ensure_runtime_ready(self):
        return None

    def _build_generation_result(self, audio, **kwargs):
        return {
            "audio": np.array(audio),
            **kwargs,
        }


def test_stream_mode_yields_before_full_turn_drain(monkeypatch):
    """First stream chunk should arrive before drain/close complete the turn."""
    _FakeSession.instances.clear()

    monkeypatch.setattr(realtime_model_module, "MossTTSRealtimeInference", _FakeInferencer)
    monkeypatch.setattr(realtime_model_module, "RealtimeSession", _FakeSession)

    model = _ModelShim()
    stream_iter = realtime_model_module.Model.generate(model, text="hello", stream=True)

    first_chunk = next(stream_iter)
    session = _FakeSession.instances[-1]

    assert first_chunk["is_streaming_chunk"] is True
    assert first_chunk["is_final_chunk"] is False
    assert session.calls[:2] == ["reset", "push"]
    assert "drain" not in session.calls
    assert "close" not in session.calls

    list(stream_iter)
    assert session.calls == ["reset", "push", "end", "drain", "close"]


def test_square_codebook_major_tokens_transpose_to_time_major():
    """Square `(RVQ, T)` preencoded references must normalize to `(T, RVQ)`."""
    rvq = 4
    matrix = mx.array(
        [
            [0, 1, 2, 3],
            [10, 11, 12, 13],
            [20, 21, 22, 23],
            [30, 31, 32, 33],
        ],
        dtype=mx.int32,
    )

    normalized = _normalize_preencoded_audio_tokens(
        matrix,
        rvq=rvq,
        audio_pad_token=32767,
    )

    assert normalized is not None
    np.testing.assert_array_equal(np.array(normalized), np.array(matrix).T)
