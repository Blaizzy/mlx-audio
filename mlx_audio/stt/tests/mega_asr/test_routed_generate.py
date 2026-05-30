"""Synthetic tests for Mega-ASR routed generate / stream_transcribe (Task 4.3).

No real weights and no downloads: a TINY ``Qwen3ASRModel`` is built with random
weights, the router's ``route`` is monkeypatched, and the inner ``_asr`` decode
method is replaced by a spy. Real transcription parity is a separate, later
(``requires_weights``) task.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from mlx_audio.stt.models.mega_asr.config import MegaASRConfig
from mlx_audio.stt.models.mega_asr.lora import materialize_delta, resolve_linear
from mlx_audio.stt.models.mega_asr.mega_asr import Model
from mlx_audio.stt.models.qwen3_asr.config import AudioEncoderConfig, TextConfig

# Resolves _asr.model.layers[0].self_attn.q_proj -- a real nn.Linear, so that
# apply/remove actually mutates then restores a weight.
TARGET = "model.layers.0.self_attn.q_proj"

SENTINEL = object()


def _tiny_config() -> MegaASRConfig:
    audio_config = AudioEncoderConfig(
        num_mel_bins=32,
        encoder_layers=1,
        encoder_attention_heads=2,
        encoder_ffn_dim=64,
        d_model=32,
        max_source_positions=64,
        output_dim=32,
        downsample_hidden_size=8,
        n_window=2,
        n_window_infer=8,
    )
    text_config = TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,  # q_proj = Linear(32, 2*16=32) -> weight [32, 32]
        tie_word_embeddings=True,
    )
    return MegaASRConfig(
        audio_config=audio_config,
        text_config=text_config,
        router_config={
            "d_model": 8,
            "nhead": 2,
            "dim_feedforward": 16,
            "num_layers": 1,
            "frontend_hidden_dim": 8,
            "classifier_hidden_dim": 8,
            "max_len": 64,
        },
    )


def _set_route(model, use_lora):
    setattr(model._router, "route", lambda wav: {"use_lora": use_lora})


def _build():
    mx.random.seed(0)
    model = Model(_tiny_config())

    r, in_dim, out_dim = 4, 32, 32
    a = (mx.random.normal((r, in_dim)) * 0.1).astype(mx.float32)
    b = (mx.random.normal((out_dim, r)) * 0.1).astype(mx.float32)
    model._deltas = {TARGET: {"A": a, "B": b, "scaling": 2.0}}

    base = np.array(resolve_linear(model._asr, TARGET).weight)
    delta = np.array(materialize_delta(model._deltas[TARGET]))

    calls = []

    def spy(audio, **kwargs):
        calls.append((audio, kwargs))
        return SENTINEL

    setattr(model._asr, "generate", spy)
    return model, base, delta, calls


def _target_weight(model) -> np.ndarray:
    return np.array(resolve_linear(model._asr, TARGET).weight)


def test_routed_generate_toggles_lora_per_utterance():
    model, base, delta, calls = _build()
    audio = mx.zeros((16000,))

    assert model._lora_active is False
    assert np.allclose(_target_weight(model), base, atol=1e-4)

    _set_route(model, False)
    out = model.generate(audio, language="English")
    assert out is SENTINEL
    assert model._lora_active is False
    assert calls[-1][0] is audio
    assert calls[-1][1] == {"language": "English"}
    assert np.allclose(_target_weight(model), base, atol=1e-4)

    _set_route(model, True)
    out = model.generate(audio)
    assert out is SENTINEL
    assert model._lora_active is True
    assert np.allclose(_target_weight(model), base + delta, atol=1e-4)

    model.generate(audio)
    assert model._lora_active is True
    assert np.allclose(_target_weight(model), base + delta, atol=1e-4)

    _set_route(model, False)
    model.generate(audio)
    assert model._lora_active is False
    assert np.allclose(_target_weight(model), base, atol=1e-4)

    model.generate(audio)
    assert model._lora_active is False
    assert np.allclose(_target_weight(model), base, atol=1e-4)

    assert len(calls) == 5


def test_generate_is_explicit_method_not_delegated():
    assert "generate" in vars(Model)
    assert "stream_transcribe" in vars(Model)
    assert "_set_lora" in vars(Model)
    assert Model.generate.__qualname__ == "Model.generate"


def test_generate_routes_before_delegating():
    model, *_ = _build()
    routed = []

    def fake_route(wav):
        routed.append(wav)
        return {"use_lora": False}

    setattr(model._router, "route", fake_route)
    audio = mx.zeros((8000,))
    assert model.generate(audio) is SENTINEL
    assert routed and routed[0] is audio


def test_stream_transcribe_routes_and_delegates():
    model, base, delta, _ = _build()
    _set_route(model, True)

    stream_calls = []

    def stream_spy(audio, **kwargs):
        stream_calls.append((audio, kwargs))
        return SENTINEL

    setattr(model._asr, "stream_transcribe", stream_spy)

    audio = mx.zeros((8000,))
    out = model.stream_transcribe(audio, max_tokens=5)
    assert out is SENTINEL
    assert model._lora_active is True
    assert stream_calls[-1][0] is audio
    assert stream_calls[-1][1] == {"max_tokens": 5}
    assert np.allclose(_target_weight(model), base + delta, atol=1e-4)
