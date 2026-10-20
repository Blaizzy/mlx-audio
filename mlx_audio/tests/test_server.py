import asyncio
import functools
import gc
import inspect
import io
import json
import queue
from unittest.mock import AsyncMock, MagicMock, patch

import httpx2
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from huggingface_hub.errors import RepositoryNotFoundError

from mlx_audio.audio_io import read as audio_read
from mlx_audio.audio_io import write as audio_write

# python-multipart is required for FastAPI file uploads
pytest.importorskip("multipart", reason="python-multipart is required for server tests")

from fastapi.testclient import TestClient

from mlx_audio.server import _stream_inference_results, app
from mlx_audio.server_inference import InferenceResultChunk


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def mock_model_provider():
    # mock the model_provider.load_model method
    with patch(
        "mlx_audio.server.model_provider", new_callable=AsyncMock
    ) as mock_provider:
        mock_provider.load_model = MagicMock()
        # In-use acquisition (acquire_model/release_model) reuses the
        # load_model mock, so per-test load_model return values still reach
        # the endpoints; release is no-op bookkeeping on the mock.
        mock_provider.acquire_model = MagicMock(
            side_effect=lambda name: mock_provider.load_model(name)
        )
        mock_provider.release_model = MagicMock()
        yield mock_provider


def test_list_models_empty(client, mock_model_provider):
    # mock the model_provider.get_available_models method
    mock_model_provider.get_available_models = AsyncMock(return_value=[])
    response = client.get("/v1/models")
    assert response.status_code == 200
    assert response.json() == {"object": "list", "data": []}


def test_list_models_with_data(client, mock_model_provider):
    # Test that the list_models endpoint
    mock_model_provider.get_available_models = AsyncMock(
        return_value=["model1", "model2"]
    )
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 2
    assert data["data"][0]["id"] == "model1"
    assert data["data"][1]["id"] == "model2"


def test_add_model(client, mock_model_provider):
    # Test that the add_model endpoint
    response = client.post("/v1/models?model_name=test_model")
    assert response.status_code == 200
    assert response.json() == {
        "status": "success",
        "message": "Model test_model added successfully",
    }
    mock_model_provider.load_model.assert_called_once_with("test_model")


def test_remove_model_success(client, mock_model_provider):
    # Test that the remove_model endpoint returns a 204 status code
    mock_model_provider.remove_model = AsyncMock(return_value=True)
    response = client.delete("/v1/models?model_name=test_model")
    assert response.status_code == 204
    mock_model_provider.remove_model.assert_called_once_with("test_model")


def test_remove_model_not_found(client, mock_model_provider):
    # Test that the remove_model endpoint returns a 404 status code
    mock_model_provider.remove_model = AsyncMock(return_value=False)
    response = client.delete("/v1/models?model_name=non_existent_model")
    assert response.status_code == 404
    assert response.json() == {"detail": "Model 'non_existent_model' not found"}
    mock_model_provider.remove_model.assert_called_once_with("non_existent_model")


def test_remove_model_with_quotes_in_name(client, mock_model_provider):
    # Test that the remove_model endpoint returns a 204 status code
    mock_model_provider.remove_model = AsyncMock(return_value=True)
    response = client.delete('/v1/models?model_name="test_model_quotes"')
    assert response.status_code == 204
    mock_model_provider.remove_model.assert_called_once_with("test_model_quotes")


class MockAudioResult:
    def __init__(self, audio_data, sample_rate):
        self.audio = audio_data
        self.sample_rate = sample_rate


def sync_mock_audio_stream_generator(input_text: str, **kwargs):
    sample_rate = 16000
    duration = 1
    frequency = 440
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = 0.5 * np.sin(2 * np.pi * frequency * t)
    yield MockAudioResult(audio_data.astype(np.float32), sample_rate)


def test_tts_speech(client, mock_model_provider):
    # Test that the tts_speech endpoint returns a 200 status code
    mock_tts_model = MagicMock()
    mock_tts_model.generate = MagicMock(wraps=sync_mock_audio_stream_generator)

    mock_model_provider.load_model = MagicMock(return_value=mock_tts_model)

    payload = {"model": "test_tts_model", "input": "Hello world", "voice": "alloy"}
    response = client.post("/v1/audio/speech", json=payload)
    assert response.status_code == 200
    assert response.headers["content-type"].lower() == "audio/mp3"
    assert (
        response.headers["content-disposition"].lower()
        == "attachment; filename=speech.mp3"
    )

    mock_model_provider.load_model.assert_any_call("test_tts_model")
    mock_tts_model.generate.assert_called_once()

    args, kwargs = mock_tts_model.generate.call_args
    assert args[0] == payload["input"]
    assert kwargs.get("voice") == payload["voice"]

    try:
        audio_data, sample_rate = audio_read(io.BytesIO(response.content))
        assert sample_rate > 0
        assert len(audio_data) > 0
    except Exception as e:
        pytest.fail(f"Failed to read or validate MP3 content: {e}")


def _hf_repo_not_found(model_name: str) -> RepositoryNotFoundError:
    """Construct a RepositoryNotFoundError shaped like the real HF client raises."""
    request = httpx2.Request("GET", f"https://huggingface.co/api/models/{model_name}")
    response = httpx2.Response(404, request=request)
    return RepositoryNotFoundError(
        f"404 Client Error. Repository Not Found", response=response
    )


def test_tts_speech_bad_model_returns_404_not_silent_200(client, mock_model_provider):
    """Regression: bad model ids must surface as 4xx, not 200 OK with empty body.

    Before the pre-flight fix, ``StreamingResponse`` committed the response
    headers (200 OK) before the worker thread tried to load the model, so any
    HuggingFace failure ended up as a clean zero-byte body. The fix loads the
    model synchronously before returning the response.
    """

    def _raise(model_name):
        raise _hf_repo_not_found(model_name)

    mock_model_provider.load_model = MagicMock(side_effect=_raise)

    payload = {
        "model": "does-not-exist-on-hf",
        "input": "hi",
        "response_format": "wav",
    }
    response = client.post("/v1/audio/speech", json=payload)

    assert response.status_code == 404, (
        f"expected 404 for bad model, got {response.status_code} "
        f"(body length={len(response.content)})"
    )
    assert response.headers["content-type"].startswith("application/json")
    body = response.json()
    assert "detail" in body
    assert "does-not-exist-on-hf" in body["detail"]


def test_tts_speech_load_failure_returns_500(client, mock_model_provider):
    """Non-HF load failures should surface as 500 with a JSON detail body."""

    def _raise(model_name):
        raise RuntimeError("checkpoint is corrupted")

    mock_model_provider.load_model = MagicMock(side_effect=_raise)

    payload = {"model": "some-model", "input": "hi", "response_format": "wav"}
    response = client.post("/v1/audio/speech", json=payload)

    assert response.status_code == 500
    assert response.headers["content-type"].startswith("application/json")
    body = response.json()
    assert "detail" in body
    assert "some-model" in body["detail"]


def test_stt_transcriptions_bad_model_returns_404(client, mock_model_provider):
    """Same silent-200 hazard applies to the default ndjson streaming response."""

    def _raise(model_name):
        raise _hf_repo_not_found(model_name)

    mock_model_provider.load_model = MagicMock(side_effect=_raise)

    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("test.mp3", _make_transcription_audio_buffer(), "audio/mp3")},
        data={"model": "does-not-exist-on-hf"},
    )

    assert response.status_code == 404, (
        f"expected 404 for bad model, got {response.status_code} "
        f"(body length={len(response.content)})"
    )
    assert response.headers["content-type"].startswith("application/json")
    body = response.json()
    assert "detail" in body
    assert "does-not-exist-on-hf" in body["detail"]


def test_stt_transcriptions(client, mock_model_provider):
    # Test that the stt_transcriptions endpoint returns a 200 status code
    mock_stt_model = MagicMock()
    mock_stt_model.generate = MagicMock(
        return_value={"text": "This is a test transcription."}
    )

    mock_model_provider.load_model = MagicMock(return_value=mock_stt_model)

    sample_rate = 16000
    duration = 1
    frequency = 440
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = 0.5 * np.sin(2 * np.pi * frequency * t).astype(np.float32)

    buffer = io.BytesIO()
    audio_write(buffer, audio_data, sample_rate, format="mp3")
    buffer.seek(0)

    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("test.mp3", buffer, "audio/mp3")},
        data={"model": "test_stt_model"},
    )

    assert response.status_code == 200
    assert response.json() == {"text": "This is a test transcription."}

    mock_model_provider.load_model.assert_any_call("test_stt_model")
    mock_stt_model.generate.assert_called_once()

    assert mock_stt_model.generate.call_args[0][0].startswith("/tmp/")


@pytest.mark.parametrize("filename", ["test.m4a", "test.aac", "test.caf", "test.mp4"])
def test_stt_transcriptions_accepts_non_writable_upload_extensions(
    client, mock_model_provider, filename
):
    mock_stt_model = MagicMock()
    mock_stt_model.generate = MagicMock(return_value={"text": "ok"})
    mock_model_provider.load_model = MagicMock(return_value=mock_stt_model)

    # Content sniffing lets WAV bytes isolate the upload filename.
    buffer = io.BytesIO()
    audio_write(buffer, np.zeros(1600, dtype=np.float32), 16000, format="wav")
    buffer.seek(0)

    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": (filename, buffer, "application/octet-stream")},
        data={"model": "test_stt_model", "response_format": "json"},
    )

    assert response.status_code == 200, response.text
    tmp_path = mock_stt_model.generate.call_args[0][0]
    assert tmp_path.endswith(".wav")


# ---------------------------------------------------------------------------
# OpenAI-compatible response_format tests for /v1/audio/transcriptions
# ---------------------------------------------------------------------------


def _make_transcription_audio_buffer():
    """Build a tiny mp3 buffer suitable as the ``file`` upload field."""
    sample_rate = 16000
    duration = 1
    frequency = 440
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = 0.5 * np.sin(2 * np.pi * frequency * t).astype(np.float32)
    buffer = io.BytesIO()
    audio_write(buffer, audio_data, sample_rate, format="mp3")
    buffer.seek(0)
    return buffer


def _post_transcription(client, mock_model_provider, mock_return, *, response_format):
    mock_stt_model = MagicMock()
    mock_stt_model.generate = MagicMock(return_value=mock_return)
    mock_model_provider.load_model = MagicMock(return_value=mock_stt_model)

    return client.post(
        "/v1/audio/transcriptions",
        files={"file": ("test.mp3", _make_transcription_audio_buffer(), "audio/mp3")},
        data={"model": "test_stt_model", "response_format": response_format},
    )


def test_stt_transcriptions_response_format_text(client, mock_model_provider):
    """response_format=text returns plain text body with the transcript."""
    response = _post_transcription(
        client,
        mock_model_provider,
        {"text": "Hello world."},
        response_format="text",
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert response.text == "Hello world."


def test_stt_transcriptions_response_format_json(client, mock_model_provider):
    """response_format=json returns the OpenAI minimal {"text": ...} shape."""
    response = _post_transcription(
        client,
        mock_model_provider,
        {"text": "Hello world."},
        response_format="json",
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == {"text": "Hello world."}


def test_stt_transcriptions_response_format_verbose_json(client, mock_model_provider):
    """response_format=verbose_json passes the full model payload through unchanged."""
    full_payload = {
        "text": "Hello world.",
        "language": "en",
        "segments": [
            {"id": 0, "text": "Hello", "start": 0.0, "end": 0.5},
            {"id": 1, "text": " world.", "start": 0.5, "end": 1.0},
        ],
    }
    response = _post_transcription(
        client,
        mock_model_provider,
        full_payload,
        response_format="verbose_json",
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    body = response.json()
    assert body["text"] == "Hello world."
    assert body["language"] == "en"
    assert [s["text"] for s in body["segments"]] == ["Hello", " world."]


def test_stt_transcriptions_default_format_preserves_ndjson(
    client, mock_model_provider
):
    """Without response_format, the legacy application/x-ndjson stream is preserved."""
    mock_stt_model = MagicMock()
    mock_stt_model.generate = MagicMock(return_value={"text": "hi"})
    mock_model_provider.load_model = MagicMock(return_value=mock_stt_model)

    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("test.mp3", _make_transcription_audio_buffer(), "audio/mp3")},
        data={"model": "test_stt_model"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    # Each line in the body should be a JSON object.
    lines = [line for line in response.text.splitlines() if line.strip()]
    assert lines, "expected at least one ndjson line"
    import json as _json

    for line in lines:
        _json.loads(line)


# ---------------------------------------------------------------------------
# WebSocket realtime streaming tests
# ---------------------------------------------------------------------------


def make_speech_audio(duration_s, sample_rate=16000):
    """Create int16 audio that reliably triggers VAD speech detection."""
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    # 300 Hz sine wave at high amplitude triggers VAD as speech
    audio = (np.sin(2 * np.pi * 300 * t) * 25000).astype(np.int16)
    return audio


def make_silence_audio(duration_s, sample_rate=16000):
    """Create int16 audio of near-zero values that VAD classifies as silence."""
    return np.zeros(int(sample_rate * duration_s), dtype=np.int16)


def _make_streaming_generate(deltas):
    """Build a mock generate function with a ``stream`` parameter that yields deltas."""

    def generate(audio, *, stream=False, language=None, verbose=False, **kwargs):
        if stream:
            return iter(deltas)
        # Non-streaming fallback (shouldn't be called in streaming tests)
        return MagicMock(
            text="".join(str(d) for d in deltas), segments=None, language=None
        )

    return generate


def _make_non_streaming_generate(text):
    """Build a mock generate without a ``stream`` parameter (legacy models)."""

    def generate(audio, *, language=None, verbose=False, **kwargs):
        return MagicMock(text=text, segments=None, language=None)

    return generate


class MockChunk:
    """Structured streaming result with a .text attribute."""

    def __init__(self, text):
        self.text = text


def _trackable(fn):
    """Wrap fn to track calls while preserving signature for inspect.signature."""
    calls = []

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        calls.append((args, kwargs))
        return fn(*args, **kwargs)

    wrapper.call_args_list = calls
    return wrapper


def _ws_send_audio_and_collect(
    client, mock_model_provider, generate_fn, config_extra=None
):
    """Connect WS, send config + 6s speech, stop, and return all messages."""
    mock_stt_model = MagicMock()
    mock_stt_model.generate = _trackable(generate_fn)
    mock_model_provider.load_model = MagicMock(return_value=mock_stt_model)

    messages = []
    config = {"model": "test-model", "sample_rate": 16000}
    if config_extra:
        config.update(config_extra)
    with client.websocket_connect("/v1/audio/transcriptions/realtime") as ws:
        ws.send_json(config)
        assert ws.receive_json()["status"] == "ready"

        # 6s of speech exceeds initial_chunk_size (1.5s) and max_chunk_size (5s)
        speech = make_speech_audio(6.0)
        chunk_size = 4800  # 300ms chunks
        for i in range(0, len(speech), chunk_size):
            ws.send_bytes(speech[i : i + chunk_size].tobytes())

        ws.send_json({"action": "stop"})

        while True:
            try:
                messages.append(ws.receive_json())
            except Exception:
                break

    return messages, mock_stt_model


def test_realtime_ws_streaming_model_sends_deltas(client, mock_model_provider):
    """Streaming model yields string deltas → delta + complete messages."""
    gen_fn = _make_streaming_generate(["Hello", " world", "!"])
    messages, _ = _ws_send_audio_and_collect(client, mock_model_provider, gen_fn)

    # Find delta and complete messages
    deltas = [m for m in messages if m.get("type") == "delta"]
    completes = [m for m in messages if m.get("type") == "complete"]

    assert len(deltas) >= 1, f"Expected delta messages, got: {messages}"
    assert len(completes) >= 1, f"Expected complete message, got: {messages}"

    # Delta messages should have 'delta' field but no 'text' field (backward compat)
    for d in deltas:
        assert "delta" in d
        assert "text" not in d

    # Complete message should have all fields
    complete = completes[-1]
    assert "text" in complete
    assert "Hello" in complete["text"] and "world" in complete["text"]
    assert "is_partial" in complete


def test_realtime_ws_non_streaming_model_fallback(client, mock_model_provider):
    """Non-streaming model → legacy format messages (no type field)."""
    gen_fn = _make_non_streaming_generate("Transcribed text")
    messages, _ = _ws_send_audio_and_collect(client, mock_model_provider, gen_fn)

    # Should have at least one message with text
    text_msgs = [m for m in messages if "text" in m and "type" not in m]
    assert len(text_msgs) >= 1, f"Expected legacy text message, got: {messages}"
    # Final message should not be partial
    final = [m for m in text_msgs if not m.get("is_partial", True)]
    assert len(final) >= 1, f"Expected final non-partial message, got: {messages}"
    assert final[-1]["text"] == "Transcribed text"


def test_realtime_ws_streaming_structured_chunks(client, mock_model_provider):
    """Streaming model yields objects with .text attribute → delta messages."""
    chunks = [MockChunk("Hello"), MockChunk(" world")]
    gen_fn = _make_streaming_generate(chunks)
    messages, _ = _ws_send_audio_and_collect(client, mock_model_provider, gen_fn)

    deltas = [m for m in messages if m.get("type") == "delta"]
    completes = [m for m in messages if m.get("type") == "complete"]

    assert len(deltas) >= 1, f"Expected delta messages, got: {messages}"
    assert len(completes) >= 1, f"Expected complete message, got: {messages}"

    # Check that delta values come from .text attribute
    delta_texts = [d["delta"] for d in deltas]
    combined = "".join(delta_texts)
    assert "Hello" in combined


def test_realtime_ws_mx_array_pass(client, mock_model_provider):
    """Streaming models receive mx.array, not file paths."""
    gen_fn = _make_streaming_generate(["test"])
    _, mock_stt_model = _ws_send_audio_and_collect(client, mock_model_provider, gen_fn)

    # Check that generate was called with an mx.array (not a string path)
    tracked = mock_stt_model.generate
    assert len(tracked.call_args_list) > 0, "generate was never called"
    first_arg = tracked.call_args_list[0][0][
        0
    ]  # first call, positional args, first arg
    assert isinstance(first_arg, mx.array), f"Expected mx.array, got {type(first_arg)}"


def test_realtime_ws_mx_array_supports_bfloat16_cast(client, mock_model_provider):
    """Regression: models like Parakeet that cast to bfloat16 must receive mx.array."""

    def gen_fn(audio, *, stream=False, language=None, verbose=False, **kwargs):
        if stream:
            # Parakeet's stream_generate does this internally
            _ = audio.astype(mx.bfloat16)
            return iter(["ok"])
        return MagicMock(text="ok", segments=None, language=None)

    messages, _ = _ws_send_audio_and_collect(client, mock_model_provider, gen_fn)
    completes = [m for m in messages if m.get("type") == "complete"]
    assert len(completes) >= 1
    assert completes[0]["text"] == "ok"


def test_realtime_ws_streaming_disabled_fallback(client, mock_model_provider):
    """Streaming-capable model with streaming=false config falls back to legacy format."""
    gen_fn = _make_streaming_generate(["Hello", " world", "!"])
    messages, _ = _ws_send_audio_and_collect(
        client, mock_model_provider, gen_fn, config_extra={"streaming": False}
    )

    # Should have legacy-format messages (no 'type' field), not delta/complete
    deltas = [m for m in messages if m.get("type") == "delta"]
    completes = [m for m in messages if m.get("type") == "complete"]
    assert (
        len(deltas) == 0
    ), f"Expected no delta messages when streaming disabled, got: {deltas}"
    assert (
        len(completes) == 0
    ), f"Expected no complete messages when streaming disabled, got: {completes}"

    # Should have at least one legacy text message
    text_msgs = [m for m in messages if "text" in m and "type" not in m]
    assert len(text_msgs) >= 1, f"Expected legacy text message, got: {messages}"


# --- /v1/realtime server-side VAD turn detection --------------------------


class _FakeStreamingSession:
    """Minimal streaming session: emits one delta once the audio is closed."""

    input_sample_rate = 16000

    def __init__(self):
        self._closed = False
        self._emitted = False

    def feed(self, samples):
        pass

    def close(self):
        self._closed = True

    def step(self, max_decode_tokens=4):
        if self._closed and not self._emitted:
            self._emitted = True
            return ["bonjour"]
        return []

    @property
    def done(self):
        return self._closed and self._emitted


class _FakeStreamingModel:
    def create_streaming_session(self, *, temperature=0.0, **kwargs):
        return _FakeStreamingSession()


class _CountingStreamingSession:
    """Streaming session that counts feed()/step() so a test can assert the
    transcription model is not run while there is no speech."""

    input_sample_rate = 16000

    def __init__(self):
        self.feed_calls = 0
        self.step_calls = 0
        self._closed = False

    def feed(self, samples):
        self.feed_calls += 1

    def close(self):
        self._closed = True

    def step(self, max_decode_tokens=4):
        self.step_calls += 1
        return []

    @property
    def done(self):
        return self._closed


class _CountingStreamingModel:
    def __init__(self, session):
        self._session = session

    def create_streaming_session(self, *, temperature=0.0, **kwargs):
        return self._session


class _FakeVadModel:
    """Returns a scripted speech probability per consumed 512-sample frame."""

    def __init__(self, probs):
        self._probs = probs

    def initial_state(self, sample_rate=16000):
        return {"i": 0}

    def feed(self, chunk, state, sample_rate=16000):
        i = state["i"]
        p = self._probs[i] if i < len(self._probs) else 0.0
        return mx.array([[p]]), {"i": i + 1}


def test_realtime_ws_server_vad_auto_commits(client, mock_model_provider, monkeypatch):
    """With server_vad the server detects end-of-speech, emits
    speech_started/stopped, auto-commits and returns the transcript — the
    client never sends input_audio_buffer.commit."""
    import base64

    from mlx_audio.realtime_vad import VAD_FRAME_SIZE

    mock_model_provider.load_model = MagicMock(return_value=_FakeStreamingModel())
    # Four speech frames, then silence — enough to open then close a turn.
    probs = [0.9] * 4 + [0.0] * 60
    monkeypatch.setattr(
        "mlx_audio.server._load_realtime_vad_model",
        lambda name: _FakeVadModel(probs),
    )

    with client.websocket_connect("/v1/realtime?model=fake-stt") as ws:
        assert ws.receive_json()["type"] == "session.created"

        ws.send_json(
            {
                "type": "session.update",
                "session": {
                    "audio": {
                        "input": {
                            "format": {"type": "audio/pcm", "rate": 16000},
                            "turn_detection": {
                                "type": "server_vad",
                                "silence_duration_ms": 160,
                            },
                        }
                    }
                },
            }
        )
        updated = ws.receive_json()
        assert updated["type"] == "session.updated"
        assert (
            updated["session"]["audio"]["input"]["turn_detection"]["type"]
            == "server_vad"
        )

        pcm = np.zeros(VAD_FRAME_SIZE * 40, dtype=np.int16).tobytes()
        ws.send_json(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm).decode(),
            }
        )

        events = []
        while True:
            evt = ws.receive_json()
            events.append(evt)
            if evt["type"] == ("conversation.item.input_audio_transcription.completed"):
                break

    types = [e["type"] for e in events]
    assert "input_audio_buffer.speech_started" in types
    assert "input_audio_buffer.speech_stopped" in types
    assert "input_audio_buffer.committed" in types
    # speech_started precedes speech_stopped.
    assert types.index("input_audio_buffer.speech_started") < types.index(
        "input_audio_buffer.speech_stopped"
    )
    assert events[-1]["transcript"] == "bonjour"


def test_realtime_ws_server_vad_idle_during_silence(
    client, mock_model_provider, monkeypatch
):
    """With server_vad, pure silence must not run the transcription model at all:
    the model is never fed and step() is never called, so the GPU stays idle
    between turns. (The earlier code fed and decoded every appended chunk, so it
    was pinned at 100% even on silence.)"""
    import base64

    from mlx_audio.realtime_vad import VAD_FRAME_SIZE

    session = _CountingStreamingSession()
    mock_model_provider.load_model = MagicMock(
        return_value=_CountingStreamingModel(session)
    )
    # The VAD never crosses the threshold: this is a silent stream.
    monkeypatch.setattr(
        "mlx_audio.server._load_realtime_vad_model",
        lambda name: _FakeVadModel([0.0] * 400),
    )

    with client.websocket_connect("/v1/realtime?model=fake-stt") as ws:
        assert ws.receive_json()["type"] == "session.created"
        ws.send_json(
            {
                "type": "session.update",
                "session": {
                    "audio": {
                        "input": {
                            "format": {"type": "audio/pcm", "rate": 16000},
                            "turn_detection": {
                                "type": "server_vad",
                                "silence_duration_ms": 160,
                            },
                        }
                    }
                },
            }
        )
        assert ws.receive_json()["type"] == "session.updated"

        pcm = np.zeros(VAD_FRAME_SIZE * 40, dtype=np.int16).tobytes()
        for _ in range(5):
            ws.send_json(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(pcm).decode(),
                }
            )

        # A no-op session.update echoes back only once the appends above have been
        # processed (WebSocket messages are ordered) — a barrier without forcing a
        # commit. It is also proof nothing leaked: any speech_started would arrive
        # before this echo and fail the assertion.
        ws.send_json({"type": "session.update", "session": {}})
        assert ws.receive_json()["type"] == "session.updated"

    assert session.step_calls == 0, "the transcription model decoded silence"
    assert session.feed_calls == 0, "the transcription model was fed silence"


def test_realtime_ws_rejects_semantic_vad(client, mock_model_provider):
    """semantic_vad is not implemented yet → the server replies with an error
    rather than silently ignoring the request."""
    mock_model_provider.load_model = MagicMock(return_value=_FakeStreamingModel())

    with client.websocket_connect("/v1/realtime?model=fake-stt") as ws:
        assert ws.receive_json()["type"] == "session.created"
        ws.send_json(
            {
                "type": "session.update",
                "session": {
                    "audio": {"input": {"turn_detection": {"type": "semantic_vad"}}}
                },
            }
        )
        evt = ws.receive_json()
        assert evt["type"] == "error"
        assert "semantic_vad" in evt["error"]["message"]


# ---------------------------------------------------------------------------
# word_timestamps form field tests
# ---------------------------------------------------------------------------


def test_transcription_request_word_timestamps_defaults():
    """TranscriptionRequest defaults word_timestamps=False, timestamp_granularities=None."""
    from mlx_audio.server import TranscriptionRequest

    req = TranscriptionRequest(model="test-model")
    assert req.word_timestamps is False
    assert req.timestamp_granularities is None


def test_transcription_request_word_timestamps_accepted():
    """TranscriptionRequest accepts word_timestamps=True."""
    from mlx_audio.server import TranscriptionRequest

    req = TranscriptionRequest(
        model="test-model", word_timestamps=True, timestamp_granularities="word"
    )
    assert req.word_timestamps is True
    assert req.timestamp_granularities == "word"


def test_stt_word_timestamps_passed_to_generate(client, mock_model_provider):
    """word_timestamps=true form field reaches stt_model.generate() as a kwarg.

    The STTExecutionAdapter allowlist (_STT_EXTRA_KWARGS) must pass word_timestamps
    through even when it isn't declared in the model's generate() signature.
    """
    captured_kwargs: dict = {}

    def mock_generate(path, **kwargs):
        captured_kwargs.update(kwargs)
        return {"text": "hello", "segments": [], "language": "en"}

    mock_stt_model = MagicMock()
    mock_stt_model.generate = mock_generate
    mock_model_provider.load_model = MagicMock(return_value=mock_stt_model)

    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("test.mp3", _make_transcription_audio_buffer(), "audio/mp3")},
        data={
            "model": "test_stt_model",
            "response_format": "verbose_json",
            "word_timestamps": "true",
        },
    )

    assert response.status_code == 200
    assert captured_kwargs.get("word_timestamps") is True


def _post_transcription_capturing_kwargs(client, mock_model_provider, generate, data):
    """Serve ``generate`` as the STT model's generate() and return its kwargs."""
    captured_kwargs: dict = {}

    def capturing_generate(path, **kwargs):
        captured_kwargs.update(kwargs)
        return {"text": "hello"}

    # Expose the real generate() signature to the server's kwarg filter.
    capturing_generate.__signature__ = inspect.signature(generate)

    mock_stt_model = MagicMock()
    mock_stt_model.generate = capturing_generate
    mock_model_provider.load_model = MagicMock(return_value=mock_stt_model)

    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("test.mp3", _make_transcription_audio_buffer(), "audio/mp3")},
        data={"model": "test_stt_model", "response_format": "json", **data},
    )
    assert response.status_code == 200
    return captured_kwargs


def _generate_default_8192(audio, *, max_tokens: int = 8192, **kwargs): ...


def _generate_default_128(audio, *, max_tokens: int = 128, **kwargs): ...


def _generate_default_none(audio, *, max_tokens=None, **kwargs): ...


def _generate_without_max_tokens(audio, **kwargs): ...


@pytest.mark.parametrize(
    "generate, data, expected",
    [
        # The model's larger default applies when the client omits max_tokens.
        (_generate_default_8192, {}, 8192),
        # Models with a smaller (or no) default keep the previous 1024 floor.
        (_generate_default_128, {}, 1024),
        (_generate_default_none, {}, 1024),
        # An explicit client value always wins.
        (_generate_default_8192, {"max_tokens": "300"}, 300),
        (_generate_default_128, {"max_tokens": "64"}, 64),
    ],
    ids=["default-8192", "default-128", "default-none", "explicit-300", "explicit-64"],
)
def test_stt_max_tokens_default_reaches_generate(
    client, mock_model_provider, generate, data, expected
):
    """A request without max_tokens must not cap a long-audio model at 1024."""
    captured_kwargs = _post_transcription_capturing_kwargs(
        client, mock_model_provider, generate, data
    )
    assert captured_kwargs.get("max_tokens") == expected


def test_stt_max_tokens_not_passed_when_generate_lacks_it(client, mock_model_provider):
    captured_kwargs = _post_transcription_capturing_kwargs(
        client, mock_model_provider, _generate_without_max_tokens, {}
    )
    assert "max_tokens" not in captured_kwargs


def test_stt_prompt_passed_to_generate(client, mock_model_provider):
    """prompt form field reaches generate() for prompt-driven STT models (e.g. MOSS-Music)."""
    captured_kwargs: dict = {}

    def mock_generate(path, prompt=None, **kwargs):
        captured_kwargs["prompt"] = prompt
        return {"text": "C major"}

    mock_stt_model = MagicMock()
    mock_stt_model.generate = mock_generate
    mock_model_provider.load_model = MagicMock(return_value=mock_stt_model)

    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("test.mp3", _make_transcription_audio_buffer(), "audio/mp3")},
        data={
            "model": "test_stt_model",
            "response_format": "json",
            "prompt": "What is the key of this music?",
        },
    )

    assert response.status_code == 200
    assert captured_kwargs["prompt"] == "What is the key of this music?"


def test_stt_prompt_dropped_for_models_without_prompt(client, mock_model_provider):
    """Models whose generate() does not declare prompt never receive it."""
    captured_kwargs: dict = {}

    def mock_generate(path, language=None, **kwargs):
        captured_kwargs.update(kwargs)
        return {"text": "hello"}

    mock_stt_model = MagicMock()
    mock_stt_model.generate = mock_generate
    mock_model_provider.load_model = MagicMock(return_value=mock_stt_model)

    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("test.mp3", _make_transcription_audio_buffer(), "audio/mp3")},
        data={
            "model": "test_stt_model",
            "response_format": "json",
            "prompt": "ignored",
        },
    )

    assert response.status_code == 200
    assert "prompt" not in captured_kwargs


def test_stt_word_timestamps_verbose_json_words_passthrough(
    client, mock_model_provider
):
    """verbose_json response includes words[] from the model when word_timestamps=True."""
    full_payload = {
        "text": "Hello world.",
        "language": "en",
        "segments": [
            {
                "id": 0,
                "text": "Hello world.",
                "start": 0.0,
                "end": 1.0,
                "words": [
                    {"word": "Hello", "start": 0.0, "end": 0.5, "probability": 0.99},
                    {"word": "world.", "start": 0.5, "end": 1.0, "probability": 0.98},
                ],
            }
        ],
    }

    mock_stt_model = MagicMock()
    mock_stt_model.generate = MagicMock(return_value=full_payload)
    mock_model_provider.load_model = MagicMock(return_value=mock_stt_model)

    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("test.mp3", _make_transcription_audio_buffer(), "audio/mp3")},
        data={
            "model": "test_stt_model",
            "response_format": "verbose_json",
            "word_timestamps": "true",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["text"] == "Hello world."
    segments = body.get("segments", [])
    assert len(segments) == 1
    words = segments[0].get("words", [])
    assert len(words) == 2
    assert words[0]["word"] == "Hello"
    assert words[1]["word"] == "world."


class _FakeInferenceHandle:
    def __init__(self, chunks):
        self.result_queue = queue.Queue()
        for chunk in chunks:
            self.result_queue.put(chunk)
        self.cancelled = False

    def cancel(self):
        self.cancelled = True


class _ConnectedRequest:
    async def is_disconnected(self):
        return False


async def test_stream_inference_results_reports_ndjson_error():
    handle = _FakeInferenceHandle(
        [
            InferenceResultChunk(kind="data", payload='{"text": "hello"}\n'),
            InferenceResultChunk(kind="error", error=ValueError("processor missing")),
        ]
    )

    payloads = [
        payload
        async for payload in _stream_inference_results(
            handle, _ConnectedRequest(), ndjson_errors=True
        )
    ]

    assert payloads[0] == '{"text": "hello"}\n'
    assert json.loads(payloads[1]) == {
        "error": {"message": "processor missing", "type": "ValueError"}
    }
    assert handle.cancelled


async def test_stream_inference_results_reraises_error_by_default():
    error = ValueError("processor missing")
    handle = _FakeInferenceHandle([InferenceResultChunk(kind="error", error=error)])

    with pytest.raises(ValueError, match="processor missing") as exc_info:
        async for _ in _stream_inference_results(
            handle, _ConnectedRequest(), ndjson_errors=False
        ):
            pass

    assert exc_info.value is error
    assert handle.cancelled


# ── Sampling defaults: omitted fields use the model's generation config ──

QWEN_GENERATE_CONFIG = {
    "temperature": 0.9,
    "top_p": 1.0,
    "top_k": 50,
    "repetition_penalty": 1.05,
    "max_new_tokens": 8192,
}
SERVER_SAMPLING_DEFAULTS = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "repetition_penalty": 1.0,
}


def _sampling_kwargs(kwargs):
    return {name: kwargs[name] for name in SERVER_SAMPLING_DEFAULTS}


def _tts_model_with_config(generate_config):
    model = MagicMock()
    model.generate = MagicMock(wraps=sync_mock_audio_stream_generator)
    model.generate_config = generate_config
    return model


def test_tts_speech_omitted_sampling_uses_model_generate_config(
    client, mock_model_provider
):
    model = _tts_model_with_config(dict(QWEN_GENERATE_CONFIG))
    mock_model_provider.load_model = MagicMock(return_value=model)

    payload = {"model": "qwen3_tts", "input": "Hello", "voice": "vivian"}
    response = client.post("/v1/audio/speech", json=payload)

    assert response.status_code == 200
    _, kwargs = model.generate.call_args
    assert _sampling_kwargs(kwargs) == {
        "temperature": 0.9,
        "top_p": 1.0,
        "top_k": 50,
        "repetition_penalty": 1.05,
    }
    # Only the four sampling fields are resolved from the config.
    assert kwargs["max_tokens"] == 1200


@pytest.mark.parametrize(
    "sent, expected",
    [
        (
            {"temperature": 0, "top_p": 0, "top_k": 0, "repetition_penalty": 1.0},
            {"temperature": 0, "top_p": 0, "top_k": 0, "repetition_penalty": 1.0},
        ),
        (
            {"top_k": 0},
            {"temperature": 0.9, "top_p": 1.0, "top_k": 0, "repetition_penalty": 1.05},
        ),
        (
            {"temperature": 0.7, "repetition_penalty": 1.2},
            {"temperature": 0.7, "top_p": 1.0, "top_k": 50, "repetition_penalty": 1.2},
        ),
    ],
)
def test_tts_speech_sent_sampling_overrides_generate_config(
    client, mock_model_provider, sent, expected
):
    model = _tts_model_with_config(dict(QWEN_GENERATE_CONFIG))
    mock_model_provider.load_model = MagicMock(return_value=model)

    payload = {"model": "qwen3_tts", "input": "Hello", **sent}
    response = client.post("/v1/audio/speech", json=payload)

    assert response.status_code == 200
    _, kwargs = model.generate.call_args
    assert _sampling_kwargs(kwargs) == expected


@pytest.mark.parametrize("generate_config", ["missing", None, {}, {"top_k": None}])
def test_tts_speech_without_generate_config_keeps_server_defaults(
    client, mock_model_provider, generate_config
):
    if generate_config == "missing":
        # spec= stops MagicMock from fabricating a generate_config attribute.
        model = MagicMock(spec=["generate"])
    else:
        model = MagicMock()
        model.generate_config = generate_config
    model.generate = MagicMock(wraps=sync_mock_audio_stream_generator)
    mock_model_provider.load_model = MagicMock(return_value=model)

    response = client.post(
        "/v1/audio/speech", json={"model": "other_tts", "input": "Hello"}
    )

    assert response.status_code == 200
    _, kwargs = model.generate.call_args
    assert _sampling_kwargs(kwargs) == SERVER_SAMPLING_DEFAULTS


def _tts_inference_request(model, **fields):
    from mlx_audio.server import SpeechRequest, SpeechTaskPayload, TTSExecutionAdapter
    from mlx_audio.server_inference import InferenceRequest

    request = InferenceRequest(
        endpoint_kind="tts",
        model_name="qwen3_tts",
        payload=SpeechTaskPayload(
            request=SpeechRequest(model="qwen3_tts", input="Hello", **fields)
        ),
    )
    setattr(request, TTSExecutionAdapter._REQUEST_MODEL_ATTR, model)
    return request


def test_tts_continuous_batch_options_use_generate_config():
    from mlx_audio.server import TTSExecutionAdapter

    adapter = TTSExecutionAdapter()
    model = _tts_model_with_config(dict(QWEN_GENERATE_CONFIG))

    options = adapter._build_batch_options(_tts_inference_request(model))
    assert (
        options.temperature,
        options.top_p,
        options.top_k,
        options.repetition_penalty,
    ) == (0.9, 1.0, 50, 1.05)

    options = adapter._build_batch_options(
        _tts_inference_request(model, temperature=0.3)
    )
    assert (options.temperature, options.top_k) == (0.3, 50)

    plain = adapter._build_batch_options(_tts_inference_request(MagicMock()))
    assert (
        plain.temperature,
        plain.top_p,
        plain.top_k,
        plain.repetition_penalty,
    ) == (0.7, 0.95, 40, 1.0)


class _BatchTTSModel:
    sample_rate = 16000

    def __init__(self, generate_config):
        self.generate_config = generate_config
        self.batch_calls = []

    def batch_generate(
        self,
        texts,
        voices=None,
        temperature=0.0,
        top_p=0.0,
        top_k=0,
        repetition_penalty=0.0,
        lang_code="auto",
        max_tokens=0,
        stream=False,
        verbose=False,
    ):
        self.batch_calls.append(
            {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
            }
        )
        for idx, _ in enumerate(texts):
            result = MockAudioResult(np.zeros(160, dtype=np.float32), 16000)
            result.sequence_idx = idx
            yield result


def test_tts_run_batch_uses_generate_config():
    from mlx_audio.server import TTSExecutionAdapter

    adapter = TTSExecutionAdapter()
    model = _BatchTTSModel(dict(QWEN_GENERATE_CONFIG))
    requests = [_tts_inference_request(model) for _ in range(2)]

    adapter.run_batch(requests)

    assert model.batch_calls == [
        {"temperature": 0.9, "top_p": 1.0, "top_k": 50, "repetition_penalty": 1.05}
    ]
    for request in requests:
        kinds = [request.result_queue.get_nowait().kind for _ in range(2)]
        assert kinds == ["data", "done"]


def test_tts_batch_key_separates_omitted_from_sent_sampling():
    from mlx_audio.server import TTSExecutionAdapter

    adapter = TTSExecutionAdapter()
    model = _tts_model_with_config(dict(QWEN_GENERATE_CONFIG))

    omitted = adapter.batch_key(_tts_inference_request(model))
    also_omitted = adapter.batch_key(_tts_inference_request(model))
    # Same raw value as the omitted default, but sent explicitly: it must not
    # share a batch with a request that resolves to the model's 0.9.
    sent_default = adapter.batch_key(_tts_inference_request(model, temperature=0.7))

    assert omitted == also_omitted
    assert omitted != sent_default
    hash(omitted)
    # Continuous sessions are keyed the same way.
    assert adapter.continuous_batch_key(
        _tts_inference_request(model)
    ) != adapter.continuous_batch_key(_tts_inference_request(model, temperature=0.7))


def test_tts_speech_streaming_uses_generate_config(client, mock_model_provider):
    model = _tts_model_with_config(dict(QWEN_GENERATE_CONFIG))
    mock_model_provider.load_model = MagicMock(return_value=model)

    payload = {"model": "qwen3_tts", "input": "Hello", "stream": True, "top_p": 0.5}
    response = client.post("/v1/audio/speech", json=payload)

    assert response.status_code == 200
    _, kwargs = model.generate.call_args
    assert kwargs["stream"] is True
    assert _sampling_kwargs(kwargs) == {
        "temperature": 0.9,
        "top_p": 0.5,
        "top_k": 50,
        "repetition_penalty": 1.05,
    }


# -- ModelProvider memory eviction ------------------------------------------


def _make_provider(max_resident=1, idle_ttl=0.0):
    from mlx_audio.server import ModelProvider

    return ModelProvider(max_resident_models=max_resident, idle_ttl_seconds=idle_ttl)


def test_model_provider_lru_evicts_least_recently_used(monkeypatch):
    loaded = []
    monkeypatch.setattr(
        "mlx_audio.server.load_model",
        lambda name: loaded.append(name) or f"model-{name}",
    )
    provider = _make_provider(max_resident=2)

    provider.load_model("a")
    provider.load_model("b")
    provider.load_model("a")  # a becomes the most recently used
    provider.load_model("c")  # evicts b (LRU), keeps a

    assert set(provider.models) == {"a", "c"}
    assert set(provider._last_used.keys()) == {"a", "c"}


def test_model_provider_evicts_when_capacity_exceeded(monkeypatch):
    loaded = []
    monkeypatch.setattr(
        "mlx_audio.server.load_model",
        lambda name: loaded.append(name) or f"model-{name}",
    )
    provider = _make_provider(max_resident=1)

    provider.load_model("a")
    provider.load_model("b")

    assert list(provider.models.keys()) == ["b"]
    assert loaded == ["a", "b"]


async def test_model_provider_remove_model_clears_memory(monkeypatch):
    cleared = []
    monkeypatch.setattr("mlx_audio.server.load_model", lambda name: f"model-{name}")
    monkeypatch.setattr("mlx_audio.server.mx.clear_cache", lambda: cleared.append(1))
    provider = _make_provider()

    provider.load_model("a")
    assert "a" in provider.models

    removed = await provider.remove_model("a")

    assert removed is True
    assert provider.models == {}
    assert cleared == [1]
    assert "a" not in provider._last_used


def test_model_provider_evict_collects_garbage_before_clearing_cache(monkeypatch):
    order = []
    monkeypatch.setattr("mlx_audio.server.load_model", lambda name: f"model-{name}")
    monkeypatch.setattr(
        "mlx_audio.server.gc.collect", lambda: order.append("gc.collect")
    )
    monkeypatch.setattr(
        "mlx_audio.server.mx.clear_cache", lambda: order.append("mx.clear_cache")
    )
    provider = _make_provider()

    provider.load_model("a")
    provider.load_model("b")

    assert order == ["gc.collect", "mx.clear_cache"]


async def test_model_provider_remove_model_missing_returns_false():
    provider = _make_provider()
    assert await provider.remove_model("missing") is False


async def test_model_provider_idle_sweeper_evicts_stale_models(monkeypatch):
    loaded = []
    monkeypatch.setattr(
        "mlx_audio.server.load_model",
        lambda name: loaded.append(name) or f"model-{name}",
    )
    clock = {"now": 0.0}
    monkeypatch.setattr("mlx_audio.server.time.monotonic", lambda: clock["now"])
    provider = _make_provider(idle_ttl=0.05)
    provider.start_sweeper()
    try:
        provider.load_model("a")
        clock["now"] = 0.06
        stale = provider.evict_stale(clock["now"])
        assert stale == ["a"]
        assert "a" not in provider.models

        provider.load_model("b")
        clock["now"] = 0.10  # b is only 0.04s old — still fresh
        assert provider.evict_stale(clock["now"]) == []
        assert "b" in provider.models

        clock["now"] = 0.20  # b is now 0.14s old — stale
        assert provider.evict_stale(clock["now"]) == ["b"]
        assert "b" not in provider.models
    finally:
        provider.stop_sweeper()


def test_model_provider_sweeper_is_noop_when_ttl_disabled(monkeypatch):
    provider = _make_provider(idle_ttl=0.0)
    provider.start_sweeper()
    assert provider._sweeper_task is None
    provider.stop_sweeper()


def test_model_provider_zero_max_resident_is_unbounded(monkeypatch):
    monkeypatch.setattr("mlx_audio.server.load_model", lambda name: f"model-{name}")
    provider = _make_provider(max_resident=0)
    assert provider.max_resident_models == 0

    for i in range(5):
        provider.load_model(f"m{i}")

    assert len(provider.models) == 5


# -- in-use tracking: a model a request holds is never an eviction candidate ---


def test_model_provider_lru_skips_in_use_models(monkeypatch):
    monkeypatch.setattr(
        "mlx_audio.server.load_model",
        lambda name: f"model-{name}",
    )
    provider = _make_provider(max_resident=2)

    # "a" is held by an in-flight request; "b" has been released.
    held = provider.acquire_model("a")
    b = provider.acquire_model("b")
    provider.release_model("b")
    provider.load_model("a")  # a stays most-recently-used either way
    provider.load_model("c")  # capacity exceeded: must evict b, not the held a

    assert set(provider.models) == {"a", "c"}
    assert provider._in_use == {"a": 1}
    assert held == "model-a" and b == "model-b"


def test_model_provider_lru_all_in_use_overflows_and_warns(monkeypatch, capsys):
    monkeypatch.setattr(
        "mlx_audio.server.load_model",
        lambda name: f"model-{name}",
    )
    provider = _make_provider(max_resident=1)

    provider.acquire_model("a")
    provider.acquire_model("b")  # nothing evictable: load proceeds anyway

    assert set(provider.models) == {"a", "b"}
    assert "in use" in capsys.readouterr().out


def test_model_provider_release_makes_model_evictable_again(monkeypatch):
    monkeypatch.setattr(
        "mlx_audio.server.load_model",
        lambda name: f"model-{name}",
    )
    provider = _make_provider(max_resident=2)

    provider.acquire_model("a")
    provider.acquire_model("b")
    provider.release_model("a")
    provider.release_model("a")  # an extra release must not go negative
    provider.load_model("c")

    assert "a" not in provider.models
    assert provider._in_use == {"b": 1}


async def test_model_provider_sweeper_skips_in_use_models(monkeypatch):
    monkeypatch.setattr("mlx_audio.server.load_model", lambda name: f"model-{name}")
    clock = {"now": 0.0}
    monkeypatch.setattr("mlx_audio.server.time.monotonic", lambda: clock["now"])
    provider = _make_provider(idle_ttl=0.05)

    provider.acquire_model("a")  # acquired long "ago" at the same clock time
    clock["now"] = 0.20
    assert provider.evict_stale(clock["now"]) == []
    assert "a" in provider.models

    provider.release_model("a")
    # release_model re-arms the idle clock at "now" (0.20); past the TTL it is evictable
    assert provider.evict_stale(0.26) == ["a"]
    assert "a" not in provider.models


async def test_model_provider_remove_model_warns_when_in_use(monkeypatch, capsys):
    monkeypatch.setattr("mlx_audio.server.load_model", lambda name: f"model-{name}")
    provider = _make_provider()

    provider.acquire_model("a")
    removed = await provider.remove_model("a")

    assert removed is True
    assert provider.models == {}
    assert "still hold it" in capsys.readouterr().out


# -- small-weights nn.Module: eviction must reclaim cyclic model memory ------


class _SmallModule(nn.Module):
    """Tiny ``nn.Module`` holding a small MLX weight buffer (repo convention:
    small weights in tests — real weights are hundreds of MB, this is ~1 MB).

    ``partner`` mirrors how real models have nested modules whose graphs
    cross-reference each other (model ↔ processor), so a loaded model can be
    unreachable-but-uncollectable: refcounting alone cannot free its cycle,
    which is what ``_evict()``'s ``gc.collect()`` step is for.
    """

    def __init__(self, size: int):
        super().__init__()
        self.weight = mx.zeros((size, size), dtype=mx.float32)
        mx.eval(self.weight)
        self.partner = None


def _cyclic_small_model(size: int) -> _SmallModule:
    """Build one model object: a root with a small internal reference cycle."""
    root = _SmallModule(size)
    partner = _SmallModule(size // 2)
    root.partner = partner
    partner.partner = root
    return root


async def test_model_provider_eviction_frees_cyclic_nn_module_weights(monkeypatch):
    """Eviction reclaims a loaded ``nn.Module``'s weight memory.

    ``nn.Module`` graphs contain reference cycles, so dropping the provider's
    reference does not free the weights through refcounting alone: the cycle
    (and its buffers) stay alive. ``_evict()`` runs ``gc.collect()`` before
    ``mx.clear_cache()`` so eviction really releases them.
    """
    monkeypatch.setattr(
        "mlx_audio.server.load_model", lambda name: _cyclic_small_model(512)
    )
    gc.collect()  # settle ambient cyclic garbage, so baselines are stable
    provider = _make_provider(max_resident=2)

    baseline = mx.get_active_memory()
    provider.load_model("a")
    provider.load_model("b")
    assert mx.get_active_memory() - baseline >= 2 * 4 * (512**2 + 256**2)

    removed = await provider.remove_model("a")
    assert removed is True
    # ``b``'s weights are still active; ``a``'s cycle-held buffer was reclaimed
    # by ``_evict()``'s ``gc.collect()`` + ``mx.clear_cache()`` ordering.
    assert mx.get_active_memory() - baseline <= 4 * (512**2 + 256**2) + 4096


def test_small_nn_module_cycle_needs_gc_before_clear_cache():
    """``mx.clear_cache()`` alone cannot free a dropped cyclic ``nn.Module``;
    only ``gc.collect()`` can. This is the leak ``_evict()``'s ordering fixed."""
    gc.collect()  # settle ambient cyclic garbage, so baselines are stable
    baseline = mx.get_active_memory()
    model = _cyclic_small_model(512)  # ~1.5 MB of weights behind a cycle

    del model
    mx.clear_cache()
    # Refcounting cannot break the cycle: the model's weights are still active.
    assert mx.get_active_memory() - baseline >= 4 * (512**2 + 256**2)

    gc.collect()
    mx.clear_cache()
    assert abs(mx.get_active_memory() - baseline) <= 4096
