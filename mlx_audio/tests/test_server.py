import functools
import io
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from mlx_audio.audio_io import read as audio_read
from mlx_audio.audio_io import write as audio_write

# python-multipart is required for FastAPI file uploads
pytest.importorskip("multipart", reason="python-multipart is required for server tests")

from fastapi.testclient import TestClient

from mlx_audio.server import app


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


def sync_mock_realtime_generator(input_text: str, **kwargs):
    del input_text, kwargs
    sample_rate = 24000
    duration = 0.25
    frequency = 523.25
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = 0.5 * np.sin(2 * np.pi * frequency * t)
    yield MockAudioResult(audio_data.astype(np.float32), sample_rate)


def strict_non_realtime_generator(input_text: str, *, voice=None, stream=False):
    del input_text, voice, stream
    sample_rate = 16000
    duration = 0.25
    frequency = 330.0
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = 0.5 * np.sin(2 * np.pi * frequency * t)
    yield MockAudioResult(audio_data.astype(np.float32), sample_rate)


def test_tts_speech(client, mock_model_provider):
    # Test that the tts_speech endpoint returns a 200 status code
    mock_tts_model = MagicMock()
    mock_tts_model.generate = MagicMock(wraps=sync_mock_audio_stream_generator)

    mock_model_provider.load_model = MagicMock(return_value=mock_tts_model)

    payload = {
        "model": "test_tts_model",
        "input": "Hello world",
        "voice": "alloy",
        "input_type": "pinyin",
        "instruct": "calm",
        "tokens": 80,
        "seconds": 2.0,
        "n_vq_for_inference": 8,
        "quality": "studio",
        "sound_event": "speech",
        "ambient_sound": "office",
        "language": "zh",
        "preset": "moss_tts_local",
        "model_kwargs": {"decode_chunk_duration": 0.2},
    }
    response = client.post("/v1/audio/speech", json=payload)
    assert response.status_code == 200
    assert response.headers["content-type"].lower() == "audio/mp3"
    assert (
        response.headers["content-disposition"].lower()
        == "attachment; filename=speech.mp3"
    )

    mock_model_provider.load_model.assert_called_once_with("test_tts_model")
    mock_tts_model.generate.assert_called_once()

    args, kwargs = mock_tts_model.generate.call_args
    assert args[0] == payload["input"]
    assert kwargs.get("voice") == payload["voice"]
    assert kwargs.get("tokens") == payload["tokens"]
    assert kwargs.get("duration_s") == payload["seconds"]
    assert kwargs.get("n_vq_for_inference") == payload["n_vq_for_inference"]
    assert kwargs.get("input_type") == payload["input_type"]
    assert kwargs.get("instruct") == payload["instruct"]
    assert kwargs.get("quality") == payload["quality"]
    assert kwargs.get("sound_event") == payload["sound_event"]
    assert kwargs.get("ambient_sound") == payload["ambient_sound"]
    assert kwargs.get("language") == payload["language"]
    assert kwargs.get("preset") == payload["preset"]
    assert kwargs.get("decode_chunk_duration") == payload["model_kwargs"]["decode_chunk_duration"]

    try:
        audio_data, sample_rate = audio_read(io.BytesIO(response.content))
        assert sample_rate > 0
        assert len(audio_data) > 0
    except Exception as e:
        pytest.fail(f"Failed to read or validate MP3 content: {e}")


def test_tts_speech_realtime_single_turn_compat(client, mock_model_provider):
    mock_tts_model = MagicMock()
    mock_tts_model.model_type = "moss_tts_realtime"
    mock_tts_model.generate = MagicMock(wraps=sync_mock_realtime_generator)
    mock_model_provider.load_model = MagicMock(return_value=mock_tts_model)

    payload = {
        "model": "test_realtime_model",
        "input": "Realtime one-shot",
        "response_format": "wav",
        "include_system_prompt": False,
        "reset_cache": False,
        "chunk_frames": 12,
        "overlap_frames": 2,
        "decode_chunk_duration": 0.18,
        "max_pending_frames": 128,
        "decode_kwargs": {"chunk_duration": 0.11},
    }
    response = client.post("/v1/audio/speech", json=payload)
    assert response.status_code == 200
    assert response.headers["content-type"].lower() == "audio/wav"
    mock_tts_model.generate.assert_called_once()

    args, kwargs = mock_tts_model.generate.call_args
    assert args[0] == payload["input"]
    assert kwargs.get("include_system_prompt") is False
    assert kwargs.get("reset_cache") is False
    assert kwargs.get("chunk_frames") == 12
    assert kwargs.get("overlap_frames") == 2
    assert kwargs.get("decode_chunk_duration") == 0.18
    assert kwargs.get("max_pending_frames") == 128
    assert kwargs.get("decode_kwargs") == {"chunk_duration": 0.11}

    audio_data, sample_rate = audio_read(io.BytesIO(response.content))
    assert sample_rate == 24000
    assert len(audio_data) > 0


def test_tts_speech_filters_unsupported_kwargs_for_strict_models(
    client, mock_model_provider
):
    mock_tts_model = MagicMock()
    captured = {"kwargs": None}

    def tracked_strict_generate(input_text: str, *, voice=None, stream=False):
        captured["kwargs"] = {"voice": voice, "stream": stream}
        yield from strict_non_realtime_generator(
            input_text,
            voice=voice,
            stream=stream,
        )

    mock_tts_model.generate = tracked_strict_generate
    mock_model_provider.load_model = MagicMock(return_value=mock_tts_model)

    payload = {
        "model": "test_strict_model",
        "input": "Strict signature",
        "voice": "alloy",
        "chunk_frames": 16,
        "overlap_frames": 4,
        "decode_chunk_duration": 0.2,
        "max_pending_frames": 64,
        "decode_kwargs": {"chunk_duration": 0.15},
    }
    response = client.post("/v1/audio/speech", json=payload)
    assert response.status_code == 200

    assert captured["kwargs"] is not None
    assert captured["kwargs"]["voice"] == "alloy"
    assert captured["kwargs"]["stream"] is False

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

    mock_model_provider.load_model.assert_called_once_with("test_stt_model")
    mock_stt_model.generate.assert_called_once()

    assert mock_stt_model.generate.call_args[0][0].startswith("/tmp/")


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


def test_realtime_ws_numpy_direct_pass(client, mock_model_provider):
    """Streaming models receive numpy arrays directly, not file paths."""
    gen_fn = _make_streaming_generate(["test"])
    _, mock_stt_model = _ws_send_audio_and_collect(client, mock_model_provider, gen_fn)

    # Check that generate was called with a numpy array (not a string path)
    tracked = mock_stt_model.generate
    assert len(tracked.call_args_list) > 0, "generate was never called"
    first_arg = tracked.call_args_list[0][0][
        0
    ]  # first call, positional args, first arg
    assert isinstance(
        first_arg, np.ndarray
    ), f"Expected numpy array, got {type(first_arg)}"


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
