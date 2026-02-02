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

    mock_model_provider.load_model.assert_called_once_with("test_tts_model")
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
