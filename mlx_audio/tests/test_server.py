import io
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from mlx_audio.server import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def mock_model_provider():
    with patch(
        "mlx_audio.server.model_provider", new_callable=AsyncMock
    ) as mock_provider:
        yield mock_provider


def test_list_models_empty(client, mock_model_provider):
    mock_model_provider.get_available_models = AsyncMock(return_value=[])
    response = client.get("/v1/models")
    assert response.status_code == 200
    assert response.json() == {"object": "list", "data": []}


def test_list_models_with_data(client, mock_model_provider):
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
    mock_model_provider.load_model = AsyncMock()  # Mocking the instance method
    response = client.post("/v1/models?model_name=test_model")
    assert response.status_code == 200
    assert response.json() == {
        "status": "success",
        "message": "Model test_model added successfully",
    }
    # Check if the instance method load_model was called correctly
    # Since model_provider itself is a mock, its methods are also mocks.
    # We need to assert that the method on the *mocked instance* was called.
    mock_model_provider.load_model.assert_called_once_with("test_model")


def test_remove_model_success(client, mock_model_provider):
    mock_model_provider.remove_model = AsyncMock(return_value=True)
    response = client.delete("/v1/models?model_name=test_model")
    assert response.status_code == 204
    mock_model_provider.remove_model.assert_called_once_with("test_model")


def test_remove_model_not_found(client, mock_model_provider):
    mock_model_provider.remove_model = AsyncMock(return_value=False)
    response = client.delete("/v1/models?model_name=non_existent_model")
    assert response.status_code == 404
    assert response.json() == {"detail": "Model 'non_existent_model' not found"}
    mock_model_provider.remove_model.assert_called_once_with("non_existent_model")


def test_remove_model_with_quotes_in_name(client, mock_model_provider):
    # Test case for model names that might come in with surrounding quotes
    # from unquote, e.g. model_name='"model_with_quotes"'
    mock_model_provider.remove_model = AsyncMock(return_value=True)
    response = client.delete('/v1/models?model_name="test_model_quotes"')
    assert response.status_code == 204
    # The unquote logic in the endpoint should strip the quotes
    mock_model_provider.remove_model.assert_called_once_with("test_model_quotes")


class MockAudioResult:
    def __init__(self, audio_data, sample_rate):
        self.audio = audio_data
        self.sample_rate = sample_rate


async def mock_generate_audio_stream(input_text: str, **kwargs):
    # Simulate audio generation
    sample_rate = 16000
    duration = 1  # 1 second
    frequency = 440  # A4 note
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = 0.5 * np.sin(2 * np.pi * frequency * t)
    # Ensure it's a list of AudioResult or similar structure expected by generate_audio
    yield MockAudioResult(audio_data.astype(np.float32), sample_rate)


def test_tts_speech(client, mock_model_provider):
    mock_tts_model = AsyncMock()
    # Use wraps for async generator, ensuring the signature matches the actual generate method
    mock_tts_model.generate = AsyncMock(wraps=mock_generate_audio_stream)

    mock_model_provider.load_model = AsyncMock(return_value=mock_tts_model)

    payload = {"model": "test_tts_model", "input": "Hello world", "voice": "alloy"}
    response = client.post("/v1/audio/speech", json=payload)
    assert response.status_code == 200
    assert response.headers["media-type"] == "audio/wav"
    assert response.headers["content-disposition"] == "attachment; filename=speech.wav"

    # Check if the model was loaded and generate was called
    mock_model_provider.load_model.assert_called_once_with("test_tts_model")
    mock_tts_model.generate.assert_called_once()
    # We can also check the arguments passed to generate if needed
    args, kwargs = mock_tts_model.generate.call_args
    assert args[0] == payload["input"]
    assert kwargs.get("voice") == payload["voice"]

    # Verify the audio content (optional, but good for sanity check)
    # This part is a bit more involved as it requires parsing the WAV
    try:
        audio_data, sample_rate = sf.read(io.BytesIO(response.content))
        assert sample_rate > 0
        assert len(audio_data) > 0
    except Exception as e:
        pytest.fail(f"Failed to read or validate WAV content: {e}")


def test_stt_transcriptions(client, mock_model_provider):
    mock_stt_model = AsyncMock()
    mock_stt_model.generate = AsyncMock(
        return_value={"text": "This is a test transcription."}
    )
    mock_model_provider.load_model = AsyncMock(return_value=mock_stt_model)

    # Create a dummy WAV file in memory
    sample_rate = 16000
    duration = 1
    frequency = 440
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = 0.5 * np.sin(2 * np.pi * frequency * t).astype(np.float32)

    buffer = io.BytesIO()
    sf.write(buffer, audio_data, sample_rate, format="WAV")
    buffer.seek(0)

    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("test.wav", buffer, "audio/wav")},
        data={"model": "test_stt_model"},
    )

    assert response.status_code == 200
    assert response.json() == {"text": "This is a test transcription."}

    mock_model_provider.load_model.assert_called_once_with("test_stt_model")
    mock_stt_model.generate.assert_called_once()
    # The generate method for STT receives the file path of the temp audio
    # We can check that it was called, but checking the exact temp path is tricky and brittle
    assert mock_stt_model.generate.call_args[0][0].startswith("/tmp/")
    assert mock_stt_model.generate.call_args[0][0].endswith(".wav")
