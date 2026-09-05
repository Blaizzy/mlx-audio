"""Exercise the actual ASGI realtime route with a tiny Nemotron model."""

import base64

import numpy as np
from fastapi.testclient import TestClient

from mlx_audio import server
from mlx_audio.stt.tests.test_nemotron_session import model


def test_realtime_partials_commit_and_second_turn(monkeypatch):
    monkeypatch.setattr(server.model_provider, "load_model", lambda name: model())
    client = TestClient(server.app)
    with client.websocket_connect("/v1/realtime?model=nemotron-test") as ws:
        assert ws.receive_json()["type"] == "session.created"
        ws.send_json(
            {
                "type": "session.update",
                "session": {
                    "audio": {"input": {"format": {"type": "audio/pcm", "rate": 16000}}}
                },
            }
        )
        assert ws.receive_json()["type"] == "session.updated"
        for _ in range(2):
            for _ in range(80):
                audio = base64.b64encode(np.zeros(320, dtype=np.int16)).decode()
                ws.send_json({"type": "input_audio_buffer.append", "audio": audio})
            assert ws.receive_json()["type"] == "conversation.item.added"
            event = ws.receive_json()
            assert event["type"] == "conversation.item.input_audio_transcription.delta"
            assert event["delta"]
            ws.send_json({"type": "input_audio_buffer.commit"})
            committed = False
            while True:
                event = ws.receive_json()
                assert event["type"] != "error", event
                if event["type"] == "input_audio_buffer.committed":
                    committed = True
                if (
                    event["type"]
                    == "conversation.item.input_audio_transcription.completed"
                ):
                    assert committed
                    assert event["transcript"]
                    break
