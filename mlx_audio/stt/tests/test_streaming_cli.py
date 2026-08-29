import json

from mlx_audio.stt.generate import generate_transcription
from mlx_audio.stt.models.qwen3_asr.qwen3_asr import StreamingResult


class _StreamingModel:
    def generate(self, audio, *, stream=False, verbose=False):
        assert stream
        yield StreamingResult(
            text="Hello",
            is_final=False,
            start_time=None,
            end_time=None,
            language="English",
        )
        yield StreamingResult(
            text="",
            is_final=False,
            start_time=0.0,
            end_time=1.0,
            language="English",
        )
        yield StreamingResult(
            text=" world",
            is_final=False,
            start_time=None,
            end_time=None,
            language="English",
        )
        yield StreamingResult(
            text="",
            is_final=True,
            start_time=1.0,
            end_time=2.0,
            language="English",
            prompt_tokens=5,
            generation_tokens=2,
        )


def test_streaming_cli_groups_untimed_text_at_timed_boundary(tmp_path):
    output_path = tmp_path / "transcript"

    result = generate_transcription(
        model=_StreamingModel(),
        audio="ignored.wav",
        output_path=str(output_path),
        format="json",
        stream=True,
    )

    assert result.text == "Hello world"
    assert result.segments == [
        {
            "text": "Hello",
            "start": 0.0,
            "end": 1.0,
            "is_final": False,
        },
        {
            "text": " world",
            "start": 1.0,
            "end": 2.0,
            "is_final": True,
        },
    ]
    assert result.prompt_tokens == 5
    assert result.generation_tokens == 2

    saved = json.loads(output_path.with_suffix(".json").read_text())
    assert saved["text"] == "Hello world"
    assert saved["segments"] == [
        {
            "text": "Hello",
            "start": 0.0,
            "end": 1.0,
            "duration": 1.0,
        },
        {
            "text": " world",
            "start": 1.0,
            "end": 2.0,
            "duration": 1.0,
        },
    ]
