from pathlib import Path

from mlx_audio.stt.eval.standard import (
    iter_standard_eval_samples,
    sample_from_standard_row,
)


def test_sample_from_standard_row_accepts_default_columns(tmp_path):
    sample = sample_from_standard_row(
        {
            "utt_id": "utt",
            "audio_path": "audio.wav",
            "reference_text": "hello",
            "speaker": "abc",
        },
        base_dir=tmp_path,
    )

    assert sample.utt_id == "utt"
    assert sample.audio_path == tmp_path / "audio.wav"
    assert sample.reference_text == "hello"
    assert sample.source_path == "audio.wav"
    assert sample.metadata == {"speaker": "abc"}


def test_sample_from_standard_row_accepts_alias_columns():
    sample = sample_from_standard_row(
        {
            "id": "utt",
            "audio": {"path": "/tmp/audio.wav"},
            "text": "hello",
        }
    )

    assert sample.utt_id == "utt"
    assert sample.audio_path == Path("/tmp/audio.wav")
    assert sample.reference_text == "hello"


def test_iter_standard_eval_samples_maps_rows():
    samples = list(
        iter_standard_eval_samples(
            [
                {"utt_id": "a", "audio_path": "a.wav", "reference_text": "A"},
                {"sample_id": "b", "path": "b.wav", "transcript": "B"},
            ]
        )
    )

    assert [sample.utt_id for sample in samples] == ["a", "b"]
