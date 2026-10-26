import json
from types import SimpleNamespace

import pytest

from mlx_audio.stt.generate import save_as_json, save_as_srt, save_as_vtt


@pytest.mark.parametrize(
    "writer,extension", [(save_as_srt, "srt"), (save_as_vtt, "vtt")]
)
@pytest.mark.parametrize(
    "untimed",
    [
        {"text": "hello", "speaker_id": 1},
        {"text": "hello", "start": None, "end": None},
        {"text": "hello", "start": 0.0},
        {"text": "hello", "end": 1.0},
        {"text": "hello", "start": 0.0, "end": None},
        {"text": "hello", "start": None, "end": 1.0},
    ],
)
def test_subtitles_reject_untimed_text_before_writing(
    tmp_path, capsys, writer, extension, untimed
):
    output = SimpleNamespace(
        text="timed hello",
        segments=[{"text": "timed", "start": 0.0, "end": 0.5}, untimed],
    )
    path = tmp_path / "transcript"
    target = path.with_suffix(f".{extension}")
    for existing in (False, True):
        if existing:
            target.write_text("keep this")
        with pytest.raises(ValueError, match="requires timestamps.*TXT or JSON"):
            writer(output, str(path))
        if existing:
            assert target.read_text() == "keep this"
        else:
            assert not target.exists()
    with pytest.raises(ValueError, match="requires timestamps"):
        writer(output, "-")
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("writer", [save_as_srt, save_as_vtt])
def test_subtitles_reject_transcript_without_segments(writer, tmp_path):
    with pytest.raises(ValueError, match="requires timestamps"):
        writer(SimpleNamespace(text="hello", segments=[]), str(tmp_path / "transcript"))


@pytest.mark.parametrize(
    "writer,extension,separator", [(save_as_srt, "srt", ","), (save_as_vtt, "vtt", ".")]
)
def test_subtitles_preserve_timed_text(writer, extension, separator, tmp_path):
    output = SimpleNamespace(
        text="hello",
        segments=[
            {"text": "", "speaker_id": 1},
            {"text": "hello", "start": 0.25, "end": 1.5},
        ],
    )
    path = tmp_path / "transcript"
    writer(output, str(path))
    header = "WEBVTT\n\n" if extension == "vtt" else ""
    assert path.with_suffix(f".{extension}").read_text() == (
        header + f"1\n00:00:00{separator}250 --> 00:00:01{separator}500\nhello\n\n"
    )


def test_json_preserves_missing_null_and_numeric_timestamps(tmp_path):
    output = SimpleNamespace(
        text="one two three",
        segments=[
            {"text": "one", "speaker_id": 1},
            {"text": "two", "start": None, "end": None},
            {
                "text": "three",
                "start": 1.0,
                "end": 2.5,
                "words": [{"word": "three", "start": 1.0, "end": 2.5}],
            },
        ],
    )
    path = tmp_path / "transcript"
    save_as_json(output, str(path))
    saved = json.loads(path.with_suffix(".json").read_text())
    assert saved["text"] == output.text
    assert saved["segments"] == [
        output.segments[0],
        {**output.segments[1], "duration": None},
        {**output.segments[2], "duration": 1.5},
    ]
