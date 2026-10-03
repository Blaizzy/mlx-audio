"""Tests for the music-specific generation interface."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mlx.core as mx
import numpy as np
import pytest

from mlx_audio.music.generate import configure_parser, generate_music, main


def test_parser_exposes_only_music_generation_controls() -> None:
    parser = configure_parser()
    args = parser.parse_args(
        [
            "--model",
            "mlx-community/MiniMax-Music3-mxfp8",
            "--caption",
            "Warm acoustic pop",
            "--lyrics",
            "[verse]\nMorning light",
            "--duration",
            "30",
            "--steps",
            "20",
            "--seed",
            "7",
            "--output",
            "song.wav",
        ]
    )

    assert args.caption == "Warm acoustic pop"
    assert args.lyrics == "[verse]\nMorning light"
    assert args.duration == 30
    assert args.steps == 20
    assert args.seed == 7
    assert str(args.output) == "song.wav"

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--model",
                "model",
                "--caption",
                "caption",
                "--lyrics",
                "[instrumental]",
                "--voice",
                "speaker",
            ]
        )


@patch("mlx_audio.music.generate.audio_write")
def test_generate_music_joins_results_and_writes_audio(mock_write, tmp_path) -> None:
    model = MagicMock()
    model.generate.return_value = [
        SimpleNamespace(audio=mx.array([[0.1, -0.1]]), sample_rate=44_100),
        SimpleNamespace(audio=mx.array([[0.2, -0.2]]), sample_rate=44_100),
    ]
    output = tmp_path / "nested" / "song.wav"

    result = generate_music(
        caption="Warm acoustic pop",
        lyrics="[verse]\nMorning light",
        model=model,
        duration=30,
        steps=20,
        seed=7,
        output_path=output,
        verbose=False,
    )

    assert result == output
    model.generate.assert_called_once_with(
        text="Warm acoustic pop",
        lyrics="[verse]\nMorning light",
        steps=20,
        seed=7,
        duration=30,
    )
    written_path, audio, sample_rate = mock_write.call_args.args
    assert written_path == output
    assert sample_rate == 44_100
    np.testing.assert_allclose(
        np.asarray(audio),
        np.array([[0.1, -0.1], [0.2, -0.2]], dtype=np.float32),
    )


def test_main_reads_lyrics_file(tmp_path) -> None:
    lyrics_path = tmp_path / "lyrics.txt"
    lyrics_path.write_text("[chorus]\nSing with me", encoding="utf-8")

    with (
        patch(
            "sys.argv",
            [
                "generate.py",
                "--model",
                "model",
                "--caption",
                "caption",
                "--lyrics-file",
                str(lyrics_path),
            ],
        ),
        patch("mlx_audio.music.generate.generate_music") as mock_generate,
    ):
        main()

    mock_generate.assert_called_once_with(
        caption="caption",
        lyrics="[chorus]\nSing with me",
        model="model",
        duration=None,
        steps=30,
        seed=0,
        output_path=configure_parser().get_default("output"),
        verbose=True,
    )
