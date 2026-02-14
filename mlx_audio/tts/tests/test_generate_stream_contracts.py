import io
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx

from mlx_audio.tts.generate import generate_audio


class _DummyModel:
    sample_rate = 24000

    def __init__(self, results):
        self._results = list(results)
        self.generate_call_count = 0

    def generate(self, **kwargs):
        self.generate_call_count += 1
        for result in self._results:
            yield result


class _DummyPlayer:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self.queued = 0
        self.waited = False
        self.stopped = False

    def queue_audio(self, _audio):
        self.queued += 1

    def wait_for_drain(self):
        self.waited = True

    def stop(self):
        self.stopped = True


def _result(audio_scale):
    audio = mx.array([0.1 * audio_scale, 0.2 * audio_scale], dtype=mx.float32)
    return SimpleNamespace(
        audio=audio,
        sample_rate=24000,
        audio_duration="00:00:00.000",
        token_count=4,
        prompt={"tokens-per-sec": 42.0},
        audio_samples={"samples": int(audio.shape[0]), "samples-per-sec": 24000.0},
        real_time_factor=0.1,
        processing_time_seconds=0.01,
        peak_memory_usage=0.01,
    )


class TestGenerateStreamContracts(unittest.TestCase):
    def test_stream_writes_chunks_with_output_path_and_no_playback(self):
        model = _DummyModel([_result(1), _result(2)])

        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "mlx_audio.tts.generate.audio_write"
        ) as audio_write_mock:
            generate_audio(
                text="stream me",
                model=model,
                stream=True,
                play=False,
                output_path=tmpdir,
                file_prefix="stream_case",
                verbose=False,
            )

        self.assertEqual(model.generate_call_count, 1)
        self.assertEqual(audio_write_mock.call_count, 2)
        written_files = [call.args[0] for call in audio_write_mock.call_args_list]
        self.assertEqual(
            written_files,
            [
                os.path.join(tmpdir, "stream_case_000.wav"),
                os.path.join(tmpdir, "stream_case_001.wav"),
            ],
        )

    def test_stream_without_sink_fails_before_generation(self):
        model = _DummyModel([_result(1)])
        output = io.StringIO()

        with (
            patch("mlx_audio.tts.generate.audio_write") as audio_write_mock,
            redirect_stdout(output),
            redirect_stderr(output),
        ):
            generate_audio(
                text="missing sink",
                model=model,
                stream=True,
                play=False,
                output_path=None,
                verbose=False,
            )

        self.assertEqual(model.generate_call_count, 0)
        audio_write_mock.assert_not_called()
        self.assertIn(
            "Streaming mode requires at least one sink",
            output.getvalue(),
        )

    def test_stream_with_playback_allows_no_output_path(self):
        model = _DummyModel([_result(1), _result(2)])
        created_players: list[_DummyPlayer] = []

        def _make_player(sample_rate):
            player = _DummyPlayer(sample_rate=sample_rate)
            created_players.append(player)
            return player

        with (
            patch("mlx_audio.tts.generate.AudioPlayer", side_effect=_make_player),
            patch("mlx_audio.tts.generate.audio_write") as audio_write_mock,
        ):
            generate_audio(
                text="playback sink",
                model=model,
                stream=True,
                play=True,
                output_path=None,
                verbose=False,
            )

        self.assertEqual(model.generate_call_count, 1)
        audio_write_mock.assert_not_called()
        self.assertEqual(len(created_players), 1)
        self.assertEqual(created_players[0].queued, 2)
        self.assertTrue(created_players[0].waited)
        self.assertTrue(created_players[0].stopped)


if __name__ == "__main__":
    unittest.main()
