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
        self.last_generate_kwargs = None

    def generate(self, **kwargs):
        self.generate_call_count += 1
        self.last_generate_kwargs = dict(kwargs)
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

    def test_join_audio_verbose_does_not_reference_chunk_file_name(self):
        model = _DummyModel([_result(1), _result(2)])
        output = io.StringIO()

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch("mlx_audio.tts.generate.audio_write") as audio_write_mock,
            redirect_stdout(output),
            redirect_stderr(output),
        ):
            generate_audio(
                text="join case",
                model=model,
                stream=False,
                join_audio=True,
                output_path=tmpdir,
                file_prefix="joined_case",
                verbose=True,
            )

        self.assertEqual(model.generate_call_count, 1)
        self.assertEqual(audio_write_mock.call_count, 1)
        self.assertEqual(
            audio_write_mock.call_args.args[0],
            os.path.join(tmpdir, "joined_case.wav"),
        )
        self.assertNotIn("Error loading model", output.getvalue())
        self.assertIn("✅ Audio successfully generated and saving as:", output.getvalue())

    def test_generate_passes_duration_and_n_vq_controls(self):
        model = _DummyModel([_result(1)])

        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "mlx_audio.tts.generate.audio_write"
        ):
            generate_audio(
                text="controls",
                model=model,
                stream=False,
                output_path=tmpdir,
                file_prefix="controls_case",
                duration_s=None,
                seconds=3.0,
                n_vq_for_inference=8,
                verbose=False,
            )

        self.assertEqual(model.generate_call_count, 1)
        self.assertIsNotNone(model.last_generate_kwargs)
        assert model.last_generate_kwargs is not None
        self.assertEqual(model.last_generate_kwargs.get("duration_s"), 3.0)
        self.assertEqual(model.last_generate_kwargs.get("n_vq_for_inference"), 8)

    def test_generate_forwards_preset_and_model_kwargs_json(self):
        model = _DummyModel([_result(1)])

        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "mlx_audio.tts.generate.audio_write"
        ):
            generate_audio(
                text="preset controls",
                model=model,
                stream=False,
                output_path=tmpdir,
                file_prefix="preset_case",
                preset="moss_tts_local",
                model_kwargs_json='{"decode_chunk_duration": 0.25, "top_k": 17}',
                verbose=False,
            )

        self.assertEqual(model.generate_call_count, 1)
        assert model.last_generate_kwargs is not None
        self.assertEqual(model.last_generate_kwargs.get("preset"), "moss_tts_local")
        self.assertEqual(model.last_generate_kwargs.get("decode_chunk_duration"), 0.25)
        self.assertEqual(model.last_generate_kwargs.get("top_k"), 17)

    def test_long_form_controls_forward_only_when_enabled(self):
        model = _DummyModel([_result(1)])

        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "mlx_audio.tts.generate.audio_write"
        ):
            generate_audio(
                text="no long form",
                model=model,
                stream=False,
                output_path=tmpdir,
                file_prefix="no_long_form_case",
                verbose=False,
            )
        assert model.last_generate_kwargs is not None
        self.assertNotIn("long_form", model.last_generate_kwargs)

        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "mlx_audio.tts.generate.audio_write"
        ):
            generate_audio(
                text="with long form",
                model=model,
                stream=False,
                output_path=tmpdir,
                file_prefix="long_form_case",
                long_form=True,
                long_form_min_chars=111,
                long_form_retry_attempts=2,
                verbose=False,
            )
        assert model.last_generate_kwargs is not None
        self.assertTrue(model.last_generate_kwargs.get("long_form"))
        self.assertEqual(model.last_generate_kwargs.get("long_form_min_chars"), 111)
        self.assertEqual(model.last_generate_kwargs.get("long_form_retry_attempts"), 2)


if __name__ == "__main__":
    unittest.main()
