import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mlx.core as mx
import numpy as np

from mlx_audio.tts.generate import generate_audio, parse_args


class TestGenerateArgs(unittest.TestCase):
    def test_save_requires_stream(self):
        test_args = [
            "--model",
            "dummy-model",
            "--text",
            "hello",
            "--save",
        ]

        with patch.object(sys, "argv", ["generate.py"] + test_args):
            with self.assertRaises(SystemExit) as exc:
                parse_args()

        self.assertEqual(exc.exception.code, 2)

    def test_save_with_stream_is_allowed(self):
        test_args = [
            "--model",
            "dummy-model",
            "--text",
            "hello",
            "--stream",
            "--save",
        ]

        with patch.object(sys, "argv", ["generate.py"] + test_args):
            args = parse_args()

        self.assertTrue(args.stream)
        self.assertTrue(args.save)


class TestGenerateAudio(unittest.TestCase):
    @staticmethod
    def _result(audio, sample_rate=24000, segment_idx=0):
        return SimpleNamespace(
            audio=mx.array(audio),
            sample_rate=sample_rate,
            segment_idx=segment_idx,
            audio_duration="00:00:00.100",
            audio_samples={"samples": len(audio), "samples-per-sec": 1000.0},
            token_count=1,
            prompt={"tokens-per-sec": 10.0},
            real_time_factor=1.0,
            processing_time_seconds=0.1,
            peak_memory_usage=0.1,
        )

    @patch("builtins.print")
    @patch("mlx_audio.tts.generate.audio_write")
    @patch("mlx_audio.tts.generate.AudioPlayer")
    def test_stream_save_writes_joined_audio(
        self, mock_audio_player, mock_audio_write, _mock_print
    ):
        player = MagicMock()
        mock_audio_player.return_value = player

        model = MagicMock()
        model.sample_rate = 24000
        model.generate.return_value = [
            self._result([0.1, 0.2]),
            self._result([0.3, 0.4]),
        ]

        generate_audio(
            text="hello",
            model=model,
            stream=True,
            save=True,
            verbose=False,
        )

        mock_audio_player.assert_called_once_with(sample_rate=24000)
        self.assertEqual(player.queue_audio.call_count, 2)
        player.wait_for_drain.assert_called_once()
        player.stop.assert_called_once()

        mock_audio_write.assert_called_once()
        args, kwargs = mock_audio_write.call_args
        self.assertEqual(args[0], "audio_000.wav")
        np.testing.assert_allclose(np.array(args[1]), np.array([0.1, 0.2, 0.3, 0.4]))
        self.assertEqual(args[2], 24000)
        self.assertEqual(kwargs["format"], "wav")

        self.assertTrue(model.generate.call_args.kwargs["stream"])

    @patch("builtins.print")
    @patch("mlx_audio.tts.generate.audio_write")
    @patch("mlx_audio.tts.generate.AudioPlayer")
    def test_stream_save_groups_chunks_by_segment(
        self, mock_audio_player, mock_audio_write, _mock_print
    ):
        player = MagicMock()
        mock_audio_player.return_value = player

        model = MagicMock()
        model.sample_rate = 24000
        model.generate.return_value = [
            self._result([0.1, 0.2]),
            self._result([0.3, 0.4]),
            self._result([0.5, 0.6], segment_idx=1),
        ]

        generate_audio(
            text="hello",
            model=model,
            stream=True,
            save=True,
            verbose=False,
        )

        self.assertEqual(mock_audio_write.call_count, 2)

        first_args, first_kwargs = mock_audio_write.call_args_list[0]
        self.assertEqual(first_args[0], "audio_000.wav")
        np.testing.assert_allclose(
            np.array(first_args[1]), np.array([0.1, 0.2, 0.3, 0.4])
        )
        self.assertEqual(first_args[2], 24000)
        self.assertEqual(first_kwargs["format"], "wav")

        second_args, second_kwargs = mock_audio_write.call_args_list[1]
        self.assertEqual(second_args[0], "audio_001.wav")
        np.testing.assert_allclose(np.array(second_args[1]), np.array([0.5, 0.6]))
        self.assertEqual(second_args[2], 24000)
        self.assertEqual(second_kwargs["format"], "wav")

    @patch("builtins.print")
    @patch("mlx_audio.tts.generate.os.makedirs")
    @patch("mlx_audio.tts.generate.audio_write")
    @patch("mlx_audio.tts.generate.AudioPlayer")
    def test_stream_save_join_audio_uses_output_path_and_prefix(
        self,
        mock_audio_player,
        mock_audio_write,
        mock_makedirs,
        _mock_print,
    ):
        player = MagicMock()
        mock_audio_player.return_value = player

        model = MagicMock()
        model.sample_rate = 24000
        model.generate.return_value = [
            self._result([0.1, 0.2]),
            self._result([0.3, 0.4]),
        ]

        generate_audio(
            text="hello",
            model=model,
            output_path="./out",
            file_prefix="watch",
            stream=True,
            save=True,
            join_audio=True,
            verbose=False,
        )

        mock_makedirs.assert_called_once_with("./out", exist_ok=True)
        mock_audio_write.assert_called_once()
        args, kwargs = mock_audio_write.call_args
        self.assertEqual(args[0], "./out/watch.wav")
        np.testing.assert_allclose(np.array(args[1]), np.array([0.1, 0.2, 0.3, 0.4]))
        self.assertEqual(args[2], 24000)
        self.assertEqual(kwargs["format"], "wav")

    @patch("builtins.print")
    @patch("mlx_audio.tts.generate.audio_write")
    @patch("mlx_audio.tts.generate.AudioPlayer")
    def test_stream_without_save_does_not_write(
        self, mock_audio_player, mock_audio_write, _mock_print
    ):
        player = MagicMock()
        mock_audio_player.return_value = player

        model = MagicMock()
        model.sample_rate = 24000
        model.generate.return_value = [self._result([0.1, 0.2])]

        generate_audio(
            text="hello",
            model=model,
            stream=True,
            save=False,
            verbose=False,
        )

        mock_audio_write.assert_not_called()
        mock_audio_player.assert_called_once_with(sample_rate=24000)
        player.queue_audio.assert_called_once()
        player.wait_for_drain.assert_called_once()
        player.stop.assert_called_once()
