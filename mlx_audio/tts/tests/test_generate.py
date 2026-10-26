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

    def test_generation_options_default_to_none(self):
        test_args = [
            "--model",
            "dummy-model",
            "--text",
            "hello",
        ]

        with patch.object(sys, "argv", ["generate.py"] + test_args):
            args = parse_args()

        for name in (
            "voice",
            "temperature",
            "max_tokens",
            "top_p",
            "top_k",
            "repetition_penalty",
        ):
            with self.subTest(option=name):
                self.assertIsNone(getattr(args, name))

    def test_explicit_generation_options(self):
        test_args = [
            "--model",
            "dummy-model",
            "--text",
            "hello",
            "--voice",
            "Vivian",
            "--temperature",
            "0",
            "--max_tokens",
            "42",
            "--top_p",
            "1",
            "--top_k",
            "0",
            "--repetition_penalty",
            "1",
        ]
        with patch.object(sys, "argv", ["generate.py"] + test_args):
            args = parse_args()

        self.assertEqual(args.voice, "Vivian")
        self.assertEqual(args.temperature, 0.0)
        self.assertEqual(args.max_tokens, 42)
        self.assertEqual(args.top_p, 1.0)
        self.assertEqual(args.top_k, 0)
        self.assertEqual(args.repetition_penalty, 1.0)

    def test_cfg_scale_defaults_to_none_for_model_defaults(self):
        test_args = [
            "--model",
            "dummy-model",
            "--text",
            "hello",
        ]

        with patch.object(sys, "argv", ["generate.py"] + test_args):
            args = parse_args()

        self.assertIsNone(args.cfg_scale)

    def test_model_specific_controls_are_optional(self):
        test_args = [
            "--model",
            "dummy-model",
            "--text",
            "hello",
            "--gen_duration",
            "3.2",
            "--steps",
            "30",
            "--stg_scale",
            "1.5",
        ]

        with patch.object(sys, "argv", ["generate.py"] + test_args):
            args = parse_args()

        self.assertEqual(args.gen_duration, 3.2)
        self.assertEqual(args.steps, 30)
        self.assertEqual(args.stg_scale, 1.5)

    def test_repeated_reference_args_are_lists(self):
        test_args = [
            "--model",
            "dummy-model",
            "--text",
            "hello",
            "--ref_audio",
            "s1.wav",
            "--ref_audio",
            "s2.wav",
            "--ref_text",
            "speaker one",
            "--ref_text",
            "speaker two",
        ]

        with patch.object(sys, "argv", ["generate.py"] + test_args):
            args = parse_args()

        self.assertEqual(args.ref_audio, ["s1.wav", "s2.wav"])
        self.assertEqual(args.ref_text, ["speaker one", "speaker two"])


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
    @patch("mlx_audio.tts.generate.audio_write")
    @patch("mlx_audio.tts.generate.AudioPlayer")
    def test_stream_save_skips_an_empty_final_chunk(
        self, mock_audio_player, mock_audio_write, _mock_print
    ):
        mock_audio_player.return_value = MagicMock()
        model = MagicMock()
        model.sample_rate = 24000
        model.generate.return_value = [
            self._result([0.1, 0.2]),
            self._result([0.3, 0.4], segment_idx=1),
            self._result([], segment_idx=2),
        ]

        generate_audio(
            text="hello",
            model=model,
            stream=True,
            save=True,
            verbose=False,
        )

        self.assertEqual(
            [args[0] for args, _ in mock_audio_write.call_args_list],
            ["audio_000.wav", "audio_001.wav"],
        )

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

    @patch("builtins.print")
    @patch("mlx_audio.tts.generate.audio_write")
    def test_generate_audio_omits_none_max_tokens(self, mock_audio_write, _mock_print):
        model = MagicMock()
        model.sample_rate = 24000
        model.generate.return_value = [self._result([0.1, 0.2])]

        generate_audio(
            text="hello",
            model=model,
            max_tokens=None,
            gen_duration=None,
            stg_scale=None,
            verbose=False,
        )

        self.assertNotIn("max_tokens", model.generate.call_args.kwargs)
        self.assertNotIn("cfg_scale", model.generate.call_args.kwargs)
        self.assertNotIn("ddpm_steps", model.generate.call_args.kwargs)
        self.assertNotIn("gen_duration", model.generate.call_args.kwargs)
        self.assertNotIn("stg_scale", model.generate.call_args.kwargs)
        mock_audio_write.assert_called_once()

    @patch("builtins.print")
    def test_generate_audio_leaves_generation_defaults_to_model(self, _mock_print):
        received = []

        class Model:
            sample_rate = 24000

            def generate(
                self,
                *,
                voice="model-voice",
                temperature=0.3,
                max_tokens=512,
                top_p=0.95,
                top_k=40,
                repetition_penalty=1.2,
                **kwargs,
            ):
                received.append(
                    (voice, temperature, max_tokens, top_p, top_k, repetition_penalty)
                )
                return iter(())

        generate_audio(text="hello", model=Model(), verbose=False)
        generate_audio(
            text="hello",
            model=Model(),
            verbose=False,
            voice=None,
            temperature=None,
            max_tokens=None,
            top_p=None,
            top_k=None,
            repetition_penalty=None,
        )
        self.assertEqual(
            received,
            [("model-voice", 0.3, 512, 0.95, 40, 1.2)] * 2,
        )

    @patch("builtins.print")
    def test_cli_forwards_only_explicit_generation_options(self, _mock_print):
        model = MagicMock(sample_rate=24000)
        model.generate.return_value = iter(())
        argv = [
            "generate.py",
            "--model",
            "dummy-model",
            "--text",
            "hello",
            "--voice",
            "Vivian",
            "--temperature",
            "0",
            "--max_tokens",
            "42",
            "--top_p",
            "1",
            "--top_k",
            "0",
            "--repetition_penalty",
            "1",
        ]
        with patch.object(sys, "argv", argv):
            options = vars(parse_args())
        options["model"] = model
        generate_audio(**options)

        forwarded = model.generate.call_args.kwargs
        self.assertEqual(forwarded["voice"], "Vivian")
        self.assertEqual(forwarded["temperature"], 0.0)
        self.assertEqual(forwarded["max_tokens"], 42)
        self.assertEqual(forwarded["top_p"], 1.0)
        self.assertEqual(forwarded["top_k"], 0)
        self.assertEqual(forwarded["repetition_penalty"], 1.0)

    @patch("builtins.print")
    @patch("mlx_audio.tts.generate.audio_write")
    @patch("mlx_audio.tts.generate.load_audio")
    @patch("mlx_audio.tts.generate.os.path.exists")
    def test_generate_audio_preserves_reference_paths_for_models_that_opt_in(
        self,
        mock_exists,
        mock_load_audio,
        mock_audio_write,
        _mock_print,
    ):
        mock_exists.return_value = True
        model = MagicMock()
        model.sample_rate = 48000
        model.preserve_ref_audio_path = True
        model.generate.return_value = [self._result([0.1, 0.2], sample_rate=48000)]

        generate_audio(
            text="hello",
            model=model,
            ref_audio="speaker.wav",
            verbose=False,
        )

        mock_exists.assert_called_once_with("speaker.wav")
        mock_load_audio.assert_not_called()
        self.assertEqual(model.generate.call_args.kwargs["ref_audio"], "speaker.wav")
        mock_audio_write.assert_called_once()

    @patch("builtins.print")
    @patch("mlx_audio.tts.generate.audio_write")
    @patch("mlx_audio.tts.generate.load_audio")
    @patch("mlx_audio.tts.generate.os.path.exists")
    def test_generate_audio_passes_multiple_references(
        self,
        mock_exists,
        mock_load_audio,
        mock_audio_write,
        _mock_print,
    ):
        mock_exists.return_value = True
        mock_load_audio.side_effect = [
            np.array([0.1, 0.2], dtype=np.float32),
            np.array([0.3, 0.4], dtype=np.float32),
        ]
        model = MagicMock()
        model.sample_rate = 24000
        model.generate.return_value = [self._result([0.1, 0.2])]

        generate_audio(
            text="[S1] hello [S2] hi",
            model=model,
            ref_audio=["s1.wav", "s2.wav"],
            ref_text=["speaker one", "speaker two"],
            verbose=False,
        )

        self.assertEqual(mock_load_audio.call_count, 2)
        call_kwargs = model.generate.call_args.kwargs
        self.assertEqual(len(call_kwargs["ref_audio"]), 2)
        self.assertEqual(call_kwargs["ref_text"], ["speaker one", "speaker two"])
        mock_audio_write.assert_called_once()

    @patch("builtins.print")
    @patch("mlx_audio.tts.generate.audio_write")
    def test_voice_is_omitted_when_none(self, mock_audio_write, _mock_print):
        model = MagicMock()
        model.sample_rate = 24000
        model.generate.return_value = [self._result([0.1, 0.2])]

        generate_audio(text="hello", model=model, voice=None, verbose=False)

        self.assertNotIn("voice", model.generate.call_args.kwargs)

    @patch("builtins.print")
    @patch("mlx_audio.tts.generate.audio_write")
    def test_voice_is_forwarded_when_set(self, mock_audio_write, _mock_print):
        model = MagicMock()
        model.sample_rate = 24000
        model.generate.return_value = [self._result([0.1, 0.2])]

        generate_audio(text="hello", model=model, voice="af_heart", verbose=False)

        self.assertEqual(model.generate.call_args.kwargs["voice"], "af_heart")
