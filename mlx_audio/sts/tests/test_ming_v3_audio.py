import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock

import mlx.core as mx
import numpy as np

from mlx_audio.sts.models.ming_v3_audio import Model, ModelConfig
from mlx_audio.sts.models.ming_v3_audio.conversion import copy_supporting_files
from mlx_audio.sts.models.ming_v3_audio.talker import TalkerFlowLoss
from mlx_audio.sts.utils import _resolve_model_type, infer_model_type_from_config
from mlx_audio.stt.tests.test_ming_v3_asr import tiny_text_config
from mlx_audio.tts.tests.test_lingv3_tts import ConstantDiT

PLACEHOLDER = "<|audio_start|><|audio_pad|><|audio_end|>"


def tiny_model():
    return Model(
        ModelConfig(
            text_config=asdict(tiny_text_config()),
            audio_config=dict(
                n_mels=4,
                n_audio_ctx=16,
                n_audio_state=8,
                n_audio_head=2,
                n_audio_layer=1,
                whisper_chunk_size=16,
            ),
            audio_token_id=5,
        )
    )


class TestMingAudioMessages(unittest.TestCase):
    def test_multiple_clips_with_single_token_first_clip(self):
        model = tiny_model()
        model.tokenizer = MagicMock()
        model.tokenizer.apply_chat_template.return_value = (
            "first:" + PLACEHOLDER + "\nsecond:" + PLACEHOLDER
        )
        rendered = []

        def encode(text, **kwargs):
            rendered.append(text)
            return [5] * text.count("<|audio_pad|>")

        model.tokenizer.encode.side_effect = encode
        model.prepare_audio = MagicMock(return_value=mx.zeros((4, 4)))
        model.encode_audio = MagicMock(
            side_effect=[mx.full((1, 1, 16), 3.0), mx.full((1, 2, 16), 7.0)]
        )
        ids, embeds = model.prepare_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": np.zeros(400)},
                        {"type": "audio", "audio": np.zeros(800)},
                    ],
                }
            ]
        )
        self.assertEqual(
            rendered[0],
            "first:"
            + PLACEHOLDER
            + "\nsecond:<|audio_start|><|audio_pad|><|audio_pad|><|audio_end|>",
        )
        self.assertEqual(ids, [5, 5, 5])
        np.testing.assert_array_equal(np.array(embeds[0, :, 0]), [3, 7, 7])

    def test_invalid_roles_modalities_and_placeholder_counts(self):
        model = tiny_model()
        for messages in (
            [{"role": "assistant", "content": "Hi"}],
            [{"role": "developer", "content": "Hi"}, {"role": "user", "content": "Hi"}],
            [{"role": "user", "content": [{"type": "image", "image": "x"}]}],
        ):
            with self.assertRaises(ValueError):
                model.prepare_messages(messages)
        model.tokenizer = MagicMock()
        model.tokenizer.apply_chat_template.return_value = PLACEHOLDER
        with self.assertRaisesRegex(ValueError, "placeholders"):
            model.prepare_messages([{"role": "user", "content": "Hi"}])

    def test_thinking_does_not_speak_unfinished_reasoning(self):
        model = tiny_model()
        model.tokenizer = MagicMock()
        model.tokenizer.decode.return_value = "Let me think about this"
        model.prepare_messages = MagicMock(return_value=([1, 2], mx.zeros((1, 2, 16))))
        model.decode_tokens = MagicMock(return_value=([3, 4], 0.0, 0.0))
        model._get_talker = MagicMock()
        result = model.generate(
            text="Hello", task="spoken_dialogue", thinking=True, max_new_tokens=2
        )
        self.assertEqual(result.text, "")
        self.assertEqual(result.reasoning, "Let me think about this")
        self.assertEqual(result.finish_reason, "length")
        self.assertIsNone(result.audio)
        model._get_talker.assert_not_called()
        model.tokenizer.decode.return_value = "Consider this. </think> Hello!"
        model._get_talker.return_value.generate.return_value = iter(
            [MagicMock(audio=mx.zeros(100), sample_rate=44100)]
        )
        result = model.generate(text="Hello", task="spoken_dialogue", thinking=True)
        self.assertEqual(result.text, "Hello!")
        self.assertEqual(result.reasoning, "Consider this.")
        self.assertEqual(result.audio.shape, (100,))

    def test_tts_bypasses_thinker_and_text_does_not_load_talker(self):
        model = tiny_model()
        model.tokenizer = MagicMock()
        model.prepare_messages = MagicMock(return_value=([1], mx.zeros((1, 1, 16))))
        model.decode_tokens = MagicMock(return_value=([3], 0.0, 0.0))
        model.tokenizer.decode.return_value = "Hi"
        model._get_talker = MagicMock()
        model.generate(text="Hello", task="text")
        model._get_talker.assert_not_called()
        model.decode_tokens.reset_mock()
        model._get_talker.return_value.generate.return_value = iter(
            [MagicMock(audio=mx.zeros(100), sample_rate=44100)]
        )
        result = model.generate(text="Read this.", task="tts")
        model.decode_tokens.assert_not_called()
        self.assertEqual(result.text, "Read this.")
        self.assertEqual(result.prompt_tokens, 0)


class TestTalkerSampling(unittest.TestCase):
    def test_guidance_zero_and_one_keep_both_branches_and_native_dtype(self):
        flow = TalkerFlowLoss(2, 4, dict(hidden_size=8, depth=1, num_heads=2))
        flow.cfm.model = ConstantDiT()
        hidden = mx.ones((1, 1, 4), mx.bfloat16)
        history = mx.zeros((1, 1, 2), mx.bfloat16)
        for cfg in (0, 1, 2):
            mx.random.seed(42)
            initial = mx.random.normal((1, 1, 2), dtype=mx.bfloat16)
            mx.random.seed(42)
            actual = flow.sample(hidden, history, cfg=cfg, steps=1, sigma=0)
            expected = initial + (1 + cfg)
            np.testing.assert_allclose(
                np.array(actual.astype(mx.float32)),
                np.array(expected.astype(mx.float32)),
                atol=0.02,
            )
            self.assertEqual(actual.dtype, mx.bfloat16)

    def test_talker_includes_terminal_sde_noise(self):
        flow = TalkerFlowLoss(2, 4, dict(hidden_size=8, depth=1, num_heads=2))
        flow.cfm.model = ConstantDiT()
        mx.random.seed(19)
        initial = mx.random.normal((1, 1, 2))
        terminal_noise = mx.random.normal((1, 1, 2))
        mx.random.seed(19)
        actual = flow.sample(
            mx.ones((1, 1, 4)),
            mx.zeros((1, 1, 2)),
            cfg=0,
            steps=1,
            sigma=1,
            temperature=1,
        )
        np.testing.assert_allclose(
            np.array(actual), np.array(initial + 1 + terminal_noise), atol=1e-6
        )


class TestMingAudioIntegration(unittest.TestCase):
    def test_converter_resolves_native_metadata_without_path_hints(self):
        from mlx_audio.convert import Domain, detect_model_domain, get_model_type

        for family, domain, canonical in (
            ("bailing_moe_v3_audio", Domain.STS, "ming_v3_audio"),
            ("ming_v3_asr", Domain.STT, "ming_v3_asr"),
            ("lingv3_tts", Domain.TTS, "lingv3_tts"),
        ):
            config = {"model_type": family}
            path = Path("anonymous-checkpoint")
            self.assertEqual(detect_model_domain(config, path), domain)
            self.assertEqual(get_model_type(config, path, domain), canonical)

    def test_loader_detects_native_and_converted_metadata(self):
        for family in ("bailing_moe_v3_audio", "ming_v3_audio"):
            self.assertEqual(
                infer_model_type_from_config({"model_type": family}), "ming_v3_audio"
            )
            with tempfile.TemporaryDirectory() as directory:
                (Path(directory) / "config.json").write_text(
                    json.dumps({"model_type": family})
                )
                self.assertEqual(_resolve_model_type(directory), "ming_v3_audio")

    def test_conversion_includes_talker_and_excludes_remote_code(self):
        with tempfile.TemporaryDirectory() as directory:
            source, destination = Path(directory) / "source", Path(directory) / "out"
            source.mkdir()
            destination.mkdir()
            for name in (
                "tokenizer.json",
                "model.safetensors",
                "modeling.py",
                "talker/config.json",
                "talker/inference.py",
                "talker/model.safetensors",
                "talker/campplus.onnx",
                "talker/speaker_encoder/campplus.safetensors",
                "talker/llm/tokenizer.json",
                "talker/vae/config.json",
                "talker/vae/model.safetensors",
            ):
                path = source / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture")
            copy_supporting_files(source, destination)
            copied = {
                str(p.relative_to(destination))
                for p in destination.rglob("*")
                if p.is_file()
            }
            self.assertIn("talker/vae/model.safetensors", copied)
            self.assertIn("talker/llm/tokenizer.json", copied)
            self.assertIn("talker/speaker_encoder/campplus.safetensors", copied)
            self.assertNotIn("talker/campplus.onnx", copied)
            self.assertNotIn("model.safetensors", copied)
            self.assertNotIn("modeling.py", copied)
            self.assertNotIn("talker/inference.py", copied)


if __name__ == "__main__":
    unittest.main()
