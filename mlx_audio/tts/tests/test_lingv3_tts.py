import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_audio.stt.tests.test_ming_v3_asr import tiny_text_config
from mlx_audio.tts.models.bailingmm.bailingmm import Decoder, DiT
from mlx_audio.tts.models.lingv3_tts import Model
from mlx_audio.tts.models.lingv3_tts.conversion import copy_supporting_files
from mlx_audio.tts.models.lingv3_tts.flow import FlowLoss
from mlx_audio.tts.models.lingv3_tts.prompt import (
    build_prompt,
    control_instruction,
    podcast_transcript,
    validate_podcast,
)

TOKENS = dict(
    audio_start_token_id=21,
    audio_patch_token_id=22,
    audio_end_token_id=23,
    speaker_start_token_id=24,
    speaker_end_token_id=25,
)


class Tokenizer:
    eos_token_id = 26

    def __init__(self):
        self.pieces = []

    def encode(self, text, **kwargs):
        self.pieces.append(text)
        return [1] * len(text)


class ConstantDiT(nn.Module):
    forward_with_cfg = DiT.forward_with_cfg

    def __init__(self):
        super().__init__()
        self.x_embedder = nn.Linear(2, 2)

    def __call__(self, x, t, c, latent_history):
        return mx.broadcast_to(c.mean(axis=-1, keepdims=True), x.shape)


class TestMingTinyPrompt(unittest.TestCase):
    def test_structured_caption_order_and_eos(self):
        instruction = control_instruction("ip", ip="US_James")
        item = json.loads(instruction)["audio_sequence"][0]
        self.assertEqual(
            list(item),
            [
                "序号",
                "说话人",
                "方言",
                "情感",
                "风格",
                "语速",
                "基频",
                "音量",
                "BGM",
                "IP",
            ],
        )
        tokenizer = Tokenizer()
        ids, positions = build_prompt(
            tokenizer, TOKENS, "Hello", instruction=instruction
        )
        self.assertEqual(ids[-2:], [26, 21])
        self.assertEqual(ids[positions[0] - 1 : positions[0] + 2], [24, 22, 25])
        self.assertEqual(tokenizer.pieces[-1], instruction)
        rendered = "".join(tokenizer.pieces)
        self.assertTrue(rendered.startswith("<role>HUMAN</role>"))
        self.assertNotIn("SYSTEM", rendered)
        self.assertNotIn("<think>", rendered)

    def test_freeform_is_in_user_turn_and_has_no_caption_eos(self):
        instruction = control_instruction("freeform", instruct="Warm and calm.")
        tokenizer = Tokenizer()
        ids, _ = build_prompt(tokenizer, TOKENS, "Hello", instruction=instruction)
        rendered = "".join(tokenizer.pieces)
        self.assertLess(
            rendered.index("Speaking instruction:"), rendered.index("Text input:")
        )
        self.assertNotIn(26, ids)
        self.assertEqual(ids[-1], 21)
        self.assertEqual(
            instruction, "Speaking instruction: Speak in this style: Warm and calm."
        )

    def test_reference_and_podcast_continuation_slots(self):
        tokenizer = Tokenizer()
        ids, positions = build_prompt(
            tokenizer,
            TOKENS,
            " speaker_2:Hello\n",
            speakers=2,
            prompt_text=podcast_transcript(["One", "Two"]),
            prompt_steps=3,
            language="English",
        )
        self.assertEqual(len(positions), 2)
        self.assertEqual(ids[-4:], [21, 22, 22, 22])
        self.assertEqual(
            podcast_transcript(["One", "Two"]), " speaker_1:One\n speaker_2:Two\n"
        )
        self.assertIn("Target language: English.\n", tokenizer.pieces[0])
        validate_podcast("speaker_1:Hi\nspeaker_2:Hello", 2)
        for invalid in ("speaker_3:Hi", "Hello", "speaker_1:Hi"):
            with self.assertRaises(ValueError):
                validate_podcast(invalid, 2)

    def test_control_contract_rejects_conflicting_tasks(self):
        for task, controls in [
            ("seed", dict(emotion="happy")),
            ("emotion", dict(emotion="happy", speed="快速")),
            ("basic", dict(speed="slow", pitch="中", volume="中")),
            ("freeform", dict(instruct="calm", ip="US_James")),
        ]:
            with self.assertRaises(ValueError):
                control_instruction(task, **controls)
        value = json.loads(control_instruction("emotion", emotion="happy"))
        self.assertEqual(value["audio_sequence"][0]["情感"], "高兴")


class TestMingTinySampling(unittest.TestCase):
    def test_neutral_cli_pitch_does_not_become_a_style_control(self):
        model = Model.__new__(Model)
        nn.Module.__init__(model)
        model.tokenizer = Tokenizer()
        model.prepare_inputs = MagicMock(side_effect=RuntimeError("inputs prepared"))
        with self.assertRaisesRegex(RuntimeError, "inputs prepared"):
            next(model.generate("Hello", voice="US_James", pitch=1.0, speed=1.0))
        self.assertIsNotNone(model.prepare_inputs.call_args.kwargs["instruction"])
        with self.assertRaisesRegex(ValueError, "pitch control"):
            next(model.generate("Hello", pitch=1.2))

    def test_cfg_zero_one_and_two_follow_reference_euler(self):
        flow = FlowLoss(2, 4, dict(hidden_size=8, depth=1, num_heads=2))
        flow.cfm.model = ConstantDiT()
        for cfg, delta in ((0, 0), (1, 1), (2, 3)):
            mx.random.seed(42)
            initial = mx.random.normal((1, 2, 1)).transpose(0, 2, 1)
            mx.random.seed(42)
            # A single step excludes the terminal noise draw even at nonzero temperature.
            actual = flow.sample(
                mx.ones((1, 1, 4)),
                mx.zeros((1, 1, 2)),
                cfg=cfg,
                steps=1,
                sigma=10,
                temperature=1,
            )
            np.testing.assert_allclose(
                np.array(actual), np.array(initial + delta), atol=1e-6
            )
            self.assertEqual(actual.dtype, mx.float32)

    def test_bfloat16_vae_upsampling_matches_half_pixel_interpolation(self):
        decoder = SimpleNamespace(patch_size=3)
        x = mx.array([[[1, 3, 9]]], dtype=mx.bfloat16)
        actual = Decoder._linear_upsample_1d(decoder, x)
        position = (np.arange(9) + 0.5) / 3 - 0.5
        expected = np.interp(position, np.arange(3), [1, 3, 9])[None, None]
        np.testing.assert_allclose(np.array(actual), expected, atol=1e-6)
        singleton = Decoder._linear_upsample_1d(decoder, x[..., :1])
        np.testing.assert_array_equal(np.array(singleton), [[[1, 1, 1]]])

    def test_freeform_uses_supplied_embedding_and_rejects_extra_speakers(self):
        model = Model.__new__(Model)
        nn.Module.__init__(model)
        model.config = SimpleNamespace(llm_config=tiny_text_config(), tts_config=TOKENS)
        model.model = SimpleNamespace(
            model=SimpleNamespace(
                word_embeddings=lambda ids: mx.zeros((*ids.shape, 16))
            )
        )
        model.spk_head = nn.Linear(192, 16)
        model.tokenizer = Tokenizer()
        model._speaker_embeddings = MagicMock(return_value=mx.full((1, 16), 7.0))
        ids, embeds, latent = model.prepare_inputs(
            "Hello",
            task="freeform",
            instruction="Speaking instruction: Calm",
            spk_emb=mx.ones(192),
        )
        model._speaker_embeddings.assert_called_once()
        self.assertTrue(bool(mx.any(embeds == 7)))
        self.assertIsNone(latent)
        model._speaker_embeddings.return_value = mx.zeros((2, 16))
        with self.assertRaisesRegex(ValueError, "speaker count"):
            model.prepare_inputs(
                "Hello",
                task="freeform",
                instruction="Speaking instruction: Calm",
                spk_emb=mx.ones((2, 192)),
            )

    def test_conversion_copies_only_runtime_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            source, dest = Path(directory) / "source", Path(directory) / "output"
            source.mkdir()
            dest.mkdir()
            for name in (
                "config.json",
                "tokenizer.json",
                "campplus.onnx",
                "model.safetensors",
                "modeling.py",
                "requirements.txt",
            ):
                (source / name).write_text("fixture")
            speaker = source / "speaker_encoder/campplus.safetensors"
            speaker.parent.mkdir()
            speaker.write_text("fixture")
            copy_supporting_files(source, dest)
            self.assertEqual(
                {str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file()},
                {"tokenizer.json", "speaker_encoder/campplus.safetensors"},
            )


if __name__ == "__main__":
    unittest.main()
