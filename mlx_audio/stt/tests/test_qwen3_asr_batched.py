import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_audio.stt.models.qwen3_asr.qwen3_asr import (
    Qwen3ASRModel,
    _rope_safe,
)


class _FakeTokenizer:
    eos_token_id = 99
    eos_token_ids = [99]
    unk_token_id = -1

    def convert_tokens_to_ids(self, token):
        return {"<|im_end|>": 98, "<|endoftext|>": 99}.get(token, self.unk_token_id)

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(str(token_id) for token_id in token_ids)


class _FakeEmbeddings:
    def __call__(self, input_ids):
        return mx.zeros((*input_ids.shape, 1))

    def as_linear(self, hidden_states):
        return mx.zeros((*hidden_states.shape[:2], 128))


class _FakeTextModel:
    def __init__(self):
        self.embed_tokens = _FakeEmbeddings()

    def __call__(self, *, inputs_embeds, cache=None):
        return inputs_embeds


def _make_minimal_model():
    model = Qwen3ASRModel.__new__(Qwen3ASRModel)
    model.config = SimpleNamespace(
        text_config=SimpleNamespace(num_hidden_layers=0),
    )
    model._tokenizer = _FakeTokenizer()
    model._feature_extractor = object()
    model.model = _FakeTextModel()
    model.lm_head = None
    model.get_audio_features = Mock(return_value=mx.zeros((1, 1)))
    model._preprocess_audio = Mock(return_value=(mx.zeros((1, 1)), None, 1))
    model._build_prompt = Mock(return_value=mx.array([[0]]))
    model._build_inputs_embeds = Mock(return_value=mx.zeros((1, 1, 1)))
    model._forward_with_embeds = Mock(return_value=mx.zeros((2, 1, 128)))
    return model


class TestRopeSafe(unittest.TestCase):
    """Regression test for the mx.fast.rope batched single-token bug.

    nn.RoPE on a (B, heads, 1, dim) tensor with B > 1 corrupts every row but
    the first, which silently breaks batched single-token decode. _rope_safe
    must return identical outputs for identical batch rows and match the
    single-row reference exactly.
    """

    def test_batched_single_token_matches_single_row(self):
        rope = nn.RoPE(128, traditional=False, base=1_000_000.0)
        row = mx.random.normal((1, 16, 1, 128))
        batched = mx.concatenate([row, row], axis=0)  # two identical rows

        ref = rope(row, offset=300)
        out = _rope_safe(rope, batched, 300)

        self.assertTrue(mx.allclose(out[0], out[1], rtol=0, atol=1e-6).item())
        self.assertTrue(mx.allclose(out[0], ref[0], rtol=0, atol=1e-6).item())

    def test_multi_token_unchanged(self):
        rope = nn.RoPE(128, traditional=False, base=1_000_000.0)
        x = mx.random.normal((2, 16, 4, 128))
        self.assertTrue(
            mx.allclose(_rope_safe(rope, x, 300), rope(x, offset=300)).item()
        )


class TestBatchedGeneration(unittest.TestCase):
    def make_minimal_model(self):
        return _make_minimal_model()

    def test_batched_generation_respects_global_token_budget(self):
        model = self.make_minimal_model()
        sampler_outputs = iter(
            [
                mx.array([10, 20]),
                mx.array([11, 21]),
                mx.array([12, 22]),
            ]
        )

        texts, gen_tokens, prompt_tokens, processed = model._generate_chunks_batched(
            [(np.zeros(4), 0.0), (np.zeros(4), 1.0)],
            max_tokens=3,
            sampler=lambda logits: next(sampler_outputs),
            logits_processors=None,
            language="en",
            system_prompt=None,
            batch_size=2,
            verbose=False,
        )

        self.assertEqual(sum(gen_tokens), 3)
        self.assertEqual(texts, ["10 11", "20"])
        self.assertEqual(prompt_tokens, [1, 1])
        self.assertEqual(processed, [True, True])

    def test_batched_generation_applies_logits_processors_per_row(self):
        model = self.make_minimal_model()
        sampler_outputs = iter(
            [
                mx.array([10, 20]),
                mx.array([11, 21]),
                mx.array([12, 22]),
            ]
        )
        histories = []
        processed_logits = []

        def record_history(tokens, logits):
            histories.append(tokens.tolist())
            return logits + len(histories)

        def sample(logits):
            processed_logits.append(logits[:, 0].tolist())
            return next(sampler_outputs)

        model._generate_chunks_batched(
            [(np.zeros(4), 0.0), (np.zeros(4), 1.0)],
            max_tokens=3,
            sampler=sample,
            logits_processors=[record_history],
            language="en",
            system_prompt=None,
            batch_size=2,
            verbose=False,
        )

        self.assertEqual(
            histories,
            [[0], [0], [0, 10], [0, 20], [0, 10, 11], [0, 20, 21]],
        )
        self.assertEqual(processed_logits, [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    def test_generate_rejects_invalid_batch_size(self):
        model = self.make_minimal_model()

        with self.assertRaisesRegex(ValueError, "batch_size"):
            model.generate(np.zeros(16000), batch_size=0)

    def test_generate_uses_batched_path_for_multiple_chunks(self):
        model = self.make_minimal_model()
        model._generate_chunks_batched = Mock(
            return_value=(["first", "second"], [1, 1], [5, 5], [True, True])
        )

        out = model.generate(
            np.zeros(32000),
            batch_size=2,
            max_tokens=2,
            chunk_duration=1,
            min_chunk_duration=1,
            language="en",
        )

        self.assertEqual(out.text, "first second")
        self.assertEqual(out.generation_tokens, 2)
        self.assertEqual(len(out.segments), 2)
        model._generate_chunks_batched.assert_called_once()
        self.assertIsNone(
            model._generate_chunks_batched.call_args.kwargs["logits_processors"]
        )

    def test_short_audio_segment_uses_unpadded_duration(self):
        model = self.make_minimal_model()
        model._generate_single_chunk = Mock(return_value=("text", 5, 1))

        out = model.generate(
            np.zeros(4000, dtype=np.float32),
            min_chunk_duration=1.0,
            language="English",
        )

        self.assertEqual(len(out.segments), 1)
        self.assertEqual(out.segments[0]["start"], 0.0)
        self.assertEqual(out.segments[0]["end"], 0.25)


class TestAudioInputValidation(unittest.TestCase):
    def test_empty_audio_is_rejected_before_padding(self):
        model = _make_minimal_model()
        model._generate_single_chunk = Mock()

        with self.assertRaisesRegex(ValueError, "at least one sample"):
            model.generate(np.array([], dtype=np.float32))

        model._generate_single_chunk.assert_not_called()

    def test_non_finite_audio_is_rejected(self):
        model = _make_minimal_model()
        model._generate_single_chunk = Mock()

        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "finite"):
                    model.generate(np.array([0.0, value], dtype=np.float32))

        model._generate_single_chunk.assert_not_called()

    def test_direct_preprocessing_rejects_empty_audio(self):
        model = Qwen3ASRModel.__new__(Qwen3ASRModel)
        model._feature_extractor = Mock()

        with self.assertRaisesRegex(ValueError, "at least one sample"):
            model._preprocess_audio(np.array([], dtype=np.float32))

        model._feature_extractor.assert_not_called()


class TestLanguageParsing(unittest.TestCase):
    def setUp(self):
        self.model = Qwen3ASRModel.__new__(Qwen3ASRModel)

    def test_language_none_with_empty_text_is_no_speech(self):
        self.assertEqual(
            self.model.extract_language("language None<asr_text>"),
            ("", ""),
        )

    def test_language_none_preserves_returned_text(self):
        self.assertEqual(
            self.model.extract_language("language None<asr_text>quiet speech"),
            ("", "quiet speech"),
        )

    def test_named_language_is_unchanged(self):
        self.assertEqual(
            self.model.extract_language("language English<asr_text>Hello"),
            ("English", "Hello"),
        )

    def test_language_marker_tolerates_case_and_whitespace(self):
        variants = (
            " language None<asr_text> ",
            "\nlanguage None<asr_text>",
            "Language None<asr_text>",
            "language None\n<asr_text>",
        )

        for text in variants:
            with self.subTest(text=text):
                self.assertEqual(self.model.extract_language(text), ("", ""))


class TestPerChunkLanguageParsing(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            (np.zeros(16000, dtype=np.float32), 0.0),
            (np.zeros(16000, dtype=np.float32), 1.0),
        ]

    def test_sequential_chunks_keep_auto_detection_enabled(self):
        model = _make_minimal_model()
        model._generate_single_chunk = Mock(
            side_effect=[
                ("language None<asr_text>", 1, 1),
                ("language English<asr_text>Hello", 1, 1),
            ]
        )

        with patch(
            "mlx_audio.stt.models.qwen3_asr.qwen3_asr.split_audio_into_chunks",
            return_value=self.chunks,
        ):
            out = model.generate(np.zeros(32000, dtype=np.float32))

        self.assertEqual(out.text, "Hello")
        self.assertEqual([segment["text"] for segment in out.segments], ["", "Hello"])
        self.assertEqual(
            [segment["language"] for segment in out.segments], ["", "English"]
        )
        self.assertEqual(
            [
                call.kwargs["language"]
                for call in model._generate_single_chunk.call_args_list
            ],
            [None, None],
        )

    def test_batched_chunks_parse_each_detected_language(self):
        model = _make_minimal_model()
        model._generate_chunks_batched = Mock(
            return_value=(
                ["language None<asr_text>", "language English<asr_text>Hello"],
                [1, 1],
                [1, 1],
                [True, True],
            )
        )

        with patch(
            "mlx_audio.stt.models.qwen3_asr.qwen3_asr.split_audio_into_chunks",
            return_value=self.chunks,
        ):
            out = model.generate(
                np.zeros(32000, dtype=np.float32),
                batch_size=2,
            )

        self.assertEqual(out.text, "Hello")
        self.assertEqual([segment["text"] for segment in out.segments], ["", "Hello"])
        self.assertEqual(
            [segment["language"] for segment in out.segments], ["", "English"]
        )
        self.assertIsNone(model._generate_chunks_batched.call_args.kwargs["language"])

    def test_streaming_chunks_reset_language_detection(self):
        model = _make_minimal_model()
        token_text = {
            1: "language ",
            2: "None",
            3: "<asr_text>",
            4: "English",
            5: "Hello",
        }
        model._tokenizer.decode = lambda token_ids: token_text[token_ids[0]]
        token_state = mx.zeros((1,))
        model.stream_generate = Mock(
            side_effect=[
                iter(
                    [
                        (mx.array(1), token_state),
                        (mx.array(2), token_state),
                        (mx.array(3), token_state),
                    ]
                ),
                iter(
                    [
                        (mx.array(1), token_state),
                        (mx.array(4), token_state),
                        (mx.array(3), token_state),
                        (mx.array(5), token_state),
                    ]
                ),
            ]
        )

        with patch(
            "mlx_audio.stt.models.qwen3_asr.qwen3_asr.split_audio_into_chunks",
            return_value=self.chunks,
        ):
            results = list(model.stream_transcribe(np.zeros(32000, dtype=np.float32)))

        self.assertEqual([result.text for result in results if result.text], ["Hello"])
        boundaries = [result for result in results if not result.text]
        self.assertEqual([result.language for result in boundaries], ["", "English"])
        self.assertEqual(
            [call.kwargs["language"] for call in model.stream_generate.call_args_list],
            [None, None],
        )


if __name__ == "__main__":
    unittest.main()
