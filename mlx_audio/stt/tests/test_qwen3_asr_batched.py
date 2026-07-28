import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_audio.stt.models.qwen3_asr.qwen3_asr import Qwen3ASRModel, _rope_safe


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


if __name__ == "__main__":
    unittest.main()
