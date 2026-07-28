import unittest

import mlx.core as mx
import numpy as np

from mlx_audio.tts.continuous import TTSBatchItem, TTSBatchOptions
from mlx_audio.tts.models.higgs_audio_v3.config import (
    HiggsAudioV3Config,
    HiggsAudioV3TextConfig,
)
from mlx_audio.tts.models.higgs_audio_v3.generation import (
    HiggsSamplerState,
    apply_delay_pattern,
    reverse_delay_pattern,
    sample_batch,
    step,
)
from mlx_audio.tts.models.higgs_audio_v3.model import Model
from mlx_audio.tts.models.higgs_audio_v3.prompt import (
    AUDIO_PLACEHOLDER_ID,
    HiggsAudioV3PromptBuilder,
    ReferenceCodes,
)
from mlx_audio.tts.utils import get_model_and_args
from mlx_audio.utils import get_model_name_parts


def _tiny_config() -> HiggsAudioV3Config:
    return HiggsAudioV3Config(
        text_config=HiggsAudioV3TextConfig(
            hidden_size=32,
            num_hidden_layers=2,
            intermediate_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
            rope_theta=10000.0,
            head_dim=8,
            rms_norm_eps=1e-6,
            vocab_size=64,
            tie_word_embeddings=True,
        ),
        audio_num_codebooks=4,
        audio_codebook_size=10,
        audio_boc_token_id=8,
        audio_eoc_token_id=9,
    )


class FakeTokenizer:
    def __init__(self):
        self.vocab = {
            "<|tts|>": 1,
            "<|ref_audio|>": 2,
            "<|text|>": 3,
            "<|audio|>": 4,
            "<|ref_text|>": 5,
        }

    def get_added_vocab(self):
        return self.vocab

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [10 + (ord(ch) % 20) for ch in text]


class FakeCodec:
    def __init__(self):
        self.encode_calls = 0

    def encode(self, waveform):
        self.encode_calls += 1
        del waveform
        codes = mx.array(
            [[[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]],
            dtype=mx.int32,
        )
        return codes

    def decode(self, codes):
        length = int(codes.shape[0]) * 10
        return mx.ones((length,), dtype=mx.float32)


class TestHiggsAudioV3Config(unittest.TestCase):
    def test_parses_source_config_shape(self):
        cfg = HiggsAudioV3Config.from_dict(
            {
                "model_type": "higgs_multimodal_qwen3",
                "audio_token_id": -100,
                "audio_encoder_config": {
                    "num_codebooks": 8,
                    "vocab_size": 1026,
                    "use_delay_pattern": True,
                },
                "text_config": {
                    "model_type": "qwen3",
                    "hidden_size": 2560,
                    "num_hidden_layers": 36,
                    "intermediate_size": 9728,
                    "num_attention_heads": 32,
                    "num_key_value_heads": 8,
                    "max_position_embeddings": 32768,
                    "head_dim": 128,
                    "rms_norm_eps": 1e-6,
                    "vocab_size": 151936,
                    "tie_word_embeddings": True,
                    "rope_parameters": {"rope_theta": 1000000},
                },
            }
        )
        self.assertEqual(cfg.model_type, "higgs_multimodal_qwen3")
        self.assertEqual(cfg.text_config.hidden_size, 2560)
        self.assertEqual(cfg.text_config.rope_theta, 1000000)
        self.assertEqual(cfg.audio_num_codebooks, 8)
        self.assertEqual(cfg.audio_codebook_size, 1026)
        self.assertEqual(cfg.audio_boc_token_id, 1024)
        self.assertEqual(cfg.audio_eoc_token_id, 1025)

    def test_loader_remapping(self):
        model_name = get_model_name_parts("bosonai/higgs-audio-v3-tts-4b")
        arch, model_type = get_model_and_args(
            "higgs_multimodal_qwen3",
            model_name,
        )
        self.assertEqual(model_type, "higgs_audio_v3")
        self.assertTrue(hasattr(arch, "Model"))


class TestHiggsAudioV3DelayPattern(unittest.TestCase):
    def test_delay_pattern_round_trip(self):
        codes = mx.arange(1, 13, dtype=mx.int32).reshape(3, 4)
        delayed = apply_delay_pattern(codes, boc_id=98, eoc_id=99)
        self.assertEqual(delayed.shape, (6, 4))
        np.testing.assert_array_equal(np.array(reverse_delay_pattern(delayed)), codes)

    def test_delay_pattern_padding(self):
        codes = mx.array([[1, 2, 3]], dtype=mx.int32)
        delayed = apply_delay_pattern(codes, boc_id=8, eoc_id=9)
        np.testing.assert_array_equal(
            np.array(delayed),
            np.array([[1, 8, 8], [9, 2, 8], [9, 9, 3]], dtype=np.int32),
        )


class TestHiggsAudioV3Sampler(unittest.TestCase):
    def test_batch_sampler_greedy_shape(self):
        logits = mx.zeros((2, 4, 10))
        logits[:, :, 3] = 1.0
        out = sample_batch(logits, temperature=0.0, top_p=None, top_k=None)
        self.assertEqual(out.shape, (2, 4))
        np.testing.assert_array_equal(np.array(out), np.full((2, 4), 3))

    def test_ramp_in_forces_later_codebooks_to_boc(self):
        state = HiggsSamplerState(num_codebooks=4)
        logits = mx.zeros((4, 10))
        logits[:, 1] = 10.0
        out = step(
            logits,
            state,
            temperature=0.0,
            top_p=None,
            top_k=None,
            boc_id=8,
            eoc_id=9,
        )
        np.testing.assert_array_equal(np.array(out), [1, 8, 8, 8])
        self.assertEqual(state.delay_count, 1)

    def test_eoc_wind_down(self):
        state = HiggsSamplerState(num_codebooks=4, delay_count=4)
        logits = mx.zeros((4, 10))
        logits[:, 1] = 10.0
        logits[0, 9] = 20.0
        step(logits, state, temperature=0.0, top_p=None, top_k=None, boc_id=8, eoc_id=9)
        self.assertEqual(state.eoc_countdown, 2)
        self.assertFalse(state.generation_done)

        logits[0, 1] = 30.0
        step(logits, state, temperature=0.0, top_p=None, top_k=None, boc_id=8, eoc_id=9)
        self.assertEqual(state.eoc_countdown, 1)
        self.assertFalse(state.generation_done)
        step(logits, state, temperature=0.0, top_p=None, top_k=None, boc_id=8, eoc_id=9)
        self.assertTrue(state.generation_done)


class TestHiggsAudioV3Prompt(unittest.TestCase):
    def test_prompt_placeholders_match_delayed_rows(self):
        builder = HiggsAudioV3PromptBuilder(FakeTokenizer())
        delayed = mx.zeros((5, 4), dtype=mx.int32)
        prompt = builder.build_prompt(
            "target",
            references=[ReferenceCodes(codes=delayed, text="ref")],
        )
        self.assertIn(AUDIO_PLACEHOLDER_ID, prompt.token_ids)
        self.assertEqual(prompt.token_ids.count(AUDIO_PLACEHOLDER_ID), 5)
        self.assertEqual(len(prompt.audio_segments), 1)
        self.assertEqual(prompt.audio_segments[0][0], prompt.token_ids.index(-100))


class TestHiggsAudioV3Model(unittest.TestCase):
    def test_sanitize_maps_source_keys(self):
        model = Model(_tiny_config())
        weights = {
            "tied.embedding.text_embedding.weight": mx.zeros((64, 32)),
            "body.layers.0.input_layernorm.weight": mx.ones((32,)),
            "body.norm.weight": mx.ones((32,)),
            "tied.embedding.modality_embeddings.0.embedding.weight": mx.zeros((40, 32)),
            "tied.embedding.modality_embeddings.0.model.fc.weight": mx.zeros((1, 1)),
        }
        sanitized = model.sanitize(weights)
        self.assertIn("backbone.embed_tokens.weight", sanitized)
        self.assertIn("backbone.layers.0.input_layernorm.weight", sanitized)
        self.assertIn("backbone.norm.weight", sanitized)
        self.assertIn("multimodal_embedding.weight", sanitized)
        self.assertNotIn(
            "tied.embedding.modality_embeddings.0.model.fc.weight", sanitized
        )

    def test_fused_embedding_and_head_shapes(self):
        cfg = _tiny_config()
        model = Model(cfg)
        weight = mx.arange(40 * 32, dtype=mx.float32).reshape(40, 32)
        model.multimodal_embedding.weight = weight
        codes = mx.zeros((3, 4), dtype=mx.int32)
        out = model._embed_audio_codes(codes)
        self.assertEqual(out.shape, (3, 32))
        expected = weight[0] + weight[10] + weight[20] + weight[30]
        np.testing.assert_allclose(np.array(out[0]), np.array(expected))

        logits = model._audio_logits(mx.zeros((1, 32)))
        self.assertEqual(logits.shape, (1, 4, 10))

    def test_tiny_forward_shape(self):
        cfg = _tiny_config()
        model = Model(cfg)
        mx.eval(model.parameters())
        hidden = model(mx.zeros((1, 3), dtype=mx.int32))
        self.assertEqual(hidden.shape, (1, 3, cfg.text_config.hidden_size))

    def test_encode_reference_audio_returns_delayed_codes(self):
        model = Model(_tiny_config())
        codec = FakeCodec()
        model._codec = codec

        audio = np.linspace(-0.1, 0.1, 100, dtype=np.float32)
        encoded = model.encode_reference_audio(audio)

        self.assertEqual(codec.encode_calls, 1)
        raw_codes = mx.array(
            [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]],
            dtype=mx.int32,
        )
        expected = apply_delay_pattern(
            raw_codes,
            boc_id=model.config.audio_boc_token_id,
            eoc_id=model.config.audio_eoc_token_id,
        )
        np.testing.assert_array_equal(np.array(encoded), np.array(expected))

    def test_ref_audio_codes_skip_reference_encoding(self):
        model = Model(_tiny_config())
        codec = FakeCodec()
        model._codec = codec

        encoded = mx.ones((4, 4), dtype=mx.int32)
        refs = model._normalize_references(
            ref_audio_codes=encoded,
            ref_text="Reference transcript.",
        )

        self.assertEqual(codec.encode_calls, 0)
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0].text, "Reference transcript.")
        np.testing.assert_array_equal(np.array(refs[0].codes), np.ones((4, 4)))

    def test_references_dict_accepts_preencoded_codes(self):
        model = Model(_tiny_config())
        codec = FakeCodec()
        model._codec = codec

        encoded = np.ones((4, 4), dtype=np.int32)
        refs = model._normalize_references(
            references=[{"codes": encoded, "text": "Reference transcript."}]
        )

        self.assertEqual(codec.encode_calls, 0)
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0].text, "Reference transcript.")
        np.testing.assert_array_equal(np.array(refs[0].codes), encoded)

    def test_ref_audio_codes_list_matches_ref_texts(self):
        model = Model(_tiny_config())
        refs = model._normalize_references(
            ref_audio_codes_list=[
                mx.ones((4, 4), dtype=mx.int32),
                mx.zeros((5, 4), dtype=mx.int32),
            ],
            ref_texts=["first", "second"],
        )

        self.assertEqual(len(refs), 2)
        self.assertEqual([ref.text for ref in refs], ["first", "second"])

    def test_ref_audio_codes_accepts_python_2d_list_as_single_reference(self):
        model = Model(_tiny_config())

        refs = model._normalize_references(
            ref_audio_codes=[[1, 2, 3, 4], [2, 3, 4, 5]],
            ref_text="single",
        )

        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0].text, "single")
        self.assertEqual(refs[0].codes.shape, (2, 4))

    def test_ref_audio_and_ref_audio_codes_are_mutually_exclusive(self):
        model = Model(_tiny_config())

        with self.assertRaisesRegex(ValueError, "either ref_audio or ref_audio_codes"):
            model._normalize_references(
                ref_audio=np.zeros(100, dtype=np.float32),
                ref_audio_codes=mx.ones((4, 4), dtype=mx.int32),
            )

    def test_ref_audio_codes_validate_shape(self):
        model = Model(_tiny_config())

        with self.assertRaisesRegex(ValueError, "must have 4 codebooks"):
            model._normalize_references(
                ref_audio_codes=mx.ones((4, 3), dtype=mx.int32),
            )

    def test_supports_tts_batch_for_reference_cloning(self):
        model = Model(_tiny_config())

        self.assertTrue(
            model.supports_tts_batch(
                ref_audio=np.zeros(100, dtype=np.float32),
                ref_text="Reference transcript.",
            )
        )
        self.assertFalse(model.supports_tts_batch(stream=True))
        self.assertFalse(model.supports_tts_batch(voice="speaker"))
        self.assertFalse(model.supports_tts_batch(speed=1.2))
        self.assertTrue(
            model.supports_tts_continuous_batch(
                ref_audio=np.zeros(100, dtype=np.float32),
                ref_text="Reference transcript.",
            )
        )
        self.assertFalse(model.supports_tts_continuous_batch(stream=True))

    def test_batch_generate_reuses_shared_reference_audio(self):
        model = Model(_tiny_config())
        model._prompt_builder = HiggsAudioV3PromptBuilder(FakeTokenizer())
        codec = FakeCodec()
        model._codec = codec
        mx.eval(model.parameters())

        results = list(
            model.batch_generate(
                ["first", "second"],
                ref_audio=np.zeros(100, dtype=np.float32),
                ref_text="Reference transcript.",
                max_new_tokens=5,
                temperature=0.0,
                top_k=1,
                fade_in_ms=0.0,
                fade_out_ms=0.0,
            )
        )

        self.assertEqual(codec.encode_calls, 1)
        self.assertEqual([result.sequence_idx for result in results], [0, 1])
        self.assertEqual(len(results), 2)
        self.assertTrue(
            all(result.sample_rate == model.sample_rate for result in results)
        )
        self.assertTrue(all(result.token_count <= 5 for result in results))

    def test_batch_generate_accepts_preencoded_reference_codes(self):
        model = Model(_tiny_config())
        model._prompt_builder = HiggsAudioV3PromptBuilder(FakeTokenizer())
        codec = FakeCodec()
        model._codec = codec
        mx.eval(model.parameters())

        ref_codes = mx.ones((4, 4), dtype=mx.int32)
        results = list(
            model.batch_generate(
                ["first", "second"],
                ref_audio_codes=ref_codes,
                ref_text="Reference transcript.",
                max_new_tokens=5,
                temperature=0.0,
                top_k=1,
                fade_in_ms=0.0,
                fade_out_ms=0.0,
            )
        )

        self.assertEqual(codec.encode_calls, 0)
        self.assertEqual([result.sequence_idx for result in results], [0, 1])

    def test_batch_generate_matches_serial_for_different_prompt_lengths(self):
        model = Model(_tiny_config())
        model._prompt_builder = HiggsAudioV3PromptBuilder(FakeTokenizer())
        model._codec = FakeCodec()
        mx.eval(model.parameters())

        texts = [
            "short",
            "this is a much longer text input for the second batch item",
        ]
        ref_codes = mx.ones((4, 4), dtype=mx.int32)
        common_kwargs = {
            "ref_audio_codes": ref_codes,
            "ref_text": "Reference transcript.",
            "max_new_tokens": 5,
            "temperature": 0.0,
            "top_k": 1,
            "fade_in_ms": 0.0,
            "fade_out_ms": 0.0,
        }

        serial = [next(model.generate(text, **common_kwargs)) for text in texts]
        batch = list(model.batch_generate(texts, **common_kwargs))

        self.assertEqual([result.sequence_idx for result in batch], [0, 1])
        self.assertEqual(
            [result.token_count for result in batch],
            [result.token_count for result in serial],
        )
        self.assertEqual(
            [result.samples for result in batch],
            [result.samples for result in serial],
        )

    def test_batch_generate_rejects_streaming(self):
        model = Model(_tiny_config())

        with self.assertRaisesRegex(NotImplementedError, "batch streaming"):
            list(model.batch_generate(["text"], stream=True))

    def test_continuous_batch_session_matches_serial(self):
        model = Model(_tiny_config())
        model._prompt_builder = HiggsAudioV3PromptBuilder(FakeTokenizer())
        model._codec = FakeCodec()
        mx.eval(model.parameters())

        texts = [
            "short",
            "this is a much longer text input for the second batch item",
        ]
        ref_audio = np.zeros(100, dtype=np.float32)
        ref_text = "Reference transcript."
        common_kwargs = {
            "ref_audio": ref_audio,
            "ref_text": ref_text,
            "max_new_tokens": 5,
            "temperature": 0.0,
            "top_k": 1,
            "fade_in_ms": 30.0,
            "fade_out_ms": 15.0,
        }
        serial = [next(model.generate(text, **common_kwargs)) for text in texts]

        session = model.create_tts_batch_session(
            TTSBatchOptions(
                temperature=0.0,
                top_p=None,
                top_k=1,
                max_tokens=5,
                max_batch_size=2,
            )
        )
        self.assertEqual(session.available_slots, 2)
        session.add(
            [
                TTSBatchItem(
                    sequence_id=index,
                    text=text,
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                )
                for index, text in enumerate(texts)
            ]
        )

        events = []
        while not session.idle:
            events.extend(session.step())

        events = sorted(events, key=lambda event: event.sequence_id)
        self.assertEqual([event.sequence_id for event in events], [0, 1])
        self.assertTrue(all(event.done for event in events))
        self.assertEqual(
            [event.token_count for event in events],
            [result.token_count for result in serial],
        )
        self.assertEqual(
            [event.samples for event in events],
            [result.samples for result in serial],
        )

    def test_continuous_batch_session_rejects_streaming(self):
        model = Model(_tiny_config())

        self.assertFalse(model.supports_tts_continuous_batch(stream=True))

    def test_continuous_batch_session_cancels_pending(self):
        model = Model(_tiny_config())
        session = model.create_tts_batch_session(TTSBatchOptions(max_tokens=5))

        session.add([TTSBatchItem(sequence_id=1, text="cancel me")])
        session.cancel(1)

        self.assertTrue(session.idle)
        self.assertEqual(session.step(), [])


if __name__ == "__main__":
    unittest.main()
