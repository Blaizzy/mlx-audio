import unittest

import mlx.core as mx
import numpy as np

from mlx_audio.tts.models.zonos2.config import Zonos2Config
from mlx_audio.tts.models.zonos2.convert import convert_state_dict
from mlx_audio.tts.models.zonos2.generation import (
    TTSSamplingParams,
    Zonos2GenerationState,
    sample_frame,
)
from mlx_audio.tts.models.zonos2.model import Model
from mlx_audio.tts.models.zonos2.prompt import (
    TTSPromptBuilder,
    TTSPromptConfig,
    accurate_mode_token_id,
    quality_token_id,
    shear,
    shear_up,
    speaker_background_token_id,
    speaking_rate_token_id,
    text_to_byte_ids,
)
from mlx_audio.tts.models.zonos2.speaker_encoder import (
    sanitize_speaker_encoder_weights,
    speaker_log_mel_spectrogram,
)
from mlx_audio.tts.models.zonos2.textnorm import TTSTextNormalizer
from mlx_audio.tts.utils import get_model_and_args
from mlx_audio.utils import get_model_name_parts


def _tiny_config() -> Zonos2Config:
    return Zonos2Config(
        n_layers=1,
        dim=16,
        head_dim=4,
        n_kv_heads=2,
        ffn_dim_multiplier=1.0,
        multiple_of=8,
        n_codebooks=2,
        codebook_size=4,
        eoa_id=4,
        audio_pad_id=5,
        text_vocab=519,
        speaker_enabled=False,
        speaker_background_token_enabled=False,
        accurate_mode_token_enabled=False,
        moe_n_experts=1,
    )


class TestZonos2Config(unittest.TestCase):
    def test_parses_source_config_shape(self):
        cfg = Zonos2Config.from_dict(
            {
                "model_type": "zonos2",
                "dtype": "bfloat16",
                "n_layers": 28,
                "dim": 2048,
                "head_dim": 128,
                "n_kv_heads": 4,
                "n_codebooks": 9,
                "codebook_size": 1024,
                "eoa_id": 1024,
                "audio_pad_id": 1025,
                "text_vocab": 519,
                "speaker_enabled": True,
                "speaker_embedding_dim": 2048,
                "speaker_lda_dim": 1024,
                "speaker_encoder_model_id": "example/speaker",
                "speaker_encoder_path": "speaker_encoder",
                "speaker_encoder_sample_rate": 24000,
                "speaking_rate_num_buckets": 8,
                "quality_num_buckets": 60,
                "moe_n_experts": 16,
                "moe_start_from_layer": 3,
                "moe_end_from_layer": 1,
                "norm_topk_prob": True,
                "special_topk_layers": {"26": 2},
            }
        )
        self.assertEqual(cfg.model_type, "zonos2")
        self.assertEqual(cfg.num_heads, 16)
        self.assertEqual(cfg.num_kv_heads, 4)
        self.assertEqual(cfg.intermediate_size, 3072)
        self.assertEqual(cfg.audio_vocab_size, 1026)
        self.assertTrue(cfg.is_moe_layer(3))
        self.assertFalse(cfg.is_moe_layer(2))
        self.assertFalse(cfg.is_moe_layer(27))
        self.assertEqual(cfg.num_experts_per_tok(26), 2)
        self.assertTrue(cfg.norm_topk_prob)
        self.assertEqual(cfg.speaker_encoder_model_id, "example/speaker")
        self.assertEqual(cfg.speaker_encoder_path, "speaker_encoder")
        self.assertEqual(cfg.speaker_encoder_sample_rate, 24000)

    def test_loader_finds_model(self):
        arch, model_type = get_model_and_args(
            "zonos2",
            get_model_name_parts("mlx-community/ZONOS2-bf16"),
        )
        self.assertEqual(model_type, "zonos2")
        self.assertTrue(hasattr(arch, "Model"))


class TestZonos2Prompt(unittest.TestCase):
    def test_conditioning_token_ids_match_reference_layout(self):
        counts = (12, 12, 12, 8, 8, 8)
        self.assertEqual(
            speaking_rate_token_id(519, 8, 3, counts, 2, 1),
            451,
        )
        self.assertEqual(
            quality_token_id(
                519,
                8,
                counts,
                feature_idx=1,
                quality_bucket=4,
                speaker_background_num_buckets=2,
                accurate_mode_num_buckets=1,
            ),
            472,
        )
        self.assertEqual(
            speaker_background_token_id(
                519,
                8,
                counts,
                clean=False,
                speaker_background_num_buckets=2,
                accurate_mode_num_buckets=1,
            ),
            517,
        )
        self.assertEqual(accurate_mode_token_id(519, 8, counts, 2, 1), 518)

    def test_shear_matches_reference_triangular_delay(self):
        codes = mx.arange(1, 13, dtype=mx.int32).reshape(3, 4)
        delayed = shear(codes, pad=99)
        np.testing.assert_array_equal(
            np.array(delayed),
            np.array(
                [
                    [1, 99, 99, 99],
                    [5, 2, 99, 99],
                    [9, 6, 3, 99],
                ],
                dtype=np.int32,
            ),
        )

    def test_shear_up_matches_reference_de_delay(self):
        delayed = mx.array(
            [
                [1, 99, 99, 99],
                [5, 2, 99, 99],
                [9, 6, 3, 99],
            ],
            dtype=mx.int32,
        )
        codes = shear_up(delayed, pad=99)
        np.testing.assert_array_equal(
            np.array(codes),
            np.array(
                [
                    [1, 2, 3, 99],
                    [5, 6, 99, 99],
                    [9, 99, 99, 99],
                ],
                dtype=np.int32,
            ),
        )

    def test_prompt_builder_adds_text_then_silence(self):
        cfg = TTSPromptConfig()
        builder = TTSPromptBuilder(cfg)
        prompt = builder.build_list(
            "A",
            speaking_rate_bucket=3,
            quality_buckets=[None, None, None, None, None, 3],
        )
        self.assertEqual(prompt[0][-1], 451)
        self.assertEqual(prompt[1][-1], 511)
        self.assertEqual([row[-1] for row in prompt[2:5]], text_to_byte_ids("A"))
        self.assertEqual(len(prompt), 2 + 3 + 17)

    def test_speaker_marker_prefix(self):
        cfg = TTSPromptConfig()
        prefix = TTSPromptBuilder(cfg).speaker_marker_prefix(
            clean_speaker_background=False,
            accurate_mode=True,
        )
        self.assertEqual([row[-1] for row in prefix], [519, 517, 518])
        self.assertTrue(all(token == 1025 for row in prefix for token in row[:-1]))


class TestZonos2TextNorm(unittest.TestCase):
    def test_common_english_written_forms(self):
        normalizer = TTSTextNormalizer()
        text = "On Jan. 5, 2026 at 3:45pm, NASA paid $12.50 for 2 kg " "and saved 7.5%."
        self.assertEqual(
            normalizer.normalize(text, "en_us"),
            "On january fifth twenty twenty six at three forty five p m, "
            "n a s a paid twelve dollars and fifty cents for two kilograms "
            "and saved seven point five percent.",
        )

    def test_numeric_dates_ordinals_units_and_phone_numbers(self):
        normalizer = TTSTextNormalizer()
        self.assertEqual(
            normalizer.normalize("Call 555-1212 on 2026-06-13 at 10:05.", "en"),
            "Call five five five one two one two on june thirteenth "
            "twenty twenty six at ten oh five.",
        )
        self.assertEqual(
            normalizer.normalize("The 21st lap was 3.2 km at 70°F.", "en"),
            "The twenty first lap was three point two kilometers at "
            "seventy degrees fahrenheit.",
        )
        self.assertEqual(
            normalizer.normalize("Mix 1/2 cup on 06/13/2026.", "en"),
            "Mix one half cup on june thirteenth twenty twenty six.",
        )

    def test_non_english_text_passes_through(self):
        normalizer = TTSTextNormalizer()
        self.assertFalse(normalizer.supported("fr"))
        self.assertEqual(
            normalizer.normalize("Le prix est 12,50 €.", "fr"), "Le prix est 12,50 €."
        )


class TestZonos2Sampler(unittest.TestCase):
    def test_greedy_sampling_appends_text_placeholder(self):
        state = Zonos2GenerationState(n_codebooks=2, eoa_id=4, text_vocab=519)
        logits = mx.zeros((2, 6))
        logits[0, 1] = 5
        logits[1, 2] = 6
        frame = sample_frame(
            logits,
            state,
            TTSSamplingParams(temperature=0.0),
            key=mx.random.key(0),
        )
        self.assertEqual(frame, [1, 2, 519])

    def test_filtered_sampling_uses_mlx_categorical(self):
        state = Zonos2GenerationState(n_codebooks=2, eoa_id=7, text_vocab=519)
        logits = mx.zeros((2, 8), dtype=mx.float32)
        logits[:, 3] = 4.0
        logits[:, 4] = 3.0
        frame = sample_frame(
            logits,
            state,
            TTSSamplingParams(temperature=1.0, top_k=2, top_p=0.9, min_p=0.0),
            key=mx.random.key(3),
        )
        self.assertEqual(len(frame), 3)
        self.assertIn(frame[0], {3, 4})
        self.assertIn(frame[1], {3, 4})
        self.assertEqual(frame[-1], 519)

    def test_eos_countdown_uses_delayed_codebook_position(self):
        state = Zonos2GenerationState(n_codebooks=3, eoa_id=4, text_vocab=519)
        state.append([1, 2, 3, 519])
        state.append([1, 4, 3, 519])
        self.assertEqual(state.eos_frame, 0)
        self.assertFalse(state.finished)
        for _ in range(3):
            state.append([1, 2, 3, 519])
        self.assertTrue(state.finished)


class TestZonos2Model(unittest.TestCase):
    def test_attention_uses_interleaved_rope(self):
        cfg = _tiny_config()
        model = Model(cfg)
        self.assertTrue(model.layers[0].attention.rope.traditional)

    def test_preserves_ref_audio_paths_for_speaker_encoder(self):
        self.assertTrue(Model.preserve_ref_audio_path)

    def test_rejects_ref_audio_and_precomputed_speaker_embedding_together(self):
        model = Model(_tiny_config())
        with self.assertRaisesRegex(
            ValueError, "either speaker_embedding or ref_audio"
        ):
            model._resolve_speaker_embedding(
                speaker_embedding=np.zeros((16,), dtype=np.float32),
                ref_audio="ref.wav",
                ref_audio_sample_rate=None,
            )

    def test_tiny_forward_shape(self):
        cfg = _tiny_config()
        model = Model(cfg)
        input_ids = mx.array(
            [
                [
                    [0, 5, 2],
                    [1, 0, 257],
                    [5, 1, 519],
                ]
            ],
            dtype=mx.int32,
        )
        logits = model(input_ids)
        mx.eval(logits)
        self.assertEqual(logits.shape, (1, 3, 2, 6))

    def test_streaming_yields_chunks_and_final_marker(self):
        model = Model(_tiny_config())
        decode_calls = []

        def fake_decode(delayed_rows, eos_frame, frame_limit=None):
            del eos_frame
            decode_calls.append((len(delayed_rows), frame_limit))
            frames = int(frame_limit) if frame_limit is not None else len(delayed_rows)
            return mx.ones((max(0, frames) * 512,), dtype=mx.float32)

        model._decode_audio = fake_decode
        results = list(
            model.generate(
                "stream",
                max_tokens=5,
                stream=True,
                streaming_interval=512 / model.sample_rate,
                ignore_eos=True,
                temperature=0.0,
            )
        )
        self.assertGreaterEqual(len(results), 2)
        self.assertTrue(all(result.is_streaming_chunk for result in results))
        self.assertTrue(results[-1].is_final_chunk)
        self.assertTrue(any(call[1] is not None for call in decode_calls[:-1]))
        self.assertGreater(sum(result.samples for result in results), 0)

    def test_batch_generate_yields_one_result_per_text(self):
        model = Model(_tiny_config())
        decode_calls = []

        def fake_decode(delayed_rows, eos_frame, frame_limit=None):
            decode_calls.append((len(delayed_rows), eos_frame, frame_limit))
            return mx.ones((len(delayed_rows) * 512,), dtype=mx.float32)

        model._decode_audio = fake_decode
        results = list(
            model.batch_generate(
                ["a", "longer text"],
                max_tokens=3,
                ignore_eos=True,
                temperature=0.0,
            )
        )
        self.assertEqual([result.sequence_idx for result in results], [0, 1])
        self.assertEqual([result.token_count for result in results], [3, 3])
        self.assertEqual([result.samples for result in results], [1536, 1536])
        self.assertFalse(any(result.is_streaming_chunk for result in results))
        self.assertEqual(len(decode_calls), 2)

    def test_batch_generate_offsets_speaker_position_after_left_padding(self):
        model = Model(_tiny_config())
        texts = ["a", "longer text"]
        prompt_rows = [
            model._build_prompt_rows(
                text,
                speaking_rate_bucket=None,
                quality_buckets=None,
                speaker_conditioned=True,
                clean_speaker_background=False,
                accurate_mode=True,
            )[0]
            for text in texts
        ]
        _, left_padding = model._batch_prompt(prompt_rows)
        captured_positions = []
        original_inject = model._inject_speaker

        def capture_inject(x, speaker_embedding, speaker_token_position):
            if speaker_token_position is not None:
                captured_positions.append(speaker_token_position)
            return original_inject(x, speaker_embedding, speaker_token_position)

        def fake_decode(delayed_rows, eos_frame, frame_limit=None):
            del eos_frame, frame_limit
            return mx.ones((len(delayed_rows) * 512,), dtype=mx.float32)

        model._inject_speaker = capture_inject
        model._decode_audio = fake_decode
        speaker_embedding = mx.zeros(
            (model.config.speaker_embedding_dim,), dtype=mx.float32
        )
        list(
            model.batch_generate(
                texts,
                speaker_embedding=speaker_embedding,
                max_tokens=1,
                ignore_eos=True,
                temperature=0.0,
                text_normalization=False,
            )
        )
        self.assertEqual(captured_positions[0], left_padding)


class TestZonos2SpeakerEncoder(unittest.TestCase):
    def test_sanitize_root_conv_weights_transposes_to_mlx_layout(self):
        weight = mx.array(np.arange(2 * 5 * 3, dtype=np.float32).reshape(2, 5, 3))
        converted = sanitize_speaker_encoder_weights({"blocks.0.conv.weight": weight})
        self.assertEqual(converted["blocks.0.conv.weight"].shape, (2, 3, 5))
        np.testing.assert_array_equal(
            np.array(converted["blocks.0.conv.weight"]),
            np.arange(2 * 5 * 3, dtype=np.float32).reshape(2, 5, 3).transpose(0, 2, 1),
        )

    def test_speaker_log_mel_spectrogram_shape(self):
        audio = mx.zeros((24000,), dtype=mx.float32)
        mel = speaker_log_mel_spectrogram(audio)
        mx.eval(mel)
        self.assertEqual(mel.shape, (1, 93, 128))
        self.assertTrue(np.isfinite(np.array(mel)).all())


class TestZonos2Converter(unittest.TestCase):
    def test_sonic_expert_and_router_keys_are_mapped(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is required for converter mapping tests")

        state = {
            "layers.3.feed_forward.experts.w13": torch.arange(
                2 * 4 * 3,
                dtype=torch.float32,
            ).reshape(2, 4, 3),
            "layers.3.feed_forward.experts.w2": torch.ones(
                (2, 3, 2),
                dtype=torch.float32,
            ),
            "layers.3.feed_forward.router.router_mlp.0.weight": torch.zeros(
                (4, 4),
                dtype=torch.float32,
            ),
        }
        converted = convert_state_dict(state, dtype="float32")
        self.assertIn("layers.3.feed_forward.experts.gate_proj.weight", converted)
        self.assertIn("layers.3.feed_forward.experts.up_proj.weight", converted)
        self.assertIn("layers.3.feed_forward.experts.down_proj.weight", converted)
        self.assertIn("layers.3.feed_forward.router.router_mlp.l0.weight", converted)
        np.testing.assert_array_equal(
            np.array(converted["layers.3.feed_forward.experts.gate_proj.weight"]),
            state["layers.3.feed_forward.experts.w13"][:, 0::2, :].numpy(),
        )
        np.testing.assert_array_equal(
            np.array(converted["layers.3.feed_forward.experts.up_proj.weight"]),
            state["layers.3.feed_forward.experts.w13"][:, 1::2, :].numpy(),
        )


if __name__ == "__main__":
    unittest.main()
