import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from mlx_audio.codec.models.moss_audio_tokenizer import (
    MossAudioTokenizer,
    MossAudioTokenizerConfig,
    MossAudioTokenizerDecoderOutput,
    MossAudioTokenizerModuleConfig,
    MossAudioTokenizerQuantizerConfig,
)


def _tiny_moss_config() -> MossAudioTokenizerConfig:
    return MossAudioTokenizerConfig(
        model_type="moss_audio_tokenizer",
        sampling_rate=24,
        downsample_rate=4,
        causal_transformer_context_duration=1.0,
        encoder_modules=[
            MossAudioTokenizerModuleConfig(
                module_type="PatchedPretransform", patch_size=2
            ),
            MossAudioTokenizerModuleConfig(
                module_type="Transformer",
                input_dimension=2,
                output_dimension=2,
                d_model=4,
                num_heads=1,
                num_layers=1,
                dim_feedforward=8,
                causal=True,
                norm="layer_norm",
                positional_embedding="rope",
                max_period=10000,
                gating="none",
                layer_scale=0.01,
                conv_layout=True,
            ),
            MossAudioTokenizerModuleConfig(
                module_type="PatchedPretransform", patch_size=2
            ),
            MossAudioTokenizerModuleConfig(
                module_type="Transformer",
                input_dimension=4,
                output_dimension=4,
                d_model=4,
                num_heads=1,
                num_layers=1,
                dim_feedforward=8,
                causal=True,
                norm="layer_norm",
                positional_embedding="rope",
                max_period=10000,
                gating="none",
                layer_scale=0.01,
                conv_layout=True,
            ),
        ],
        decoder_modules=[
            MossAudioTokenizerModuleConfig(
                module_type="Transformer",
                input_dimension=4,
                output_dimension=4,
                d_model=4,
                num_heads=1,
                num_layers=1,
                dim_feedforward=8,
                causal=True,
                norm="layer_norm",
                positional_embedding="rope",
                max_period=10000,
                gating="none",
                layer_scale=0.01,
                conv_layout=True,
            ),
            MossAudioTokenizerModuleConfig(
                module_type="PatchedPretransform", patch_size=2
            ),
            MossAudioTokenizerModuleConfig(
                module_type="Transformer",
                input_dimension=2,
                output_dimension=2,
                d_model=4,
                num_heads=1,
                num_layers=1,
                dim_feedforward=8,
                causal=True,
                norm="layer_norm",
                positional_embedding="rope",
                max_period=10000,
                gating="none",
                layer_scale=0.01,
                conv_layout=True,
            ),
            MossAudioTokenizerModuleConfig(
                module_type="PatchedPretransform", patch_size=2
            ),
        ],
        quantizer=MossAudioTokenizerQuantizerConfig(
            input_dim=4,
            rvq_dim=2,
            output_dim=4,
            num_quantizers=2,
            codebook_size=8,
            codebook_dim=2,
            quantizer_type="rlfq",
        ),
    )


class TestMossAudioTokenizerModel(unittest.TestCase):
    def test_encode_decode_shape_contract(self):
        model = MossAudioTokenizer(_tiny_moss_config())
        audio = mx.random.normal((1, 1, 37))

        enc = model.encode(audio, return_dict=True)
        self.assertIsNotNone(enc.audio_codes)
        self.assertIsNotNone(enc.audio_codes_lengths)

        assert enc.audio_codes is not None
        assert enc.audio_codes_lengths is not None
        self.assertEqual(enc.audio_codes.shape[0], 2)
        self.assertEqual(enc.audio_codes.shape[1], 1)
        self.assertEqual(int(enc.audio_codes_lengths[0]), 37 // 4)

        dec = model.decode(enc.audio_codes, return_dict=True)
        self.assertIsNotNone(dec.audio)
        self.assertIsNotNone(dec.audio_lengths)

        assert dec.audio is not None
        assert dec.audio_lengths is not None
        self.assertEqual(dec.audio.shape[0], 1)
        self.assertEqual(dec.audio.shape[1], 1)
        self.assertEqual(int(dec.audio_lengths[0]), int(enc.audio_codes.shape[-1]) * 4)

    def test_batch_encode_and_batch_decode(self):
        model = MossAudioTokenizer(_tiny_moss_config())
        wav_a = mx.random.normal((13,))
        wav_b = mx.random.normal((27,))

        enc = model.batch_encode([wav_a, wav_b], num_quantizers=2)
        self.assertIsNotNone(enc.audio_codes)
        self.assertIsNotNone(enc.audio_codes_lengths)

        assert enc.audio_codes is not None
        assert enc.audio_codes_lengths is not None
        self.assertEqual(enc.audio_codes.shape[:2], (2, 2))
        self.assertEqual(int(enc.audio_codes_lengths[0]), 13 // 4)
        self.assertEqual(int(enc.audio_codes_lengths[1]), 27 // 4)

        codes_list = [
            enc.audio_codes[:, i, : int(enc.audio_codes_lengths[i])]
            for i in range(enc.audio_codes.shape[1])
        ]
        dec = model.batch_decode(codes_list, num_quantizers=2)
        self.assertIsNotNone(dec.audio)
        self.assertIsNotNone(dec.audio_lengths)

        assert dec.audio is not None
        assert dec.audio_lengths is not None
        self.assertEqual(dec.audio.shape[0], 2)
        self.assertEqual(dec.audio.shape[1], 1)
        self.assertEqual(int(dec.audio_lengths[0]), (13 // 4) * 4)
        self.assertEqual(int(dec.audio_lengths[1]), (27 // 4) * 4)

    def test_decode_accepts_b_t_nq_layout(self):
        model = MossAudioTokenizer(_tiny_moss_config())
        audio = mx.random.normal((1, 1, 40))
        enc = model.encode(audio, return_dict=True)
        assert enc.audio_codes is not None
        assert enc.audio_codes_lengths is not None

        codes_b_t_nq = enc.audio_codes.transpose(1, 2, 0)
        dec = model.decode(codes_b_t_nq, return_dict=True)
        assert dec.audio is not None
        assert dec.audio_lengths is not None
        self.assertEqual(int(dec.audio_lengths[0]), int(enc.audio_codes_lengths[0]) * 4)

    def test_decode_accepts_quantizer_prefix_for_full_and_prefix_inputs(self):
        model = MossAudioTokenizer(_tiny_moss_config())
        audio = mx.random.normal((1, 1, 40))
        enc = model.encode(audio, return_dict=True)
        assert enc.audio_codes is not None
        assert enc.audio_codes_lengths is not None

        dec_from_full = model.decode(
            enc.audio_codes,
            num_quantizers=1,
            return_dict=True,
        )
        dec_from_prefix = model.decode(
            enc.audio_codes[:1],
            num_quantizers=1,
            return_dict=True,
        )
        assert dec_from_full.audio is not None
        assert dec_from_full.audio_lengths is not None
        assert dec_from_prefix.audio is not None
        assert dec_from_prefix.audio_lengths is not None
        self.assertEqual(
            int(dec_from_full.audio_lengths[0]),
            int(dec_from_prefix.audio_lengths[0]),
        )

    def test_batch_decode_accepts_quantizer_prefix(self):
        model = MossAudioTokenizer(_tiny_moss_config())
        wav_a = mx.random.normal((17,))
        wav_b = mx.random.normal((29,))

        enc = model.batch_encode([wav_a, wav_b], num_quantizers=2)
        assert enc.audio_codes is not None
        assert enc.audio_codes_lengths is not None
        prefix_codes_list = [
            enc.audio_codes[:1, i, : int(enc.audio_codes_lengths[i])]
            for i in range(enc.audio_codes.shape[1])
        ]
        dec = model.batch_decode(prefix_codes_list, num_quantizers=1)
        assert dec.audio is not None
        assert dec.audio_lengths is not None
        self.assertEqual(int(dec.audio_lengths[0]), (17 // 4) * 4)
        self.assertEqual(int(dec.audio_lengths[1]), (29 // 4) * 4)

    def test_streaming_decode_matches_non_streaming_length(self):
        model = MossAudioTokenizer(_tiny_moss_config())
        audio = mx.random.normal((1, 1, 64))
        enc = model.encode(audio, return_dict=True)
        assert enc.audio_codes is not None

        full = model.decode(enc.audio_codes, return_dict=True)
        assert full.audio is not None
        assert full.audio_lengths is not None

        chunks = list(model.streaming_decode(enc.audio_codes, chunk_tokens=2))
        stream_concat = mx.concatenate(chunks, axis=-1)
        self.assertEqual(stream_concat.shape[0], 1)
        self.assertEqual(stream_concat.shape[1], 1)
        self.assertEqual(stream_concat.shape[-1], int(full.audio_lengths[0]))

    def test_streaming_decode_accepts_quantizer_prefix(self):
        model = MossAudioTokenizer(_tiny_moss_config())
        audio = mx.random.normal((1, 1, 64))
        enc = model.encode(audio, return_dict=True)
        assert enc.audio_codes is not None

        full = model.decode(enc.audio_codes, num_quantizers=1, return_dict=True)
        assert full.audio is not None
        assert full.audio_lengths is not None

        chunks = list(
            model.streaming_decode(enc.audio_codes[:1], chunk_tokens=2, num_quantizers=1)
        )
        stream_concat = mx.concatenate(chunks, axis=-1)
        self.assertEqual(stream_concat.shape[-1], int(full.audio_lengths[0]))

    def test_decode_accepts_encode_output_when_time_equals_quantizers(self):
        model = MossAudioTokenizer(_tiny_moss_config())
        # downsample_rate=4 => 8 samples encodes to 2 frames, matching num_quantizers=2.
        audio = mx.random.normal((1, 1, 8))
        enc = model.encode(audio, return_dict=True)
        assert enc.audio_codes is not None
        assert enc.audio_codes_lengths is not None
        self.assertEqual(tuple(enc.audio_codes.shape), (2, 1, 2))

        dec = model.decode(enc.audio_codes, return_dict=True)
        assert dec.audio is not None
        assert dec.audio_lengths is not None
        self.assertEqual(dec.audio.shape[0], 1)
        self.assertEqual(int(dec.audio_lengths[0]), int(enc.audio_codes_lengths[0]) * 4)

    def test_decode_prefers_nq_first_when_3d_shape_matches_both_orientations(self):
        model = MossAudioTokenizer(_tiny_moss_config())
        tie_shape_codes = mx.zeros((2, 5, 2), dtype=mx.int32)

        dec = model.decode(tie_shape_codes, return_dict=True)
        assert dec.audio is not None
        assert dec.audio_lengths is not None
        self.assertEqual(dec.audio.shape[0], 5)
        self.assertTrue(np.all(np.array(dec.audio_lengths) == 8))

    def test_decode_preserves_canonical_3d_tie_for_explicit_configured_num_quantizers(
        self,
    ):
        model = MossAudioTokenizer(_tiny_moss_config())
        tie_shape_codes = mx.array(
            [
                [[1, 2], [3, 4], [5, 6]],
                [[7, 8], [9, 10], [11, 12]],
            ],
            dtype=mx.int32,
        )
        captured: list[np.ndarray] = []

        def fake_decode_frame(audio_codes, audio_codes_lengths=None, caches=None):
            del audio_codes_lengths, caches
            captured.append(np.array(audio_codes))
            batch = int(audio_codes.shape[1])
            time_steps = int(audio_codes.shape[2])
            sample_count = time_steps * model.downsample_rate
            return MossAudioTokenizerDecoderOutput(
                audio=mx.zeros((batch, 1, sample_count), dtype=mx.float32),
                audio_lengths=mx.full((batch,), sample_count, dtype=mx.int32),
            )

        with patch.object(model, "_decode_frame", side_effect=fake_decode_frame):
            dec = model.decode(tie_shape_codes, num_quantizers=2, return_dict=True)

        assert dec.audio is not None
        self.assertEqual(len(captured), 1)
        np.testing.assert_array_equal(captured[0], np.array(tie_shape_codes))

    def test_decode_prefers_requested_quantizer_match_for_2d_nq_last_prefix_tie(self):
        model = MossAudioTokenizer(_tiny_moss_config())
        tie_shape_codes = mx.zeros((2, 1), dtype=mx.int32)

        dec = model.decode(tie_shape_codes, num_quantizers=1, return_dict=True)
        assert dec.audio is not None
        assert dec.audio_lengths is not None
        self.assertEqual(dec.audio.shape[0], 1)
        self.assertEqual(int(dec.audio_lengths[0]), 8)

    def test_decode_prefers_requested_quantizer_match_for_3d_nq_last_prefix_tie(self):
        model = MossAudioTokenizer(_tiny_moss_config())
        tie_shape_codes = mx.zeros((2, 1, 1), dtype=mx.int32)

        dec = model.decode(tie_shape_codes, num_quantizers=1, return_dict=True)
        assert dec.audio is not None
        assert dec.audio_lengths is not None
        self.assertEqual(dec.audio.shape[0], 2)
        self.assertTrue(np.all(np.array(dec.audio_lengths) == 4))

    def test_decode_preserves_canonical_orientation_for_true_prefix_3d_tie(self):
        model = MossAudioTokenizer(_tiny_moss_config())
        tie_shape_codes = mx.array(
            [
                [[1], [2], [3]],
            ],
            dtype=mx.int32,
        )
        captured: list[np.ndarray] = []

        def fake_decode_frame(audio_codes, audio_codes_lengths=None, caches=None):
            del audio_codes_lengths, caches
            captured.append(np.array(audio_codes))
            batch = int(audio_codes.shape[1])
            time_steps = int(audio_codes.shape[2])
            sample_count = time_steps * model.downsample_rate
            return MossAudioTokenizerDecoderOutput(
                audio=mx.zeros((batch, 1, sample_count), dtype=mx.float32),
                audio_lengths=mx.full((batch,), sample_count, dtype=mx.int32),
            )

        with patch.object(model, "_decode_frame", side_effect=fake_decode_frame):
            dec = model.decode(tie_shape_codes, num_quantizers=1, return_dict=True)

        assert dec.audio is not None
        self.assertEqual(len(captured), 1)
        np.testing.assert_array_equal(captured[0], np.array(tie_shape_codes))

    def test_decode_preserves_nq_first_for_explicit_square_tie(self):
        model = MossAudioTokenizer(_tiny_moss_config())
        tie_shape_codes = mx.array([[1, 2], [3, 4]], dtype=mx.int32)
        captured: list[np.ndarray] = []

        def fake_decode_frame(audio_codes, audio_codes_lengths=None, caches=None):
            del audio_codes_lengths, caches
            captured.append(np.array(audio_codes))
            batch = int(audio_codes.shape[1])
            time_steps = int(audio_codes.shape[2])
            sample_count = time_steps * model.downsample_rate
            return MossAudioTokenizerDecoderOutput(
                audio=mx.zeros((batch, 1, sample_count), dtype=mx.float32),
                audio_lengths=mx.full((batch,), sample_count, dtype=mx.int32),
            )

        with patch.object(model, "_decode_frame", side_effect=fake_decode_frame):
            dec = model.decode(tie_shape_codes, num_quantizers=2, return_dict=True)

        assert dec.audio is not None
        self.assertEqual(len(captured), 1)
        expected = np.array([[[1, 2]], [[3, 4]]], dtype=np.int32)
        np.testing.assert_array_equal(captured[0], expected)

    def test_batch_decode_prefers_requested_quantizer_match_for_2d_nq_last_prefix_tie(
        self,
    ):
        model = MossAudioTokenizer(_tiny_moss_config())
        tie_shape_codes = mx.zeros((2, 1), dtype=mx.int32)

        dec = model.batch_decode([tie_shape_codes], num_quantizers=1)
        assert dec.audio is not None
        assert dec.audio_lengths is not None
        self.assertEqual(dec.audio.shape[0], 1)
        self.assertEqual(int(dec.audio_lengths[0]), 8)

    def test_batch_decode_rejects_canonical_batched_true_prefix_3d_tie(
        self,
    ):
        model = MossAudioTokenizer(_tiny_moss_config())
        tie_shape_codes = mx.zeros((1, 3, 1), dtype=mx.int32)

        with self.assertRaisesRegex(
            ValueError,
            "batch_decode\\(\\) expects each codes_list entry to resolve to batch_size=1",
        ):
            _ = model.batch_decode([tie_shape_codes], num_quantizers=1)

    def test_batch_decode_preserves_nq_first_for_explicit_square_tie(self):
        model = MossAudioTokenizer(_tiny_moss_config())
        tie_shape_codes = mx.array([[1, 2], [3, 4]], dtype=mx.int32)
        captured: list[np.ndarray] = []

        def fake_decode_frame(audio_codes, audio_codes_lengths=None, caches=None):
            del audio_codes_lengths, caches
            captured.append(np.array(audio_codes))
            batch = int(audio_codes.shape[1])
            time_steps = int(audio_codes.shape[2])
            sample_count = time_steps * model.downsample_rate
            return MossAudioTokenizerDecoderOutput(
                audio=mx.zeros((batch, 1, sample_count), dtype=mx.float32),
                audio_lengths=mx.full((batch,), sample_count, dtype=mx.int32),
            )

        with patch.object(model, "_decode_frame", side_effect=fake_decode_frame):
            dec = model.batch_decode([tie_shape_codes], num_quantizers=2)

        assert dec.audio is not None
        self.assertEqual(len(captured), 1)
        expected = np.array([[[1, 2]], [[3, 4]]], dtype=np.int32)
        np.testing.assert_array_equal(captured[0], expected)

    def test_streaming_decode_prefers_requested_quantizer_match_for_2d_nq_last_prefix_tie(
        self,
    ):
        model = MossAudioTokenizer(_tiny_moss_config())
        tie_shape_codes = mx.zeros((2, 1), dtype=mx.int32)

        chunks = list(
            model.streaming_decode(
                tie_shape_codes,
                chunk_tokens=1,
                num_quantizers=1,
            )
        )
        stream_concat = mx.concatenate(chunks, axis=-1)
        self.assertEqual(stream_concat.shape[0], 1)
        self.assertEqual(stream_concat.shape[1], 1)
        self.assertEqual(stream_concat.shape[-1], 8)

    def test_streaming_decode_preserves_nq_first_for_explicit_square_tie(self):
        model = MossAudioTokenizer(_tiny_moss_config())
        tie_shape_codes = mx.array(
            [
                [[1, 2]],
                [[3, 4]],
            ],
            dtype=mx.int32,
        )
        captured: list[np.ndarray] = []

        def fake_decode_frame(audio_codes, audio_codes_lengths=None, caches=None):
            del audio_codes_lengths, caches
            captured.append(np.array(audio_codes))
            batch = int(audio_codes.shape[1])
            time_steps = int(audio_codes.shape[2])
            sample_count = time_steps * model.downsample_rate
            return MossAudioTokenizerDecoderOutput(
                audio=mx.zeros((batch, 1, sample_count), dtype=mx.float32),
                audio_lengths=mx.full((batch,), sample_count, dtype=mx.int32),
            )

        with patch.object(model, "_decode_frame", side_effect=fake_decode_frame):
            _ = list(
                model.streaming_decode(
                    tie_shape_codes,
                    chunk_tokens=1,
                    num_quantizers=2,
                )
            )

        self.assertGreaterEqual(len(captured), 1)
        reconstructed = np.concatenate(captured, axis=2)
        expected = np.array([[[1, 2]], [[3, 4]]], dtype=np.int32)
        np.testing.assert_array_equal(reconstructed, expected)

    def test_streaming_decode_rejects_canonical_batched_true_prefix_3d_tie(
        self,
    ):
        model = MossAudioTokenizer(_tiny_moss_config())
        tie_shape_codes = mx.zeros((1, 3, 1), dtype=mx.int32)

        with self.assertRaisesRegex(
            ValueError,
            "streaming_decode currently only supports batch_size=1",
        ):
            _ = list(
                model.streaming_decode(
                    tie_shape_codes,
                    chunk_tokens=1,
                    num_quantizers=1,
                )
            )

    def test_sanitize_reconstructs_weight_norm(self):
        model = MossAudioTokenizer(_tiny_moss_config())
        expected_shapes = {
            name: tuple(value.shape) for name, value in tree_flatten(model.parameters())
        }
        linear_key = next(
            key for key in expected_shapes if key.endswith("encoder.1.transformer.layers.0.linear1.weight")
        )
        linear_shape = expected_shapes[linear_key]

        expected_input_proj_shape = expected_shapes["quantizer.input_proj.weight"]
        pytorch_v_shape = (
            expected_input_proj_shape[0],
            expected_input_proj_shape[2],
            expected_input_proj_shape[1],
        )
        g = mx.ones((pytorch_v_shape[0], 1, 1), dtype=mx.float32)
        v = mx.arange(
            1,
            1 + int(np.prod(np.array(pytorch_v_shape))),
            dtype=mx.float32,
        ).reshape(pytorch_v_shape)
        weights = {
            "quantizer.input_proj.parametrizations.weight.original0": g,
            "quantizer.input_proj.parametrizations.weight.original1": v,
            linear_key: mx.ones(linear_shape, dtype=mx.float32),
        }
        sanitized = model.sanitize(weights)

        self.assertIn("quantizer.input_proj.weight", sanitized)
        self.assertEqual(
            sanitized["quantizer.input_proj.weight"].shape,
            expected_shapes["quantizer.input_proj.weight"],
        )
        self.assertNotIn(
            "quantizer.input_proj.parametrizations.weight.original0", sanitized
        )
        self.assertNotIn(
            "quantizer.input_proj.parametrizations.weight.original1", sanitized
        )
        self.assertTrue(
            np.allclose(
                np.array(sanitized[linear_key]), np.array(weights[linear_key])
            )
        )

    def test_from_pretrained_loads_local_directory(self):
        model = MossAudioTokenizer(_tiny_moss_config())
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_path = root / "config.json"
            config_payload = asdict(model.config)
            config_payload["model_type"] = "speech_tokenizer"
            config_path.write_text(json.dumps(config_payload), encoding="utf-8")

            weight_path = root / "model.safetensors"
            mx.save_safetensors(
                weight_path.as_posix(),
                dict(tree_flatten(model.parameters())),
            )

            loaded = MossAudioTokenizer.from_pretrained(root)
            audio = mx.random.normal((1, 1, 20))
            enc = loaded.encode(audio, return_dict=True)
            self.assertIsNotNone(enc.audio_codes)
            self.assertIsNotNone(enc.audio_codes_lengths)


class TestMossAudioTokenizerQuantPredicate(unittest.TestCase):
    def test_model_quant_predicate_skips_embeddings(self):
        model = MossAudioTokenizer(_tiny_moss_config())
        codebook_module = model.quantizer.quantizers[0].codebook
        linear_module = model.encoder[1].transformer.layers[0].linear1

        self.assertFalse(
            model.model_quant_predicate(
                "quantizer.quantizers.0.codebook",
                codebook_module,
            )
        )
        self.assertTrue(
            model.model_quant_predicate(
                "encoder.1.transformer.layers.0.linear1",
                linear_module,
            )
        )


if __name__ == "__main__":
    unittest.main()
