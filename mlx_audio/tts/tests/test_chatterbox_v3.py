import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mlx.core as mx
import numpy as np
from safetensors.numpy import load_file

from mlx_audio.tts.models.chatterbox.scripts.convert import (
    CHATTERBOX_VARIANTS,
    download_chatterbox_weights,
    generate_readme,
    mlx_to_numpy,
    resolve_variant,
    save_mlx_safetensors,
)
from mlx_audio.tts.models.chatterbox.t3.t3 import T3


class ChatterboxV3ConversionTest(unittest.TestCase):
    def test_transposed_weights_round_trip_in_logical_order(self):
        source = mx.arange(24).reshape(2, 3, 4)
        transposed = mx.swapaxes(source, 1, 2)

        converted = mlx_to_numpy({"weight": transposed})
        self.assertTrue(converted["weight"].flags.c_contiguous)

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.safetensors"
            save_mlx_safetensors(converted, path)
            restored = load_file(path)["weight"]

        expected = np.arange(24).reshape(2, 3, 4).swapaxes(1, 2)
        np.testing.assert_array_equal(restored, expected)

    def test_safetensors_writer_materializes_strided_numpy_views(self):
        source = np.arange(24).reshape(2, 3, 4)
        transposed = source.swapaxes(1, 2)
        self.assertFalse(transposed.flags.c_contiguous)

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.safetensors"
            save_mlx_safetensors({"weight": transposed}, path)
            restored = load_file(path)["weight"]

        np.testing.assert_array_equal(restored, transposed)

    def test_v3_resolves_exact_source_files(self):
        name, variant = resolve_variant("v3")

        self.assertEqual(name, "v3")
        self.assertEqual(variant.t3_filename, "t3_mtl23ls_v3.safetensors")
        self.assertEqual(
            variant.tokenizer_filename,
            "grapheme_mtl_merged_expanded_v1.json",
        )
        self.assertTrue(variant.multilingual)
        self.assertEqual(variant.text_preprocessing, "NFKD,fullcase")

    @patch("huggingface_hub.snapshot_download")
    def test_v3_download_does_not_fetch_other_large_checkpoints(
        self, snapshot_download
    ):
        snapshot_download.return_value = "/tmp/chatterbox-v3"

        result = download_chatterbox_weights(
            "ResembleAI/chatterbox",
            Path("/tmp/cache"),
            CHATTERBOX_VARIANTS["v3"],
        )

        self.assertEqual(result, Path("/tmp/chatterbox-v3"))
        patterns = snapshot_download.call_args.kwargs["allow_patterns"]
        self.assertEqual(
            patterns,
            [
                "ve.safetensors",
                "t3_mtl23ls_v3.safetensors",
                "s3gen.safetensors",
                "grapheme_mtl_merged_expanded_v1.json",
                "Cangjie5_TC.json",
            ],
        )

    def test_v3_model_card_records_variant_and_language(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            generate_readme(path, "local/chatterbox-v3", "v3")
            card = (path / "README.md").read_text()

        self.assertIn("Chatterbox Multilingual V3", card)
        self.assertIn('lang_code="fr"', card)
        self.assertIn("license: mit", card)


class ChatterboxT3InferenceParityTest(unittest.TestCase):
    def test_inference_context_contains_two_bos_positions(self):
        embeds = mx.arange(2 * 5 * 3).reshape(2, 5, 3)
        prepare_input_embeds = MagicMock(return_value=(embeds, 3))
        fake = SimpleNamespace(
            hp=SimpleNamespace(start_speech_token=6561),
            prepare_input_embeds=prepare_input_embeds,
            speech_emb=lambda _: mx.full((1, 1, 3), 7.0),
            speech_pos_emb=SimpleNamespace(
                get_fixed_embedding=lambda _: mx.full((1, 1, 3), 2.0)
            ),
        )
        text_tokens = mx.array([[10, 20], [10, 20]])

        context = T3._prepare_inference_context(
            fake,
            t3_cond=None,
            text_tokens=text_tokens,
            cfg_weight=0.5,
        )

        self.assertEqual(context.shape, (2, 6, 3))
        self.assertTrue(mx.array_equal(context[:, :-1], embeds))
        self.assertTrue(mx.array_equal(context[:, -1:], mx.full((2, 1, 3), 9.0)))
        speech_tokens = prepare_input_embeds.call_args.kwargs["speech_tokens"]
        self.assertEqual(speech_tokens.shape, (2, 1))
        self.assertTrue(mx.all(speech_tokens == 6561).item())


if __name__ == "__main__":
    unittest.main()
