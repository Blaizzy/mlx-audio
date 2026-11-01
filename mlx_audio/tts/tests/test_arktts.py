"""Tests for the arktts (Audio8-TTS) model: sanitize contract + tiny-model forward."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import mlx.core as mx
import numpy as np
from mlx import nn

from mlx_audio.tts.models.arktts import Model, ModelConfig
from mlx_audio.tts.models.arktts.arktts import ArkttsModel
from mlx_audio.tts.models.arktts.convert import (
    copy_supporting_files,
    load_source_weights,
    prepare_config,
    write_model_card,
)


def tiny_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=320,
        dim=32,
        n_layer=2,
        n_head=4,
        n_local_heads=2,
        head_dim=8,
        intermediate_size=64,
        max_seq_len=64,
        codebook_size=16,
        num_codebooks=4,
        semantic_begin_id=256,
        semantic_end_id=271,
        n_fast_layer=2,
        fast_dim=32,
        fast_n_head=4,
        fast_n_local_heads=2,
        fast_head_dim=8,
        fast_intermediate_size=64,
        eos_token_id=300,
        pad_token_id=301,
    )


class TestArkttsSanitize(unittest.TestCase):
    def test_weight_norm_folding(self):
        model = Model.__new__(Model)  # sanitize is self-contained
        g = mx.array(np.random.randn(4, 1, 1).astype(np.float32) ** 2 + 0.5)
        v = mx.array(np.random.randn(4, 3, 7).astype(np.float32))
        weights = {
            "encoder.block.0.conv.parametrizations.weight.original0": g,
            "encoder.block.0.conv.parametrizations.weight.original1": v,
        }
        out = Model.sanitize(model, weights)
        key = "codec.encoder.block.0.conv.weight"
        self.assertIn(key, out)
        # fold: g * v / ||v||_per_out_channel, then (O, I, K) -> (O, K, I)
        v_np = np.asarray(v)
        norm = np.sqrt((v_np.reshape(4, -1) ** 2).sum(axis=1)).reshape(4, 1, 1)
        expected = (np.asarray(g) * v_np / norm).transpose(0, 2, 1)
        self.assertTrue(np.allclose(np.asarray(out[key]), expected, atol=1e-6))

    def test_legacy_weight_norm_and_vq_transpose(self):
        model = Model.__new__(Model)
        g = mx.ones((5, 1, 1))
        v = mx.array(np.random.randn(5, 8, 1).astype(np.float32))
        out = Model.sanitize(
            model,
            {
                "quantizer.quantizer.quantizers.0.in_proj.weight_g": g,
                "quantizer.quantizer.quantizers.0.in_proj.weight_v": v,
            },
        )
        key = "codec.quantizer.quantizer.quantizers.0.in_proj.weight"
        self.assertIn(key, out)
        self.assertEqual(out[key].shape, (5, 1, 8))  # (O, I, 1) -> (O, 1, I)

    def test_transpose_routing(self):
        model = Model.__new__(Model)
        out = Model.sanitize(
            model,
            {
                # ConvTranspose sites: torch (I, O, K) -> mlx (O, K, I)
                "decoder.model.1.block.1.conv.weight": mx.zeros((16, 8, 4)),
                "quantizer.upsample.0.0.conv.weight": mx.zeros((16, 8, 2)),
                # regular conv sites: torch (O, I, K) -> mlx (O, K, I)
                "quantizer.upsample.0.1.dwconv.conv.weight": mx.zeros((16, 1, 7)),
                "quantizer.downsample.0.0.conv.weight": mx.zeros((16, 16, 2)),
                # snake alpha: (1, C, 1) -> (1, 1, C)
                "encoder.block.1.block.0.block.0.alpha": mx.zeros((1, 64, 1)),
            },
        )
        self.assertEqual(
            out["codec.decoder.model.1.block.1.conv.weight"].shape, (8, 4, 16)
        )
        self.assertEqual(
            out["codec.quantizer.upsample.0.0.conv.weight"].shape, (8, 2, 16)
        )
        self.assertEqual(
            out["codec.quantizer.upsample.0.1.dwconv.conv.weight"].shape, (16, 7, 1)
        )
        self.assertEqual(
            out["codec.quantizer.downsample.0.0.conv.weight"].shape, (16, 2, 16)
        )
        self.assertEqual(
            out["codec.encoder.block.1.block.0.block.0.alpha"].shape, (1, 1, 64)
        )

    def test_idempotent_on_converted_weights(self):
        model = Model.__new__(Model)
        converted = {
            "model.embeddings.weight": mx.zeros((8, 4)),
            "codec.x": mx.zeros((2,)),
        }
        out = Model.sanitize(model, converted)
        self.assertEqual(set(out), set(converted))
        self.assertEqual(out["model.embeddings.weight"].shape, (8, 4))

    def test_rope_buffers_dropped(self):
        model = Model.__new__(Model)
        out = Model.sanitize(model, {"freqs_cis": mx.zeros((4, 4, 2))})
        self.assertEqual(out, {})


class TestArkttsTinyModel(unittest.TestCase):
    def test_quantized_tied_logits_and_cached_generation(self):
        config = tiny_config()
        lm = ArkttsModel(config)
        nn.quantize(lm, group_size=32, bits=4)
        mx.eval(lm.parameters())
        self.assertEqual(lm.embeddings.weight.dtype, mx.uint32)
        prompt = mx.zeros((1, config.num_codebooks + 1, 3), dtype=mx.int32)
        logits, _ = lm(prompt)
        expected = lm.embeddings.as_linear(lm.norm(lm(prompt)[1]))
        np.testing.assert_allclose(np.asarray(logits), np.asarray(expected))
        codes = lm.generate_codes(
            [np.arange(3)], [np.arange(2)], max_new_tokens=2, do_sample=False
        )
        self.assertEqual(codes.shape[:2], (1, config.num_codebooks))
        self.assertTrue(bool(mx.all((codes >= 0) & (codes < config.codebook_size))))

    def test_prefill_and_greedy_generation_shapes(self):
        config = tiny_config()
        lm = ArkttsModel(config)
        mx.eval(lm.parameters())
        batch, width = 1, 12
        prompt = np.random.randint(
            0, 255, size=(batch, config.num_codebooks + 1, width)
        )
        prompt[:, 0, -3:] = np.random.randint(
            config.semantic_begin_id, config.semantic_end_id, size=(batch, 3)
        )
        logits, hidden = lm(mx.array(prompt))
        self.assertEqual(logits.shape, (batch, width, config.vocab_size))
        self.assertEqual(hidden.shape, (batch, width, config.dim))

        prefix = np.arange(5, dtype=np.int64)
        suffix = np.arange(3, dtype=np.int64)
        codes = lm.generate_codes([prefix], [suffix], max_new_tokens=4, do_sample=False)
        self.assertEqual(codes.shape[0], batch)
        self.assertEqual(codes.shape[1], config.num_codebooks)
        self.assertLessEqual(codes.shape[2], 4)

    def test_sampled_generation_with_seed_is_deterministic(self):
        config = tiny_config()
        lm = ArkttsModel(config)
        mx.eval(lm.parameters())
        prefix = np.arange(5, dtype=np.int64)
        suffix = np.arange(3, dtype=np.int64)
        a = lm.generate_codes(
            [prefix], [suffix], max_new_tokens=4, do_sample=True, seed=11
        )
        b = lm.generate_codes(
            [prefix], [suffix], max_new_tokens=4, do_sample=True, seed=11
        )
        self.assertTrue(np.array_equal(np.asarray(a), np.asarray(b)))


class TestArkttsConversion(unittest.TestCase):
    def test_native_tokenizer_and_reference_prompt(self):
        from tokenizers import Tokenizer, models, pre_tokenizers

        tokenizer = Tokenizer(
            models.WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]")
        )
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        special = ["<|im_start|>", "<|im_end|>", "<|voice|>", "<|speaker:0|>"]
        tokenizer.add_special_tokens(special)
        with tempfile.TemporaryDirectory() as directory:
            model = Model.__new__(Model)
            model.model_path = Path(directory)
            model._tokenizer = None
            tokenizer.save(str(model.model_path / "tokenizer.json"))
            with patch.dict("sys.modules", {"torch": None, "transformers": None}):
                prefix, suffix = model._prompt_segments("hello", "hello", True)
            self.assertIn(tokenizer.token_to_id("<|speaker:0|>"), prefix)
            self.assertEqual(suffix[-1], tokenizer.token_to_id("<|voice|>"))
            plain, empty = model._prompt_segments("hello", None, False)
            self.assertNotIn(tokenizer.token_to_id("<|speaker:0|>"), plain)
            self.assertEqual(empty.size, 0)

    def test_realistic_config_and_loader_dispatch(self):
        from mlx_audio.convert import Domain, get_model_type
        from mlx_audio.tts.utils import get_model_and_args

        config = {"model_type": "arktts", "dtype": "bfloat16", "auto_map": {}}
        prepared = prepare_config(config, Path("unused"))
        self.assertEqual(prepared["torch_dtype"], "bfloat16")
        self.assertEqual(ModelConfig.from_dict(prepared).dim, 896)
        self.assertNotIn("auto_map", prepared)
        self.assertIn("auto_map", config)
        module, model_type = get_model_and_args("arktts", ["Audio8", "Realistic"])
        self.assertIs(module.Model, Model)
        self.assertEqual(model_type, "arktts")
        with patch("mlx_audio.convert.get_model_types", return_value={"arktts"}):
            self.assertEqual(
                get_model_type(prepared, Path("model"), Domain.TTS), "arktts"
            )

    def test_bundled_codec_keeps_source_precision(self):
        self.assertTrue(Model.cast_predicate("model.embeddings.weight"))
        self.assertFalse(Model.cast_predicate("codec.encoder.conv.weight"))
        self.assertTrue(Model.model_quant_predicate("model.layers.0", None))
        self.assertFalse(Model.model_quant_predicate("codec.quantizer", None))

    def test_original_codec_is_loaded_and_sanitized(self):
        tensor = MagicMock()
        tensor.detach.return_value.cpu.return_value.float.return_value.numpy.return_value = np.arange(
            24, dtype=np.float32
        ).reshape(
            2, 3, 4
        )
        torch = MagicMock()
        torch.load.return_value = {
            "state_dict": {
                "generator.encoder.conv.weight": tensor,
                "generator.freqs_cis": tensor,
                "discriminator.weight": tensor,
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            (source / "config.json").write_text('{"codec_filename": "custom.pth"}')
            (source / "custom.pth").touch()
            mx.save_safetensors(
                str(source / "model.safetensors"),
                {"embeddings.weight": mx.ones((8, 4))},
            )
            with patch.dict("sys.modules", {"torch": torch}):
                weights = load_source_weights(source)
            torch.load.assert_called_once_with(
                source / "custom.pth", map_location="cpu", weights_only=True
            )
            sanitized = Model.sanitize(Model.__new__(Model), weights)
            self.assertEqual(
                set(sanitized), {"model.embeddings.weight", "codec.encoder.conv.weight"}
            )
            np.testing.assert_array_equal(
                np.asarray(sanitized["codec.encoder.conv.weight"]),
                np.arange(24).reshape(2, 3, 4).transpose(0, 2, 1),
            )

    def test_converted_weights_do_not_need_pytorch_or_pth(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            mx.save_safetensors(
                str(source / "model.safetensors"),
                {
                    "model.embeddings.weight": mx.ones((8, 4)),
                    "codec.encoder.conv.weight": mx.ones((2, 4, 3)),
                },
            )
            with patch.dict("sys.modules", {"torch": None}):
                weights = load_source_weights(source)
            self.assertEqual(len(weights), 2)

    def test_missing_codec_reports_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            (source / "config.json").write_text("{}")
            mx.save_safetensors(
                str(source / "model.safetensors"),
                {"embeddings.weight": mx.zeros((2, 2))},
            )
            with self.assertRaisesRegex(FileNotFoundError, "codec.pth"):
                load_source_weights(source)

    def test_runtime_files_and_model_card(self):
        from huggingface_hub import ModelCard

        with tempfile.TemporaryDirectory() as directory:
            source, dest = Path(directory) / "source", Path(directory) / "output"
            source.mkdir()
            dest.mkdir()
            for name in (
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "LICENSE",
                "NOTICE",
                "modeling_arktts.py",
                "codec.pth",
                "processor_config.json",
                "generation_config.json",
            ):
                (source / name).write_text("{}")
            (source / "README.md").write_text(
                "---\nlicense: apache-2.0\nlanguage:\n- en\n- ar\n---\n"
            )
            copy_supporting_files(source, dest)
            self.assertEqual(
                {file.name for file in dest.iterdir()},
                {
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                    "LICENSE",
                    "NOTICE",
                },
            )
            write_model_card(
                dest,
                str(source),
                "local-audio8",
                {"_name_or_path": "Edge0/Audio8-TTS-Realistic-0.6b"},
            )
            card = ModelCard.load(str(dest / "README.md"))
            self.assertEqual(card.data.license, "apache-2.0")
            self.assertEqual(card.data.language, ["en", "ar"])
            self.assertEqual(card.data.base_model, "Edge0/Audio8-TTS-Realistic-0.6b")
            self.assertIn('ref_audio="reference.wav"', card.text)


if __name__ == "__main__":
    unittest.main()
