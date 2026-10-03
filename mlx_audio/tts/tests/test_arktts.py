"""Tests for the arktts (Audio8-TTS) model: sanitize contract + tiny-model forward."""

import unittest

import mlx.core as mx
import numpy as np

from mlx_audio.tts.models.arktts import Model, ModelConfig
from mlx_audio.tts.models.arktts.arktts import ArkttsModel


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


if __name__ == "__main__":
    unittest.main()
