"""Tests for ACE-Step music generation model."""

import unittest

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_audio.tts.models.ace_step.config import ModelConfig, TASK_TYPES
from mlx_audio.tts.models.ace_step.modules import (
    Attention,
    DiTLayer,
    EncoderLayer,
    KVCache,
    MLP,
    RMSNorm,
    RotaryEmbedding,
    TimestepEmbedding,
    create_4d_mask,
    make_cache,
)
from mlx_audio.tts.models.ace_step.dit import DiTModel


def tiny_config(**overrides):
    """Create a small config for fast tests."""
    defaults = dict(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_lyric_encoder_hidden_layers=1,
        num_timbre_encoder_hidden_layers=1,
        num_attention_pooler_hidden_layers=1,
        in_channels=192,
        patch_size=2,
        audio_acoustic_hidden_dim=64,
        text_hidden_dim=32,
        timbre_hidden_dim=64,
        timbre_fix_frame=10,
        pool_window_size=5,
        max_position_embeddings=512,
        sliding_window=16,
        layer_types=["sliding_attention", "full_attention"],
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


class TestConfig(unittest.TestCase):
    def test_default_config(self):
        config = ModelConfig()
        self.assertEqual(config.model_type, "acestep")
        self.assertEqual(config.hidden_size, 2048)
        self.assertEqual(config.patch_size, 2)

    def test_layer_types_auto_generated(self):
        config = ModelConfig(num_hidden_layers=4)
        self.assertEqual(len(config.layer_types), 4)

    def test_task_types(self):
        self.assertIn("text2music", TASK_TYPES)
        self.assertIn("cover", TASK_TYPES)


class TestKVCache(unittest.TestCase):
    def test_empty_cache(self):
        cache = KVCache()
        self.assertFalse(cache.is_set)

    def test_update_and_fetch(self):
        cache = KVCache()
        k = mx.ones((1, 4, 10, 16))
        v = mx.ones((1, 4, 10, 16))
        cache.update_and_fetch(k, v)
        self.assertTrue(cache.is_set)
        fetched_k, fetched_v = cache.fetch()
        self.assertEqual(fetched_k.shape, k.shape)

    def test_reset(self):
        cache = KVCache()
        cache.update_and_fetch(mx.ones((1, 4, 10, 16)), mx.ones((1, 4, 10, 16)))
        cache.reset()
        self.assertFalse(cache.is_set)


class TestCreateMask(unittest.TestCase):
    def test_bidirectional_no_mask(self):
        # With no attention_mask and is_causal=False, should be all zeros
        mask = create_4d_mask(seq_len=8, dtype=mx.float32, is_causal=False)
        self.assertEqual(mask.shape, (1, 1, 8, 8))
        np.testing.assert_allclose(np.array(mask), 0.0)

    def test_causal_mask(self):
        mask = create_4d_mask(seq_len=4, dtype=mx.float32, is_causal=True)
        mask_np = np.array(mask)[0, 0]
        # Upper triangle should be -inf (large negative)
        self.assertTrue(mask_np[0, 1] < -1e8)
        # Lower triangle and diagonal should be 0
        self.assertEqual(mask_np[1, 0], 0.0)
        self.assertEqual(mask_np[0, 0], 0.0)

    def test_sliding_window_mask(self):
        mask = create_4d_mask(
            seq_len=8, dtype=mx.float32,
            sliding_window=2, is_sliding_window=True, is_causal=False,
        )
        mask_np = np.array(mask)[0, 0]
        # Adjacent positions should be visible (0)
        self.assertEqual(mask_np[3, 3], 0.0)
        self.assertEqual(mask_np[3, 4], 0.0)
        # Distant positions should be masked
        self.assertTrue(mask_np[0, 7] < -1e8)


class TestRMSNorm(unittest.TestCase):
    def test_output_shape(self):
        norm = RMSNorm(32)
        x = mx.random.normal((2, 10, 32))
        out = norm(x)
        self.assertEqual(out.shape, x.shape)


class TestTimestepEmbedding(unittest.TestCase):
    def test_output_shapes(self):
        emb = TimestepEmbedding(in_channels=64, time_embed_dim=128)
        t = mx.array([0.5, 0.8])
        temb, proj = emb(t)
        self.assertEqual(temb.shape, (2, 128))
        self.assertEqual(proj.shape, (2, 6, 128))


class TestRotaryEmbedding(unittest.TestCase):
    def test_output_shapes(self):
        rope = RotaryEmbedding(dim=16, max_position_embeddings=512)
        x = mx.random.normal((1, 10, 64))
        pos_ids = mx.arange(10)[None, :]
        cos, sin = rope(x, pos_ids)
        self.assertEqual(cos.shape[-1], 16)
        self.assertEqual(sin.shape[-1], 16)


class TestAttention(unittest.TestCase):
    def test_self_attention(self):
        attn = Attention(
            hidden_size=64, num_attention_heads=4,
            num_key_value_heads=2, head_dim=16,
        )
        x = mx.random.normal((1, 10, 64))
        rope = RotaryEmbedding(16, 512)
        pos_emb = rope(x, mx.arange(10)[None, :])
        out = attn(x, position_embeddings=pos_emb)
        self.assertEqual(out.shape, (1, 10, 64))

    def test_cross_attention_with_cache(self):
        attn = Attention(
            hidden_size=64, num_attention_heads=4,
            num_key_value_heads=2, head_dim=16,
            is_cross_attention=True,
        )
        x = mx.random.normal((1, 10, 64))
        enc = mx.random.normal((1, 20, 64))
        cache = KVCache()

        # First call populates cache
        out1 = attn(x, encoder_hidden_states=enc, cache=cache)
        self.assertTrue(cache.is_set)

        # Second call reuses cache
        out2 = attn(x, encoder_hidden_states=enc, cache=cache)
        self.assertEqual(out1.shape, out2.shape)


class TestMLP(unittest.TestCase):
    def test_output_shape(self):
        mlp = MLP(hidden_size=64, intermediate_size=128)
        x = mx.random.normal((1, 10, 64))
        out = mlp(x)
        self.assertEqual(out.shape, (1, 10, 64))


class TestDiTLayer(unittest.TestCase):
    def test_forward(self):
        layer = DiTLayer(
            hidden_size=64, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        )
        x = mx.random.normal((1, 10, 64))
        enc = mx.random.normal((1, 20, 64))
        rope = RotaryEmbedding(16, 512)
        pos_emb = rope(x, mx.arange(10)[None, :])
        # timestep_proj: [batch, 6, hidden_size]
        tp = mx.random.normal((1, 6, 64))

        out = layer(x, pos_emb, tp, encoder_hidden_states=enc)
        self.assertEqual(out.shape, (1, 10, 64))


class TestDiTModel(unittest.TestCase):
    def test_forward(self):
        config = tiny_config()
        dit = DiTModel(config)

        B, T = 1, 10
        hidden = mx.random.normal((B, T, 64))
        enc = mx.random.normal((B, 20, 64))
        ctx = mx.random.normal((B, T, 128))
        t = mx.array([0.5])

        out = dit(
            hidden_states=hidden, timestep=t, timestep_r=t,
            attention_mask=None, encoder_hidden_states=enc,
            encoder_attention_mask=None, context_latents=ctx,
        )
        self.assertEqual(out.shape, (B, T, 64))

    def test_with_cache(self):
        config = tiny_config()
        dit = DiTModel(config)
        caches = make_cache(config.num_hidden_layers)

        B, T = 1, 10
        hidden = mx.random.normal((B, T, 64))
        enc = mx.random.normal((B, 20, 64))
        ctx = mx.random.normal((B, T, 128))
        t = mx.array([0.5])

        # First call populates cache
        out1 = dit(
            hidden_states=hidden, timestep=t, timestep_r=t,
            attention_mask=None, encoder_hidden_states=enc,
            encoder_attention_mask=None, context_latents=ctx,
            cache=caches,
        )
        # Second call reuses cache
        out2 = dit(
            hidden_states=hidden, timestep=t, timestep_r=t,
            attention_mask=None, encoder_hidden_states=enc,
            encoder_attention_mask=None, context_latents=ctx,
            cache=caches,
        )
        self.assertEqual(out1.shape, out2.shape)

    def test_conv_transpose1d_matches_non_overlapping(self):
        """Verify transposed conv produces correct output length."""
        config = tiny_config()
        dit = DiTModel(config)

        B, T = 1, 6
        x = mx.random.normal((B, T, 64))
        out = dit._conv_transpose1d_forward(
            x, dit.proj_out_weight, dit.proj_out_bias, config.patch_size
        )
        self.assertEqual(out.shape, (B, T * config.patch_size, 64))


class TestModelSanitize(unittest.TestCase):
    def test_dit_prefix_remap(self):
        """Legacy 'dit.' prefix should be remapped to 'decoder.'."""
        from mlx_audio.tts.models.ace_step.ace_step import Model

        weights = {"dit.layers.0.self_attn.q_proj.weight": mx.ones((64, 64))}
        sanitized = Model.sanitize(weights)
        self.assertIn("decoder.layers.0.self_attn.q_proj.weight", sanitized)
        self.assertNotIn("dit.layers.0.self_attn.q_proj.weight", sanitized)

    def test_proj_in_rename_scoped_to_decoder(self):
        """proj_in/proj_out renaming should only apply to decoder prefix."""
        from mlx_audio.tts.models.ace_step.ace_step import Model

        weights = {
            "decoder.proj_in.weight": mx.ones((64, 2, 192)),
            "detokenizer.proj_out.weight": mx.ones((64, 32)),
        }
        sanitized = Model.sanitize(weights)
        # Decoder proj_in should be renamed to underscore
        self.assertIn("decoder.proj_in_weight", sanitized)
        # Detokenizer proj_out should NOT be renamed
        self.assertIn("detokenizer.proj_out.weight", sanitized)

    def test_scale_shift_table_unsqueeze(self):
        """2D scale_shift_table should be expanded to 3D."""
        from mlx_audio.tts.models.ace_step.ace_step import Model

        weights = {"decoder.scale_shift_table": mx.ones((6, 64))}
        sanitized = Model.sanitize(weights)
        self.assertEqual(sanitized["decoder.scale_shift_table"].shape, (1, 6, 64))


if __name__ == "__main__":
    unittest.main()
