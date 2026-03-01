import unittest

import mlx.core as mx

from mlx_audio.tts.models.echo_tts.audio import PCAState
from mlx_audio.tts.models.echo_tts.config import EchoDiTConfig
from mlx_audio.tts.models.echo_tts.config import ModelConfig as EchoModelConfig
from mlx_audio.tts.models.echo_tts.config import SamplerConfig
from mlx_audio.tts.models.echo_tts.echo_tts import Model as EchoModel
from mlx_audio.tts.models.echo_tts.model import EchoDiT
from mlx_audio.tts.models.echo_tts.text import (
    get_text_input_ids_and_mask,
    normalize_text_prompt,
    tokenizer_encode,
)


class TestEchoTTS(unittest.TestCase):
    def test_text_normalization(self):
        text = "Hello: world\nnew line"
        normalized = normalize_text_prompt(text)
        self.assertTrue(normalized.startswith("[S1] "))
        self.assertIn(",", normalized)
        self.assertNotIn("\n", normalized)

    def test_tokenizer_encode(self):
        tokens = tokenizer_encode("hello", append_bos=True, normalize=False)
        self.assertEqual(int(tokens.shape[0]), 6)  # BOS + 5 bytes
        self.assertEqual(int(tokens[0]), 0)

    def test_text_input_ids_and_mask(self):
        ids, mask, normalized = get_text_input_ids_and_mask(
            ["hello", "world"],
            max_length=10,
            normalize=True,
            return_normalized_text=True,
            pad_to_max=True,
        )
        self.assertEqual(tuple(ids.shape), (2, 10))
        self.assertEqual(tuple(mask.shape), (2, 10))
        self.assertEqual(len(normalized), 2)

    def test_echo_dit_forward_shapes(self):
        config = EchoDiTConfig(
            latent_size=8,
            model_size=32,
            num_layers=2,
            num_heads=4,
            intermediate_size=64,
            norm_eps=1e-5,
            text_vocab_size=256,
            text_model_size=32,
            text_num_layers=1,
            text_num_heads=4,
            text_intermediate_size=64,
            speaker_patch_size=2,
            speaker_model_size=32,
            speaker_num_layers=1,
            speaker_num_heads=4,
            speaker_intermediate_size=64,
            timestep_embed_size=16,
            adaln_rank=8,
        )
        model = EchoDiT(**config.__dict__)

        x = mx.random.normal((1, 6, config.latent_size))
        t = mx.array([0.7], dtype=mx.float32)
        text_input_ids = mx.array([[0, 1, 2, 3, 4]], dtype=mx.int32)
        text_mask = mx.array([[True, True, True, True, True]], dtype=mx.bool_)
        speaker_latent = mx.random.normal((1, 8, config.latent_size))
        speaker_mask = mx.ones((1, 8), dtype=mx.bool_)

        kv_text = model.get_kv_cache_text(text_input_ids, text_mask)
        kv_speaker = model.get_kv_cache_speaker(speaker_latent)

        y = model(
            x=x,
            t=t,
            text_mask=text_mask,
            speaker_mask=speaker_mask,
            kv_cache_text=kv_text,
            kv_cache_speaker=kv_speaker,
        )
        self.assertEqual(tuple(y.shape), (1, 6, config.latent_size))

    def test_sanitize_and_generate_smoke(self):
        class FakeFishAE:
            def decode_zq(self, z_q):
                b, _, t = z_q.shape
                return mx.zeros((b, 1, t * 2048), dtype=mx.float32)

        cfg = EchoModelConfig(
            dit=EchoDiTConfig(
                latent_size=8,
                model_size=32,
                num_layers=2,
                num_heads=4,
                intermediate_size=64,
                norm_eps=1e-5,
                text_vocab_size=256,
                text_model_size=32,
                text_num_layers=1,
                text_num_heads=4,
                text_intermediate_size=64,
                speaker_patch_size=2,
                speaker_model_size=32,
                speaker_num_layers=1,
                speaker_num_heads=4,
                speaker_intermediate_size=64,
                timestep_embed_size=16,
                adaln_rank=8,
            ),
            sampler=SamplerConfig(
                num_steps=1,
                cfg_scale_text=1.0,
                cfg_scale_speaker=1.0,
                sequence_length=4,
            ),
        )
        model = EchoModel(cfg)
        model.fish_ae = FakeFishAE()
        model.pca_state = PCAState(
            pca_components=mx.eye(8, dtype=mx.float32),
            pca_mean=mx.zeros((8,), dtype=mx.float32),
            latent_scale=1.0,
        )

        sanitized = model.sanitize(
            {
                "cond_module.0.weight": mx.zeros((1, 1), dtype=mx.float32),
                "pca_components": mx.zeros((1,), dtype=mx.float32),
            }
        )
        self.assertIn("model.cond_module.layers.0.weight", sanitized)
        self.assertNotIn("model.pca_components", sanitized)

        result = next(model.generate("hi", rng_seed=0))
        self.assertEqual(result.sample_rate, 44100)
        self.assertTrue(result.samples > 0)

    def test_delete_blockwise_modules(self):
        cfg = EchoModelConfig(
            delete_blockwise_modules=True,
            dit=EchoDiTConfig(
                latent_size=8,
                model_size=32,
                num_layers=2,
                num_heads=4,
                intermediate_size=64,
                norm_eps=1e-5,
                text_vocab_size=256,
                text_model_size=32,
                text_num_layers=1,
                text_num_heads=4,
                text_intermediate_size=64,
                speaker_patch_size=2,
                speaker_model_size=32,
                speaker_num_layers=1,
                speaker_num_heads=4,
                speaker_intermediate_size=64,
                timestep_embed_size=16,
                adaln_rank=8,
            ),
            sampler=SamplerConfig(num_steps=1, sequence_length=4),
        )
        model = EchoModel(cfg)

        sanitized = model.sanitize(
            {
                "latent_encoder.in_proj.weight": mx.zeros((1, 1), dtype=mx.float32),
                "blocks.0.attention.wk_latent.weight": mx.zeros(
                    (1, 1), dtype=mx.float32
                ),
                "blocks.0.attention.wv_latent.weight": mx.zeros(
                    (1, 1), dtype=mx.float32
                ),
                "out_proj.weight": mx.zeros((8, 32), dtype=mx.float32),
            }
        )
        self.assertIn("model.out_proj.weight", sanitized)
        self.assertFalse(any("latent_encoder" in k for k in sanitized))
        self.assertFalse(any("wk_latent" in k for k in sanitized))
        self.assertFalse(any("wv_latent" in k for k in sanitized))

        with self.assertRaises(ValueError):
            model.model.get_kv_cache_latent(mx.zeros((1, 0, 8), dtype=mx.float32))

        with self.assertRaises(ValueError):
            model.generate_latents("hi", block_sizes=[2], rng_seed=0)


if __name__ == "__main__":
    unittest.main()
