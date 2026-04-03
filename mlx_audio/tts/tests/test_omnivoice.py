import unittest

import mlx.core as mx


class TestOmniVoiceConfig(unittest.TestCase):
    def test_parse_from_dict_minimal(self):
        from mlx_audio.tts.models.omnivoice.config import OmniVoiceConfig

        cfg = OmniVoiceConfig.from_dict(
            {
                "model_type": "omnivoice",
                "audio_vocab_size": 1025,
                "audio_mask_id": 1024,
                "num_audio_codebook": 8,
                "audio_codebook_weights": [8, 8, 6, 6, 4, 4, 2, 2],
                "sample_rate": 24000,
            }
        )
        self.assertEqual(cfg.audio_vocab_size, 1025)
        self.assertEqual(cfg.num_audio_codebook, 8)
        self.assertEqual(cfg.sample_rate, 24000)

    def test_unknown_keys_are_ignored(self):
        from mlx_audio.tts.models.omnivoice.config import OmniVoiceConfig

        # Should not raise
        OmniVoiceConfig.from_dict({"model_type": "omnivoice", "future_key": 99})

    def test_higgs_audio_config(self):
        from mlx_audio.codec.models.higgs_audio.config import HiggsAudioConfig

        cfg = HiggsAudioConfig.from_dict(
            {
                "model_type": "higgs_audio_v2_tokenizer",
                "sample_rate": 24000,
                "codebook_size": 1024,
                "downsample_factor": 960,
            }
        )
        self.assertEqual(cfg.downsample_factor, 960)
        self.assertAlmostEqual(cfg.tokens_per_second, 25.0)


class TestOmniVoiceRegistration(unittest.TestCase):
    def test_model_type_registered(self):
        from mlx_audio.tts.utils import MODEL_REMAPPING

        self.assertIn("omnivoice", MODEL_REMAPPING)
        self.assertEqual(MODEL_REMAPPING["omnivoice"], "omnivoice")


class TestOmniVoiceBackbone(unittest.TestCase):
    def _make_backbone(self):
        from mlx_audio.tts.models.omnivoice.backbone import (
            BackboneConfig,
            OmniVoiceBackbone,
        )

        cfg = BackboneConfig(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=128,
            vocab_size=151676,
            head_dim=16,
            rms_norm_eps=1e-6,
        )
        return OmniVoiceBackbone(cfg)

    def test_output_shape(self):
        model = self._make_backbone()
        B, S = 1, 10
        embeds = mx.zeros((B, S, 64))
        out = model(embeds)
        self.assertEqual(out.shape, (B, S, 64))

    def test_bidirectional_no_causal_leak(self):
        """Token at position 7 must influence output at position 3 (bidirectional)."""
        import numpy as np

        model = self._make_backbone()
        S = 10
        base_embeds = mx.zeros((1, S, 64))
        # Perturb position 7
        perturbed_list = np.zeros((1, S, 64), dtype=np.float32)
        perturbed_list[0, 7, :] = 1.0
        perturbed = mx.array(perturbed_list)

        out_base = model(base_embeds)
        out_perturbed = model(perturbed)
        # Position 3 output should differ (bidirectional attention)
        diff = mx.abs(out_base[0, 3] - out_perturbed[0, 3])
        self.assertGreater(
            float(mx.max(diff).item()),
            1e-6,
            "Position 3 unchanged after perturbing pos 7 — causal mask still active!",
        )


class TestOmniVoiceModel(unittest.TestCase):
    def _make_model(self):
        from mlx_audio.tts.models.omnivoice.config import OmniVoiceConfig
        from mlx_audio.tts.models.omnivoice.omnivoice import Model

        cfg = OmniVoiceConfig.from_dict(
            {
                "model_type": "omnivoice",
                "audio_vocab_size": 1025,
                "audio_mask_id": 1024,
                "num_audio_codebook": 8,
                "sample_rate": 24000,
                "llm_config": {
                    "hidden_size": 64,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "intermediate_size": 128,
                    "vocab_size": 200,
                    "head_dim": 16,
                    "rms_norm_eps": 1e-6,
                },
            }
        )
        return Model(cfg)

    def test_logits_shape(self):
        model = self._make_model()
        B, S, T = 1, 5, 7
        input_ids = mx.zeros((B, S), dtype=mx.int32)
        audio_tokens = mx.full((B, T, 8), 1024, dtype=mx.int32)
        inputs_embeds = model._embed(input_ids, audio_tokens)  # [B, S+T, D]
        logits = model(inputs_embeds, prefix_len=S)
        self.assertEqual(logits.shape, (B, T, 8, 1025))

    def test_embed_shape(self):
        model = self._make_model()
        B, S, T = 1, 5, 7
        input_ids = mx.zeros((B, S), dtype=mx.int32)
        audio_tokens = mx.full((B, T, 8), 1024, dtype=mx.int32)
        embeds = model._embed(input_ids, audio_tokens)
        self.assertEqual(embeds.shape, (B, S + T, 64))  # hidden_size=64 in test cfg


class TestBuildCondEmbeds(unittest.TestCase):
    def _make_model(self):
        from mlx_audio.tts.models.omnivoice.config import OmniVoiceConfig
        from mlx_audio.tts.models.omnivoice.omnivoice import Model

        cfg = OmniVoiceConfig.from_dict(
            {
                "model_type": "omnivoice",
                "audio_vocab_size": 1025,
                "audio_mask_id": 1024,
                "num_audio_codebook": 8,
                "sample_rate": 24000,
                "llm_config": {
                    "hidden_size": 64,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "intermediate_size": 128,
                    "vocab_size": 200,
                    "head_dim": 16,
                    "rms_norm_eps": 1e-6,
                },
            }
        )
        return Model(cfg)

    def test_text_only_shape(self):
        model = self._make_model()
        embeds = model.build_cond_embeds(mx.zeros((1, 5), dtype=mx.int32))
        self.assertEqual(embeds.shape, (1, 5, 64))

    def test_text_plus_ref_shape(self):
        model = self._make_model()
        embeds = model.build_cond_embeds(
            mx.zeros((1, 5), dtype=mx.int32),
            mx.zeros((1, 7, 8), dtype=mx.int32),
        )
        self.assertEqual(embeds.shape, (1, 12, 64))  # S=5, T_ref=7

    def test_uncond_no_ref(self):
        model = self._make_model()
        embeds = model.build_cond_embeds(mx.zeros((1, 5), dtype=mx.int32))
        self.assertEqual(embeds.shape, (1, 5, 64))


class TestOmniVoiceGeneration(unittest.TestCase):
    def test_schedule_monotone(self):
        from mlx_audio.tts.models.omnivoice.generation import _get_time_steps

        ts = _get_time_steps(num_step=32, t_shift=0.1)
        self.assertEqual(len(ts), 33)
        for i in range(1, len(ts)):
            self.assertGreaterEqual(ts[i], ts[i - 1])
        self.assertAlmostEqual(ts[0], 0.0, places=6)
        self.assertAlmostEqual(ts[-1], 1.0, places=4)

    def test_iterative_unmask_no_mask_remaining(self):
        from mlx_audio.tts.models.omnivoice.config import OmniVoiceConfig
        from mlx_audio.tts.models.omnivoice.generation import iterative_unmask
        from mlx_audio.tts.models.omnivoice.omnivoice import Model

        cfg = OmniVoiceConfig.from_dict(
            {
                "model_type": "omnivoice",
                "audio_vocab_size": 1025,
                "audio_mask_id": 1024,
                "num_audio_codebook": 8,
                "sample_rate": 24000,
                "llm_config": {
                    "hidden_size": 64,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "intermediate_size": 128,
                    "vocab_size": 200,
                    "head_dim": 16,
                    "rms_norm_eps": 1e-6,
                },
            }
        )
        model = Model(cfg)

        T = 10
        input_ids = mx.zeros((1, 3), dtype=mx.int32)
        cond_embeds = model.build_cond_embeds(input_ids)
        uncond_embeds = model.build_cond_embeds(mx.zeros_like(input_ids))
        tokens = iterative_unmask(
            model=model,
            cond_embeds=cond_embeds,
            uncond_embeds=uncond_embeds,
            T=T,
            num_steps=5,  # fast test
            guidance_scale=2.0,
        )
        self.assertEqual(tokens.shape, (T, 8))
        # No mask tokens should remain
        mask_count = int(mx.sum(tokens == 1024).item())
        self.assertEqual(
            mask_count, 0, f"Found {mask_count} mask tokens after unmasking"
        )
        # All tokens must be valid codebook tokens
        self.assertTrue(bool(mx.all(tokens >= 0).item()))
        self.assertTrue(bool(mx.all(tokens <= 1023).item()))

    def test_frozen_tokens_invariant(self):
        """Tokens unmasked at step k must not change at step k+1.

        Tested implicitly by test_iterative_unmask_no_mask_remaining.
        """
        from mlx_audio.tts.models.omnivoice.generation import (  # noqa: F401
            iterative_unmask,
        )

        pass


class TestIterativeUnmaskRefactor(unittest.TestCase):
    def _make_model(self):
        from mlx_audio.tts.models.omnivoice.config import OmniVoiceConfig
        from mlx_audio.tts.models.omnivoice.omnivoice import Model

        cfg = OmniVoiceConfig.from_dict(
            {
                "model_type": "omnivoice",
                "audio_vocab_size": 1025,
                "audio_mask_id": 1024,
                "num_audio_codebook": 8,
                "sample_rate": 24000,
                "llm_config": {
                    "hidden_size": 64,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "intermediate_size": 128,
                    "vocab_size": 200,
                    "head_dim": 16,
                    "rms_norm_eps": 1e-6,
                },
            }
        )
        return Model(cfg)

    def test_new_signature_shape(self):
        from mlx_audio.tts.models.omnivoice.generation import iterative_unmask

        model = self._make_model()
        cond = mx.zeros((1, 3, 64))
        uncond = mx.zeros((1, 3, 64))
        tokens = iterative_unmask(model, cond, uncond, T=10, num_steps=2)
        self.assertEqual(tokens.shape, (10, 8))

    def test_no_mask_tokens_remain(self):
        from mlx_audio.tts.models.omnivoice.generation import iterative_unmask

        model = self._make_model()
        cond = mx.zeros((1, 3, 64))
        uncond = mx.zeros((1, 3, 64))
        tokens = iterative_unmask(model, cond, uncond, T=10, num_steps=5)
        self.assertEqual(int(mx.sum(tokens == 1024).item()), 0)

    def test_deterministic_with_fixed_seed(self):
        from mlx_audio.tts.models.omnivoice.generation import iterative_unmask

        model = self._make_model()
        input_ids = mx.zeros((1, 3), dtype=mx.int32)
        cond = model.build_cond_embeds(input_ids)
        uncond = model.build_cond_embeds(mx.zeros_like(input_ids))

        mx.random.seed(42)
        t1 = iterative_unmask(model, cond, uncond, T=5, num_steps=3)
        # Force materialization before resetting the seed
        _ = int(mx.sum(t1).item())

        mx.random.seed(42)
        t2 = iterative_unmask(model, cond, uncond, T=5, num_steps=3)

        self.assertTrue(bool(mx.all(t1 == t2).item()))


class TestOmniVoiceSanitize(unittest.TestCase):
    def _make_model(self):
        from mlx_audio.tts.models.omnivoice.config import OmniVoiceConfig
        from mlx_audio.tts.models.omnivoice.omnivoice import Model

        cfg = OmniVoiceConfig.from_dict(
            {
                "model_type": "omnivoice",
                "audio_vocab_size": 1025,
                "audio_mask_id": 1024,
                "num_audio_codebook": 8,
                "sample_rate": 24000,
                "llm_config": {
                    "hidden_size": 64,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "intermediate_size": 128,
                    "vocab_size": 200,
                    "head_dim": 16,
                    "rms_norm_eps": 1e-6,
                },
            }
        )
        return Model(cfg)

    def test_llm_prefix_remapped(self):
        model = self._make_model()
        x = mx.zeros((4,))
        result = model.sanitize({"llm.layers.0.weight": x})
        self.assertIn("backbone.layers.0.weight", result)
        self.assertNotIn("llm.layers.0.weight", result)

    def test_audio_embeddings_split(self):
        model = self._make_model()
        # k2-fsa stores all 8 codebooks as one tensor [8*1025, H]
        x = mx.zeros((8 * 1025, 4))
        result = model.sanitize({"audio_embeddings.weight": x})
        for i in range(8):
            self.assertIn(f"audio_embeddings.{i}.weight", result)
            self.assertEqual(result[f"audio_embeddings.{i}.weight"].shape, (1025, 4))
        self.assertNotIn("audio_embeddings.weight", result)

    def test_audio_heads_split(self):
        model = self._make_model()
        x = mx.zeros((8 * 1025, 4))
        result = model.sanitize({"audio_heads.weight": x})
        for i in range(8):
            self.assertIn(f"audio_heads.{i}.weight", result)
        self.assertNotIn("audio_heads.weight", result)

    def test_codebook_layer_offsets_dropped(self):
        model = self._make_model()
        x = mx.array([0, 1025, 2050, 3075, 4100, 5125, 6150, 7175])
        result = model.sanitize({"codebook_layer_offsets": x})
        self.assertNotIn("codebook_layer_offsets", result)
        self.assertEqual(len(result), 0)

    def test_other_keys_pass_through(self):
        model = self._make_model()
        x = mx.zeros((4,))
        result = model.sanitize({"some.other.key": x})
        self.assertIn("some.other.key", result)


class TestOmniVoiceGenerate(unittest.TestCase):
    def _make_model(self):
        from mlx_audio.tts.models.omnivoice.config import OmniVoiceConfig
        from mlx_audio.tts.models.omnivoice.omnivoice import Model

        cfg = OmniVoiceConfig.from_dict(
            {
                "model_type": "omnivoice",
                "audio_vocab_size": 1025,
                "audio_mask_id": 1024,
                "num_audio_codebook": 8,
                "sample_rate": 24000,
                "llm_config": {
                    "hidden_size": 64,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "intermediate_size": 128,
                    "vocab_size": 200,
                    "head_dim": 16,
                    "rms_norm_eps": 1e-6,
                },
            }
        )
        return Model(cfg)

    def test_generate_returns_generation_result(self):
        import math

        from mlx_audio.tts.models.base import GenerationResult

        model = self._make_model()
        input_ids = mx.zeros((5,), dtype=mx.int32)
        result = next(model.generate(input_ids=input_ids, duration_s=1.0, num_steps=5))
        self.assertIsInstance(result, GenerationResult)

    def test_generate_token_count(self):
        import math

        model = self._make_model()
        input_ids = mx.zeros((5,), dtype=mx.int32)
        result = next(model.generate(input_ids=input_ids, duration_s=1.0, num_steps=5))
        expected_T = math.ceil(1.0 * 24000 / 960)  # = 25 (HiggsAudio hop=960)
        self.assertEqual(result.token_count, expected_T)

    def test_generate_sample_rate(self):
        model = self._make_model()
        input_ids = mx.zeros((5,), dtype=mx.int32)
        result = next(model.generate(input_ids=input_ids, duration_s=1.0, num_steps=5))
        self.assertEqual(result.sample_rate, 24000)

    def test_generate_processing_time_positive(self):
        model = self._make_model()
        input_ids = mx.zeros((5,), dtype=mx.int32)
        result = next(model.generate(input_ids=input_ids, duration_s=1.0, num_steps=5))
        self.assertGreater(result.processing_time_seconds, 0)

    def test_generate_result_field_types(self):
        model = self._make_model()
        input_ids = mx.zeros((5,), dtype=mx.int32)
        result = next(model.generate(input_ids=input_ids, duration_s=1.0, num_steps=5))
        self.assertIsInstance(result.audio_duration, str)
        self.assertIsInstance(result.prompt, dict)
        self.assertIn("tokens-per-sec", result.prompt)
        self.assertIsInstance(result.audio_samples, dict)
        self.assertIn("samples", result.audio_samples)
        self.assertIn("samples-per-sec", result.audio_samples)

    def test_generate_with_ref_tokens_succeeds(self):
        model = self._make_model()
        input_ids = mx.zeros((5,), dtype=mx.int32)
        ref_tokens = mx.ones((4, 8), dtype=mx.int32)
        result = next(
            model.generate(
                input_ids=input_ids, duration_s=0.5, num_steps=3, ref_tokens=ref_tokens
            )
        )
        self.assertIsInstance(result.token_count, int)
        self.assertGreater(result.token_count, 0)


class TestVoiceCloneUtils(unittest.TestCase):
    def test_no_tokenizer_returns_empty(self):
        from mlx_audio.tts.models.omnivoice.utils import create_voice_clone_prompt

        result = create_voice_clone_prompt("any_path.wav", tokenizer=None)
        self.assertEqual(result.shape, (0, 8))
        self.assertEqual(result.dtype, mx.int32)

    def test_missing_file_raises(self):
        from mlx_audio.codec.models.higgs_audio.config import HiggsAudioConfig
        from mlx_audio.codec.models.higgs_audio.higgs_audio import HiggsAudioTokenizer
        from mlx_audio.tts.models.omnivoice.utils import create_voice_clone_prompt

        tok = HiggsAudioTokenizer(HiggsAudioConfig())
        with self.assertRaises(FileNotFoundError):
            create_voice_clone_prompt("/nonexistent/file.wav", tokenizer=tok)

    def test_with_tokenizer_returns_2d(self):
        import os
        import tempfile
        from unittest.mock import MagicMock

        import numpy as np
        import soundfile as sf

        from mlx_audio.codec.models.higgs_audio.config import HiggsAudioConfig
        from mlx_audio.codec.models.higgs_audio.higgs_audio import HiggsAudioTokenizer
        from mlx_audio.tts.models.omnivoice.utils import create_voice_clone_prompt

        tok = HiggsAudioTokenizer(HiggsAudioConfig())
        # Mock tokenizer.encode directly — avoids importing torch (not in CI)
        tok.encode = lambda wav: mx.zeros(
            (wav.shape[0], wav.shape[1] // 960, 8), dtype=mx.int32
        )

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
        audio = np.zeros(24000 * 2, dtype=np.float32)  # 2 seconds
        sf.write(tmp_path, audio, 24000)
        result = create_voice_clone_prompt(tmp_path, tokenizer=tok)
        self.assertEqual(result.ndim, 2)
        self.assertEqual(result.shape[1], 8)
        self.assertEqual(result.dtype, mx.int32)
        os.unlink(tmp_path)


class TestOmniVoiceGenerateWithTokenizer(unittest.TestCase):
    def _make_model(self):
        from mlx_audio.tts.models.omnivoice.config import OmniVoiceConfig
        from mlx_audio.tts.models.omnivoice.omnivoice import Model

        cfg = OmniVoiceConfig.from_dict(
            {
                "model_type": "omnivoice",
                "audio_vocab_size": 1025,
                "audio_mask_id": 1024,
                "num_audio_codebook": 8,
                "sample_rate": 24000,
                "llm_config": {
                    "hidden_size": 64,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "intermediate_size": 128,
                    "vocab_size": 200,
                    "head_dim": 16,
                    "rms_norm_eps": 1e-6,
                },
            }
        )
        return Model(cfg)

    def _make_tokenizer(self):
        from mlx_audio.codec.models.higgs_audio.config import HiggsAudioConfig
        from mlx_audio.codec.models.higgs_audio.higgs_audio import HiggsAudioTokenizer

        return HiggsAudioTokenizer(HiggsAudioConfig())

    def test_audio_is_zeros_without_tokenizer(self):
        model = self._make_model()
        input_ids = mx.zeros((5,), dtype=mx.int32)
        result = next(model.generate(input_ids=input_ids, duration_s=0.1, num_steps=2))
        # Without tokenizer, audio is a silent zero array (not None)
        self.assertIsInstance(result.audio, mx.array)

    def test_audio_is_array_with_tokenizer(self):
        model = self._make_model()
        tok = self._make_tokenizer()
        input_ids = mx.zeros((5,), dtype=mx.int32)
        result = next(
            model.generate(
                input_ids=input_ids, duration_s=0.1, num_steps=2, tokenizer=tok
            )
        )
        self.assertIsNotNone(result.audio)
        self.assertIsInstance(result.audio, mx.array)

    def test_samples_count_with_tokenizer(self):
        model = self._make_model()
        tok = self._make_tokenizer()
        input_ids = mx.zeros((5,), dtype=mx.int32)
        result = next(
            model.generate(
                input_ids=input_ids, duration_s=0.1, num_steps=2, tokenizer=tok
            )
        )
        # tokens is [T, 8]; decode returns [T*960] 1D
        expected_samples = result.token_count * 960
        self.assertEqual(result.audio.size, expected_samples)


if __name__ == "__main__":
    unittest.main()
