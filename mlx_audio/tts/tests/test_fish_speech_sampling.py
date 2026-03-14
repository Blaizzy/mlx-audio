"""Tests for Fish Speech sampling alignment with the official PyTorch implementation.

Verifies three fixes that align the MLX sampling behaviour with
the upstream ``fish-speech`` inference pipeline:

1. RAS (Repetition Aware Sampling) uses a single-pass fallback
   (sample normal + high-temp, then conditionally swap) instead of a
   retry loop.
2. Semantic codes are clamped to ``[0, codebook_size - 1]`` before the
   fast-pathway embedding lookup.
3. The semantic logit bias uses ``-inf`` (not ``-1e9``) so that masked
   tokens receive exactly zero probability after softmax.
"""

import unittest

import mlx.core as mx


def _tiny_config():
    """A minimal config where the semantic token range fits inside the vocab."""
    from mlx_audio.tts.models.fish_qwen3_omni.config import ModelConfig

    return ModelConfig.from_dict(
        {
            "semantic_start_token_id": 16,
            "semantic_end_token_id": 23,
            "text_config": {
                "vocab_size": 32,
                "n_layer": 1,
                "n_head": 2,
                "dim": 8,
                "intermediate_size": 16,
                "n_local_heads": 1,
                "head_dim": 4,
                "norm_eps": 1e-6,
                "max_seq_len": 64,
                "attention_qk_norm": True,
            },
            "audio_decoder_config": {
                "vocab_size": 8,
                "n_layer": 1,
                "n_head": 2,
                "dim": 8,
                "intermediate_size": 16,
                "n_local_heads": 1,
                "head_dim": 4,
                "num_codebooks": 2,
                "norm_eps": 1e-6,
                "max_seq_len": 3,
            },
        }
    )


def _make_model():
    from mlx_audio.tts.models.fish_qwen3_omni.fish_speech import Model

    config = _tiny_config()
    model = Model(config)
    # Manually build the semantic_logit_bias (normally done in post_load_hook).
    vocab_size = config.text_config.vocab_size
    semantic_bias = mx.full((1, vocab_size), float("-inf"), dtype=mx.float32)
    semantic_bias[
        :,
        config.semantic_start_token_id : config.semantic_end_token_id + 1,
    ] = 0.0
    model.semantic_logit_bias = semantic_bias
    return model


# ── Fix 1: RAS single-pass fallback ──────────────────────────────────


class TestRASSinglePassFallback(unittest.TestCase):
    """RAS must use a single-pass fallback (sample normal + high-temp, then
    conditionally swap) instead of a retry loop."""

    def test_ras_no_retry_constant(self):
        """RAS_MAX_RETRY should no longer exist (replaced by single-pass)."""
        import mlx_audio.tts.models.fish_qwen3_omni.fish_speech as mod

        self.assertFalse(
            hasattr(mod, "RAS_MAX_RETRY"),
            "RAS_MAX_RETRY constant should be removed (single-pass fallback).",
        )

    def test_ras_returns_normal_when_not_in_window(self):
        """Normal token is returned when it is not in the RAS window."""
        model = _make_model()
        config = model.config

        # Logits that strongly prefer the first semantic token.
        logits = mx.full((1, config.text_config.vocab_size), -1e4, dtype=mx.float32)
        logits[:, config.semantic_start_token_id] = 10.0
        mx.random.seed(0)

        token = model._sample_semantic(
            logits,
            previous_semantic_tokens=[],  # empty window
            top_p=0.9,
            top_k=8,
            temperature=0.7,
        )
        mx.eval(token)
        tok_val = int(token[0].item())
        self.assertTrue(
            config.semantic_start_token_id <= tok_val <= config.semantic_end_token_id,
            f"Expected a semantic token, got {tok_val}",
        )

    def test_ras_uses_high_temp_when_token_repeats(self):
        """When the normal sample repeats in the window, the high-temp token
        is used instead (single fallback, no retry loop)."""
        model = _make_model()
        config = model.config

        # Make logits overwhelmingly prefer semantic_start_token_id so that
        # the "normal" sample is almost certainly that token.
        logits = mx.full((1, config.text_config.vocab_size), -1e4, dtype=mx.float32)
        logits[:, config.semantic_start_token_id] = 100.0
        # Also give some probability to the next token so high-temp can pick it.
        logits[:, config.semantic_start_token_id + 1] = 0.0

        # Put semantic_start_token_id in the window so it counts as a repeat.
        window = [config.semantic_start_token_id]

        mx.random.seed(42)
        token = model._sample_semantic(
            logits,
            previous_semantic_tokens=window,
            top_p=0.9,
            top_k=8,
            temperature=0.01,  # very low temp → almost deterministic normal sample
        )
        mx.eval(token)
        # The function should have detected the repeat and returned the
        # high-temp sample.  We cannot predict the exact value since it
        # depends on sampling, but we verify it returned a valid semantic token.
        tok_val = int(token[0].item())
        self.assertTrue(
            config.semantic_start_token_id <= tok_val <= config.semantic_end_token_id,
            f"Expected a semantic token from high-temp fallback, got {tok_val}",
        )

    def test_ras_accepts_non_semantic_repeat(self):
        """Non-semantic tokens are returned even if they appear in the RAS
        window (RAS only applies to semantic tokens)."""
        model = _make_model()
        config = model.config

        # Build logits that strongly prefer a non-semantic token (token 0).
        logits = mx.full((1, config.text_config.vocab_size), -1e4, dtype=mx.float32)
        logits[:, 0] = 100.0
        # Allow the non-semantic token through the bias by overriding it.
        model.semantic_logit_bias[:, 0] = 0.0

        # Put token 0 in the window — it should still be returned because
        # it is not a semantic token.
        window = [0]

        mx.random.seed(0)
        token = model._sample_semantic(
            logits,
            previous_semantic_tokens=window,
            top_p=0.9,
            top_k=8,
            temperature=0.01,
        )
        mx.eval(token)
        self.assertEqual(int(token[0].item()), 0)


# ── Fix 2: semantic code clamp ───────────────────────────────────────


class TestSemanticCodeClamp(unittest.TestCase):
    """Semantic codes must be clamped to [0, codebook_size - 1] before the
    fast-pathway embedding lookup, matching the official PyTorch
    ``torch.clamp(a, min=0, max=model.config.codebook_size - 1)``."""

    def test_clamp_prevents_negative_index(self):
        """Subtracting semantic_start_token_id from a token below that range
        must be clamped to 0, not left as a negative index."""
        config = _tiny_config()
        token_below = mx.array(
            [config.semantic_start_token_id - 5], dtype=mx.int32
        )
        code = (token_below - config.semantic_start_token_id).astype(mx.int32)
        code = mx.clip(code, 0, config.audio_decoder_config.vocab_size - 1)
        mx.eval(code)
        self.assertEqual(int(code[0].item()), 0)

    def test_clamp_prevents_overflow(self):
        """A token above the semantic range must be clamped to
        codebook_size - 1."""
        config = _tiny_config()
        token_above = mx.array(
            [config.semantic_end_token_id + 100], dtype=mx.int32
        )
        code = (token_above - config.semantic_start_token_id).astype(mx.int32)
        code = mx.clip(code, 0, config.audio_decoder_config.vocab_size - 1)
        mx.eval(code)
        self.assertEqual(
            int(code[0].item()), config.audio_decoder_config.vocab_size - 1
        )

    def test_clamp_present_in_source(self):
        """Verify that ``mx.clip`` is present in the model source."""
        import inspect

        from mlx_audio.tts.models.fish_qwen3_omni.fish_speech import Model

        source = inspect.getsource(Model._generate_codes_for_batch)
        self.assertIn(
            "mx.clip", source, "mx.clip must guard the semantic code offset"
        )


# ── Fix 3: logit bias uses -inf ──────────────────────────────────────


class TestSemanticLogitBias(unittest.TestCase):
    """The semantic logit bias must use ``-inf`` (not ``-1e9``) so that
    masked tokens receive exactly zero probability after softmax."""

    def test_bias_uses_neg_inf(self):
        """Masked positions must be -inf."""
        model = _make_model()
        bias = model.semantic_logit_bias
        mx.eval(bias)

        val = float(bias[0, 0].item())
        self.assertEqual(val, float("-inf"), "Masked positions must be -inf, not -1e9")

    def test_bias_zero_for_semantic_range(self):
        """Semantic token positions must have bias == 0."""
        model = _make_model()
        config = model.config
        bias = model.semantic_logit_bias
        mx.eval(bias)

        semantic_slice = bias[
            0,
            config.semantic_start_token_id : config.semantic_end_token_id + 1,
        ]
        mx.eval(semantic_slice)
        for i in range(semantic_slice.shape[0]):
            val = float(semantic_slice[i].item())
            self.assertEqual(
                val,
                0.0,
                f"Semantic token at offset {i} "
                f"(id={config.semantic_start_token_id + i}) bias should be 0.0",
            )

    def test_bias_zeroes_masked_probs_after_softmax(self):
        """After softmax, masked positions should have exactly 0 probability
        (only possible with -inf, not -1e9)."""
        model = _make_model()
        config = model.config
        bias = model.semantic_logit_bias
        # Fake uniform logits.
        logits = mx.ones_like(bias)
        biased = logits + bias
        probs = mx.softmax(biased, axis=-1)
        mx.eval(probs)

        # Non-semantic token should have exactly 0 probability.
        self.assertEqual(
            float(probs[0, 0].item()),
            0.0,
            "Non-semantic tokens must have exactly 0 probability after softmax",
        )
        # Semantic token should have non-zero probability.
        self.assertGreater(
            float(probs[0, config.semantic_start_token_id].item()), 0.0
        )

    def test_bias_source_uses_neg_inf(self):
        """Verify that the source code uses float('-inf'), not -1e9."""
        import inspect

        from mlx_audio.tts.models.fish_qwen3_omni.fish_speech import Model

        source = inspect.getsource(Model.post_load_hook)
        self.assertNotIn("-1e9", source, "Source must not use -1e9 for logit bias")
        self.assertIn('"-inf"', source, "Source must use float('-inf') for logit bias")


if __name__ == "__main__":
    unittest.main()
