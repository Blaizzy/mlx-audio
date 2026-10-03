import unittest
from types import SimpleNamespace

import numpy as np

DEFAULT_LFM_CONFIG = {
    "model_type": "lfm2",
    "vocab_size": 65536,
    "hidden_size": 512,
    "num_hidden_layers": 4,
    "num_attention_heads": 8,
    "num_key_value_heads": 4,
    "max_position_embeddings": 32768,
    "norm_eps": 1e-5,
    "conv_bias": False,
    "conv_L_cache": 3,
    "block_dim": 512,
    "block_ff_dim": 1536,
    "block_multiple_of": 256,
    "block_ffn_dim_multiplier": None,
    "block_auto_adjust_ff_dim": True,
    "rope_theta": 10000.0,
    # layer_types: one per layer - "conv" or "full_attention"
    "layer_types": ["conv", "full_attention", "conv", "full_attention"],
}


def get_test_config():
    """Create a test config dict with all required fields."""
    return {
        "model_type": "lfm_audio",
        "sample_rate": 24000,
        "codebooks": 8,
        "audio_vocab_size": 2049,
        "preprocessor": {"sample_rate": 16000, "features": 128},
        "encoder": {"d_model": 512, "n_layers": 2},  # Small for testing
        "depthformer": {"layers": 2, "dim": 256},  # Small for testing
        "lfm": DEFAULT_LFM_CONFIG,
    }


class TestPreprocessorConfig(unittest.TestCase):

    def test_defaults(self):
        """Test PreprocessorConfig default values."""
        from mlx_audio.sts.models.lfm_audio.config import PreprocessorConfig

        config = PreprocessorConfig()
        self.assertEqual(config.sample_rate, 16000)
        self.assertEqual(config.features, 128)
        self.assertEqual(config.n_fft, 512)
        self.assertEqual(config.hop_length, 160)
        self.assertEqual(config.win_length, 400)


class TestConformerEncoderConfig(unittest.TestCase):

    def test_defaults(self):
        from mlx_audio.sts.models.lfm_audio.config import ConformerEncoderConfig

        config = ConformerEncoderConfig()
        self.assertEqual(config.feat_in, 128)
        self.assertEqual(config.d_model, 512)
        self.assertEqual(config.n_layers, 17)
        self.assertEqual(config.n_heads, 8)


class TestDepthformerConfig(unittest.TestCase):

    def test_defaults(self):
        from mlx_audio.sts.models.lfm_audio.config import DepthformerConfig

        config = DepthformerConfig()
        self.assertEqual(config.layers, 6)
        self.assertEqual(config.dim, 1024)
        self.assertEqual(config.num_heads, 32)
        self.assertEqual(config.num_kv_heads, 8)
        # config.json carries no theta, so this default is the only thing keeping
        # the depthformer's RoPE aligned with the reference MHA (1e6, not 1e4).
        self.assertEqual(config.rope_theta, 1000000.0)


class TestLFM2AudioConfig(unittest.TestCase):

    def test_from_dict(self):
        from mlx_audio.sts.models.lfm_audio.config import LFM2AudioConfig

        config_dict = get_test_config()
        config = LFM2AudioConfig.from_dict(config_dict)

        self.assertEqual(config.model_type, "lfm_audio")
        self.assertEqual(config.sample_rate, 24000)
        self.assertEqual(config.codebooks, 8)
        self.assertEqual(config.audio_vocab_size, 2049)
        self.assertIsNotNone(config.preprocessor)
        self.assertIsNotNone(config.encoder)
        self.assertIsNotNone(config.depthformer)
        self.assertIsNotNone(config.lfm)


class TestLFM2AudioModelOutput(unittest.TestCase):

    def test_call_returns_text_and_audio_logits(self):
        import mlx.core as mx

        from mlx_audio.sts.models.lfm_audio.config import LFM2AudioConfig
        from mlx_audio.sts.models.lfm_audio.model import LFM2AudioModel

        config = LFM2AudioConfig.from_dict(get_test_config())
        model = LFM2AudioModel(config)
        model.eval()

        # Create dummy text input
        batch_size = 1
        seq_len = 5
        text_tokens = mx.zeros((batch_size, seq_len), dtype=mx.int32)

        # Call model
        text_logits, audio_logits = model(text_tokens=text_tokens)
        mx.eval(text_logits)

        # Check text_logits is an array with correct shape
        self.assertIsInstance(text_logits, mx.array)
        self.assertEqual(text_logits.shape[0], batch_size)
        self.assertEqual(text_logits.shape[1], seq_len)

        # Check audio_logits is a list of 8 arrays (one per codebook)
        self.assertIsInstance(audio_logits, list)
        self.assertEqual(len(audio_logits), config.codebooks)

        for logit in audio_logits:
            mx.eval(logit)
            self.assertIsInstance(logit, mx.array)
            self.assertEqual(logit.shape[0], batch_size)
            self.assertEqual(logit.shape[1], seq_len)
            self.assertEqual(logit.shape[2], config.audio_vocab_size)


class TestLFM2AudioModelDtype(unittest.TestCase):

    def test_model_components_exist(self):
        from mlx_audio.sts.models.lfm_audio.config import LFM2AudioConfig
        from mlx_audio.sts.models.lfm_audio.model import LFM2AudioModel

        config = LFM2AudioConfig.from_dict(get_test_config())
        model = LFM2AudioModel(config)

        # Check components exist
        self.assertIsNotNone(model.audio_encoder)
        self.assertIsNotNone(model.audio_adapter)
        self.assertIsNotNone(model.lfm)
        self.assertIsNotNone(model.audio_embedding)
        self.assertIsNotNone(model.audio_head)
        self.assertIsNotNone(model.depth_embeddings)
        self.assertIsNotNone(model.depth_linear)

        # Check depth_embeddings count matches codebooks
        self.assertEqual(len(model.depth_embeddings), config.codebooks)

    def test_sample_rate_property(self):
        from mlx_audio.sts.models.lfm_audio.config import LFM2AudioConfig
        from mlx_audio.sts.models.lfm_audio.model import LFM2AudioModel

        config = LFM2AudioConfig.from_dict(get_test_config())
        model = LFM2AudioModel(config)

        self.assertEqual(model.sample_rate, 24000)


class TestLFMModality(unittest.TestCase):

    def test_modality_values(self):
        from mlx_audio.sts.models.lfm_audio.model import LFMModality

        self.assertEqual(LFMModality.TEXT, 1)
        self.assertEqual(LFMModality.AUDIO_IN, 2)
        self.assertEqual(LFMModality.AUDIO_OUT, 3)


class TestSpecialTokens(unittest.TestCase):

    def test_token_values(self):
        from mlx_audio.sts.models.lfm_audio.model import (
            AUDIO_EOS_TOKEN,
            AUDIO_START_TOKEN,
            IM_END_TOKEN,
            TEXT_END_TOKEN,
        )

        self.assertEqual(AUDIO_START_TOKEN, 128)
        self.assertEqual(IM_END_TOKEN, 7)
        self.assertEqual(TEXT_END_TOKEN, 130)
        self.assertEqual(AUDIO_EOS_TOKEN, 2048)


class TestGenerationConfig(unittest.TestCase):

    def test_defaults(self):
        from mlx_audio.sts.models.lfm_audio.model import GenerationConfig

        config = GenerationConfig()
        self.assertEqual(config.max_new_tokens, 512)
        # liquid-audio defaults every sampling knob to None (greedy decoding)
        self.assertIsNone(config.temperature)
        self.assertIsNone(config.top_k)
        self.assertIsNone(config.audio_temperature)
        self.assertIsNone(config.audio_top_k)


class TestAudioHead(unittest.TestCase):
    """The depthformer must treat codebooks (not time steps) as its sequence.

    Both of these need a sequence-*mixing* stub or real attention: with an
    identity depthformer the old (B*C, L, D) transposes cancel out, so shape and
    value assertions pass under either orientation.
    """

    def _config(self, dim, heads=2, kv=1):
        from mlx_audio.sts.models.lfm_audio.config import DepthformerConfig

        return DepthformerConfig(layers=1, dim=dim, num_heads=heads, num_kv_heads=kv)

    def test_depthformer_mixes_along_the_codebook_axis(self):
        import mlx.core as mx

        from mlx_audio.sts.models.lfm_audio.model import AudioHead

        C, D, L = 4, 2, 3
        head = AudioHead(self._config(D), num_codebooks=C)
        # Causal cumulative sum over the sequence axis: reveals which axis the
        # depthformer actually walks.
        head.depthformer = lambda x, cache=None, use_cache=False, mask=None: (
            mx.cumsum(x, axis=1),
            None,
        )

        proj = mx.arange(L * C * D, dtype=mx.float32).reshape(1, L, C * D)
        out = head(proj)
        mx.eval(out)
        self.assertEqual(out.shape, (1, L, C, D))

        per_codebook = proj.reshape(1, L, C, D)
        for t in range(L):
            for c in range(C):
                expected = per_codebook[0, t, : c + 1].sum(axis=0)
                self.assertEqual(
                    out[0, t, c].tolist(),
                    expected.tolist(),
                    f"time {t}, codebook {c}: depthformer did not accumulate over codebooks",
                )

    def test_codebooks_attend_causally_to_each_other(self):
        import mlx.core as mx

        from mlx_audio.sts.models.lfm_audio.model import AudioHead

        C, D = 4, 8
        head = AudioHead(self._config(D), num_codebooks=C)
        mx.random.seed(0)
        proj = mx.random.normal((1, 1, C * D))

        def bump(codebook):
            lo, hi = codebook * D, (codebook + 1) * D
            return mx.concatenate(
                [proj[:, :, :lo], proj[:, :, lo:hi] + 10.0, proj[:, :, hi:]], axis=-1
            )

        base = head(proj)
        late = head(bump(C - 1))
        early = head(bump(0))
        mx.eval(base, late, early)

        # Causality: a later codebook must not leak into an earlier one.
        self.assertTrue(
            mx.allclose(base[0, 0, 0], late[0, 0, 0], atol=1e-4).item(),
            "codebook 0 changed when the last codebook was perturbed",
        )
        # ...but there must be real cross-codebook mixing, which the old
        # orientation (codebooks folded into the batch) could not produce.
        self.assertFalse(
            mx.allclose(base[0, 0, C - 1], early[0, 0, C - 1], atol=1e-4).item(),
            "codebook 0 did not influence the last codebook - no mixing at all",
        )


class TestAudioEmbeddingWithNorm(unittest.TestCase):

    def test_get_logits_applies_embedding_norm(self):
        """liquid-audio's SharedEmbedding.get_logits normalizes before to_logits."""
        import mlx.core as mx

        from mlx_audio.sts.models.lfm_audio.model import AudioEmbeddingWithNorm

        layer = AudioEmbeddingWithNorm(vocab_size=4, dim=8)
        layer.embedding_norm.weight = mx.full((8,), 3.0)

        x = mx.arange(8, dtype=mx.float32)[None, :]

        expected = layer.to_logits(layer.embedding_norm(x))
        self.assertTrue(mx.allclose(layer.get_logits(x), expected).item())
        # ...and the norm actually changes the result
        self.assertFalse(mx.allclose(layer.get_logits(x), layer.to_logits(x)).item())

    def test_embed_is_unnormalized(self):
        """Token embeddings feeding the depthformer are raw, per the reference."""
        import mlx.core as mx

        from mlx_audio.sts.models.lfm_audio.model import AudioEmbeddingWithNorm

        layer = AudioEmbeddingWithNorm(vocab_size=4, dim=8)
        tokens = mx.array([0, 3])

        self.assertTrue(
            mx.allclose(layer.embed(tokens), layer.embedding(tokens)).item()
        )


class TestSampling(unittest.TestCase):

    def _model(self):
        from mlx_audio.sts.models.lfm_audio.model import LFM2AudioModel

        return LFM2AudioModel.__new__(LFM2AudioModel)

    def test_greedy_conditions_match_reference(self):
        """None/<=0 temperature and top_k == 1 all mean greedy."""
        model = self._model()

        self.assertTrue(model._is_greedy(None, 4))
        self.assertTrue(model._is_greedy(0.0, 4))
        self.assertTrue(model._is_greedy(-1.0, 4))
        self.assertTrue(model._is_greedy(1.0, 1))
        self.assertFalse(model._is_greedy(1.0, 4))
        self.assertFalse(model._is_greedy(1.0, None))

    def test_sample_text_token_defaults_to_argmax(self):
        import mlx.core as mx

        model = self._model()
        logits = mx.array([[0.1, 5.0, 0.2, 0.3]])

        self.assertEqual(model._sample_text_token(logits).tolist(), [1])

    def test_sample_text_token_top_k_restricts_support(self):
        """Sampling with top_k=2 can only ever return the two best tokens."""
        import mlx.core as mx

        model = self._model()
        logits = mx.array([[0.0, 1.0, 5.0, 4.0]])

        sampled = {
            model._sample_text_token(logits, temperature=1.0, top_k=2).item()
            for _ in range(50)
        }
        self.assertTrue(sampled <= {2, 3}, sampled)

    def test_apply_top_k_passthrough(self):
        """top_k of None or >= vocab leaves logits untouched."""
        import mlx.core as mx

        model = self._model()
        logits = mx.array([[0.0, 1.0, 5.0, 4.0]])

        for top_k in (None, 0, 4, 99):
            self.assertEqual(
                model._apply_top_k(logits, top_k).tolist(), logits.tolist()
            )


def _stub_model(text_tokens, audio_frames):
    """LFM2AudioModel with the heavy pieces replaced by scripted token streams.

    Scripts are consumed in order. Running off the end means generation took a
    different path than the test expected, so fail with that rather than the
    RuntimeError a bare StopIteration becomes inside a generator.
    """
    import mlx.core as mx

    from mlx_audio.sts.models.lfm_audio.model import LFM2AudioModel

    model = LFM2AudioModel.__new__(LFM2AudioModel)
    # SimpleNamespace is not callable, so use a tiny class for the LFM stub.
    model.lfm = type(
        "LFMStub",
        (),
        {
            "embed_tokens": SimpleNamespace(
                as_linear=lambda hidden: mx.zeros((1, 1, 140))
            ),
            "__call__": lambda self, inputs=None, cache=None, input_embeddings=None: mx.zeros(
                (1, 1, 1)
            ),
        },
    )()

    def script(values, kind):
        pending = list(values)

        def take():
            if not pending:
                raise AssertionError(
                    f"generation asked for more {kind} than the test scripted "
                    f"({len(values)}); it took an unexpected path"
                )
            return pending.pop(0)

        return take

    next_text = script(text_tokens, "text tokens")
    next_frame = script(audio_frames, "audio frames")

    model._prefill = lambda **kwargs: (mx.zeros((1, 1, 1)), [])
    model._sample_text_token = lambda *a, **kw: mx.array([next_text()])
    model._sample_audio_frame = lambda *a, **kw: (
        mx.array([next_frame()], dtype=mx.int32),
        None,
    )
    model._embed_text = lambda token: mx.zeros((1, 1, 1))
    model._embed_audio_out = lambda frame: mx.zeros((1, 1))
    return model


class TestInterleavedGeneration(unittest.TestCase):

    def test_audio_eos_updates_lfm_state_before_resuming_text(self):
        import mlx.core as mx

        from mlx_audio.sts.models.lfm_audio.model import (
            AUDIO_EOS_TOKEN,
            LFM2AudioModel,
            LFMModality,
        )

        class LFMStub:
            def __init__(self):
                self.calls = []
                self.embed_tokens = SimpleNamespace(as_linear=self.as_linear)

            def as_linear(self, hidden):
                state = int(hidden.tolist()[0][0][0])
                token_by_state = {
                    1: 10,  # first text token
                    2: 11,  # stale text token if audio EOS is not embedded
                    3: 12,  # text token after audio EOS updates state
                }
                logits = np.full((1, 1, 140), -100.0, dtype=np.float32)
                logits[0, 0, token_by_state[state]] = 100.0
                return mx.array(logits)

            def __call__(self, inputs=None, cache=None, input_embeddings=None):
                value = int(input_embeddings.tolist()[0][0][0])
                self.calls.append(value)
                if value == 99:
                    return mx.array([[[3.0]]])
                return mx.array([[[2.0]]])

        model = LFM2AudioModel.__new__(LFM2AudioModel)
        model.config = SimpleNamespace(
            interleaved_n_text=1,
            interleaved_n_audio=1,
            codebooks=8,
        )
        model.lfm = LFMStub()
        model._prefill = lambda **kwargs: (mx.array([[[1.0]]]), [])
        model._sample_text_token = lambda logits, temperature, top_k: mx.argmax(
            logits, axis=-1
        )
        model._embed_text = lambda token: token.astype(mx.float32)[..., None]
        model._sample_audio_frame = lambda *args, **kwargs: (
            mx.full((1, 8), AUDIO_EOS_TOKEN, dtype=mx.int32),
            None,
        )
        model._embed_audio_out = lambda audio_frame: mx.array([[99.0]])

        outputs = list(
            model.generate_interleaved(
                text_tokens=mx.array([[1]]),
                modalities=mx.array([[LFMModality.TEXT]]),
                max_new_tokens=3,
            )
        )

        self.assertEqual(
            [(modality, token.tolist()) for token, modality in outputs],
            [
                (LFMModality.TEXT, [10]),
                (
                    LFMModality.AUDIO_OUT,
                    [AUDIO_EOS_TOKEN] * 8,
                ),
                (LFMModality.TEXT, [12]),
            ],
        )
        self.assertIn(99, model.lfm.calls)

    def test_modality_group_sizes_match_config(self):
        """n_text text tokens, then n_audio frames, then back to text."""
        import mlx.core as mx

        from mlx_audio.sts.models.lfm_audio.model import LFMModality

        model = _stub_model(text_tokens=[20] * 12, audio_frames=[[5] * 8] * 12)
        model.config = SimpleNamespace(
            interleaved_n_text=3,
            interleaved_n_audio=2,
            codebooks=8,
        )

        modalities = [
            modality
            for _, modality in model.generate_interleaved(
                text_tokens=mx.array([[1]]),
                modalities=mx.array([[LFMModality.TEXT]]),
                max_new_tokens=10,
            )
        ]

        self.assertEqual(
            modalities,
            [LFMModality.TEXT] * 3
            + [LFMModality.AUDIO_OUT] * 2
            + [LFMModality.TEXT] * 3
            + [LFMModality.AUDIO_OUT] * 2,
        )

    def test_text_end_switches_to_audio_immediately(self):
        """<|text_end|> ends the text span before n_text is exhausted."""
        import mlx.core as mx

        from mlx_audio.sts.models.lfm_audio.model import TEXT_END_TOKEN, LFMModality

        model = _stub_model(
            text_tokens=[20, TEXT_END_TOKEN, 21],
            audio_frames=[[5] * 8] * 8,
        )
        model.config = SimpleNamespace(
            interleaved_n_text=6,
            interleaved_n_audio=2,
            codebooks=8,
        )

        outputs = [
            (modality, token.tolist())
            for token, modality in model.generate_interleaved(
                text_tokens=mx.array([[1]]),
                modalities=mx.array([[LFMModality.TEXT]]),
                max_new_tokens=5,
            )
        ]

        # Once text is done the model stays in audio mode, unlike a plain
        # n_text/n_audio alternation.
        self.assertEqual(
            outputs,
            [
                (LFMModality.TEXT, [20]),
                (LFMModality.TEXT, [TEXT_END_TOKEN]),
                (LFMModality.AUDIO_OUT, [5] * 8),
                (LFMModality.AUDIO_OUT, [5] * 8),
                (LFMModality.AUDIO_OUT, [5] * 8),
            ],
        )

    def test_audio_eos_does_not_refill_the_text_budget(self):
        """Reference quirk: EOS mid-span returns to text without resetting the counter.

        `liquid-audio` only resets `modality_left` when the audio span runs to
        completion; an early EOS leaves the remaining audio budget in place and
        text spends it down. Resetting to n_text here would shorten the run of
        text that follows an early EOS.
        """
        import mlx.core as mx

        from mlx_audio.sts.models.lfm_audio.model import AUDIO_EOS_TOKEN, LFMModality

        model = _stub_model(
            text_tokens=[20] * 10,
            audio_frames=[[AUDIO_EOS_TOKEN] * 8] + [[5] * 8] * 6,
        )
        model.config = SimpleNamespace(
            interleaved_n_text=2, interleaved_n_audio=6, codebooks=8
        )

        pattern = "".join(
            "T" if m == LFMModality.TEXT else "A"
            for _, m in model.generate_interleaved(
                text_tokens=mx.array([[1]]),
                modalities=mx.array([[LFMModality.TEXT]]),
                max_new_tokens=9,
            )
        )

        # 2 text -> audio span opens -> immediate EOS -> the 5 remaining audio
        # slots are spent on text -> audio resumes.
        self.assertEqual(pattern, "TTATTTTTA")


class TestSequentialGeneration(unittest.TestCase):

    def test_audio_start_is_yielded_and_opens_audio_span(self):
        """The reference yields <|audio_start|> before switching modality."""
        import mlx.core as mx

        from mlx_audio.sts.models.lfm_audio.model import (
            AUDIO_EOS_TOKEN,
            AUDIO_START_TOKEN,
            IM_END_TOKEN,
            LFMModality,
        )

        model = _stub_model(
            text_tokens=[20, AUDIO_START_TOKEN, IM_END_TOKEN],
            audio_frames=[[5] * 8, [AUDIO_EOS_TOKEN] * 8],
        )
        model.config = SimpleNamespace(codebooks=8)

        outputs = [
            (modality, token.tolist())
            for token, modality in model.generate_sequential(
                text_tokens=mx.array([[1]]),
                modalities=mx.array([[LFMModality.TEXT]]),
                max_new_tokens=10,
            )
        ]

        self.assertEqual(
            outputs,
            [
                (LFMModality.TEXT, [20]),
                (LFMModality.TEXT, [AUDIO_START_TOKEN]),
                (LFMModality.AUDIO_OUT, [5] * 8),
                (LFMModality.AUDIO_OUT, [AUDIO_EOS_TOKEN] * 8),
                (LFMModality.TEXT, [IM_END_TOKEN]),
            ],
        )


if __name__ == "__main__":
    unittest.main()
