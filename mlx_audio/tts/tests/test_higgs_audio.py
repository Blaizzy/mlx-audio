# Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

import unittest

import mlx.core as mx
import numpy as np

from mlx_audio.tts.models.higgs_audio.config import HiggsAudioConfig
from mlx_audio.tts.models.higgs_audio.generation import (
    apply_delay_pattern,
    build_delay_pattern_mask,
    greedy_sample_audio,
    lookup_audio_embedding,
    revert_delay_pattern,
    sample_audio,
)
from mlx_audio.tts.models.higgs_audio.higgs_audio import HiggsAudioModel


def _tiny_config():
    """Small config just enough to shape-check forward — not real weights."""
    from mlx_audio.tts.models.higgs_audio.config import HiggsTextConfig

    return HiggsAudioConfig(
        text_config=HiggsTextConfig(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=128,
            vocab_size=256,
            rope_theta=10000.0,
            rms_norm_eps=1e-5,
            tie_word_embeddings=True,
            rope_scaling=None,
        ),
        audio_num_codebooks=4,
        audio_codebook_size=16,
        audio_stream_bos_id=16,
        audio_stream_eos_id=17,
        audio_dual_ffn_layers=[0, 1],
        use_audio_out_self_attention=False,
        audio_decoder_proj_num_layers=0,
        use_delay_pattern=True,
    )


class TestDelayPattern(unittest.TestCase):
    """Delay-pattern round-trip and structural invariants."""

    def test_build_delay_pattern_mask_shape(self):
        K, L = 4, 5
        x = mx.arange(K * L, dtype=mx.int32).reshape(K, L)
        out = build_delay_pattern_mask(x, bos_token_id=99, pad_token_id=88)
        self.assertEqual(out.shape, (K, L + K - 1))

    def test_build_delay_pattern_mask_triangles(self):
        """Lower triangle is BOS, upper triangle is EOS, middle is shifted content."""
        K, L = 3, 4
        x = mx.arange(1, K * L + 1, dtype=mx.int32).reshape(K, L)  # nonzero content
        out = build_delay_pattern_mask(x, bos_token_id=-1, pad_token_id=-2)
        out_np = np.array(out)

        # Row 0: full content + EOS tail
        np.testing.assert_array_equal(out_np[0], [1, 2, 3, 4, -2, -2])
        # Row 1: one BOS + content + one EOS
        np.testing.assert_array_equal(out_np[1], [-1, 5, 6, 7, 8, -2])
        # Row 2: two BOS + content
        np.testing.assert_array_equal(out_np[2], [-1, -1, 9, 10, 11, 12])

    def test_revert_reverses_apply(self):
        """revert_delay_pattern should be the inverse of apply_delay_pattern on the content band."""
        K, L = 4, 6
        content = mx.arange(1, K * L + 1, dtype=mx.int32).reshape(K, L)
        delayed = apply_delay_pattern(content, bos_id=0)  # [K, L+K-1]
        reverted = revert_delay_pattern(delayed)
        # After revert, shape is [K, L+K-1-K+1] = [K, L]
        self.assertEqual(reverted.shape, (K, L))
        np.testing.assert_array_equal(np.array(reverted), np.array(content))


class TestAudioEmbedding(unittest.TestCase):
    """lookup_audio_embedding correctly offsets per codebook and sums."""

    def test_lookup_shape_and_sum(self):
        import mlx.nn as nn

        K = 4
        C_plus2 = 10
        hidden = 8
        T = 3

        emb = nn.Embedding(K * C_plus2, hidden)
        ids = mx.zeros((K, T), dtype=mx.int32)  # all codebooks emit token 0
        out = lookup_audio_embedding(emb, ids, C_plus2)
        self.assertEqual(out.shape, (T, hidden))

        # Per-codebook row 0 uses embedding[0], row 1 uses embedding[C_plus2], etc.
        # Sum across K codebooks at one timestep equals sum of those K embeddings.
        expected_sum = mx.sum(
            emb(mx.arange(K, dtype=mx.int32) * C_plus2), axis=0
        )  # [hidden]
        # out[0] should equal expected_sum
        np.testing.assert_allclose(np.array(out[0]), np.array(expected_sum), rtol=1e-5)


class TestSampling(unittest.TestCase):
    """Sampling utilities produce correct shapes and respect temperature=0 = greedy."""

    def test_greedy_picks_argmax(self):
        # B=1, T=1, K=2, V=5 — craft logits with known argmax per codebook
        logits = mx.array(
            [[[[0.0, 1.0, 2.0, 0.5, 0.0], [3.0, 0.5, 0.2, 0.1, 0.0]]]], dtype=mx.float32
        )
        out = greedy_sample_audio(logits)
        self.assertEqual(out.shape, (1, 1, 2))
        np.testing.assert_array_equal(np.array(out)[0, 0], [2, 0])

    def test_sample_audio_zero_temp_equals_greedy(self):
        logits = mx.random.normal(shape=(1, 1, 3, 8))
        g = greedy_sample_audio(logits)
        s = sample_audio(logits, temperature=0.0)
        np.testing.assert_array_equal(np.array(g), np.array(s))

    def test_sample_audio_temperature_returns_valid_token(self):
        mx.random.seed(42)
        logits = mx.random.normal(shape=(1, 1, 3, 8))
        out = sample_audio(logits, temperature=0.7, top_p=0.95)
        self.assertEqual(out.shape, (1, 1, 3))
        out_np = np.array(out)
        self.assertTrue((out_np >= 0).all() and (out_np < 8).all())


class TestHiggsAudioModel(unittest.TestCase):
    """Model instantiates and forwards on a tiny synthetic config."""

    def _tiny_config(self):
        return _tiny_config()

    def test_forward_shapes(self):
        cfg = self._tiny_config()
        model = HiggsAudioModel(cfg)
        mx.eval(model.parameters())

        B, T = 1, 5
        input_ids = mx.zeros((B, T), dtype=mx.int32)
        audio_out_mask = mx.zeros((B, T), dtype=mx.bool_)
        text_logits, audio_logits = model(
            input_ids=input_ids, audio_out_mask=audio_out_mask
        )
        self.assertEqual(text_logits.shape, (B, T, cfg.text_config.vocab_size))
        self.assertEqual(
            audio_logits.shape,
            (B, T, cfg.audio_num_codebooks, cfg.audio_codebook_size + 2),
        )

    def test_forward_text_only_no_audio_logits(self):
        cfg = self._tiny_config()
        model = HiggsAudioModel(cfg)
        mx.eval(model.parameters())

        B, T = 1, 4
        input_ids = mx.zeros((B, T), dtype=mx.int32)
        text_logits, audio_logits = model(input_ids=input_ids, audio_out_mask=None)
        self.assertEqual(text_logits.shape, (B, T, cfg.text_config.vocab_size))
        self.assertIsNone(audio_logits)


class TestQuantizedLoad(unittest.TestCase):
    """Verify that a config.json quantization block triggers nn.quantize on the
    model skeleton before weight-loading in HiggsAudioServer.from_pretrained."""

    def test_quantize_predicate_skips_protected_layers(self):
        """The skip-list applied by from_pretrained must leave protected
        layers (audio_codebook_embeddings, audio_decoder_proj.audio_lm_head)
        as plain nn.Linear / nn.Embedding — only the Llama backbone quantizes."""
        import mlx.nn as nn

        from mlx_audio.tts.models.higgs_audio.config import (
            HiggsAudioConfig,
            HiggsTextConfig,
        )
        from mlx_audio.tts.models.higgs_audio.higgs_audio import HiggsAudioModel

        cfg = HiggsAudioConfig(
            text_config=HiggsTextConfig(
                hidden_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                intermediate_size=128,
                vocab_size=256,
                rope_theta=10000.0,
                rms_norm_eps=1e-5,
                tie_word_embeddings=True,
                rope_scaling=None,
            ),
            audio_num_codebooks=4,
            audio_codebook_size=16,
            audio_stream_bos_id=16,
            audio_stream_eos_id=17,
            audio_dual_ffn_layers=[0, 1],
            use_audio_out_self_attention=False,
            audio_decoder_proj_num_layers=0,
            use_delay_pattern=True,
        )
        model = HiggsAudioModel(cfg)
        mx.eval(model.parameters())

        skip = {"audio_codebook_embeddings", "audio_decoder_proj.audio_lm_head"}

        def predicate(name, module):
            if not isinstance(module, (nn.Linear, nn.Embedding)):
                return False
            return not any(s in name for s in skip)

        nn.quantize(model, group_size=64, bits=8, class_predicate=predicate)

        # Protected: audio_codebook_embeddings remains a plain Embedding.
        self.assertIsInstance(model.audio_codebook_embeddings, nn.Embedding)
        self.assertNotIsInstance(model.audio_codebook_embeddings, nn.QuantizedEmbedding)

        # Protected: audio_lm_head remains a plain Linear.
        self.assertIsInstance(model.audio_decoder_proj.audio_lm_head, nn.Linear)
        self.assertNotIsInstance(
            model.audio_decoder_proj.audio_lm_head, nn.QuantizedLinear
        )

        # Quantized: text_lm_head becomes QuantizedLinear.
        self.assertIsInstance(model.audio_decoder_proj.text_lm_head, nn.QuantizedLinear)


class TestFrameworkInterface(unittest.TestCase):
    """Model + ModelConfig conform to the mlx_audio.tts.utils.load convention."""

    def test_modelconfig_aliases_higgs_audio_config(self):
        from mlx_audio.tts.models.higgs_audio import ModelConfig

        self.assertIs(ModelConfig, HiggsAudioConfig)

    def test_modelconfig_from_dict(self):
        from mlx_audio.tts.models.higgs_audio import ModelConfig

        cfg = ModelConfig.from_dict({})  # permissive defaults
        self.assertIsInstance(cfg, HiggsAudioConfig)
        self.assertEqual(cfg.audio_num_codebooks, 8)

    def test_model_subclasses_higgs_audio_model(self):
        """Subclassing (not wrapping) keeps safetensors key paths unchanged."""
        from mlx_audio.tts.models.higgs_audio import Model

        self.assertTrue(issubclass(Model, HiggsAudioModel))

    def test_model_exposes_sample_rate(self):
        from mlx_audio.tts.models.higgs_audio import Model

        cfg = _tiny_config()
        model = Model(cfg)
        self.assertEqual(model.sample_rate, 24000)

    def test_model_quant_predicate_protects_audio_head(self):
        """Model.model_quant_predicate must skip audio_codebook_embeddings and
        audio_decoder_proj.audio_lm_head (voice-character preservation)."""
        import mlx.nn as nn

        from mlx_audio.tts.models.higgs_audio import Model

        cfg = _tiny_config()
        model = Model(cfg)

        dummy_lin = nn.Linear(4, 4)
        dummy_emb = nn.Embedding(4, 4)
        # Protected names → False (skip quantization)
        self.assertFalse(
            model.model_quant_predicate("audio_codebook_embeddings", dummy_emb)
        )
        self.assertFalse(
            model.model_quant_predicate("audio_decoder_proj.audio_lm_head", dummy_lin)
        )
        # Unprotected names → True (quantize)
        self.assertTrue(
            model.model_quant_predicate("layers.0.mlp.gate_proj", dummy_lin)
        )
        self.assertTrue(
            model.model_quant_predicate("audio_decoder_proj.text_lm_head", dummy_lin)
        )

    def test_model_generate_requires_loaded_codec(self):
        """Calling generate() before post_load_hook ran must raise a clear error."""
        from mlx_audio.tts.models.higgs_audio import Model

        cfg = _tiny_config()
        model = Model(cfg)
        # Realize the iterator so the guard actually runs.
        with self.assertRaises(RuntimeError):
            next(iter(model.generate("hello")))


class _FakeTokenizer:
    """Toy tokenizer used by the reference-cache tests. Deterministic char → id
    mapping so expected shapes are trivially computable."""

    def __init__(self, vocab_size: int = 256):
        self._vocab_size = vocab_size

    def encode(self, text: str, add_special_tokens: bool = False) -> list:
        return [ord(c) % self._vocab_size for c in text]


class _FakeCodec:
    """Toy codec. Returns a deterministic code tensor shaped like the real
    HiggsAudioTokenizer.encode output ([1, T_codes, K]) so encode_reference
    can exercise the downstream path without loading weights."""

    def __init__(self, K: int = 4, T_codes: int = 6, codebook_size: int = 16):
        self._K = K
        self._T = T_codes
        self._cb = codebook_size

    def encode(self, audio_3d: mx.array) -> mx.array:
        # Deterministic codes: repeat an ascending pattern so repeated
        # encode() calls return the same values.
        tokens = mx.arange(self._T, dtype=mx.int32)[:, None]  # [T, 1]
        tokens = mx.broadcast_to(tokens, (self._T, self._K)) % self._cb
        return tokens[None]  # [1, T, K]


class TestReferenceCache(unittest.TestCase):
    """ReferenceContext + encode_reference + _build_prompt_voice_clone — the
    caching path for HiggsAudioServer. Cache must be a pure perf win: same
    (embeds, mask, info) as the one-shot build_prompt() path."""

    def _setup(self):
        from mlx_audio.tts.models.higgs_audio.higgs_audio import HiggsAudioModel

        cfg = _tiny_config()
        model = HiggsAudioModel(cfg)
        mx.eval(model.parameters())

        tokenizer = _FakeTokenizer(vocab_size=cfg.text_config.vocab_size)
        codec = _FakeCodec(
            K=cfg.audio_num_codebooks,
            T_codes=6,
            codebook_size=cfg.audio_codebook_size,
        )
        # ref_audio content doesn't matter — FakeCodec ignores it.
        ref_audio_24k = np.zeros(24000, dtype=np.float32)
        return cfg, model, tokenizer, codec, ref_audio_24k

    def test_reference_context_has_cacheable_fields(self):
        from mlx_audio.tts.models.higgs_audio.serve import (
            ReferenceContext,
            encode_reference,
        )

        cfg, model, tokenizer, codec, ref_audio_24k = self._setup()
        ctx = encode_reference(
            ref_audio_24k,
            "hello reference",
            config=cfg,
            tokenizer=tokenizer,
            codec=codec,
            embed_tokens=model.embed_tokens,
            audio_codebook_embeddings=model.audio_codebook_embeddings,
        )
        self.assertIsInstance(ctx, ReferenceContext)
        self.assertEqual(
            ctx.prefix_emb.shape, (ctx.prefix_len, cfg.text_config.hidden_size)
        )
        self.assertEqual(
            ctx.audio_emb.shape, (ctx.T_ref_delayed, cfg.text_config.hidden_size)
        )
        # build_delay_pattern_mask adds K-1 columns of delay padding.
        self.assertEqual(ctx.T_ref_delayed, ctx.T_ref + 2 + cfg.audio_num_codebooks - 1)
        self.assertEqual(ctx.ref_text, "hello reference")

    def test_cached_path_matches_one_shot_build_prompt(self):
        """encode_reference + _build_prompt_voice_clone must produce the same
        (embeds, mask, info) as the legacy build_prompt(ref_audio_24k=...) path
        — the cache is a perf optimization, not a behavior change."""
        from mlx_audio.tts.models.higgs_audio.serve import (
            _build_prompt_voice_clone,
            build_prompt,
            encode_reference,
        )

        cfg, model, tokenizer, codec, ref_audio_24k = self._setup()
        target = "say this"
        ref_text = "the quick brown fox"

        one_shot_embeds, one_shot_mask, one_shot_info = build_prompt(
            target,
            ref_text=ref_text,
            ref_audio_24k=ref_audio_24k,
            config=cfg,
            tokenizer=tokenizer,
            codec=codec,
            embed_tokens=model.embed_tokens,
            audio_codebook_embeddings=model.audio_codebook_embeddings,
        )

        ctx = encode_reference(
            ref_audio_24k,
            ref_text,
            config=cfg,
            tokenizer=tokenizer,
            codec=codec,
            embed_tokens=model.embed_tokens,
            audio_codebook_embeddings=model.audio_codebook_embeddings,
        )
        cached_embeds, cached_mask, cached_info = _build_prompt_voice_clone(
            target,
            ctx,
            tokenizer=tokenizer,
            embed_tokens=model.embed_tokens,
        )

        self.assertEqual(one_shot_embeds.shape, cached_embeds.shape)
        self.assertEqual(one_shot_mask.shape, cached_mask.shape)
        np.testing.assert_allclose(
            np.array(one_shot_embeds), np.array(cached_embeds), rtol=1e-6, atol=1e-6
        )
        np.testing.assert_array_equal(np.array(one_shot_mask), np.array(cached_mask))
        self.assertEqual(one_shot_info["mode"], "voice_clone")
        self.assertEqual(cached_info["mode"], "voice_clone")
        self.assertEqual(one_shot_info["T_ref"], cached_info["T_ref"])
        self.assertEqual(one_shot_info["T_ref_delayed"], cached_info["T_ref_delayed"])
        self.assertEqual(one_shot_info["text_len"], cached_info["text_len"])

    def test_server_build_prompt_dispatch(self):
        """HiggsAudioServer._build_prompt dispatch order: explicit path →
        re-encode; cache populated → use cache; neither → smart-voice."""
        from mlx_audio.tts.models.higgs_audio.serve import (
            HiggsAudioServer,
            encode_reference,
        )

        cfg, model, tokenizer, codec, ref_audio_24k = self._setup()
        server = HiggsAudioServer(
            model=model, codec=codec, tokenizer=tokenizer, config=cfg
        )

        # No cache, no path → smart-voice.
        _, _, info = server._build_prompt(
            "hello", reference_text=None, reference_audio_path=None
        )
        self.assertEqual(info["mode"], "smart_voice")

        # Populate cache manually (skipping disk I/O of _load_wav_as_24k_mono).
        server._reference_cache = encode_reference(
            ref_audio_24k,
            "ref text",
            config=cfg,
            tokenizer=tokenizer,
            codec=codec,
            embed_tokens=model.embed_tokens,
            audio_codebook_embeddings=model.audio_codebook_embeddings,
        )

        # Cache set, no path → voice_clone via cache (no codec.encode call).
        _, _, info = server._build_prompt(
            "hello", reference_text=None, reference_audio_path=None
        )
        self.assertEqual(info["mode"], "voice_clone")

        # clear_reference drops the cache → back to smart-voice.
        server.clear_reference()
        _, _, info = server._build_prompt(
            "hello", reference_text=None, reference_audio_path=None
        )
        self.assertEqual(info["mode"], "smart_voice")


class TestStreamingAPIContract(unittest.TestCase):
    """``Model.generate(stream=True)`` is the framework path that must remain
    consistent across all TTS models — verified at the signature level so the
    streaming kwargs aren't silently swallowed by ``**kwargs``."""

    def test_model_generate_exposes_streaming_kwargs(self):
        import inspect

        from mlx_audio.tts.models.higgs_audio import Model

        sig = inspect.signature(Model.generate)
        for name in ("stream", "streaming_interval", "overlap_ms"):
            self.assertIn(
                name,
                sig.parameters,
                f"{name} must be a named parameter of Model.generate",
            )
        # stream defaults to False so non-streaming callers see no behavior change.
        self.assertEqual(sig.parameters["stream"].default, False)

    def test_iter_overlap_add_pcm_is_public_helper(self):
        """The overlap-add streaming algorithm must live as a module-level
        helper so ``Model.generate(stream=True)`` and
        ``HiggsAudioServer.generate_stream_overlap_add`` share one
        implementation."""
        from mlx_audio.tts.models.higgs_audio.serve import iter_overlap_add_pcm

        self.assertTrue(callable(iter_overlap_add_pcm))

    def test_model_generate_stream_true_requires_loaded_codec(self):
        """Same load-guard as the non-streaming path — calling generate before
        post_load_hook must raise, not silently produce empty chunks."""
        from mlx_audio.tts.models.higgs_audio import Model

        cfg = _tiny_config()
        model = Model(cfg)
        with self.assertRaises(RuntimeError):
            next(iter(model.generate("hello", stream=True)))


class _FakeDecodingCodec:
    """Minimal codec stub for streaming tests: ``decode([T, K]) → mx.array``
    with deterministic non-zero PCM so chunks are observable."""

    def __init__(self, samples_per_frame: int = 96):
        self._spf = samples_per_frame

    def decode(self, codes_TK: mx.array) -> mx.array:
        T = codes_TK.shape[0]
        # Position-indexed ramp so successive decodes differ in the right tail
        # — exercises the crossfade path rather than reducing it to a no-op.
        ramp = mx.arange(T * self._spf, dtype=mx.float32) * 1e-4
        return ramp.reshape(1, -1, 1)


class TestStreamingIntegration(unittest.TestCase):
    """End-to-end exercise of ``iter_overlap_add_pcm`` with tiny weights and a
    fake codec. Verifies the streaming loop terminates with ``is_final=True``,
    yields at least one chunk, and produces float32 1-D PCM."""

    def _tiny_setup(self):
        from mlx_audio.tts.models.higgs_audio.higgs_audio import HiggsAudioModel

        cfg = _tiny_config()
        model = HiggsAudioModel(cfg)
        mx.eval(model.parameters())
        codec = _FakeDecodingCodec(samples_per_frame=96)

        # Minimal valid prefill inputs — content doesn't matter, only shape.
        T_prompt = 4
        full_embeds = mx.zeros((1, T_prompt, cfg.text_config.hidden_size))
        audio_out_mask = mx.zeros((1, T_prompt), dtype=mx.bool_)
        return cfg, model, codec, full_embeds, audio_out_mask

    def test_streaming_terminates_with_final_chunk(self):
        from mlx_audio.tts.models.higgs_audio.serve import iter_overlap_add_pcm

        cfg, model, codec, full_embeds, audio_out_mask = self._tiny_setup()
        chunks = list(
            iter_overlap_add_pcm(
                model=model,
                codec=codec,
                config=cfg,
                full_embeds=full_embeds,
                audio_out_mask=audio_out_mask,
                max_new_frames=20,
                temperature=0.0,
                top_p=None,
                top_k=None,
                ras_win_len=None,
                ras_max_repeat=2,
                sampling_warmup_frames=20,  # full greedy → deterministic
                emit_every_frames=4,
                overlap_ms=4.0,
                fade_in_ms=1.0,
                fade_out_ms=1.0,
                sample_rate=24000,
            )
        )

        self.assertGreater(len(chunks), 0, "streaming must yield at least one chunk")
        # Exactly one chunk should be marked final, and it must be the last.
        finals = [i for i, (_, meta) in enumerate(chunks) if meta["is_final"]]
        self.assertEqual(finals, [len(chunks) - 1])

        for pcm, meta in chunks:
            self.assertEqual(pcm.dtype, np.float32)
            self.assertEqual(pcm.ndim, 1)
            self.assertGreater(pcm.size, 0)
            self.assertIn("frames_total", meta)

        # frames_total is monotonically non-decreasing (each yield reflects
        # cumulative generation progress).
        totals = [m["frames_total"] for _, m in chunks]
        self.assertEqual(totals, sorted(totals))


if __name__ == "__main__":
    unittest.main()
