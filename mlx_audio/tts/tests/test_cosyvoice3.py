"""Unit tests for the CosyVoice3 MLX skeleton/implementation.

These tests exercise the model with tiny, randomly-initialized configs (no
real checkpoint required) to lock in shape contracts and control flow:
  * DiT forward pass shape.
  * CFM + DiT end-to-end Euler solve.
  * Flow token -> mel end-to-end.
  * HiFT mel -> waveform.
  * LLM autoregressive decode with a small Qwen2 backbone.
  * ras_sampling / nucleus_sampling correctness on edge cases.
  * ModelConfig (de)serialization, including nested dataclasses.
  * convert.py mechanical weight transforms.
  * Model.sanitize prefix routing.
  * generate() end-to-end with a mocked frontend.

Numerical parity against the PyTorch reference is intentionally NOT covered
here (it requires the real `iic/CosyVoice3-0.5B` checkpoint) — see the
package README's "Remaining work" section.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_audio.tts.models.cosyvoice3.config import (
    FlowConfig,
    HiFTConfig,
    LLMConfig,
    ModelConfig,
)
from mlx_audio.tts.models.cosyvoice3.convert import (
    _merge_weight_norm,
    conv1d_to_mlx,
    conv_transpose1d_to_mlx,
    fold_weight_norm,
)
from mlx_audio.tts.models.cosyvoice3.dit import DiT
from mlx_audio.tts.models.cosyvoice3.flow import CausalMaskedDiffWithDiT
from mlx_audio.tts.models.cosyvoice3.flow_matching import CausalConditionalCFM
from mlx_audio.tts.models.cosyvoice3.frontend import _first_existing_optional
from mlx_audio.tts.models.cosyvoice3.hift import CausalHiFTGenerator
from mlx_audio.tts.models.cosyvoice3.llm import CosyVoice3LM
from mlx_audio.tts.models.cosyvoice3.sampling import (
    nucleus_sampling,
    ras_sampling,
)


def _tiny_llm_config() -> LLMConfig:
    return LLMConfig(
        hidden_size=64,
        num_hidden_layers=2,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=200,
        llm_input_size=64,
        llm_output_size=64,
        speech_token_size=50,
        speech_vocab_extra=10,
    )


def _tiny_flow_config() -> FlowConfig:
    return FlowConfig(
        dit_hidden_size=128,
        dit_depth=2,
        dit_num_heads=4,
        dit_head_dim=32,
        dit_mlp_ratio=2.0,
        pre_lookahead_channels=256,
        vocab_size=60,
    )


class TestModelConfig(unittest.TestCase):
    def test_defaults(self):
        config = ModelConfig()
        self.assertEqual(config.model_type, "cosyvoice3")
        self.assertEqual(config.sample_rate, 24000)
        self.assertEqual(config.llm.speech_token_size, 6561)
        self.assertEqual(config.flow.dit_depth, 22)
        self.assertEqual(config.flow.dit_hidden_size, 1024)
        self.assertEqual(config.flow.dit_num_heads, 16)
        self.assertEqual(config.flow.dit_mlp_ratio, 2.0)
        self.assertEqual(config.hift.upsample_rates, [8, 5, 3])

    def test_nested_from_dict(self):
        config = ModelConfig.from_dict(
            {
                "sample_rate": 22050,
                "llm": {"hidden_size": 1024},
                "flow": {"dit_depth": 20},
                "hift": {"sampling_rate": 22050},
            }
        )
        self.assertEqual(config.sample_rate, 22050)
        self.assertEqual(config.llm.hidden_size, 1024)
        self.assertEqual(config.flow.dit_depth, 20)
        self.assertEqual(config.hift.sampling_rate, 22050)
        # unspecified nested fields keep their defaults
        self.assertEqual(config.llm.speech_token_size, 6561)


class TestDiT(unittest.TestCase):
    def test_forward_shape(self):
        dit = DiT(
            dim=128,
            depth=2,
            heads=4,
            dim_head=32,
            ff_mult=2.0,
            mel_dim=80,
            mu_dim=80,
            spk_dim=80,
            out_channels=80,
        )
        mx.eval(dit.parameters())

        batch, seq_len = 2, 20
        x = mx.random.normal((batch, 80, seq_len))
        mu = mx.random.normal((batch, 80, seq_len))
        cond = mx.zeros((batch, 80, seq_len))
        mask = mx.ones((batch, 1, seq_len))
        spks = mx.random.normal((batch, 80))
        t = mx.array([0.5, 0.5])

        out = dit(x, mask, mu, t, spks, cond)
        mx.eval(out)
        self.assertEqual(out.shape, (batch, 80, seq_len))

    def test_streaming_not_implemented(self):
        dit = DiT(dim=32, depth=1, heads=2, dim_head=16, ff_mult=2.0)
        mx.eval(dit.parameters())
        x = mx.random.normal((1, 80, 5))
        mu = mx.random.normal((1, 80, 5))
        cond = mx.zeros((1, 80, 5))
        mask = mx.ones((1, 1, 5))
        spks = mx.random.normal((1, 80))
        with self.assertRaises(NotImplementedError):
            dit(x, mask, mu, mx.array(0.5), spks, cond, streaming=True)


class TestFlowMatching(unittest.TestCase):
    def test_cfm_with_dit_end_to_end(self):
        dit = DiT(dim=128, depth=2, heads=4, dim_head=32, ff_mult=2.0, out_channels=80)
        cfm = CausalConditionalCFM(
            estimator=dit, inference_cfg_rate=0.7, max_len=50 * 10
        )
        mx.eval(cfm.parameters())

        batch, seq_len = 1, 20
        mu = mx.random.normal((batch, 80, seq_len))
        cond = mx.zeros((batch, 80, seq_len))
        mask = mx.ones((batch, 1, seq_len))
        spks = mx.random.normal((batch, 80))

        mel = cfm(mu, mask, spks, cond, n_timesteps=4)
        mx.eval(mel)
        self.assertEqual(mel.shape, (batch, 80, seq_len))


class TestFlow(unittest.TestCase):
    def test_token_to_mel_end_to_end(self):
        config = _tiny_flow_config()
        flow = CausalMaskedDiffWithDiT(config)
        mx.eval(flow.parameters())

        prompt_token = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)
        token = mx.array([[10, 11, 12, 13, 14, 15]], dtype=mx.int32)
        prompt_token_len = mx.array([5], dtype=mx.int32)
        token_len = mx.array([6], dtype=mx.int32)
        prompt_feat = mx.random.normal((1, 10, 80))  # 5 prompt tokens * ratio 2
        embedding = mx.random.normal((1, 192))

        mel = flow.inference(
            token, token_len, prompt_token, prompt_token_len, prompt_feat, None,
            embedding, n_timesteps=3,
        )
        mx.eval(mel)
        expected_target_frames = (5 + 6) * config.token_mel_ratio - 10
        self.assertEqual(mel.shape, (1, 80, expected_target_frames))

    def test_rejects_batch_size_greater_than_one(self):
        config = _tiny_flow_config()
        flow = CausalMaskedDiffWithDiT(config)
        mx.eval(flow.parameters())
        token = mx.zeros((2, 4), dtype=mx.int32)
        with self.assertRaises(ValueError):
            flow.inference(
                token,
                mx.array([4, 4], dtype=mx.int32),
                mx.zeros((2, 2), dtype=mx.int32),
                mx.array([2, 2], dtype=mx.int32),
                mx.zeros((2, 4, 80)),
                None,
                mx.zeros((2, 192)),
            )

    def test_sanitize_transposes_conv_weights(self):
        config = _tiny_flow_config()
        flow = CausalMaskedDiffWithDiT(config)
        # PyTorch Conv1d weight layout: (out_channels, in_channels, kernel)
        pt_weight = mx.zeros((config.pre_lookahead_channels, config.input_size, 4))
        out = flow.sanitize({"pre_lookahead_layer.conv1.weight": pt_weight})
        # MLX Conv1d layout: (out_channels, kernel, in_channels)
        self.assertEqual(
            out["pre_lookahead_layer.conv1.weight"].shape,
            (config.pre_lookahead_channels, 4, config.input_size),
        )


class TestFrontEnd(unittest.TestCase):
    def test_first_existing_optional_returns_none_when_absent(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.assertIsNone(
                _first_existing_optional(root, "campplus.safetensors", "campplus.onnx")
            )

    def test_first_existing_optional_prefers_listed_order(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "campplus.onnx").touch()
            found = _first_existing_optional(
                root, "campplus.safetensors", "campplus.onnx"
            )
            self.assertEqual(found, root / "campplus.onnx")

            (root / "campplus.safetensors").touch()
            found = _first_existing_optional(
                root, "campplus.safetensors", "campplus.onnx"
            )
            self.assertEqual(found, root / "campplus.safetensors")

    def test_extract_speaker_embedding_requires_encoder(self):
        from mlx_audio.tts.models.cosyvoice3.frontend import CosyVoice3FrontEnd

        fe = CosyVoice3FrontEnd()
        with self.assertRaises(RuntimeError):
            fe.extract_speaker_embedding("dummy.wav")

    def test_encode_text_requires_tokenizer(self):
        from mlx_audio.tts.models.cosyvoice3.frontend import CosyVoice3FrontEnd

        fe = CosyVoice3FrontEnd()
        with self.assertRaises(RuntimeError):
            fe.encode_text("hello")


class TestHiFT(unittest.TestCase):
    def test_mel_to_waveform(self):
        hift = CausalHiFTGenerator(HiFTConfig())
        mx.eval(hift.parameters())

        mel = mx.random.normal((1, 80, 20))
        wav, _ = hift.inference(mel)
        mx.eval(wav)

        upsample_total = 8 * 5 * 3 * HiFTConfig().istft_params["hop_len"]
        self.assertEqual(wav.shape, (1, 20 * upsample_total))


class TestLLM(unittest.TestCase):
    def test_autoregressive_decode(self):
        config = _tiny_llm_config()
        lm = CosyVoice3LM(config)  # backbone is built eagerly in __init__
        mx.eval(lm.parameters())

        text = mx.array([[5, 6, 151646, 8]], dtype=mx.int32)
        prompt_text = mx.array([[1, 2, 3]], dtype=mx.int32)
        prompt_speech = mx.array([[10, 11]], dtype=mx.int32)

        tokens = lm.inference(
            text, prompt_text, prompt_speech,
            min_token_text_ratio=0.5, max_token_text_ratio=5.0,
        )
        self.assertIsInstance(tokens, list)
        self.assertLessEqual(len(tokens), int(4 * 5.0))
        for t in tokens:
            self.assertLess(t, config.speech_token_size)  # eos/specials excluded

    def test_special_token_offsets(self):
        config = _tiny_llm_config()
        lm = CosyVoice3LM(config)
        self.assertEqual(lm.sos, config.speech_token_size)
        self.assertEqual(lm.eos_token, config.speech_token_size + 1)
        self.assertEqual(lm.task_id, config.speech_token_size + 2)
        self.assertEqual(lm.fill_token, config.speech_token_size + 3)
        self.assertEqual(len(lm.stop_token_ids), config.speech_vocab_extra)

    def test_backbone_built_eagerly(self):
        """The Qwen2 backbone must exist right after __init__ — the
        mlx-audio loader calls model.load_weights(...) immediately after
        construction, before any explicit build_backbone() call, so a lazily
        built (None) backbone would silently drop its weights."""
        config = _tiny_llm_config()
        lm = CosyVoice3LM(config)
        self.assertIsNotNone(lm.llm)

    def test_sanitize_remaps_qwen_backbone_keys(self):
        config = _tiny_llm_config()
        lm = CosyVoice3LM(config)
        weights = {
            "llm.model.model.embed_tokens.weight": mx.zeros((10, 32)),
            "llm_decoder.weight": mx.zeros((25, 32)),
        }
        out = lm.sanitize(weights)
        self.assertIn("llm.model.embed_tokens.weight", out)
        self.assertNotIn("llm.model.model.embed_tokens.weight", out)
        self.assertIn("llm_decoder.weight", out)


class TestSampling(unittest.TestCase):
    def test_nucleus_sampling_always_includes_top_candidate(self):
        logits = mx.array([10.0, 0.0, 0.0, 0.0])
        # even with a vanishingly small top_p, the top candidate must be kept
        result = nucleus_sampling(logits, top_p=0.01, top_k=25)
        self.assertEqual(result, 0)

    def test_nucleus_sampling_respects_top_k(self):
        logits = mx.array([10.0, 9.0, 0.0, 0.0])
        result = nucleus_sampling(logits, top_p=0.99, top_k=1)
        self.assertEqual(result, 0)

    def test_ras_sampling_suppresses_heavy_repetition(self):
        logits = mx.array([100.0, 0.0, 0.0])
        decoded = [0] * 10  # window full of the dominant candidate
        result = ras_sampling(
            logits, decoded, sampling=25, top_p=0.8, top_k=25,
            win_size=10, tau_r=0.1,
        )
        self.assertNotEqual(result, 0)

    def test_ras_sampling_without_history_matches_nucleus(self):
        logits = mx.array([10.0, 0.0, 0.0])
        result = ras_sampling(logits, [], sampling=25)
        self.assertEqual(result, 0)


class TestConvert(unittest.TestCase):
    def test_conv1d_to_mlx(self):
        w = mx.zeros((16, 8, 3))  # (out, in, k)
        out = conv1d_to_mlx(w)
        self.assertEqual(out.shape, (16, 3, 8))

    def test_conv_transpose1d_to_mlx(self):
        w = mx.zeros((8, 16, 4))  # (in, out, k)
        out = conv_transpose1d_to_mlx(w)
        self.assertEqual(out.shape, (16, 4, 8))

    def test_fold_weight_norm_shape(self):
        g = mx.ones((16, 1, 1))
        v = mx.random.normal((16, 8, 3))
        out = fold_weight_norm(g, v)
        self.assertEqual(out.shape, (16, 8, 3))

    def test_merge_weight_norm_drops_batch_tracked(self):
        state = {
            "c.weight_g": mx.ones((4, 1, 1)),
            "c.weight_v": mx.random.normal((4, 2, 3)),
            "b.num_batches_tracked": mx.array(5),
        }
        merged = _merge_weight_norm(state)
        self.assertIn("c.weight", merged)
        self.assertNotIn("c.weight_g", merged)
        self.assertNotIn("c.weight_v", merged)
        self.assertNotIn("b.num_batches_tracked", merged)

    def test_convert_flow_weights_matches_real_checkpoint_key_layout(self):
        """Regression test pinned to the real flow.pt state_dict layout
        (verified 2026-07-24 against FunAudioLLM/Fun-CosyVoice3-0.5B-2512).

        The reference DiT stores a few blocks as nn.Sequential with a
        non-parametric layer (GELU/SiLU/Mish/Dropout) interleaved, which
        shifts the numeric indices PyTorch assigns vs. this module's flat
        parameter lists. Exercise convert_flow_weights against exactly the
        key shapes seen in the real checkpoint and load the result into a
        real CausalMaskedDiffWithDiT to catch drift in either direction.
        """
        from mlx_audio.tts.models.cosyvoice3.convert import convert_flow_weights
        from mlx_audio.tts.models.cosyvoice3.flow import CausalMaskedDiffWithDiT

        torch_state = {
            # FeedForward.ff = Sequential(Sequential(Linear, GELU), Dropout, Linear)
            "decoder.estimator.transformer_blocks.0.ff.ff.0.0.weight": mx.zeros((2048, 1024)),
            "decoder.estimator.transformer_blocks.0.ff.ff.0.0.bias": mx.zeros((2048,)),
            "decoder.estimator.transformer_blocks.0.ff.ff.2.weight": mx.zeros((1024, 2048)),
            "decoder.estimator.transformer_blocks.0.ff.ff.2.bias": mx.zeros((1024,)),
            # CausalConvPositionEmbedding.conv{1,2} = Sequential(Conv1d, Mish)
            "decoder.estimator.input_embed.conv_pos_embed.conv1.0.weight": mx.zeros((1024, 64, 31)),
            "decoder.estimator.input_embed.conv_pos_embed.conv1.0.bias": mx.zeros((1024,)),
            "decoder.estimator.input_embed.conv_pos_embed.conv2.0.weight": mx.zeros((1024, 64, 31)),
            "decoder.estimator.input_embed.conv_pos_embed.conv2.0.bias": mx.zeros((1024,)),
            # TimestepEmbedding.time_mlp = Sequential(Linear, SiLU, Linear)
            "decoder.estimator.time_embed.time_mlp.0.weight": mx.zeros((1024, 256)),
            "decoder.estimator.time_embed.time_mlp.0.bias": mx.zeros((1024,)),
            "decoder.estimator.time_embed.time_mlp.2.weight": mx.zeros((1024, 1024)),
            "decoder.estimator.time_embed.time_mlp.2.bias": mx.zeros((1024,)),
            # non-persistent buffer: never learned, must be dropped
            "decoder.estimator.rotary_embed.inv_freq": mx.zeros((32,)),
        }
        converted = convert_flow_weights(torch_state)

        self.assertIn("decoder.estimator.transformer_blocks.0.ff.ff.0.weight", converted)
        self.assertIn("decoder.estimator.transformer_blocks.0.ff.ff.1.weight", converted)
        self.assertNotIn(
            "decoder.estimator.transformer_blocks.0.ff.ff.0.0.weight", converted
        )
        self.assertIn(
            "decoder.estimator.input_embed.conv_pos_embed.conv1.weight", converted
        )
        self.assertIn("decoder.estimator.time_embed.time_mlp.1.weight", converted)
        self.assertNotIn("decoder.estimator.rotary_embed.inv_freq", converted)

        # conv weights land in MLX's (out, k, in) layout, not raw PyTorch (out, in, k)
        self.assertEqual(
            converted["decoder.estimator.input_embed.conv_pos_embed.conv1.weight"].shape,
            (1024, 31, 64),
        )

        # every converted key must resolve into the real module's parameter tree
        flow = CausalMaskedDiffWithDiT(_tiny_flow_config())
        real_keys = set(dict(tree_flatten(flow.parameters())).keys())
        for k in converted:
            self.assertIn(k, real_keys, f"{k} has no matching parameter in CausalMaskedDiffWithDiT")

    def test_convert_llm_weights_matches_real_checkpoint_key_layout(self):
        """Regression test pinned to the real llm.pt state_dict layout
        (verified 2026-07-24 against FunAudioLLM/Fun-CosyVoice3-0.5B-2512).

        With tie_word_embeddings=True (the v3 default), the checkpoint's
        ``llm.model.lm_head.weight`` is bit-for-bit identical to
        ``llm.model.model.embed_tokens.weight`` and must be dropped: mlx_lm's
        Qwen2 does not allocate a separate lm_head parameter when tied, so
        keeping it would break ``load_weights(strict=True)``.
        """
        from mlx_audio.tts.models.cosyvoice3.convert import convert_llm_weights

        embed = mx.zeros((100, 64))
        torch_state = {
            "llm.model.model.embed_tokens.weight": embed,
            "llm.model.model.norm.weight": mx.zeros((64,)),
            "llm.model.lm_head.weight": embed,  # tied: identical to embed_tokens
            "llm_decoder.weight": mx.zeros((60, 64)),
            "speech_embedding.weight": mx.zeros((60, 64)),
        }

        tied = convert_llm_weights(torch_state, tie_word_embeddings=True)
        self.assertIn("llm.model.embed_tokens.weight", tied)
        self.assertNotIn("llm.lm_head.weight", tied)
        self.assertIn("llm_decoder.weight", tied)
        self.assertIn("speech_embedding.weight", tied)

        untied = convert_llm_weights(torch_state, tie_word_embeddings=False)
        self.assertIn("llm.lm_head.weight", untied)

        # loads cleanly (strict) into the real module's parameter tree
        cfg = _tiny_llm_config()
        m = CosyVoice3LM(cfg)
        real_keys = set(dict(tree_flatten(m.parameters())).keys())
        for k in tied:
            self.assertIn(k, real_keys, f"{k} has no matching parameter in CosyVoice3LM")

    def test_convert_hift_weights_matches_real_checkpoint_key_layout(self):
        """Regression test pinned to the real hift.pt state_dict layout
        (verified 2026-07-24 against FunAudioLLM/Fun-CosyVoice3-0.5B-2512).

        Unlike the non-causal HiFTGenerator, ``CausalConv1dUpsample`` (used
        for ``ups.*``) subclasses ``torch.nn.Conv1d`` directly rather than
        ``ConvTranspose1d`` — its weight is (out, in, k), same layout as every
        other conv here, so all ndim==3 weights take the identical
        conv1d_to_mlx transpose. ``f0_predictor.condnet`` skips the
        PyTorch-Sequential ELU indices (0,2,4,6,8 -> 0,1,2,3,4).
        """
        from mlx_audio.tts.models.cosyvoice3.convert import convert_hift_weights
        from mlx_audio.tts.models.cosyvoice3.hift import CausalHiFTGenerator

        torch_state = {
            "conv_pre.parametrizations.weight.original0": mx.ones((512, 1, 1)),
            "conv_pre.parametrizations.weight.original1": mx.ones((512, 80, 5)) * 0.1,
            "conv_pre.bias": mx.zeros((512,)),
            # ups.0: CausalConv1dUpsample is a plain Conv1d subclass -> (out, in, k)
            "ups.0.parametrizations.weight.original0": mx.ones((256, 1, 1)),
            "ups.0.parametrizations.weight.original1": mx.ones((256, 512, 16)) * 0.1,
            "ups.0.bias": mx.zeros((256,)),
            "f0_predictor.condnet.0.parametrizations.weight.original0": mx.ones((512, 1, 1)),
            "f0_predictor.condnet.0.parametrizations.weight.original1": mx.ones((512, 80, 4)) * 0.1,
            "f0_predictor.condnet.0.bias": mx.zeros((512,)),
            "f0_predictor.condnet.2.parametrizations.weight.original0": mx.ones((512, 1, 1)),
            "f0_predictor.condnet.2.parametrizations.weight.original1": mx.ones((512, 512, 3)) * 0.1,
            "f0_predictor.condnet.2.bias": mx.zeros((512,)),
        }
        converted = convert_hift_weights(torch_state)

        self.assertEqual(converted["conv_pre.weight"].shape, (512, 5, 80))
        self.assertEqual(converted["ups.0.weight"].shape, (256, 16, 512))
        self.assertIn("f0_predictor.condnet.0.weight", converted)
        self.assertIn("f0_predictor.condnet.1.weight", converted)
        self.assertNotIn("f0_predictor.condnet.2.weight", converted)

        # loads cleanly into the real module's parameter tree (shapes match)
        hift = CausalHiFTGenerator(HiFTConfig())
        real = dict(tree_flatten(hift.parameters()))
        for k, v in converted.items():
            self.assertIn(k, real, f"{k} has no matching parameter in CausalHiFTGenerator")
            self.assertEqual(v.shape, real[k].shape, f"{k} shape mismatch")


class TestModelIntegration(unittest.TestCase):
    def test_sanitize_routes_by_prefix(self):
        """Verify the prefix-split/re-prefix routing itself, independent of
        each sub-module's own sanitize (which may drop unknown keys — see
        e.g. CausalHiFTGenerator.sanitize, which only keeps keys matching its
        real parameter tree)."""
        from mlx_audio.tts.models.cosyvoice3 import Model, ModelConfig as MC

        model = Model(MC())
        weights = {
            "llm.foo": mx.array([1.0]),
            "flow.bar": mx.array([2.0]),
            "hift.baz": mx.array([3.0]),
            "unprefixed": mx.array([4.0]),
        }
        with (
            patch.object(model.llm, "sanitize", side_effect=lambda w: w),
            patch.object(model.flow, "sanitize", side_effect=lambda w: w),
            patch.object(model.hift, "sanitize", side_effect=lambda w: w),
        ):
            routed = model.sanitize(weights)
        self.assertEqual(
            sorted(routed.keys()), ["flow.bar", "hift.baz", "llm.foo", "unprefixed"]
        )

    def test_hift_sanitize_is_a_pure_key_transform(self):
        """CausalHiFTGenerator.sanitize (convert_hift_weights) only reshapes
        keys/values — like convert_flow_weights/convert_llm_weights, it does
        not filter against the live parameter tree. Actually dropping unknown
        keys happens at ``load_weights(strict=False)`` time, not here."""
        from mlx_audio.tts.models.cosyvoice3.hift import CausalHiFTGenerator

        hift = CausalHiFTGenerator(HiFTConfig())
        routed = hift.sanitize({"totally_unknown_key": mx.array([1.0])})
        self.assertIn("totally_unknown_key", routed)

    def test_generate_end_to_end_with_mock_frontend(self):
        from mlx_audio.tts.models.cosyvoice3 import Model

        config = ModelConfig(
            llm=_tiny_llm_config(),
            flow=_tiny_flow_config(),
            hift=HiFTConfig(),
        )
        model = Model(config)  # LLM backbone is built eagerly in __init__
        mx.eval(model.parameters())

        class MockFrontEnd:
            sample_rate = 24000

            def text_normalize(self, text, split=False, text_frontend=True):
                return text

            def frontend_zero_shot(self, tts_text, prompt_text, prompt_wav, spk_id=""):
                return {
                    "text": mx.array([[5, 6, 7, 8]], dtype=mx.int32),
                    "text_len": mx.array([4], dtype=mx.int32),
                    "prompt_text": mx.array([[1, 151646, 3]], dtype=mx.int32),
                    "prompt_text_len": mx.array([3], dtype=mx.int32),
                    "prompt_speech_token": mx.array(
                        [[10, 11, 12, 13, 14]], dtype=mx.int32
                    ),
                    "prompt_feat": mx.random.normal((1, 10, 80)),
                    "embedding": mx.random.normal((1, 192)),
                }

            def frontend_instruct2(self, tts_text, instruct, prompt_wav):
                return self.frontend_zero_shot(tts_text, prompt_text="", prompt_wav=prompt_wav)

        model.frontend = MockFrontEnd()

        results = list(
            model.generate(
                text="hello", ref_audio="dummy.wav", ref_text="ref",
                n_timesteps=2, sampling=5,
            )
        )
        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result.sample_rate, 24000)
        self.assertGreater(result.samples, 0)
        self.assertGreater(result.token_count, 0)

    def test_generate_requires_ref_audio(self):
        from mlx_audio.tts.models.cosyvoice3 import Model

        config = ModelConfig(
            llm=_tiny_llm_config(), flow=_tiny_flow_config(), hift=HiFTConfig()
        )
        model = Model(config)
        model.frontend = object()  # non-None sentinel; ref_audio check comes first
        with self.assertRaises(ValueError):
            next(model.generate(text="hello", ref_audio=None))

    def test_generate_requires_frontend(self):
        from mlx_audio.tts.models.cosyvoice3 import Model

        config = ModelConfig(
            llm=_tiny_llm_config(), flow=_tiny_flow_config(), hift=HiFTConfig()
        )
        model = Model(config)
        with self.assertRaises(RuntimeError):
            next(model.generate(text="hello", ref_audio="dummy.wav"))


if __name__ == "__main__":
    unittest.main()
