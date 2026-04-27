from __future__ import annotations

import inspect
import os
import unittest
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, cast
from unittest.mock import MagicMock, patch

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_audio.codec.models.moss_audio_tokenizer import MossAudioTokenizer
from mlx_audio.tts.models.moss_tts.config import ModelConfig
from mlx_audio.tts.models.moss_tts.moss_tts import (
    Model,
    _apply_experimental_quantization,
    _apply_mixed_precision_rescue,
    _encode_reference_audio_with_codec,
    _get_experimental_quant_mode,
    _get_experimental_quant_patterns,
    _get_generated_audio_history,
    _get_mixed_precision_rescue_patterns,
    _normalize_reference_audio_for_codec,
    _path_matches_rescue_pattern,
    _requantize_module_with_mode,
    _suppress_token_ids,
    find_last_equal,
)
from mlx_audio.tts.models.moss_tts.processor import (
    AUDIO_PLACEHOLDER,
    AssistantMessage,
    MossTTSProcessor,
)
from mlx_audio.tts.models.moss_tts.processor import (
    apply_de_delay_pattern as processor_apply_de_delay_pattern,
)
from mlx_audio.tts.models.moss_tts.processor import (
    apply_delay_pattern as processor_apply_delay_pattern,
)
from mlx_audio.tts.models.moss_tts.processor import (
    build_user_message,
    parse_output,
    prepare_generation_input,
)
from mlx_audio.tts.models.moss_tts.qwen3 import Qwen3Attention
from mlx_audio.utils import apply_quantization

# pyright: reportMissingImports=false


class TestConfig(unittest.TestCase):
    def test_from_dict_merges_language_config_defaults(self):
        cfg = ModelConfig.from_dict({"language_config": {"hidden_size": 1234}})
        self.assertEqual(int(cfg.language_config["hidden_size"]), 1234)
        self.assertIn("vocab_size", cfg.language_config)
        self.assertIn("num_hidden_layers", cfg.language_config)

    def test_from_dict_accepts_dict_like_language_config(self):
        class DictLike:
            def __iter__(self):
                return iter({"hidden_size": 42}.items())

        cfg = ModelConfig.from_dict({"language_config": DictLike()})
        self.assertEqual(int(cfg.language_config["hidden_size"]), 42)

    def test_sampling_rate_and_sample_rate_are_kept_in_sync(self):
        cfg1 = ModelConfig.from_dict({"sample_rate": 22050, "sampling_rate": None})
        self.assertEqual(int(cfg1.sample_rate), 22050)
        self.assertEqual(int(cfg1.sampling_rate), 22050)

        cfg2 = ModelConfig.from_dict({"sampling_rate": 16000, "sample_rate": None})
        self.assertEqual(int(cfg2.sample_rate), 16000)
        self.assertEqual(int(cfg2.sampling_rate), 16000)


class TestSanitize(unittest.TestCase):
    def test_model_sanitize_filters_rotary_inv_freq(self):
        weights = {
            "language_model.layers.0.self_attn.rotary_emb.inv_freq": mx.array([1.0]),
            "language_model.layers.0.self_attn.q_proj.weight": mx.array([2.0]),
        }
        sanitized = Model.sanitize(weights)
        self.assertNotIn(
            "language_model.layers.0.self_attn.rotary_emb.inv_freq", sanitized
        )
        self.assertIn("language_model.layers.0.self_attn.q_proj.weight", sanitized)

    def test_codec_sanitize_transposes_conv1d_weight(self):
        pytorch_weight = mx.zeros((64, 32, 3), dtype=mx.float32)
        weights = {"decoder.modules.0.conv.weight": pytorch_weight}
        sanitized = MossAudioTokenizer.sanitize(weights)
        self.assertEqual(
            tuple(sanitized["decoder.modules.0.conv.weight"].shape), (64, 3, 32)
        )

    def test_codec_sanitize_leaves_mlx_conv1d_weight_unchanged(self):
        mlx_weight = mx.zeros((64, 3, 32), dtype=mx.float32)
        weights = {"decoder.modules.0.conv.weight": mlx_weight}
        sanitized = MossAudioTokenizer.sanitize(weights)
        self.assertEqual(
            tuple(sanitized["decoder.modules.0.conv.weight"].shape), (64, 3, 32)
        )

    def test_codec_sanitize_filters_non_decoder_quantizer_keys(self):
        weights = {
            "encoder.layers.0.weight": mx.zeros((2, 2)),
            "decoder.layers.0.weight": mx.zeros((2, 2)),
            "quantizer.quantizers.0.codebook.weight": mx.zeros((2, 2)),
        }
        sanitized = MossAudioTokenizer.sanitize(weights)
        self.assertNotIn("encoder.layers.0.weight", sanitized)
        self.assertIn("decoder.layers.0.weight", sanitized)
        self.assertIn("quantizer.quantizers.0.codebook.weight", sanitized)


class TestPrecisionPaths(unittest.TestCase):
    def test_get_input_embeddings_accumulates_in_fp32_then_casts_to_bf16(self):
        cfg = ModelConfig(
            n_vq=2,
            audio_vocab_size=8,
            language_config={
                "hidden_size": 4,
                "num_hidden_layers": 1,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "head_dim": 4,
                "intermediate_size": 8,
                "vocab_size": 16,
            },
        )
        model = Model(cfg)
        model.language_model.embed_tokens.weight = mx.full(
            model.language_model.embed_tokens.weight.shape, 1.0, dtype=mx.bfloat16
        )
        model.emb_ext[0].weight = mx.full(
            model.emb_ext[0].weight.shape, 2.0, dtype=mx.bfloat16
        )
        model.emb_ext[1].weight = mx.full(
            model.emb_ext[1].weight.shape, 3.0, dtype=mx.bfloat16
        )
        input_ids = mx.zeros((1, 4, 3), dtype=mx.int32)

        out = model.get_input_embeddings(input_ids)

        self.assertEqual(out.dtype, mx.bfloat16)
        np.testing.assert_array_equal(
            np.array(out.astype(mx.float32)), np.full((1, 4, 4), 6.0, dtype=np.float32)
        )

    def test_qwen3_attention_applies_rope_in_fp32_but_attention_in_bf16(self):
        cfg = {
            "hidden_size": 8,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "attention_bias": False,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
        }
        attn = Qwen3Attention(cfg, layer_idx=0)

        class DummyLinear(nn.Linear):
            def __init__(self, out: mx.array):
                super().__init__(1, 1, bias=False)
                self.out = out

            def __call__(self, x: mx.array) -> mx.array:
                _ = x
                return self.out

        class IdentityRMSNorm(nn.RMSNorm):
            def __init__(self, dims: int):
                super().__init__(dims)

            def __call__(self, x: mx.array) -> mx.array:
                return x

        class DummyRope(nn.RoPE):
            def __init__(self):
                super().__init__(cfg["head_dim"], traditional=False, base=10000.0)
                self.calls: list[tuple[Any, Optional[int]]] = []

            def __call__(self, x: mx.array, offset: Optional[int] = None) -> mx.array:
                self.calls.append((x.dtype, offset))
                return x

        batch = 1
        seq_len = 2
        hidden = cfg["hidden_size"]
        qkv = mx.arange(batch * seq_len * hidden, dtype=mx.float32).reshape(
            batch, seq_len, hidden
        )
        qkv = qkv.astype(mx.bfloat16)

        attn.q_proj = DummyLinear(qkv)
        attn.k_proj = DummyLinear(qkv)
        attn.v_proj = DummyLinear(qkv)
        attn.q_norm = IdentityRMSNorm(cfg["head_dim"])
        attn.k_norm = IdentityRMSNorm(cfg["head_dim"])
        attn.o_proj = DummyLinear(qkv)
        rope = DummyRope()
        attn.rope = rope

        seen: dict[str, Any] = {}

        def fake_sdpa(q: mx.array, k: mx.array, v: mx.array, **kwargs: Any) -> mx.array:
            seen["q_dtype"] = q.dtype
            seen["k_dtype"] = k.dtype
            seen["v_dtype"] = v.dtype
            seen["mask"] = kwargs.get("mask")
            return q

        x = mx.zeros((batch, seq_len, hidden), dtype=mx.bfloat16)
        with patch(
            "mlx_audio.tts.models.moss_tts.qwen3.mx.fast.scaled_dot_product_attention",
            side_effect=fake_sdpa,
        ):
            out = attn(x)

        self.assertEqual([call[0] for call in rope.calls], [mx.float32, mx.float32])
        self.assertEqual(seen["q_dtype"], mx.bfloat16)
        self.assertEqual(seen["k_dtype"], mx.bfloat16)
        self.assertEqual(seen["v_dtype"], mx.bfloat16)
        self.assertEqual(out.dtype, mx.bfloat16)

    def test_qwen3_model_uses_native_causal_mask_when_attention_mask_none(self):
        from mlx_audio.tts.models.moss_tts.qwen3 import Qwen3Model

        cfg = {
            "hidden_size": 8,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "attention_bias": False,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "num_hidden_layers": 1,
            "vocab_size": 16,
        }
        model = Qwen3Model(cfg)

        seen = {"masks": []}

        def fake_sdpa(q: mx.array, k: mx.array, v: mx.array, **kwargs: Any) -> mx.array:
            _ = (k, v)
            seen["masks"].append(kwargs.get("mask"))
            return q

        with patch(
            "mlx_audio.tts.models.moss_tts.qwen3.mx.fast.scaled_dot_product_attention",
            side_effect=fake_sdpa,
        ):
            x = mx.zeros((1, 2, 8), dtype=mx.bfloat16)
            model(inputs_embeds=x, attention_mask=None, cache=None)
            self.assertGreaterEqual(len(seen["masks"]), 1)
            self.assertIsInstance(seen["masks"][0], mx.array)

            x1 = mx.zeros((1, 1, 8), dtype=mx.bfloat16)
            model(inputs_embeds=x1, attention_mask=None, cache=None)
            self.assertGreaterEqual(len(seen["masks"]), 2)
            self.assertIsNone(seen["masks"][1])


class TestDelayPattern(unittest.TestCase):
    def test_round_trip_model_delay_pattern(self):
        codes = mx.array(
            [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
            dtype=mx.int32,
        )
        delayed = Model.apply_delay_pattern(codes, pad_code=1024)
        recovered = Model.apply_de_delay_pattern(delayed)
        np.testing.assert_array_equal(np.array(recovered), np.array(codes))

    def test_round_trip_processor_delay_pattern(self):
        codes = mx.array(
            [[10, 11, 12, 13], [14, 15, 16, 17]],
            dtype=mx.int32,
        )
        delayed = processor_apply_delay_pattern(codes, pad_code=99)
        recovered = processor_apply_de_delay_pattern(delayed)
        np.testing.assert_array_equal(np.array(recovered), np.array(codes))

    def test_delay_pattern_shape_and_padding(self):
        codes = mx.array([[1, 2, 3], [4, 5, 6]], dtype=mx.int32)
        delayed = Model.apply_delay_pattern(codes, pad_code=7)
        self.assertEqual(tuple(delayed.shape), (2 + 3 - 1, 3))
        delayed_np = np.array(delayed)
        self.assertEqual(int(delayed_np[0, 0]), 1)
        self.assertEqual(int(delayed_np[0, 2]), 7)


class TestSuppressTokenIds(unittest.TestCase):
    def test_suppresses_specified_ids(self):
        logits = mx.array([[0.1, 0.5, 0.3, 0.9]], dtype=mx.float32)
        result = _suppress_token_ids(logits, [1, 3])
        result_np = np.array(result)
        self.assertAlmostEqual(float(result_np[0, 0]), 0.1, places=6)
        self.assertAlmostEqual(float(result_np[0, 2]), 0.3, places=6)
        self.assertTrue(result_np[0, 1] == float("-inf"))
        self.assertTrue(result_np[0, 3] == float("-inf"))

    def test_empty_ids_returns_unchanged(self):
        logits = mx.array([[0.1, 0.5, 0.3]], dtype=mx.float32)
        result = _suppress_token_ids(logits, [])
        np.testing.assert_array_equal(np.array(result), np.array(logits))


class TestProcessor(unittest.TestCase):
    def test_build_user_message_contains_template_and_text(self):
        msg = build_user_message(text="Hello from tests")
        self.assertIn("<user_inst>", msg)
        self.assertIn("Hello from tests", msg)

    def test_build_user_message_includes_audio_placeholder_when_reference_given(self):
        msg = build_user_message(text="Hi", reference=[object()])
        self.assertIn(AUDIO_PLACEHOLDER, msg)

    def test_prepare_generation_input_shape(self):
        class DummyTokenizer:
            def apply_chat_template(
                self, messages, add_generation_prompt=False, tokenize=False
            ):
                _ = (add_generation_prompt, tokenize)
                return f"USER: {messages[0]['content']}"

            def encode(self, text: str) -> List[int]:
                _ = text
                return [101, 102, 103, 104]

        cfg = SimpleNamespace(n_vq=4, audio_pad_code=99, audio_user_slot_token_id=123)
        x = prepare_generation_input("Hello", tokenizer=DummyTokenizer(), config=cfg)
        self.assertEqual(tuple(x.shape), (1, 4, 1 + 4))
        self.assertEqual(int(x[0, 0, 0].item()), 101)
        pad_mask = x[0, :, 1:] == 99
        self.assertTrue(bool(np.all(np.array(pad_mask)).item()))

    def test_parse_output_segments_non_pad_runs(self):
        pad_code = 99
        cfg = SimpleNamespace(audio_pad_code=pad_code)
        audio_codes = mx.array(
            [
                [pad_code, pad_code, pad_code, pad_code],
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [pad_code, pad_code, pad_code, pad_code],
                [9, 10, 11, 12],
            ],
            dtype=mx.int32,
        )
        delay_audio = processor_apply_delay_pattern(audio_codes, pad_code=pad_code)
        text_ids = mx.zeros((int(delay_audio.shape[0]), 1), dtype=mx.int32)
        generation_ids = mx.concatenate([text_ids, delay_audio], axis=1)

        _, segments = parse_output(generation_ids, start_length=0, config=cfg)
        self.assertEqual(len(segments), 2)
        np.testing.assert_array_equal(np.array(segments[0]), np.array(audio_codes[1:3]))
        np.testing.assert_array_equal(np.array(segments[1]), np.array(audio_codes[4:5]))

    def test_processor_wrapper_methods(self):
        class DummyTokenizer:
            def apply_chat_template(
                self, messages, add_generation_prompt=False, tokenize=False
            ):
                _ = (add_generation_prompt, tokenize)
                return messages[0]["content"]

            def encode(self, text: str) -> List[int]:
                _ = text
                return [1, 2]

        cfg = SimpleNamespace(n_vq=2, audio_pad_code=7)
        proc = MossTTSProcessor(tokenizer=DummyTokenizer(), config=cfg)
        msg = proc.build_user_message(text="X")
        self.assertIn("X", msg)
        ids = proc.prepare_generation_input("X")
        self.assertEqual(tuple(ids.shape), (1, 2, 3))

    def test_decode_trims_prefix_audio_and_normalizes_placeholder(self):
        class DummyTokenizer:
            token_map = {
                10: "<|im_start|>",
                13: "<|audio_start|>",
                14: "<|audio_gen|>",
                16: "<|audio_end|>",
                15: "<|audio_delay|>",
            }

            def decode(self, ids):
                return "".join(self.token_map[int(i)] for i in ids)

            def convert_ids_to_tokens(self, token_id: int):
                return self.token_map[int(token_id)]

        class DummyCodec:
            def decode(self, audio_codes):
                length = int(audio_codes.shape[-1]) * 4
                audio = mx.arange(length, dtype=mx.float32)[None, None, :]
                return {"audio": audio}

        cfg = SimpleNamespace(
            audio_pad_code=99,
            audio_start_token_id=13,
            audio_end_token_id=16,
            audio_assistant_gen_slot_token_id=14,
            audio_assistant_delay_slot_token_id=15,
        )
        proc = MossTTSProcessor(
            tokenizer=DummyTokenizer(), config=cfg, codec=DummyCodec()
        )
        delay_audio = mx.array(
            [
                [1, 99],
                [3, 2],
                [5, 4],
                [99, 6],
            ],
            dtype=mx.int32,
        )
        generation_ids = mx.concatenate(
            [
                mx.array([[10], [13], [14], [16]], dtype=mx.int32),
                delay_audio,
            ],
            axis=1,
        )

        decoded = proc.decode([(1, generation_ids)])
        self.assertEqual(len(decoded), 1)
        message = decoded[0]
        self.assertIsInstance(message, AssistantMessage)
        assert message is not None
        self.assertEqual(message.content, AUDIO_PLACEHOLDER)
        self.assertEqual(len(message.audio), 1)
        np.testing.assert_array_equal(
            np.array(message.audio[0]),
            np.array([4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]),
        )


class TestHelpers(unittest.TestCase):
    def test_find_last_equal(self):
        x = mx.array(
            [
                [1, 2, 3, 2],
                [9, 9, 9, 9],
                [0, 1, 0, 1],
            ],
            dtype=mx.int32,
        )
        out = find_last_equal(x, 2)
        np.testing.assert_array_equal(
            np.array(out), np.array([3, -1, -1], dtype=np.int32)
        )

    def test_get_generated_audio_history_excludes_prompt_reference_prefix(self):
        generation_ids = mx.array(
            [
                [
                    [100, 1, 10],
                    [101, 2, 11],
                    [102, 3, 12],
                    [103, 4, 13],
                    [104, 5, 14],
                ]
            ],
            dtype=mx.int32,
        )

        prev_h0 = _get_generated_audio_history(
            generation_ids, generated_start=3, vq_idx=0
        )
        prev_h1 = _get_generated_audio_history(
            generation_ids, generated_start=3, vq_idx=1
        )

        np.testing.assert_array_equal(
            np.array(prev_h0), np.array([[4, 5]], dtype=np.int32)
        )
        np.testing.assert_array_equal(
            np.array(prev_h1), np.array([[13, 14]], dtype=np.int32)
        )

    def test_normalize_reference_audio_matches_hf_mono_and_gain_clamp(self):
        stereo = mx.array(
            [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]], dtype=mx.float32
        )

        normalized = _normalize_reference_audio_for_codec(stereo)

        expected = np.ones((4,), dtype=np.float32) * float(10.0 ** (-3.0 / 20.0))
        np.testing.assert_allclose(np.array(normalized), expected, rtol=1e-6, atol=1e-6)

    def test_encode_reference_audio_uses_model_n_vq_and_normalized_wav(self):
        class DummyCodec:
            def __init__(self):
                self.has_encoder = True
                self.config = SimpleNamespace(num_quantizers=16)
                self.seen_input = np.zeros((1, 1, 1), dtype=np.float32)
                self.seen_num_quantizers = -1

            def encode(self, input_values, num_quantizers=None):
                if num_quantizers is None:
                    raise AssertionError("num_quantizers must be provided")
                self.seen_input = np.array(input_values, dtype=np.float32)
                self.seen_num_quantizers = int(num_quantizers)
                audio_codes = mx.arange(
                    int(num_quantizers) * 1 * 3, dtype=mx.int32
                ).reshape(int(num_quantizers), 1, 3)
                audio_codes_lengths = mx.array([3], dtype=mx.int32)
                return {
                    "audio_codes": audio_codes,
                    "audio_codes_lengths": audio_codes_lengths,
                }

        codec = DummyCodec()

        with patch(
            "mlx_audio.tts.models.moss_tts.moss_tts.load_audio",
            return_value=mx.array([1.0, 1.0, 1.0, 1.0], dtype=mx.float32),
        ):
            codes_list = _encode_reference_audio_with_codec(
                codec=codec,
                ref_audio="dummy.wav",
                sample_rate=24000,
                n_vq=4,
            )

        self.assertEqual(codec.seen_num_quantizers, 4)
        expected_wav = np.ones((4,), dtype=np.float32) * float(10.0 ** (-3.0 / 20.0))
        np.testing.assert_allclose(
            codec.seen_input[0, 0], expected_wav, rtol=1e-6, atol=1e-6
        )
        self.assertEqual(len(codes_list), 1)
        self.assertEqual(tuple(codes_list[0].shape), (3, 4))

    def test_path_matches_rescue_pattern(self):
        self.assertTrue(_path_matches_rescue_pattern("lm_heads.0", "lm_heads.*"))
        self.assertTrue(
            _path_matches_rescue_pattern(
                "language_model.embed_tokens", "language_model.embed_tokens"
            )
        )
        self.assertFalse(
            _path_matches_rescue_pattern(
                "language_model.layers.0.self_attn.q_proj", "lm_heads.*"
            )
        )
        self.assertTrue(
            _path_matches_rescue_pattern(
                "language_model.layers.12.mlp.down_proj",
                "language_model.layers.*.mlp.*",
            )
        )

    def test_get_mixed_precision_rescue_patterns_merges_defaults_config_and_env(self):
        cfg = SimpleNamespace(
            mixed_precision_rescue_patterns=["language_model.layers.0.self_attn.q_proj"]
        )
        with patch.dict(
            os.environ,
            {
                "MOSS_TTS_MIXED_PRECISION_RESCUE": "language_model.layers.1.self_attn.q_proj"
            },
            clear=False,
        ):
            patterns = _get_mixed_precision_rescue_patterns(cfg)

        self.assertIn("language_model.embed_tokens", patterns)
        self.assertIn("lm_heads.*", patterns)
        self.assertIn("language_model.layers.0.self_attn.q_proj", patterns)
        self.assertIn("language_model.layers.1.self_attn.q_proj", patterns)

    def test_apply_mixed_precision_rescue_dequantizes_matching_quantized_linear(self):
        class DummyQuantizedLinear:
            def __init__(self):
                self.weight = mx.zeros((4, 8), dtype=mx.uint32)
                self.scales = mx.zeros((4, 1), dtype=mx.float16)
                self.biases = mx.zeros((4, 1), dtype=mx.float16)
                self.group_size = 64
                self.bits = 8

        class DummyLayer:
            def __init__(self):
                self.self_attn = SimpleNamespace(q_proj=DummyQuantizedLinear())

        class DummyLanguageModel:
            def __init__(self):
                self.layers = [DummyLayer()]
                self.embed_tokens = object()

        class DummyModel:
            def __init__(self):
                self.config = SimpleNamespace(
                    mixed_precision_rescue_patterns=[
                        "language_model.layers.0.self_attn.q_proj"
                    ]
                )
                self.language_model = DummyLanguageModel()
                self.lm_heads = []

            def named_modules(self):
                return [
                    ("", self),
                    (
                        "language_model.layers.0.self_attn.q_proj",
                        self.language_model.layers[0].self_attn.q_proj,
                    ),
                ]

        model = DummyModel()
        with patch(
            "mlx_audio.tts.models.moss_tts.moss_tts.mx.dequantize",
            return_value=mx.zeros((4, 8), dtype=mx.float16),
        ):
            _apply_mixed_precision_rescue(model)

        self.assertIsInstance(
            model.language_model.layers[0].self_attn.q_proj, nn.Linear
        )

    def test_get_experimental_quant_mode_prefers_env(self):
        cfg = SimpleNamespace(experimental_quant_mode="affine")
        with patch.dict(
            os.environ,
            {"MOSS_TTS_EXPERIMENTAL_QUANT_MODE": "mxfp8"},
            clear=False,
        ):
            mode = _get_experimental_quant_mode(cfg)
        self.assertEqual(mode, "mxfp8")

    def test_get_experimental_quant_patterns_defaults_to_lm_heads(self):
        with patch.dict(
            os.environ,
            {"MOSS_TTS_EXPERIMENTAL_QUANT_MODE": "mxfp8"},
            clear=False,
        ):
            patterns = _get_experimental_quant_patterns(SimpleNamespace())
        self.assertEqual(patterns, ["lm_heads.*"])

    def test_requantize_module_with_mode_rewrites_linear_to_mxfp8(self):
        linear = nn.Linear(32, 64, bias=False)
        rewritten = _requantize_module_with_mode(
            linear, path="lm_heads.0", mode="mxfp8"
        )

        self.assertIsInstance(rewritten, nn.QuantizedLinear)
        self.assertEqual(rewritten.mode, "mxfp8")
        self.assertEqual(int(rewritten.bits), 8)
        self.assertEqual(int(rewritten.group_size), 32)

    def test_apply_experimental_quantization_rewrites_matching_heads(self):
        class DummyModel:
            def __init__(self):
                self.config = SimpleNamespace(
                    experimental_quant_mode="mxfp8",
                    experimental_quant_patterns=["lm_heads.*"],
                )
                self.lm_heads = [nn.Linear(32, 64, bias=False)]

            def named_modules(self):
                return [("", self), ("lm_heads.0", self.lm_heads[0])]

        model = DummyModel()
        _apply_experimental_quantization(model)

        self.assertIsInstance(model.lm_heads[0], nn.QuantizedLinear)
        self.assertEqual(model.lm_heads[0].mode, "mxfp8")


class TestQuantPredicate(unittest.TestCase):
    def test_model_quant_predicate_skip_patterns(self):
        model = Model(ModelConfig())
        self.assertFalse(model.model_quant_predicate("codec.layers.0.weight", None))
        self.assertFalse(model.model_quant_predicate("emb_ext.0.weight", None))
        self.assertFalse(model.model_quant_predicate("audio_tokenizer.foo", None))
        self.assertTrue(
            model.model_quant_predicate("language_model.embed_tokens", None)
        )
        self.assertTrue(model.model_quant_predicate("lm_heads.0", None))
        self.assertTrue(
            model.model_quant_predicate(
                "language_model.layers.0.self_attn.q_proj.weight", None
            )
        )


class TestConversationInputIntegration(unittest.TestCase):
    class DummyTokenizer:
        def decode(self, ids):
            _ = ids
            return ""

        def encode(self, text: str) -> List[int]:
            _ = text
            return [1, 2, 3, 4]

        def apply_chat_template(
            self, messages, add_generation_prompt=False, tokenize=False
        ):
            _ = (messages, add_generation_prompt, tokenize)
            return ""

        def convert_ids_to_tokens(self, token_id: int):
            return str(token_id)

    class DummyProcessor(MossTTSProcessor):
        def __init__(self, config: Any):
            super().__init__(
                tokenizer=TestConversationInputIntegration.DummyTokenizer(),
                config=config,
            )
            self.prepare_generation_input_mock = MagicMock(
                return_value=mx.zeros((1, 4, 3), dtype=mx.int32)
            )
            self.prepare_conversation_input_mock = MagicMock(
                return_value=mx.zeros((1, 4, 3), dtype=mx.int32)
            )
            self.decode_mock = MagicMock(
                return_value=[
                    AssistantMessage(
                        content=AUDIO_PLACEHOLDER,
                        audio=[mx.ones((8,), dtype=mx.float32)],
                    )
                ]
            )

        def prepare_generation_input(self, *args: Any, **kwargs: Any) -> mx.array:
            return self.prepare_generation_input_mock(*args, **kwargs)

        def prepare_conversation_input(self, *args: Any, **kwargs: Any) -> mx.array:
            return self.prepare_conversation_input_mock(*args, **kwargs)

        def decode(self, *args: Any, **kwargs: Any) -> List[Optional[AssistantMessage]]:
            return self.decode_mock(*args, **kwargs)

    def _make_model(
        self,
    ) -> tuple[Model, "TestConversationInputIntegration.DummyProcessor"]:
        model = Model(
            ModelConfig(
                n_vq=2,
                language_config={
                    "hidden_size": 4,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 1,
                    "num_key_value_heads": 1,
                    "head_dim": 4,
                    "intermediate_size": 8,
                    "vocab_size": 155700,
                },
            )
        )
        processor = self.DummyProcessor(model.config)
        model.processor = processor
        model.tokenizer = self.DummyTokenizer()
        return model, processor

    def _mock_forward(self, model: Model):
        vocab_size = int(model.config.language_config["vocab_size"])
        audio_vocab = int(model.config.audio_vocab_size) + 1

        def forward(
            input_ids, attention_mask=None, past_key_values=None, use_cache=None
        ):
            _ = (input_ids, attention_mask, past_key_values, use_cache)
            text_logits = mx.full((1, 1, vocab_size), -1000.0, dtype=mx.float32)
            text_logits[0, 0, int(model.config.im_end_token_id)] = 0.0
            audio_logits = [
                mx.zeros((1, 1, audio_vocab), dtype=mx.float32)
                for _ in range(model.config.n_vq)
            ]
            return {"logits": [text_logits, *audio_logits], "past_key_values": None}

        return patch.object(model, "forward", side_effect=forward)

    def test_generate_uses_prepare_conversation_input_for_generation_mode(self):
        model, processor = self._make_model()

        with self._mock_forward(model):
            out = list(
                model.generate(
                    conversation=[{"role": "user", "text": "hello", "language": "en"}],
                    mode="generation",
                    max_tokens=4,
                    verbose=False,
                )
            )

        processor.prepare_generation_input_mock.assert_not_called()
        processor.prepare_conversation_input_mock.assert_called_once()
        args, kwargs = processor.prepare_conversation_input_mock.call_args
        self.assertEqual(kwargs["mode"], "generation")
        self.assertEqual(args[0][0]["role"], "user")
        self.assertEqual(args[0][0]["text"], "hello")
        self.assertEqual(len(out), 1)
        self.assertGreater(int(out[0].audio.shape[0]), 0)

    def test_generate_normalizes_continuation_audio_entries(self):
        model, processor = self._make_model()

        with (
            patch.object(
                model,
                "_encode_reference_audio_to_codes",
                return_value=[mx.array([[1, 2], [3, 4]], dtype=mx.int32)],
            ) as encode_mock,
            self._mock_forward(model),
        ):
            _ = list(
                model.generate(
                    conversation=[
                        {
                            "role": "user",
                            "text": "prefix and continuation",
                            "language": "en",
                        },
                        {"role": "assistant", "audio_codes_list": ["prefix.wav"]},
                    ],
                    mode="continuation",
                    max_tokens=4,
                    verbose=False,
                )
            )

        encode_mock.assert_called_once_with("prefix.wav")
        args, kwargs = processor.prepare_conversation_input_mock.call_args
        self.assertEqual(kwargs["mode"], "continuation")
        assistant_message = args[0][1]
        self.assertEqual(assistant_message["role"], "assistant")
        self.assertEqual(len(assistant_message["audio_codes_list"]), 1)
        np.testing.assert_array_equal(
            np.array(assistant_message["audio_codes_list"][0]),
            np.array([[1, 2], [3, 4]], dtype=np.int32),
        )

    def test_generate_rejects_ref_audio_with_conversation(self):
        model, _processor = self._make_model()

        with self.assertRaisesRegex(ValueError, "`ref_audio` is not supported"):
            _ = list(
                model.generate(
                    conversation=[{"role": "user", "text": "hello"}],
                    mode="generation",
                    ref_audio="ref.wav",
                )
            )

    def test_generate_rejects_continuation_without_assistant_prefix_audio(self):
        model, _processor = self._make_model()

        with self.assertRaisesRegex(
            ValueError,
            "continuation mode requires assistant prefix audio in `audio_codes_list`",
        ):
            _ = list(
                model.generate(
                    conversation=[
                        {"role": "user", "text": "hello", "language": "en"},
                        {"role": "assistant", "audio_codes_list": []},
                    ],
                    mode="continuation",
                )
            )


class TestApplyQuantization(unittest.TestCase):
    def test_apply_quantization_uses_config_group_size(self):
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(32, 64, bias=False)

            def named_modules(self):
                return [("", self), ("layer", self.layer)]

        model = DummyModel()
        config = {"quantization": {"group_size": 32, "bits": 8, "mode": "mxfp8"}}
        weights = {
            "layer.weight": model.layer.weight,
            "layer.scales": mx.ones((64, 1), dtype=mx.float32),
        }

        apply_quantization(model, config, weights)

        self.assertIsInstance(model.layer, nn.QuantizedLinear)
        self.assertEqual(getattr(model.layer, "mode", None), "mxfp8")

    def test_apply_quantization_skips_non_divisible_last_dim(self):
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(48, 64, bias=False)

            def named_modules(self):
                return [("", self), ("layer", self.layer)]

        model = DummyModel()
        config = {"quantization": {"group_size": 32, "bits": 8, "mode": "mxfp8"}}
        weights = {
            "layer.weight": model.layer.weight,
            "layer.scales": mx.ones((64, 1), dtype=mx.float32),
        }

        apply_quantization(model, config, weights)

        self.assertIsInstance(model.layer, nn.Linear)


class TestGenerateIntegration(unittest.TestCase):
    def test_post_load_hook_does_not_run_quant_rewrite_by_default(self):
        from pathlib import Path
        from unittest.mock import patch

        model_dir = os.environ.get("MOSS_TTS_MODEL_DIR", "./moss-tts-8bit")
        if not os.path.isdir(model_dir):
            self.skipTest(f"model dir not found: {model_dir} (set MOSS_TTS_MODEL_DIR)")

        from mlx_audio.tts.utils import load_model

        with (
            patch(
                "mlx_audio.tts.models.moss_tts.moss_tts._apply_mixed_precision_rescue"
            ) as rescue_mock,
            patch(
                "mlx_audio.tts.models.moss_tts.moss_tts._apply_experimental_quantization"
            ) as exp_mock,
        ):
            _ = load_model(Path(model_dir))

        rescue_mock.assert_not_called()
        exp_mock.assert_not_called()

    def test_generate_smoke_if_local_weights_exist(self):
        model_dir = os.environ.get("MOSS_TTS_MODEL_DIR", "./moss-tts-8bit")
        if not os.path.isdir(model_dir):
            self.skipTest(f"model dir not found: {model_dir} (set MOSS_TTS_MODEL_DIR)")

        from pathlib import Path

        from mlx_audio.tts.utils import load_model

        model = load_model(Path(model_dir))
        assert model.generate is not None
        results = list(model.generate("Hello from MOSS", max_tokens=64, verbose=False))
        self.assertGreaterEqual(len(results), 1)

    def test_generate_ref_audio_runs_with_fixture_q8(self):
        from pathlib import Path

        from mlx_audio.tts.utils import load_model

        model_dir = os.environ.get("MOSS_TTS_MODEL_DIR", "./moss-tts-8bit")
        if not os.path.isdir(model_dir):
            self.skipTest(f"model dir not found: {model_dir} (set MOSS_TTS_MODEL_DIR)")

        model = load_model(Path(model_dir))
        fixture = Path(__file__).resolve().parent / "fixtures" / "ref_audio.wav"
        if not fixture.exists():
            self.fail(f"fixture wav missing: {fixture}")

        assert model.generate is not None
        out = list(
            model.generate(
                text="Hello",
                ref_audio=str(fixture),
                max_tokens=16,
                verbose=False,
            )
        )
        self.assertGreaterEqual(len(out), 1)
        self.assertGreater(int(out[0].audio.shape[0]), 0)

    def test_generate_ref_audio_requires_codec_encoder(self):
        from pathlib import Path

        model_dir = os.environ.get("MOSS_TTS_MODEL_DIR", "./moss-tts-8bit")
        if not os.path.isdir(model_dir):
            self.skipTest(f"model dir not found: {model_dir} (set MOSS_TTS_MODEL_DIR)")

        fixture = Path(__file__).resolve().parent / "fixtures" / "ref_audio.wav"
        if not fixture.exists():
            self.fail(f"fixture wav missing: {fixture}")

        from mlx_audio.tts.utils import load_model

        model = load_model(Path(model_dir))

        from mlx_audio.codec.models.moss_audio_tokenizer import MossAudioTokenizer

        model.codec = MossAudioTokenizer.from_pretrained("./moss-audio-tokenizer-full")
        model.codec.has_encoder = False

        with self.assertRaisesRegex(ValueError, "requires MLX codec encoder"):
            assert model.generate is not None
            _ = list(
                model.generate(
                    text="Hello",
                    ref_audio=str(fixture),
                    max_tokens=8,
                    verbose=False,
                )
            )


class TestCodecReconstruction(unittest.TestCase):
    @staticmethod
    def _si_sdr_db(
        reference: np.ndarray, estimate: np.ndarray, eps: float = 1e-8
    ) -> float:
        reference = reference.astype(np.float64, copy=False)
        estimate = estimate.astype(np.float64, copy=False)

        reference = reference - float(np.mean(reference))
        estimate = estimate - float(np.mean(estimate))

        ref_energy = float(np.sum(reference * reference)) + float(eps)
        scale = float(np.sum(estimate * reference)) / ref_energy
        target = scale * reference
        noise = estimate - target

        target_energy = float(np.sum(target * target)) + float(eps)
        noise_energy = float(np.sum(noise * noise)) + float(eps)
        return float(10.0 * np.log10(target_energy / noise_energy))

    @staticmethod
    def _spectral_quality_metrics(
        reference: np.ndarray,
        estimate: np.ndarray,
        n_fft: int = 1024,
        hop: int = 512,
        eps: float = 1e-8,
    ) -> tuple[float, float]:
        ref = reference.astype(np.float64, copy=False)
        est = estimate.astype(np.float64, copy=False)
        n = min(int(ref.shape[0]), int(est.shape[0]))
        ref = ref[:n]
        est = est[:n]

        if n < n_fft:
            pad = int(n_fft - n)
            ref = np.pad(ref, (0, pad))
            est = np.pad(est, (0, pad))
            n = int(n_fft)

        win = np.hanning(int(n_fft)).astype(np.float64)
        frame_count = 1 + (n - int(n_fft)) // int(hop)

        diff_energy = 0.0
        ref_energy = 0.0
        cos_sum = 0.0

        for i in range(int(frame_count)):
            s = i * int(hop)
            ref_f = ref[s : s + int(n_fft)] * win
            est_f = est[s : s + int(n_fft)] * win

            ref_mag = np.abs(np.fft.rfft(ref_f)) + eps
            est_mag = np.abs(np.fft.rfft(est_f)) + eps

            diff = ref_mag - est_mag
            diff_energy += float(np.dot(diff, diff))
            ref_energy += float(np.dot(ref_mag, ref_mag))

            ref_log = np.log(ref_mag)
            est_log = np.log(est_mag)
            cos = float(
                np.dot(ref_log, est_log)
                / (np.linalg.norm(ref_log) * np.linalg.norm(est_log) + eps)
            )
            cos_sum += cos

        spectral_convergence = float(np.sqrt(diff_energy / (ref_energy + eps)))
        logmag_cosine = float(cos_sum / max(int(frame_count), 1))
        return spectral_convergence, logmag_cosine

    def test_codec_reconstructs_fixture_with_high_quality(self):
        from pathlib import Path

        from mlx_audio.audio_io import write as audio_write
        from mlx_audio.utils import load_audio

        model_dir = Path(os.environ.get("MOSS_TTS_MODEL_DIR", "./moss-tts-8bit"))
        codec_dir = model_dir / "moss-audio-tokenizer-full"
        if not codec_dir.is_dir():
            codec_dir = Path("./moss-audio-tokenizer-full")
        if not codec_dir.is_dir():
            self.fail(
                "moss-audio-tokenizer-full not found (expected <MOSS_TTS_MODEL_DIR>/moss-audio-tokenizer-full or ./moss-audio-tokenizer-full)"
            )

        codec = MossAudioTokenizer.from_pretrained(str(codec_dir))
        if not bool(getattr(codec, "has_encoder", False)):
            self.fail(
                f"codec at {codec_dir} is decode-only (no encoder.* weights); reconstruction quality test requires encoder"
            )

        fixture = Path(__file__).resolve().parent / "fixtures" / "ref_audio.wav"
        if not fixture.exists():
            self.fail(f"fixture wav missing: {fixture}")

        sr = int(codec.config.sampling_rate)
        wav = load_audio(str(fixture), sample_rate=sr)
        wav_np = np.array(wav, dtype=np.float32).reshape(-1)

        if wav_np.size < int(codec.config.downsample_rate):
            self.fail(
                f"fixture too short for codec downsample envelope: len={wav_np.size}, downsample_rate={codec.config.downsample_rate}"
            )

        encoded = codec.encode(
            mx.array(wav_np, dtype=mx.float32)[None, None, :],
            num_quantizers=int(codec.config.num_quantizers),
        )
        reconstructed = codec.decode(encoded["audio_codes"])["audio"]
        recon_np = np.array(reconstructed[0, 0], dtype=np.float32)

        self.assertGreaterEqual(
            int(recon_np.shape[0]),
            int(wav_np.shape[0]),
            msg=(
                f"reconstructed audio shorter than source: recon={recon_np.shape[0]}, "
                f"src={wav_np.shape[0]}"
            ),
        )
        recon_np = recon_np[: int(wav_np.shape[0])]
        n = int(wav_np.shape[0])

        trim = min(max(int(codec.config.downsample_rate // 2), 1), n // 10)
        if trim > 0 and n > (2 * trim + 16):
            wav_eval = wav_np[trim:-trim]
            recon_eval = recon_np[trim:-trim]
        else:
            wav_eval = wav_np
            recon_eval = recon_np

        si_sdr_db = self._si_sdr_db(wav_eval, recon_eval)
        ref_centered = wav_eval.astype(np.float64) - float(np.mean(wav_eval))
        recon_centered = recon_eval.astype(np.float64) - float(np.mean(recon_eval))
        corr = float(
            np.dot(ref_centered, recon_centered)
            / (np.linalg.norm(ref_centered) * np.linalg.norm(recon_centered) + 1e-8)
        )
        spectral_convergence, logmag_cosine = self._spectral_quality_metrics(
            wav_eval, recon_eval, n_fft=1024, hop=512
        )
        recon_rms = float(np.sqrt(np.mean(recon_eval.astype(np.float64) ** 2)))

        self.assertTrue(np.isfinite(si_sdr_db), msg=f"non-finite SI-SDR: {si_sdr_db}")
        self.assertGreater(
            recon_rms,
            1e-4,
            msg=f"reconstructed audio is near-silent (rms={recon_rms:.6f})",
        )
        self.assertGreaterEqual(
            corr,
            0.70,
            msg=(
                f"codec reconstruction quality below threshold: corr={corr:.3f} < 0.700, "
                f"si_sdr={si_sdr_db:.2f} dB, codec_dir={codec_dir}, eval_samples={wav_eval.shape[0]}"
            ),
        )
        self.assertLessEqual(
            spectral_convergence,
            0.2,
            msg=(
                f"codec reconstruction quality below threshold: spectral_convergence={spectral_convergence:.3f} > 0.10, "
                f"logmag_cosine={logmag_cosine:.3f}, si_sdr={si_sdr_db:.2f} dB, codec_dir={codec_dir}"
            ),
        )
        self.assertGreaterEqual(
            si_sdr_db,
            10,
            msg=(
                f"si_sdr_db below threshold: si_sdr_db={si_sdr_db:.3f} < 10, "
                f"si_sdr={si_sdr_db:.2f} dB, codec_dir={codec_dir}"
            ),
        )

        should_save_reconstruction = (
            os.environ.get("MOSS_SAVE_CODEC_RECONSTRUCTION", "0").strip() == "1"
        )
        if should_save_reconstruction:
            out_path_raw = os.environ.get(
                "MOSS_CODEC_RECONSTRUCTION_PATH",
                str(Path("./test_moss_output_codec") / "moss_codec_reconstruction.wav"),
            )
            out_path = Path(out_path_raw)
            if out_path.parent:
                out_path.parent.mkdir(parents=True, exist_ok=True)
            audio_write(str(out_path), recon_np, sr, format="wav")


class TestCodecPatchUpsample(unittest.TestCase):
    def _reference_patch_upsample(self, x: mx.array, patch_size: int) -> mx.array:
        b, t, c = x.shape
        p = int(patch_size)
        if p <= 0 or int(c) % p != 0:
            raise ValueError("invalid patch_size")

        y = x.reshape(int(b), int(t), int(c) // p, p)
        y = y.transpose(0, 1, 3, 2)
        y = y.reshape(int(b), int(t) * p, int(c) // p)
        return y

    def test_patchupsample_parity_against_reference_transform(self):
        from mlx_audio.codec.models.moss_audio_tokenizer.moss_audio_tokenizer import (
            _PatchUpsample,
        )

        b, t, p, d = 2, 3, 4, 5
        c = p * d

        x_unique = mx.arange(b * t * c, dtype=mx.float32).reshape(b, t, c)
        up = _PatchUpsample(patch_size=p, input_dim=c)

        y = up(x_unique)
        y_ref = self._reference_patch_upsample(x_unique, patch_size=p)

        np.testing.assert_array_equal(np.array(y), np.array(y_ref))

    def test_patchupsample_temporal_scaling_and_shape_invariants(self):
        from mlx_audio.codec.models.moss_audio_tokenizer.moss_audio_tokenizer import (
            _PatchUpsample,
        )

        b, t, p, d = 1, 7, 3, 2
        c = p * d
        x = mx.zeros((b, t, c), dtype=mx.float32)
        y = _PatchUpsample(patch_size=p, input_dim=c)(x)

        self.assertEqual(tuple(y.shape), (b, t * p, d))

        x2 = mx.arange(t, dtype=mx.float32).reshape(1, t, 1)
        x2 = mx.tile(x2, (1, 1, c))
        y2 = _PatchUpsample(patch_size=p, input_dim=c)(x2)
        y2_np = np.array(y2[0, :, 0])

        expected = np.repeat(np.arange(t, dtype=np.float32), p)
        np.testing.assert_array_equal(y2_np, expected)

    def test_patchupsample_invalid_patch_size_guard_raises(self):
        from mlx_audio.codec.models.moss_audio_tokenizer.moss_audio_tokenizer import (
            _PatchUpsample,
        )

        x = mx.zeros((1, 2, 10), dtype=mx.float32)
        up = _PatchUpsample(patch_size=4, input_dim=10)
        with self.assertRaisesRegex(ValueError, r"^invalid patch_size$"):
            _ = up(x)


if __name__ == "__main__":
    unittest.main()
