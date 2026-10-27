import unittest
from dataclasses import asdict
from unittest.mock import MagicMock

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from mlx_audio.lm.models.bailing_moe_v3 import Gate
from mlx_audio.lm.models.bailing_moe_v3 import Model as LanguageModel
from mlx_audio.lm.models.bailing_moe_v3 import ModelArgs, sanitize_backbone
from mlx_audio.stt.models.ming_v3_asr import AudioConfig, Model, ModelConfig
from mlx_audio.stt.models.ming_v3_asr.ming_v3_asr import decode_transcript


def tiny_text_config(**overrides):
    return ModelArgs(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        vocab_size=32,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        q_lora_rank=4,
        kv_lora_rank=4,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        v_head_dim=8,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        moe_shared_expert_intermediate_size=8,
        n_group=2,
        topk_group=1,
        **overrides,
    )


def tiny_audio_model():
    return Model(
        ModelConfig(
            text_config=asdict(tiny_text_config()),
            audio_config=dict(
                n_mels=4,
                n_audio_ctx=16,
                n_audio_state=8,
                n_audio_head=2,
                n_audio_layer=1,
                whisper_chunk_size=16,
            ),
            audio_token_id=5,
        )
    )


class TestMingBackbone(unittest.TestCase):
    def test_prefill_cached_and_chunked_decoding_agree(self):
        # CPU FP32 isolates the cache equations from Metal's GEMM precision modes.
        with mx.stream(mx.cpu):
            mx.random.seed(17)
            model = LanguageModel(tiny_text_config())
            model.eval()
            ids = mx.array([[1, 2, 3, 4, 5, 6]])
            expected = model(ids)
            for chunks in ((1, 1, 1, 1, 1, 1), (2, 3, 1)):
                cache = model.make_cache()
                pieces, start = [], 0
                for length in chunks:
                    pieces.append(model(ids[:, start : start + length], cache))
                    start += length
                np.testing.assert_allclose(
                    np.array(mx.concatenate(pieces, axis=1)),
                    np.array(expected),
                    atol=1e-5,
                    rtol=1e-5,
                )

    def test_direct_query_projection(self):
        args = tiny_text_config()
        args.q_lora_rank = None
        args.num_hidden_layers = 2
        args.layer_group_size = 6
        args.first_k_dense_replace = 2
        model = LanguageModel(args)
        output = model(mx.array([[1, 2]]))
        self.assertEqual(output.shape, (1, 2, 32))
        self.assertTrue(all(args.uses_mla(i) for i in range(2)))

    def test_routing_uses_bias_only_for_selection_and_masks_excluded_groups(self):
        args = tiny_text_config()
        gate = Gate(args)
        gate.weight = mx.zeros((4, 16))
        # All corrected scores are negative; excluded groups must be -inf, not zero.
        gate.expert_bias = mx.array([-2.0, -3.0, -4.0, -5.0])
        selected, weights = gate(mx.ones((1, 1, 16)))
        self.assertEqual(set(selected.flatten().tolist()), {0, 1})
        np.testing.assert_allclose(np.array(weights), [[[1.25, 1.25]]])

    def test_sanitize_packs_experts_and_excludes_mtp(self):
        args = tiny_text_config()
        weights = {"model.layers.4.input_layernorm.weight": mx.ones((16,))}
        for projection in ("gate_proj", "up_proj", "down_proj"):
            for expert in range(args.num_experts):
                weights[f"model.layers.1.mlp.experts.{expert}.{projection}.weight"] = (
                    mx.full((8, 16), expert)
                )
        packed = sanitize_backbone(weights, args)
        self.assertNotIn("model.layers.4.input_layernorm.weight", packed)
        self.assertEqual(len(packed), 3)
        actual = packed["model.layers.1.mlp.switch_mlp.gate_proj.weight"]
        self.assertEqual(actual.shape, (4, 8, 16))
        self.assertEqual(actual[:, 0, 0].tolist(), [0, 1, 2, 3])

    def test_incomplete_experts_fail(self):
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            sanitize_backbone(
                {"model.layers.1.mlp.experts.0.gate_proj.weight": mx.ones((8, 16))},
                tiny_text_config(),
            )


class TestMingAudio(unittest.TestCase):
    def test_chunk_length_contract(self):
        config = AudioConfig(whisper_chunk_size=3000)
        for frames, expected in ((1, 1), (4, 1), (5, 2), (3000, 750), (3001, 751)):
            self.assertEqual(config.token_count(frames), expected)
        multi = AudioConfig(
            downsampler_kernel_size_list=[3, 3, 3], downsampler_stride_list=[2, 2, 2]
        )
        self.assertEqual(multi.token_count(3001), 189)

    def test_chunks_are_encoded_independently(self):
        model = tiny_audio_model()
        model.eval()
        features = mx.random.normal((21, 4))
        combined = model.encode_audio(features)
        separate = mx.concatenate(
            [model.encode_audio(features[:16]), model.encode_audio(features[16:])],
            axis=1,
        )
        self.assertEqual(combined.shape[1], model.config.audio_config.token_count(21))
        np.testing.assert_allclose(np.array(combined), np.array(separate), atol=1e-6)

    def test_audio_placeholders_are_exact(self):
        model = tiny_audio_model()
        encoded = mx.full((1, 2, 16), 7.0)
        ids, embeds = model.inject_audio([1, 5, 2, 5, 3], encoded)
        self.assertEqual(ids.shape, (1, 5))
        np.testing.assert_array_equal(np.array(embeds[0, [1, 3]]), np.array(encoded[0]))
        with self.assertRaisesRegex(ValueError, "placeholder"):
            model.inject_audio([1, 5], encoded)

    def test_sanitize_conv_layout_is_idempotent(self):
        model = tiny_audio_model()
        parameters = dict(tree_flatten(model.parameters()))
        original = {
            key: (
                value.transpose(0, 2, 1)
                if value.ndim == 3
                and (
                    key.startswith("model.audio.conv")
                    or key.startswith("linear_proj_audio")
                )
                else value
            )
            for key, value in parameters.items()
        }
        converted = model.sanitize(original)
        twice = model.sanitize(converted)
        model.load_weights(list(twice.items()), strict=True)
        for key in original:
            self.assertEqual(twice[key].shape, parameters[key].shape)

    def test_language_envelope(self):
        self.assertEqual(
            decode_transcript("English\tHello there."), ("Hello there.", "English")
        )
        self.assertEqual(decode_transcript("Chinese 你好"), ("你好", "Chinese"))
        self.assertEqual(decode_transcript("Hello there."), ("Hello there.", None))

    def test_decode_respects_checkpoint_eos(self):
        model = tiny_audio_model()
        model.tokenizer = MagicMock(eos_token_id=30)
        model.tokenizer.convert_tokens_to_ids.return_value = 29
        model.config.text_config.eos_token_id = 31
        model.lm_head = lambda x: mx.concatenate(
            [mx.zeros((1, 31)), mx.ones((1, 1))], axis=-1
        )
        tokens, _, _ = model.decode_tokens(mx.ones((1, 2, 16)), max_tokens=2)
        self.assertEqual(tokens, [])

    def test_cli_token_budget_reserves_prompt_positions(self):
        model = tiny_audio_model()
        model.config.text_config.max_position_embeddings = 100
        model.tokenizer = MagicMock()
        model.tokenizer.encode.side_effect = lambda *args, **kwargs: [1]
        model.tokenizer.decode.return_value = "English\tHello"
        model.audio_tokens = ("<audio>", "</audio>")
        model.prepare_audio = MagicMock(return_value=mx.zeros((4, 4)))
        model.encode_audio = MagicMock(return_value=mx.zeros((1, 2, 16)))
        model.inject_audio = MagicMock(return_value=(None, mx.zeros((1, 4, 16))))
        model.decode_tokens = MagicMock(return_value=([1], 0.01, 0.02))
        result = model.generate(np.zeros(400), max_tokens=8192)
        self.assertEqual(model.decode_tokens.call_args.args[1], 96)
        self.assertEqual(result.text, "Hello")


if __name__ == "__main__":
    unittest.main()
