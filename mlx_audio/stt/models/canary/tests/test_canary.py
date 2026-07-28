import base64
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.stt.models.canary.canary import CanaryEncoder, Model
from mlx_audio.stt.models.canary.config import (
    DecoderConfig,
    EncoderConfig,
    ModelConfig,
    PreprocessorConfig,
)
from mlx_audio.stt.models.canary.decoder import (
    CanaryDecoder,
    FixedPositionalEncoding,
    MultiHeadCrossAttention,
    MultiHeadSelfAttention,
    TransformerDecoderBlock,
)
from mlx_audio.stt.models.canary.tokenizer import CanaryTokenizer


def _small_encoder_config():
    return EncoderConfig(
        feat_in=16,
        n_layers=2,
        d_model=32,
        n_heads=4,
        ff_expansion_factor=2,
        subsampling_factor=2,
        self_attention_model="rel_pos",
        subsampling="dw_striding",
        conv_kernel_size=3,
        subsampling_conv_channels=16,
        pos_emb_max_len=256,
        xscaling=True,
    )


def _small_decoder_config():
    return DecoderConfig(
        num_layers=2,
        hidden_size=32,
        num_attention_heads=4,
        inner_size=64,
    )


def _small_model_config():
    return ModelConfig(
        model_type="canary",
        preprocessor=PreprocessorConfig(features=16),
        encoder=_small_encoder_config(),
        transf_decoder=_small_decoder_config(),
        vocab_size=64,
        enc_output_dim=32,
    )


class TestConfig(unittest.TestCase):

    def test_preprocessor_config_defaults(self):
        config = PreprocessorConfig()
        self.assertEqual(config.sample_rate, 16000)
        self.assertEqual(config.features, 128)
        self.assertEqual(config.normalize, "per_feature")

    def test_preprocessor_win_hop_length(self):
        config = PreprocessorConfig(
            sample_rate=16000, window_size=0.025, window_stride=0.01
        )
        self.assertEqual(config.win_length, 400)
        self.assertEqual(config.hop_length, 160)

    def test_encoder_config_from_dict(self):
        d = {"feat_in": 128, "n_layers": 32, "d_model": 1024, "n_heads": 8}
        config = EncoderConfig.from_dict(d)
        self.assertEqual(config.n_layers, 32)
        self.assertEqual(config.d_model, 1024)

    def test_decoder_config_from_dict(self):
        d = {
            "num_layers": 8,
            "hidden_size": 1024,
            "num_attention_heads": 8,
            "inner_size": 4096,
        }
        config = DecoderConfig.from_dict(d)
        self.assertEqual(config.num_layers, 8)
        self.assertEqual(config.inner_size, 4096)

    def test_decoder_config_from_nested_dict(self):
        d = {
            "decoder": {
                "num_layers": 8,
                "hidden_size": 1024,
                "num_attention_heads": 8,
                "inner_size": 4096,
            }
        }
        config = DecoderConfig.from_dict(d)
        self.assertEqual(config.num_layers, 8)

    def test_model_config_from_dict(self):
        d = {
            "model_type": "canary",
            "preprocessor": {"features": 128, "sample_rate": 16000},
            "encoder": {"feat_in": 128, "n_layers": 32, "d_model": 1024, "n_heads": 8},
            "transf_decoder": {
                "num_layers": 8,
                "hidden_size": 1024,
                "num_attention_heads": 8,
                "inner_size": 4096,
            },
            "vocab_size": 16384,
            "enc_output_dim": 1024,
        }
        config = ModelConfig.from_dict(d)
        self.assertEqual(config.vocab_size, 16384)
        self.assertIsInstance(config.encoder, EncoderConfig)
        self.assertIsInstance(config.transf_decoder, DecoderConfig)

    def test_model_config_ignores_extra_keys(self):
        d = {
            "model_type": "canary",
            "unknown_field": 42,
            "preprocessor": {"features": 128, "extra": True},
        }
        config = ModelConfig.from_dict(d)
        self.assertEqual(config.model_type, "canary")


class TestMultiHeadSelfAttention(unittest.TestCase):

    def test_output_shape(self):
        attn = MultiHeadSelfAttention(d_model=32, n_heads=4)
        x = mx.random.normal((1, 10, 32))
        out, cache = attn(x)
        mx.eval(out)
        self.assertEqual(out.shape, (1, 10, 32))

    def test_cache_update(self):
        attn = MultiHeadSelfAttention(d_model=32, n_heads=4)
        x1 = mx.random.normal((1, 5, 32))
        out1, cache1 = attn(x1)
        mx.eval(out1)

        x2 = mx.random.normal((1, 1, 32))
        out2, cache2 = attn(x2, cache=cache1)
        mx.eval(out2)

        self.assertEqual(cache2[0].shape[2], 6)  # 5 + 1


class TestMultiHeadCrossAttention(unittest.TestCase):

    def test_output_shape(self):
        attn = MultiHeadCrossAttention(d_model=32, n_heads=4)
        x = mx.random.normal((1, 5, 32))
        enc = mx.random.normal((1, 20, 32))
        out, cache = attn(x, enc)
        mx.eval(out)
        self.assertEqual(out.shape, (1, 5, 32))

    def test_cache_reuse(self):
        attn = MultiHeadCrossAttention(d_model=32, n_heads=4)
        enc = mx.random.normal((1, 20, 32))

        x1 = mx.random.normal((1, 5, 32))
        out1, cache1 = attn(x1, enc)
        mx.eval(out1)

        x2 = mx.random.normal((1, 1, 32))
        out2, cache2 = attn(x2, enc, cache=cache1)
        mx.eval(out2)
        self.assertEqual(out2.shape, (1, 1, 32))


class TestTransformerDecoderBlock(unittest.TestCase):

    def test_output_shape(self):
        block = TransformerDecoderBlock(d_model=32, n_heads=4, inner_size=64)
        x = mx.random.normal((1, 5, 32))
        enc = mx.random.normal((1, 20, 32))
        out, self_cache, cross_cache = block(x, enc)
        mx.eval(out)
        self.assertEqual(out.shape, (1, 5, 32))


class TestCanaryDecoder(unittest.TestCase):

    def test_output_shape(self):
        config = _small_decoder_config()
        decoder = CanaryDecoder(config, vocab_size=64, d_model=32)
        tokens = mx.array([[1, 2, 3]], dtype=mx.int32)
        enc = mx.random.normal((1, 20, 32))
        logits, cache = decoder(tokens, enc)
        mx.eval(logits)
        self.assertEqual(logits.shape, (1, 3, 64))
        self.assertEqual(len(cache), 2)

    def test_autoregressive_step(self):
        config = _small_decoder_config()
        decoder = CanaryDecoder(config, vocab_size=64, d_model=32)
        enc = mx.random.normal((1, 20, 32))

        prompt = mx.array([[1, 2, 3]], dtype=mx.int32)
        logits, cache = decoder(prompt, enc, start_pos=0)
        mx.eval(logits)

        next_token = mx.array([[4]], dtype=mx.int32)
        logits2, cache2 = decoder(next_token, enc, cache=cache, start_pos=3)
        mx.eval(logits2)
        self.assertEqual(logits2.shape, (1, 1, 64))


class TestModelSanitize(unittest.TestCase):

    def setUp(self):
        self.config = _small_model_config()
        self.model = Model(self.config)

    def test_encoder_key_mapping(self):
        weights = {"encoder.layers.0.self_attn.linear_q.weight": mx.zeros((32, 32))}
        sanitized = self.model.sanitize(weights)
        self.assertIn("encoder.conformer.layers.0.self_attn.linear_q.weight", sanitized)

    def test_decoder_embedding_mapping(self):
        weights = {
            "transf_decoder._embedding.token_embedding.weight": mx.zeros((64, 32))
        }
        sanitized = self.model.sanitize(weights)
        self.assertIn("decoder.embedding.weight", sanitized)

    def test_decoder_position_mapping(self):
        weights = {
            "transf_decoder._embedding.position_embedding.pos_enc": mx.zeros((1024, 32))
        }
        sanitized = self.model.sanitize(weights)
        self.assertIn("decoder.position_embedding.pos_enc", sanitized)

    def test_decoder_layer_self_attn_mapping(self):
        weights = {
            "transf_decoder._decoder.layers.0.first_sub_layer.query_net.weight": mx.zeros(
                (32, 32)
            )
        }
        sanitized = self.model.sanitize(weights)
        self.assertIn("decoder.blocks.0.self_attn.q_proj.weight", sanitized)

    def test_decoder_layer_cross_attn_mapping(self):
        weights = {
            "transf_decoder._decoder.layers.0.second_sub_layer.key_net.weight": mx.zeros(
                (32, 32)
            )
        }
        sanitized = self.model.sanitize(weights)
        self.assertIn("decoder.blocks.0.cross_attn.k_proj.weight", sanitized)

    def test_decoder_layer_ffn_mapping(self):
        weights = {
            "transf_decoder._decoder.layers.0.third_sub_layer.dense_in.weight": mx.zeros(
                (64, 32)
            )
        }
        sanitized = self.model.sanitize(weights)
        self.assertIn("decoder.blocks.0.ff1.weight", sanitized)

    def test_decoder_layer_norm_mapping(self):
        weights = {
            "transf_decoder._decoder.layers.0.layer_norm_1.weight": mx.zeros((32,)),
            "transf_decoder._decoder.layers.0.layer_norm_2.weight": mx.zeros((32,)),
            "transf_decoder._decoder.layers.0.layer_norm_3.weight": mx.zeros((32,)),
        }
        sanitized = self.model.sanitize(weights)
        self.assertIn("decoder.blocks.0.self_attn_norm.weight", sanitized)
        self.assertIn("decoder.blocks.0.cross_attn_norm.weight", sanitized)
        self.assertIn("decoder.blocks.0.ff_norm.weight", sanitized)

    def test_decoder_final_norm_mapping(self):
        weights = {"transf_decoder._decoder.final_layer_norm.weight": mx.zeros((32,))}
        sanitized = self.model.sanitize(weights)
        self.assertIn("decoder.final_norm.weight", sanitized)

    def test_output_proj_mapping(self):
        weights = {"log_softmax.mlp.layer0.weight": mx.zeros((64, 32))}
        sanitized = self.model.sanitize(weights)
        self.assertIn("decoder.output_proj.weight", sanitized)

    def test_skips_dropout_keys(self):
        weights = {
            "transf_decoder._decoder.layers.0.first_sub_layer.attn_dropout.p": mx.array(
                [0.1]
            ),
            "transf_decoder._decoder.layers.0.first_sub_layer.layer_dropout.p": mx.array(
                [0.1]
            ),
        }
        sanitized = self.model.sanitize(weights)
        self.assertEqual(len(sanitized), 0)

    def test_conv1d_transpose(self):
        weights = {
            "encoder.layers.0.conv.pointwise_conv1.weight": mx.zeros((64, 32, 1))
        }
        sanitized = self.model.sanitize(weights)
        key = "encoder.conformer.layers.0.conv.pointwise_conv1.weight"
        self.assertEqual(sanitized[key].shape, (64, 1, 32))

    def test_conv2d_transpose(self):
        weights = {"encoder.pre_encode.conv.0.weight": mx.zeros((256, 1, 3, 3))}
        sanitized = self.model.sanitize(weights)
        key = "encoder.conformer.pre_encode.conv.0.weight"
        self.assertEqual(sanitized[key].shape, (256, 3, 3, 1))

    def test_skips_encoder_decoder_proj(self):
        weights = {"encoder_decoder_proj.weight": mx.zeros((32, 32))}
        sanitized = self.model.sanitize(weights)
        self.assertEqual(len(sanitized), 0)


class TestModelSanitizeMLXNative(unittest.TestCase):
    """MLX-native conversions (e.g. qfuxa/canary-mlx, Mediform/canary-1b-v2-mlx-q8)
    already use MLX tensor layouts and flattened key names, so they must be
    remapped *without* re-transposing conv weights."""

    def setUp(self):
        self.model = Model(_small_model_config())

    def _sanitize(self, extra):
        # A "head.classifier" key marks the checkpoint as MLX-native so that
        # format detection selects the right path.
        weights = {"head.classifier.weight": mx.zeros((64, 32))}
        weights.update(extra)
        return self.model.sanitize(weights)

    def test_detected_via_head_classifier(self):
        sanitized = self.model.sanitize({"head.classifier.weight": mx.zeros((64, 32))})
        self.assertIn("decoder.output_proj.weight", sanitized)

    def test_detected_via_decoder_layers(self):
        weights = {
            "transf_decoder.layers.0.first_sub_layer.linear_q.weight": mx.zeros(
                (32, 32)
            )
        }
        sanitized = self.model.sanitize(weights)
        self.assertIn("decoder.blocks.0.self_attn.q_proj.weight", sanitized)

    def test_token_embedding_and_norm_mapping(self):
        sanitized = self._sanitize(
            {
                "transf_decoder.token_embedding.weight": mx.zeros((64, 32)),
                "transf_decoder.embedding_layer_norm.weight": mx.zeros((32,)),
                "transf_decoder.final_layer_norm.weight": mx.zeros((32,)),
            }
        )
        self.assertIn("decoder.embedding.weight", sanitized)
        self.assertIn("decoder.embedding_layer_norm.weight", sanitized)
        self.assertIn("decoder.final_norm.weight", sanitized)

    def test_self_and_cross_attn_mapping(self):
        sanitized = self._sanitize(
            {
                "transf_decoder.layers.0.first_sub_layer.linear_out.weight": mx.zeros(
                    (32, 32)
                ),
                "transf_decoder.layers.0.second_sub_layer.linear_v.weight": mx.zeros(
                    (32, 32)
                ),
            }
        )
        self.assertIn("decoder.blocks.0.self_attn.out_proj.weight", sanitized)
        self.assertIn("decoder.blocks.0.cross_attn.v_proj.weight", sanitized)

    def test_ffn_and_layer_norm_mapping(self):
        sanitized = self._sanitize(
            {
                "transf_decoder.layers.0.third_sub_layer.linear1.weight": mx.zeros(
                    (64, 32)
                ),
                "transf_decoder.layers.0.third_sub_layer.linear2.weight": mx.zeros(
                    (32, 64)
                ),
                "transf_decoder.layers.0.layer_norm_1.weight": mx.zeros((32,)),
                "transf_decoder.layers.0.layer_norm_2.weight": mx.zeros((32,)),
                "transf_decoder.layers.0.layer_norm_3.weight": mx.zeros((32,)),
            }
        )
        self.assertIn("decoder.blocks.0.ff1.weight", sanitized)
        self.assertIn("decoder.blocks.0.ff2.weight", sanitized)
        self.assertIn("decoder.blocks.0.self_attn_norm.weight", sanitized)
        self.assertIn("decoder.blocks.0.cross_attn_norm.weight", sanitized)
        self.assertIn("decoder.blocks.0.ff_norm.weight", sanitized)

    def test_conv_weights_not_transposed(self):
        # Already (out, kH, kW, in) and (out, kW, in) — must be left untouched.
        sanitized = self._sanitize(
            {
                "encoder.pre_encode.conv.0.weight": mx.zeros((256, 3, 3, 1)),
                "encoder.layers.0.conv.depthwise_conv.weight": mx.zeros((1024, 9, 1)),
            }
        )
        self.assertEqual(
            sanitized["encoder.conformer.pre_encode.conv.0.weight"].shape,
            (256, 3, 3, 1),
        )
        self.assertEqual(
            sanitized["encoder.conformer.layers.0.conv.depthwise_conv.weight"].shape,
            (1024, 9, 1),
        )

    def test_preserves_quantization_tensors(self):
        # scales/biases of a quantized linear must ride along with the weight.
        sanitized = self._sanitize(
            {
                "transf_decoder.layers.0.first_sub_layer.linear_q.weight": mx.zeros(
                    (32, 8)
                ),
                "transf_decoder.layers.0.first_sub_layer.linear_q.scales": mx.zeros(
                    (32, 1)
                ),
                "transf_decoder.layers.0.first_sub_layer.linear_q.biases": mx.zeros(
                    (32, 1)
                ),
            }
        )
        self.assertIn("decoder.blocks.0.self_attn.q_proj.scales", sanitized)
        self.assertIn("decoder.blocks.0.self_attn.q_proj.biases", sanitized)

    def test_third_sub_layer_unknown_key_warns(self):
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sanitized = self._sanitize(
                {
                    "transf_decoder.layers.0.third_sub_layer.unknown_key.weight": mx.zeros(
                        (32, 32)
                    )
                }
            )
        self.assertEqual(len(w), 1)
        self.assertIn("RuntimeWarning", str(w[0].category))
        self.assertIn("third_sub_layer", str(w[0].message))
        # Key passes through with bare suffix, not prefixed with ff_
        self.assertIn("decoder.blocks.0.unknown_key.weight", sanitized)

    def test_unknown_sub_layer_prefix_warns(self):
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sanitized = self._sanitize(
                {"transf_decoder.layers.0.future_sublayer.weight": mx.zeros((32, 32))}
            )
        self.assertEqual(len(w), 1)
        self.assertIn("RuntimeWarning", str(w[0].category))
        self.assertIn("future_sublayer", str(w[0].message))
        # Key passes through unchanged
        self.assertIn("decoder.blocks.0.future_sublayer.weight", sanitized)


class TestFixedPositionalEncoding(unittest.TestCase):
    """The positional encoding is a computed sinusoidal table (NeMo divides it by
    sqrt(d_model)); it must not regress to the all-zeros stub it used to be."""

    def test_table_is_sinusoidal_not_zeros(self):
        d_model, max_len = 32, 16
        pe = FixedPositionalEncoding(d_model, max_len=max_len)
        table = pe._pos_enc
        mx.eval(table)
        self.assertEqual(table.shape, (max_len, d_model))
        self.assertGreater(float(mx.max(mx.abs(table))), 0.0)

        # Position 0: sin(0)=0 at even indices, cos(0)=1 at odd indices, all /sqrt(d).
        row0 = table[0]
        mx.eval(row0)
        self.assertAlmostEqual(float(row0[0]), 0.0, places=5)
        self.assertAlmostEqual(float(row0[1]), 1.0 / math.sqrt(d_model), places=5)

    def test_pos_enc_not_a_module_parameter(self):
        pe = FixedPositionalEncoding(32, max_len=16)
        params = pe.parameters()
        # _pos_enc must NOT appear in parameters() so it is never serialized
        # or treated as a trainable weight.
        flat = dict(pe.parameters())
        self.assertNotIn("_pos_enc", flat)
        self.assertNotIn("pos_enc", flat)

    def test_lookup_shape(self):
        pe = FixedPositionalEncoding(32, max_len=16)
        out = pe(mx.array([[0, 1, 2]]))
        mx.eval(out)
        self.assertEqual(out.shape, (1, 3, 32))


class TestEmbeddedTokenizer(unittest.TestCase):
    """Some conversions (e.g. Mediform/canary-1b-v2-mlx-q8) embed the
    SentencePiece model as base64 in config.json instead of shipping a file."""

    @staticmethod
    def _train_tiny_sp_proto() -> bytes:
        import sentencepiece as spm

        lines = [
            "the quick brown fox jumps over the lazy dog",
            "hello world this is a test of sentencepiece",
            "canary models transcribe speech to text",
            "machine learning with mlx on apple silicon",
        ] * 30
        model_io = io.BytesIO()
        spm.SentencePieceTrainer.train(
            sentence_iterator=iter(lines),
            model_writer=model_io,
            vocab_size=64,
            model_type="bpe",
            character_coverage=1.0,
            unk_id=0,
            bos_id=-1,
            eos_id=-1,
            pad_id=-1,
        )
        return model_io.getvalue()

    def test_tokenizer_from_proto_roundtrips(self):
        proto = self._train_tiny_sp_proto()
        tok = CanaryTokenizer(model_proto=proto)
        self.assertEqual(tok.vocab_size, 64)
        ids = tok.encode("hello world")
        self.assertTrue(all(isinstance(i, int) for i in ids))
        self.assertEqual(tok.decode(ids), "hello world")

    def test_post_load_hook_reads_embedded_base64(self):
        proto = self._train_tiny_sp_proto()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            with open(path / "config.json", "w") as f:
                json.dump(
                    {"tokenizer": {"model_base64": base64.b64encode(proto).decode()}}, f
                )
            self.assertEqual(Model._load_embedded_tokenizer_proto(path), proto)

            model = Model(_small_model_config())
            model = Model.post_load_hook(model, path)
            self.assertIsNotNone(model._tokenizer)
            self.assertEqual(
                model._tokenizer.decode(model._tokenizer.encode("text")), "text"
            )

    def test_post_load_hook_no_tokenizer_when_absent(self):
        with tempfile.TemporaryDirectory() as d:
            model = Model(_small_model_config())
            import warnings

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                model = Model.post_load_hook(model, Path(d))
            self.assertIsNone(model._tokenizer)
            self.assertEqual(len(w), 1)
            self.assertIn("RuntimeWarning", str(w[0].category))
            self.assertIn("No tokenizer found", str(w[0].message))

    def test_post_load_hook_warns_when_config_has_no_tokenizer_key(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            with open(path / "config.json", "w") as f:
                json.dump({"model_type": "canary"}, f)
            model = Model(_small_model_config())
            import warnings

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                model = Model.post_load_hook(model, path)
            self.assertIsNone(model._tokenizer)
            self.assertEqual(len(w), 1)
            self.assertIn("No tokenizer found", str(w[0].message))

    def test_malformed_base64_returns_none_with_warning(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            with open(path / "config.json", "w") as f:
                json.dump({"tokenizer": {"model_base64": "not-valid-base64!!!"}}, f)
            import warnings

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = Model._load_embedded_tokenizer_proto(path)
            self.assertIsNone(result)
            self.assertEqual(len(w), 1)
            self.assertIn("RuntimeWarning", str(w[0].category))

    def test_tokens_path_only_tokenizer_raises_on_decode(self):
        tok = CanaryTokenizer.__new__(CanaryTokenizer)
        tok.token2id = {}
        tok.id2token = {}
        with self.assertRaises(RuntimeError):
            tok.decode([1, 2, 3])
        with self.assertRaises(RuntimeError):
            tok.encode("hello")


class TestSanitizeAlreadySanitized(unittest.TestCase):
    """Weights already in internal MLX format (e.g. from a re-saved checkpoint)
    must pass through sanitize() unchanged — no key remapping, no conv transpose."""

    def setUp(self):
        self.model = Model(_small_model_config())

    def test_already_sanitized_passthrough(self):
        weights = {
            "decoder.blocks.0.self_attn.q_proj.weight": mx.zeros((32, 32)),
            "encoder.conformer.pre_encode.conv.0.weight": mx.zeros((256, 3, 3, 1)),
            "decoder.output_proj.weight": mx.zeros((64, 32)),
        }
        sanitized = self.model.sanitize(weights)
        # Keys must be unchanged.
        self.assertIn("decoder.blocks.0.self_attn.q_proj.weight", sanitized)
        self.assertIn("encoder.conformer.pre_encode.conv.0.weight", sanitized)
        self.assertIn("decoder.output_proj.weight", sanitized)
        # Conv weight must NOT be transposed (already in MLX layout).
        self.assertEqual(
            sanitized["encoder.conformer.pre_encode.conv.0.weight"].shape,
            (256, 3, 3, 1),
        )


class TestModel(unittest.TestCase):

    def setUp(self):
        self.config = _small_model_config()
        self.model = Model(self.config)

    def test_model_init(self):
        self.assertIsInstance(self.model.encoder, CanaryEncoder)
        self.assertIsInstance(self.model.decoder, CanaryDecoder)

    def test_sample_rate(self):
        self.assertEqual(self.model.sample_rate, 16000)


if __name__ == "__main__":
    unittest.main()
