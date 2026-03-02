import unittest
from unittest.mock import MagicMock, patch

import mlx.core as mx
import numpy as np


class TestModelConfig(unittest.TestCase):
    def setUp(self):
        from mlx_audio.lid.models.wav2vec2.config import ModelConfig

        self.ModelConfig = ModelConfig

    def test_default_values(self):
        config = self.ModelConfig()
        self.assertEqual(config.classifier_proj_size, 256)
        self.assertEqual(config.num_labels, 2)
        self.assertIsNone(config.id2label)

    def test_custom_values(self):
        config = self.ModelConfig(
            classifier_proj_size=1024,
            num_labels=256,
            hidden_size=1024,
        )
        self.assertEqual(config.classifier_proj_size, 1024)
        self.assertEqual(config.num_labels, 256)

    def test_id2label_sets_num_labels(self):
        id2label = {str(i): f"lang_{i}" for i in range(10)}
        config = self.ModelConfig(id2label=id2label)
        self.assertEqual(config.num_labels, 10)

    def test_inherits_wav2vec2_fields(self):
        config = self.ModelConfig(
            hidden_size=768,
            num_attention_heads=12,
            num_hidden_layers=12,
        )
        self.assertEqual(config.hidden_size, 768)
        self.assertEqual(config.num_attention_heads, 12)
        self.assertEqual(config.num_hidden_layers, 12)


class TestWav2Vec2ForSequenceClassification(unittest.TestCase):
    def setUp(self):
        from mlx_audio.lid.models.wav2vec2.config import ModelConfig
        from mlx_audio.lid.models.wav2vec2.wav2vec_lid import (
            Wav2Vec2ForSequenceClassification,
        )

        self.ModelConfig = ModelConfig
        self.Model = Wav2Vec2ForSequenceClassification

        self.config = ModelConfig(
            hidden_size=64,
            num_attention_heads=2,
            num_hidden_layers=2,
            intermediate_size=128,
            num_feat_extract_layers=2,
            conv_dim=[32, 64],
            conv_stride=[5, 2],
            conv_kernel=[10, 3],
            classifier_proj_size=32,
            num_labels=10,
        )

    def test_model_init(self):
        model = self.Model(self.config)
        self.assertIsNotNone(model.wav2vec2)
        self.assertIsNotNone(model.projector)
        self.assertIsNotNone(model.classifier)

    def test_classifier_output_dim(self):
        model = self.Model(self.config)
        self.assertEqual(model.classifier.weight.shape[0], self.config.num_labels)

    def test_projector_dims(self):
        model = self.Model(self.config)
        self.assertEqual(
            model.projector.weight.shape[0], self.config.classifier_proj_size
        )
        self.assertEqual(model.projector.weight.shape[1], self.config.hidden_size)

    def test_forward_output_shape(self):
        model = self.Model(self.config)
        audio = mx.random.normal((1, 1600))
        logits = model(audio)
        mx.eval(logits)
        self.assertEqual(logits.shape[0], 1)
        self.assertEqual(logits.shape[1], self.config.num_labels)

    def test_forward_batch(self):
        model = self.Model(self.config)
        audio = mx.random.normal((2, 1600))
        logits = model(audio)
        mx.eval(logits)
        self.assertEqual(logits.shape[0], 2)
        self.assertEqual(logits.shape[1], self.config.num_labels)

    def test_sanitize_keeps_wav2vec2_prefix(self):
        model = self.Model(self.config)
        weights = {
            "wav2vec2.feature_extractor.conv_layers.0.conv.weight": mx.zeros(
                (32, 1, 10)
            ),
            "projector.weight": mx.zeros((32, 64)),
            "classifier.weight": mx.zeros((10, 32)),
        }
        sanitized = model.sanitize(weights)
        self.assertIn("wav2vec2.feature_extractor.conv_layers.0.conv.weight", sanitized)
        self.assertIn("projector.weight", sanitized)
        self.assertIn("classifier.weight", sanitized)

    def test_sanitize_conv_axis_swap(self):
        model = self.Model(self.config)
        weights = {
            "wav2vec2.feature_extractor.conv_layers.0.conv.weight": mx.zeros(
                (32, 1, 10)
            ),
        }
        sanitized = model.sanitize(weights)
        self.assertEqual(
            sanitized["wav2vec2.feature_extractor.conv_layers.0.conv.weight"].shape,
            (32, 10, 1),
        )

    def test_sanitize_drops_quantizer_keys(self):
        model = self.Model(self.config)
        weights = {
            "quantizer.weight_proj.weight": mx.zeros((10, 10)),
            "project_q.weight": mx.zeros((10, 10)),
            "masked_spec_embed": mx.zeros((64,)),
            "lm_head.weight": mx.zeros((10, 10)),
            "wav2vec2.encoder.layers.0.feed_forward.output_dense.weight": mx.zeros(
                (64, 128)
            ),
        }
        sanitized = model.sanitize(weights)
        self.assertNotIn("quantizer.weight_proj.weight", sanitized)
        self.assertNotIn("project_q.weight", sanitized)
        self.assertNotIn("masked_spec_embed", sanitized)
        self.assertNotIn("lm_head.weight", sanitized)
        self.assertIn(
            "wav2vec2.encoder.layers.0.feed_forward.output_dense.weight", sanitized
        )

    def test_sanitize_parametrize_weight_norm(self):
        model = self.Model(self.config)
        weights = {
            "wav2vec2.encoder.pos_conv_embed.conv.parametrizations.weight.original0": mx.zeros(
                (64, 1, 128)
            ),
            "wav2vec2.encoder.pos_conv_embed.conv.parametrizations.weight.original1": mx.zeros(
                (64, 1, 128)
            ),
        }
        sanitized = model.sanitize(weights)
        self.assertIn("wav2vec2.encoder.pos_conv_embed.conv.weight_g", sanitized)
        self.assertIn("wav2vec2.encoder.pos_conv_embed.conv.weight_v", sanitized)
        self.assertEqual(
            sanitized["wav2vec2.encoder.pos_conv_embed.conv.weight_g"].shape,
            (64, 128, 1),
        )

    def test_predict_returns_sorted_results(self):
        model = self.Model(self.config)
        id2label = {str(i): f"lang_{i}" for i in range(10)}
        model.config.id2label = id2label

        audio = mx.random.normal((1600,))
        results = model.predict(audio, top_k=3)

        self.assertEqual(len(results), 3)
        for lang, prob in results:
            self.assertIsInstance(lang, str)
            self.assertIsInstance(prob, float)
            self.assertGreaterEqual(prob, 0.0)
            self.assertLessEqual(prob, 1.0)

        probs = [p for _, p in results]
        self.assertEqual(probs, sorted(probs, reverse=True))

    def test_predict_1d_input(self):
        model = self.Model(self.config)
        model.config.id2label = {"0": "eng"}
        audio = mx.random.normal((1600,))
        results = model.predict(audio, top_k=1)
        self.assertEqual(len(results), 1)

    def test_predict_without_id2label(self):
        model = self.Model(self.config)
        model.config.id2label = None
        audio = mx.random.normal((1600,))
        results = model.predict(audio, top_k=2)
        for label, _ in results:
            self.assertTrue(label.startswith("LABEL_"))

    def test_predict_rejects_batch(self):
        model = self.Model(self.config)
        model.config.id2label = {"0": "eng"}
        audio = mx.random.normal((2, 1600))
        with self.assertRaises(ValueError):
            model.predict(audio)


class TestLidUtils(unittest.TestCase):
    def test_model_remapping_has_ecapa(self):
        from mlx_audio.lid.utils import MODEL_REMAPPING

        self.assertIn("ecapa-tdnn", MODEL_REMAPPING)
        self.assertEqual(MODEL_REMAPPING["ecapa-tdnn"], "ecapa_tdnn")

    def test_sample_rate(self):
        from mlx_audio.lid.utils import SAMPLE_RATE

        self.assertEqual(SAMPLE_RATE, 16000)


class TestLidImports(unittest.TestCase):
    def test_top_level_imports(self):
        from mlx_audio.lid import load, load_model

        self.assertTrue(callable(load))
        self.assertTrue(callable(load_model))

    def test_model_exports(self):
        from mlx_audio.lid.models.wav2vec2 import DETECTION_HINTS, Model, ModelConfig

        self.assertTrue(hasattr(ModelConfig, "classifier_proj_size"))
        self.assertIn("config_keys", DETECTION_HINTS)
        self.assertIn("architectures", DETECTION_HINTS)

    def test_detection_hints_content(self):
        from mlx_audio.lid.models.wav2vec2 import DETECTION_HINTS

        self.assertIn("classifier_proj_size", DETECTION_HINTS["config_keys"])
        self.assertIn(
            "Wav2Vec2ForSequenceClassification", DETECTION_HINTS["architectures"]
        )


class TestLidCategoryRouting(unittest.TestCase):
    def test_get_model_category_detects_lid(self):
        from mlx_audio.utils import get_model_category

        category = get_model_category("wav2vec2", ["mms", "lid", "256"])
        self.assertEqual(category, "lid")

    def test_get_model_category_non_lid_wav2vec(self):
        from mlx_audio.utils import get_model_category

        category = get_model_category("wav2vec2", ["wav2vec2", "base"])
        self.assertNotEqual(category, "lid")


# ---- ECAPA-TDNN Tests ----


class TestEcapaTdnnConfig(unittest.TestCase):
    def setUp(self):
        from mlx_audio.lid.models.ecapa_tdnn.config import ModelConfig

        self.ModelConfig = ModelConfig

    def test_default_values(self):
        config = self.ModelConfig()
        self.assertEqual(config.n_mels, 60)
        self.assertEqual(config.channels, 1024)
        self.assertEqual(config.res2net_scale, 8)
        self.assertEqual(config.se_channels, 128)
        self.assertEqual(config.embedding_dim, 256)
        self.assertEqual(config.classifier_hidden_dim, 512)
        self.assertEqual(config.num_classes, 107)
        self.assertIsNone(config.id2label)

    def test_custom_values(self):
        config = self.ModelConfig(channels=512, embedding_dim=128)
        self.assertEqual(config.channels, 512)
        self.assertEqual(config.embedding_dim, 128)

    def test_id2label_sets_num_classes(self):
        labels = {str(i): f"{i}: lang_{i}" for i in range(50)}
        config = self.ModelConfig(id2label=labels)
        self.assertEqual(config.num_classes, 50)

    def test_kernel_sizes_and_dilations(self):
        config = self.ModelConfig()
        self.assertEqual(config.kernel_sizes, [5, 3, 3, 3, 1])
        self.assertEqual(config.dilations, [1, 2, 3, 4, 1])


class TestEcapaTdnnModel(unittest.TestCase):
    def setUp(self):
        from mlx_audio.lid.models.ecapa_tdnn.config import ModelConfig
        from mlx_audio.lid.models.ecapa_tdnn.ecapa_tdnn import EcapaTdnn

        self.ModelConfig = ModelConfig
        self.Model = EcapaTdnn
        self.config = ModelConfig(
            channels=64,
            res2net_scale=2,
            se_channels=16,
            attention_channels=16,
            embedding_dim=32,
            classifier_hidden_dim=32,
            num_classes=10,
        )

    def test_model_init(self):
        model = self.Model(self.config)
        self.assertIsNotNone(model.embedding_model)
        self.assertIsNotNone(model.classifier)

    def test_forward_output_shape(self):
        model = self.Model(self.config)
        mel = mx.random.normal((1, 100, 60))
        log_probs = model(mel)
        mx.eval(log_probs)
        self.assertEqual(log_probs.shape[0], 1)
        self.assertEqual(log_probs.shape[1], self.config.num_classes)

    def test_forward_log_probs_sum(self):
        model = self.Model(self.config)
        mel = mx.random.normal((1, 100, 60))
        log_probs = model(mel)
        probs = mx.exp(log_probs)
        mx.eval(probs)
        total = float(mx.sum(probs[0]).item())
        self.assertAlmostEqual(total, 1.0, places=3)

    def test_predict_returns_sorted(self):
        model = self.Model(self.config)
        labels = {str(i): f"lang_{i}" for i in range(10)}
        model.config.id2label = labels
        model.id2label = {int(k): v for k, v in labels.items()}

        audio = mx.random.normal((16000,))
        results = model.predict(audio, top_k=3)

        self.assertEqual(len(results), 3)
        probs = [p for _, p in results]
        self.assertEqual(probs, sorted(probs, reverse=True))

    def test_predict_without_id2label(self):
        model = self.Model(self.config)
        model.id2label = {}
        audio = mx.random.normal((16000,))
        results = model.predict(audio, top_k=2)
        for label, _ in results:
            self.assertTrue(label.startswith("LABEL_"))


class TestEcapaTdnnSanitize(unittest.TestCase):
    def setUp(self):
        from mlx_audio.lid.models.ecapa_tdnn.config import ModelConfig
        from mlx_audio.lid.models.ecapa_tdnn.ecapa_tdnn import EcapaTdnn

        self.model = EcapaTdnn(
            ModelConfig(
                channels=64,
                res2net_scale=2,
                se_channels=16,
                attention_channels=16,
                embedding_dim=32,
                classifier_hidden_dim=32,
                num_classes=10,
            )
        )

    def test_drops_num_batches_tracked(self):
        weights = {
            "embedding_model.blocks.0.norm.norm.num_batches_tracked": mx.array(0),
            "embedding_model.blocks.0.conv.conv.weight": mx.zeros((64, 5, 60)),
        }
        sanitized = self.model.sanitize(weights)
        self.assertEqual(len(sanitized), 1)
        self.assertNotIn(
            "embedding_model.blocks.0.norm.norm.num_batches_tracked", sanitized
        )

    def test_remaps_block_indices(self):
        weights = {
            "embedding_model.blocks.0.conv.conv.weight": mx.zeros((64, 5, 60)),
            "embedding_model.blocks.1.tdnn1.conv.conv.weight": mx.zeros((64, 1, 64)),
        }
        sanitized = self.model.sanitize(weights)
        self.assertIn("embedding_model.block0.conv.weight", sanitized)
        self.assertIn("embedding_model.block1.tdnn1.conv.weight", sanitized)

    def test_flattens_double_nesting(self):
        weights = {
            "embedding_model.block0.conv.conv.weight": mx.zeros((64, 5, 60)),
            "embedding_model.block0.norm.norm.weight": mx.zeros((64,)),
        }
        sanitized = self.model.sanitize(weights)
        self.assertIn("embedding_model.block0.conv.weight", sanitized)
        self.assertIn("embedding_model.block0.norm.weight", sanitized)

    def test_se_block_conv_flatten(self):
        weights = {
            "embedding_model.block1.se_block.conv1.conv.weight": mx.zeros((16, 1, 64)),
            "embedding_model.block1.se_block.conv2.conv.weight": mx.zeros((64, 1, 16)),
        }
        sanitized = self.model.sanitize(weights)
        self.assertIn("embedding_model.block1.se_block.conv1.weight", sanitized)
        self.assertIn("embedding_model.block1.se_block.conv2.weight", sanitized)

    def test_asp_bn_and_fc_flatten(self):
        weights = {
            "embedding_model.asp_bn.norm.weight": mx.zeros((384,)),
            "embedding_model.fc.conv.weight": mx.zeros((32, 1, 384)),
        }
        sanitized = self.model.sanitize(weights)
        self.assertIn("embedding_model.asp_bn.weight", sanitized)
        self.assertIn("embedding_model.fc.weight", sanitized)


class TestEcapaMelSpectrogram(unittest.TestCase):
    def test_output_shape(self):
        from mlx_audio.lid.models.ecapa_tdnn.mel import compute_mel_spectrogram

        audio = mx.random.normal((16000,))
        mel = compute_mel_spectrogram(audio)
        mx.eval(mel)
        self.assertEqual(mel.ndim, 3)
        self.assertEqual(mel.shape[0], 1)
        self.assertEqual(mel.shape[2], 60)
        self.assertGreater(mel.shape[1], 0)

    def test_empty_audio(self):
        from mlx_audio.lid.models.ecapa_tdnn.mel import compute_mel_spectrogram

        audio = mx.array([])
        mel = compute_mel_spectrogram(audio)
        mx.eval(mel)
        self.assertEqual(mel.shape[0], 1)
        self.assertEqual(mel.shape[2], 60)


class TestEcapaTdnnExports(unittest.TestCase):
    def test_model_exports(self):
        from mlx_audio.lid.models.ecapa_tdnn import DETECTION_HINTS, Model, ModelConfig

        self.assertTrue(hasattr(ModelConfig, "n_mels"))
        self.assertIn("config_keys", DETECTION_HINTS)
        self.assertIn("architectures", DETECTION_HINTS)

    def test_detection_hints_content(self):
        from mlx_audio.lid.models.ecapa_tdnn import DETECTION_HINTS

        self.assertIn("res2net_scale", DETECTION_HINTS["config_keys"])
        self.assertIn("EcapaTdnn", DETECTION_HINTS["architectures"])


if __name__ == "__main__":
    unittest.main()
