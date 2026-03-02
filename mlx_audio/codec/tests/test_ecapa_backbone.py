# pyright: reportUnknownMemberType=false

import unittest

import mlx.core as mx


class TestEcapaTdnnConfig(unittest.TestCase):
    def test_default_values(self):
        from mlx_audio.codec.models.ecapa_tdnn.config import EcapaTdnnConfig

        config = EcapaTdnnConfig()
        self.assertEqual(config.input_size, 60)
        self.assertEqual(config.channels, 1024)
        self.assertEqual(config.embed_dim, 256)
        self.assertEqual(config.kernel_sizes, [5, 3, 3, 3, 1])
        self.assertEqual(config.dilations, [1, 2, 3, 4, 1])
        self.assertEqual(config.attention_channels, 128)
        self.assertEqual(config.res2net_scale, 8)
        self.assertEqual(config.se_channels, 128)
        self.assertEqual(config.global_context, False)

    def test_custom_values(self):
        from mlx_audio.codec.models.ecapa_tdnn.config import EcapaTdnnConfig

        config = EcapaTdnnConfig(channels=512, embed_dim=192, global_context=True)
        self.assertEqual(config.channels, 512)
        self.assertEqual(config.embed_dim, 192)
        self.assertTrue(config.global_context)

    def test_spark_preset(self):
        from mlx_audio.codec.models.ecapa_tdnn.config import EcapaTdnnConfig

        config = EcapaTdnnConfig(
            input_size=80, channels=512, embed_dim=192, global_context=True
        )
        self.assertEqual(config.input_size, 80)
        self.assertTrue(config.global_context)

    def test_lid_preset(self):
        from mlx_audio.codec.models.ecapa_tdnn.config import EcapaTdnnConfig

        config = EcapaTdnnConfig(
            input_size=60, channels=1024, embed_dim=256, global_context=True
        )
        self.assertEqual(config.channels, 1024)
        self.assertTrue(config.global_context)


class TestTDNNBlock(unittest.TestCase):
    def test_output_shape(self):
        from mlx_audio.codec.models.ecapa_tdnn.ecapa_tdnn import TDNNBlock

        block = TDNNBlock(60, 1024, kernel_size=5)
        x = mx.zeros((1, 100, 60))
        out = block(x)
        self.assertEqual(out.shape, (1, 100, 1024))

    def test_dilation(self):
        from mlx_audio.codec.models.ecapa_tdnn.ecapa_tdnn import TDNNBlock

        block = TDNNBlock(1024, 1024, kernel_size=3, dilation=2)
        x = mx.zeros((1, 100, 1024))
        out = block(x)
        self.assertEqual(out.shape, (1, 100, 1024))


class TestRes2NetBlock(unittest.TestCase):
    def test_output_shape(self):
        from mlx_audio.codec.models.ecapa_tdnn.ecapa_tdnn import Res2NetBlock

        block = Res2NetBlock(1024, kernel_size=3, dilation=2, scale=8)
        x = mx.zeros((1, 100, 1024))
        out = block(x)
        self.assertEqual(out.shape, (1, 100, 1024))


class TestSEBlock(unittest.TestCase):
    def test_output_shape(self):
        from mlx_audio.codec.models.ecapa_tdnn.ecapa_tdnn import SEBlock

        block = SEBlock(1024, bottleneck=128)
        x = mx.zeros((1, 100, 1024))
        out = block(x)
        self.assertEqual(out.shape, (1, 100, 1024))

    def test_squeeze_excitation(self):
        from mlx_audio.codec.models.ecapa_tdnn.ecapa_tdnn import SEBlock

        block = SEBlock(1024, bottleneck=128)
        x = mx.ones((1, 50, 1024))
        out = block(x)
        mx.eval(out)
        self.assertFalse(mx.array_equal(x, out))


class TestSERes2NetBlock(unittest.TestCase):
    def test_output_shape(self):
        from mlx_audio.codec.models.ecapa_tdnn.ecapa_tdnn import SERes2NetBlock

        block = SERes2NetBlock(
            1024, kernel_size=3, dilation=2, res2net_scale=8, se_channels=128
        )
        x = mx.zeros((1, 100, 1024))
        out = block(x)
        self.assertEqual(out.shape, (1, 100, 1024))

    def test_residual_connection(self):
        from mlx_audio.codec.models.ecapa_tdnn.ecapa_tdnn import SERes2NetBlock

        block = SERes2NetBlock(
            1024, kernel_size=3, dilation=2, res2net_scale=8, se_channels=128
        )
        x = mx.zeros((1, 100, 1024))
        out = block(x)
        mx.eval(out)
        self.assertEqual(out.shape, x.shape)


class TestAttentiveStatisticsPooling(unittest.TestCase):
    def test_output_shape(self):
        from mlx_audio.codec.models.ecapa_tdnn.ecapa_tdnn import (
            AttentiveStatisticsPooling,
        )

        asp = AttentiveStatisticsPooling(
            1024, attention_channels=128, global_context=False
        )
        x = mx.zeros((1, 100, 1024))
        out = asp(x)
        self.assertEqual(out.shape, (1, 2048))

    def test_global_context(self):
        from mlx_audio.codec.models.ecapa_tdnn.ecapa_tdnn import (
            AttentiveStatisticsPooling,
        )

        asp = AttentiveStatisticsPooling(
            1024, attention_channels=128, global_context=True
        )
        x = mx.zeros((1, 100, 1024))
        out = asp(x)
        self.assertEqual(out.shape, (1, 2048))

    def test_global_context_changes_attention_input(self):
        from mlx_audio.codec.models.ecapa_tdnn.ecapa_tdnn import (
            AttentiveStatisticsPooling,
        )

        asp_no_gc = AttentiveStatisticsPooling(128, 64, global_context=False)
        asp_gc = AttentiveStatisticsPooling(128, 64, global_context=True)
        self.assertEqual(asp_no_gc.tdnn.conv.weight.shape[-1], 128)
        self.assertEqual(asp_gc.tdnn.conv.weight.shape[-1], 128 * 3)


class TestEcapaTdnnBackbone(unittest.TestCase):
    def test_output_shape_default_config(self):
        from mlx_audio.codec.models.ecapa_tdnn import EcapaTdnnBackbone, EcapaTdnnConfig

        config = EcapaTdnnConfig()
        model = EcapaTdnnBackbone(config)
        x = mx.zeros((1, 100, 60))
        out = model(x)
        mx.eval(out)
        self.assertEqual(out.shape, (1, config.embed_dim))

    def test_output_shape_spark_config(self):
        from mlx_audio.codec.models.ecapa_tdnn import EcapaTdnnBackbone, EcapaTdnnConfig

        config = EcapaTdnnConfig(
            input_size=80, channels=512, embed_dim=192, global_context=True
        )
        model = EcapaTdnnBackbone(config)
        x = mx.zeros((1, 200, 80))
        out = model(x)
        mx.eval(out)
        self.assertEqual(out.shape, (1, 192))

    def test_submodules_accessible(self):
        from mlx_audio.codec.models.ecapa_tdnn import EcapaTdnnBackbone, EcapaTdnnConfig

        config = EcapaTdnnConfig()
        model = EcapaTdnnBackbone(config)
        self.assertTrue(hasattr(model, "blocks"))
        self.assertTrue(hasattr(model, "mfa"))
        self.assertTrue(hasattr(model, "asp"))
        self.assertTrue(hasattr(model, "asp_bn"))
        self.assertTrue(hasattr(model, "fc"))

    def test_batch_dimension(self):
        from mlx_audio.codec.models.ecapa_tdnn import EcapaTdnnBackbone, EcapaTdnnConfig

        config = EcapaTdnnConfig(input_size=60, channels=512, embed_dim=128)
        model = EcapaTdnnBackbone(config)
        x = mx.zeros((4, 50, 60))
        out = model(x)
        mx.eval(out)
        self.assertEqual(out.shape[0], 4)
        self.assertEqual(out.shape[1], 128)


class TestCodecReExports(unittest.TestCase):
    def test_import_from_codec_models(self):
        from mlx_audio.codec.models import EcapaTdnnBackbone

        self.assertTrue(callable(EcapaTdnnBackbone))

    def test_import_from_codec_top_level(self):
        from mlx_audio.codec import EcapaTdnnBackbone

        self.assertTrue(callable(EcapaTdnnBackbone))


class TestBackboneRegressionNumerics(unittest.TestCase):
    def test_output_is_finite(self):
        from mlx_audio.codec.models.ecapa_tdnn import EcapaTdnnBackbone, EcapaTdnnConfig

        config = EcapaTdnnConfig(input_size=60, channels=512, embed_dim=128)
        model = EcapaTdnnBackbone(config)
        x = mx.random.normal((1, 100, 60))
        out = model(x)
        mx.eval(out)
        self.assertTrue(mx.all(mx.isfinite(out)).item())

    def test_output_is_nonzero(self):
        from mlx_audio.codec.models.ecapa_tdnn import EcapaTdnnBackbone, EcapaTdnnConfig

        config = EcapaTdnnConfig(input_size=60, channels=512, embed_dim=128)
        model = EcapaTdnnBackbone(config)
        x = mx.random.normal((2, 100, 60))
        out = model(x)
        mx.eval(out)
        self.assertGreater(mx.abs(out).sum().item(), 0)

    def test_different_inputs_give_different_outputs(self):
        from mlx_audio.codec.models.ecapa_tdnn import EcapaTdnnBackbone, EcapaTdnnConfig

        config = EcapaTdnnConfig(input_size=60, channels=512, embed_dim=128)
        model = EcapaTdnnBackbone(config)
        x1 = mx.random.normal((2, 100, 60))
        x2 = mx.random.normal((2, 100, 60))
        out1 = model(x1)
        out2 = model(x2)
        mx.eval(out1, out2)
        self.assertFalse(mx.allclose(out1, out2).item())


class TestLidBackboneIntegration(unittest.TestCase):
    def test_lid_ecapa_uses_shared_backbone(self):
        from mlx_audio.codec.models.ecapa_tdnn import EcapaTdnnBackbone
        from mlx_audio.lid.models.ecapa_tdnn.config import ModelConfig
        from mlx_audio.lid.models.ecapa_tdnn.ecapa_tdnn import ECAPA_TDNN

        config = ModelConfig()
        model = ECAPA_TDNN(config)
        self.assertIsInstance(model.embedding_model, EcapaTdnnBackbone)

    def test_lid_forward_pass_unchanged(self):
        from mlx_audio.lid.models.ecapa_tdnn.config import ModelConfig
        from mlx_audio.lid.models.ecapa_tdnn.ecapa_tdnn import ECAPA_TDNN

        config = ModelConfig(id2label={str(i): f"{i}: lang_{i}" for i in range(10)})
        model = ECAPA_TDNN(config)
        mel = mx.zeros((1, 100, 60))
        out = model(mel)
        mx.eval(out)
        self.assertEqual(out.shape, (1, 10))

    def test_lid_predict_unchanged(self):
        from mlx_audio.lid.models.ecapa_tdnn.config import ModelConfig
        from mlx_audio.lid.models.ecapa_tdnn.ecapa_tdnn import ECAPA_TDNN

        config = ModelConfig(id2label={str(i): f"{i}: lang_{i}" for i in range(10)})
        model = ECAPA_TDNN(config)
        audio = mx.random.normal((16000,))
        result = model.predict(audio, top_k=3)
        self.assertEqual(len(result), 3)
        for lang, prob in result:
            self.assertIsInstance(lang, str)
            self.assertIsInstance(prob, float)
