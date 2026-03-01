import unittest

import mlx.core as mx


class TestEcapaTdnnConfig(unittest.TestCase):
    def setUp(self):
        from mlx_audio.codec.models.ecapa_tdnn.config import EcapaTdnnConfig

        self.Config = EcapaTdnnConfig

    def test_default_values(self):
        config = self.Config()
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
        config = self.Config(channels=512, embed_dim=192, global_context=True)
        self.assertEqual(config.channels, 512)
        self.assertEqual(config.embed_dim, 192)
        self.assertTrue(config.global_context)

    def test_spark_preset(self):
        config = self.Config(
            input_size=80, channels=512, embed_dim=192, global_context=True
        )
        self.assertEqual(config.input_size, 80)
        self.assertTrue(config.global_context)

    def test_lid_preset(self):
        config = self.Config(
            input_size=60, channels=1024, embed_dim=256, global_context=False
        )
        self.assertEqual(config.channels, 1024)
        self.assertFalse(config.global_context)


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
