import unittest


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
