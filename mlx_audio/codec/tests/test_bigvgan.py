import math
import unittest

import mlx.core as mx

from mlx_audio.codec.models.bigvgan.bigvgan import BigVGAN, BigVGANConfig


class TestBigVGAN(unittest.TestCase):
    def test_bigvgan_22khz_80bands(self):
        cfg = BigVGANConfig(
            num_mels=80,
            upsample_rates=[4, 4, 2, 2, 2, 2],
            upsample_kernel_sizes=[8, 8, 4, 4, 4, 4],
            upsample_initial_channel=1536,
            resblock="1",
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            activation="snakebeta",
            snake_logscale=True,
            use_bias_at_final=True,
            use_tanh_at_final=True,
        )
        model = BigVGAN(cfg)

        audio = mx.zeros((1, 80, 800))
        y = model(audio)

        self.assertEqual(y.shape, (1, 1, 800 * math.prod(cfg.upsample_rates)))

    def test_bigvgan_44khz_128bands_512x(self):
        cfg = BigVGANConfig(
            num_mels=128,
            upsample_rates=[8, 4, 2, 2, 2, 2],
            upsample_kernel_sizes=[16, 8, 4, 4, 4, 4],
            upsample_initial_channel=1536,
            resblock="1",
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            activation="snakebeta",
            snake_logscale=True,
            use_bias_at_final=False,
            use_tanh_at_final=False,
        )
        model = BigVGAN(cfg)

        audio = mx.zeros((1, 128, 800))
        y = model(audio)

        self.assertEqual(y.shape, (1, 1, 800 * math.prod(cfg.upsample_rates)))


if __name__ == "__main__":
    unittest.main()
