import unittest

import mlx.core as mx

from ..models.descript import DAC


class TestDescript(unittest.TestCase):
    """Test Descript model encoding and decoding."""

    def test_descript_16khz(self):
        audio = mx.zeros((1, 1, 80_000))

        encoder_dim = 64
        encoder_rates = [2, 4, 5, 8]
        decoder_dim = 1536
        decoder_rates = [8, 5, 4, 2]
        n_codebooks = 12
        codebook_size = 1024
        codebook_dim = 8
        sample_rate = 16_000

        model = DAC(
            encoder_dim=encoder_dim,
            encoder_rates=encoder_rates,
            decoder_dim=decoder_dim,
            decoder_rates=decoder_rates,
            n_codebooks=n_codebooks,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            sample_rate=sample_rate,
        )

        x = model.preprocess(audio, sample_rate)
        z, codes, latents, _, _ = model.encode(x)
        self.assertEqual(z.shape, (1, 1024, 250))
        self.assertEqual(codes.shape, (1, 12, 250))
        self.assertEqual(latents.shape, (1, 96, 250))

        y = model.decode(z).squeeze(-1)
        self.assertEqual(y.shape, (1, 80_043))

    def test_descript_24khz(self):
        audio = mx.zeros((1, 1, 120_000))

        encoder_dim = 64
        encoder_rates = [2, 4, 5, 8]
        decoder_dim = 1536
        decoder_rates = [8, 5, 4, 2]
        n_codebooks = 32
        codebook_size = 1024
        codebook_dim = 8
        sample_rate = 24_000

        model = DAC(
            encoder_dim=encoder_dim,
            encoder_rates=encoder_rates,
            decoder_dim=decoder_dim,
            decoder_rates=decoder_rates,
            n_codebooks=n_codebooks,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            sample_rate=sample_rate,
        )

        x = model.preprocess(audio, sample_rate)
        z, codes, latents, _, _ = model.encode(x)
        self.assertEqual(z.shape, (1, 1024, 375))
        self.assertEqual(codes.shape, (1, 32, 375))
        self.assertEqual(latents.shape, (1, 256, 375))

        y = model.decode(z).squeeze(-1)
        self.assertEqual(y.shape, (1, 120_043))

    def test_descript_44khz(self):
        audio = mx.zeros((1, 1, 220_000))

        encoder_dim = 64
        encoder_rates = [2, 4, 8, 8]
        decoder_dim = 1536
        decoder_rates = [8, 8, 4, 2]
        n_codebooks = 9
        codebook_size = 1024
        codebook_dim = 8
        sample_rate = 44_100

        model = DAC(
            encoder_dim=encoder_dim,
            encoder_rates=encoder_rates,
            decoder_dim=decoder_dim,
            decoder_rates=decoder_rates,
            n_codebooks=n_codebooks,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            sample_rate=sample_rate,
        )

        x = model.preprocess(audio, sample_rate)
        z, codes, latents, _, _ = model.encode(x)
        self.assertEqual(z.shape, (1, 1024, 430))
        self.assertEqual(codes.shape, (1, 9, 430))
        self.assertEqual(latents.shape, (1, 72, 430))

        y = model.decode(z).squeeze(-1)
        self.assertEqual(y.shape, (1, 220_235))


if __name__ == "__main__":
    unittest.main()
