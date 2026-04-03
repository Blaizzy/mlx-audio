import unittest

import mlx.core as mx


class TestHiggsAudioDAC(unittest.TestCase):
    def test_residual_unit_shape(self):
        from mlx_audio.codec.models.higgs_audio.dac import ResidualUnit

        model = ResidualUnit(64)
        x = mx.zeros((1, 100, 64))
        y = model(x)
        self.assertEqual(y.shape, (1, 100, 64))

    def test_encoder_block_downsamples(self):
        from mlx_audio.codec.models.higgs_audio.dac import AcousticEncoderBlock

        model = AcousticEncoderBlock(64, 128, stride=8)
        x = mx.zeros((1, 800, 64))
        y = model(x)
        self.assertEqual(y.shape[1], 100)

    def test_acoustic_encoder_hop(self):
        from mlx_audio.codec.models.higgs_audio.dac import AcousticEncoder

        model = AcousticEncoder()
        x = mx.zeros((1, 960, 1))
        y = model(x)
        self.assertEqual(y.shape, (1, 1, 256))

    def test_acoustic_decoder_upsample(self):
        from mlx_audio.codec.models.higgs_audio.dac import AcousticDecoder

        model = AcousticDecoder()
        x = mx.zeros((1, 1, 256))
        y = model(x)
        self.assertEqual(y.shape, (1, 960, 1))

    def test_rvq_decode_shape(self):
        from mlx_audio.codec.models.higgs_audio.dac import ResidualVectorQuantizer

        model = ResidualVectorQuantizer()
        codes = mx.zeros((1, 17, 8), dtype=mx.int32)
        y = model.decode(codes)
        self.assertEqual(y.shape, (1, 17, 1024))


class TestHiggsAudioTokenizer(unittest.TestCase):
    def test_higgs_audio_instantiation(self):
        from mlx_audio.codec.models.higgs_audio import (
            HiggsAudioConfig,
            HiggsAudioTokenizer,
        )

        tokenizer = HiggsAudioTokenizer(HiggsAudioConfig())
        self.assertIsNotNone(tokenizer)

    def test_higgs_audio_config_tokens_per_second(self):
        from mlx_audio.codec.models.higgs_audio import HiggsAudioConfig

        cfg = HiggsAudioConfig()
        self.assertAlmostEqual(cfg.tokens_per_second, 25.0)


class TestHiggsAudioTokenizerFull(unittest.TestCase):
    def _tok(self):
        from mlx_audio.codec.models.higgs_audio.config import HiggsAudioConfig
        from mlx_audio.codec.models.higgs_audio.higgs_audio import HiggsAudioTokenizer

        return HiggsAudioTokenizer(HiggsAudioConfig())

    def test_instantiation(self):
        self.assertIsNotNone(self._tok())

    def test_decode_2d_shape(self):
        tok = self._tok()
        tokens = mx.zeros((4, 8), dtype=mx.int32)
        wav = tok.decode(tokens)
        self.assertEqual(wav.shape, (4 * 960,))

    def test_decode_3d_shape(self):
        tok = self._tok()
        tokens = mx.zeros((1, 4, 8), dtype=mx.int32)
        wav = tok.decode(tokens)
        self.assertEqual(wav.ndim, 3)
        self.assertEqual(wav.shape[0], 1)
        self.assertEqual(wav.shape[2], 1)

    def test_encode_raises_without_pt_tokenizer(self):
        tok = self._tok()
        wav = mx.zeros((1, 960 * 5, 1))
        with self.assertRaises(RuntimeError):
            tok.encode(wav)

    def test_sanitize_drops_semantic(self):
        tok = self._tok()
        weights = {
            "acoustic_encoder.conv1.weight_g": mx.zeros((1,)),
            "semantic_model.encoder.conv.weight": mx.zeros((1,)),
            "fc2.weight": mx.zeros((256, 1024)),
            "fc1.weight": mx.zeros((768, 1024)),
        }
        result = tok.sanitize(weights)
        self.assertIn("acoustic_encoder.conv1.weight_g", result)
        self.assertIn("fc2.weight", result)
        self.assertNotIn("semantic_model.encoder.conv.weight", result)
        self.assertNotIn("fc1.weight", result)

    def test_from_pretrained_missing_raises(self):
        from mlx_audio.codec.models.higgs_audio.higgs_audio import HiggsAudioTokenizer

        with self.assertRaises(FileNotFoundError):
            HiggsAudioTokenizer.from_pretrained("/nonexistent/path")


if __name__ == "__main__":
    unittest.main()
