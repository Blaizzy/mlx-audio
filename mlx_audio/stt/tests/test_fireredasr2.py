import unittest
from unittest import mock

import mlx.core as mx

from mlx_audio.stt.models.fireredasr2.config import (
    DecoderConfig,
    EncoderConfig,
    ModelConfig,
)
from mlx_audio.stt.models.fireredasr2.fireredasr2 import Model


class TestFireRedASR2(unittest.TestCase):
    def setUp(self):
        enc_config = EncoderConfig(
            n_layers=2,
            n_head=2,
            d_model=64,
            kernel_size=15,
            pe_maxlen=100,
        )
        dec_config = DecoderConfig(n_layers=2, n_head=2, d_model=64, pe_maxlen=100)
        self.config = ModelConfig(
            idim=80,
            odim=1000,
            d_model=64,
            sos_id=3,
            eos_id=4,
            pad_id=2,
            blank_id=0,
            encoder=enc_config,
            decoder=dec_config,
        )
        self.model = Model(self.config)

    def test_encoder(self):
        x = mx.random.normal((1, 100, 80))
        encoder_out = self.model.encoder(x)
        self.assertEqual(encoder_out.shape[0], 1)
        self.assertEqual(encoder_out.shape[-1], 64)

    def test_beam_search(self):
        encoder_out = mx.random.normal((1, 25, 64))
        hyp, score = self.model.decoder.beam_search(
            encoder_out, beam_size=1, max_len=10
        )
        self.assertIsInstance(hyp, mx.array)
        self.assertTrue(len(hyp.shape) > 0)

    def test_long_audio_is_decoded_in_chunks(self):
        config = ModelConfig(
            idim=80,
            odim=1000,
            d_model=64,
            encoder=EncoderConfig(n_layers=1, n_head=2, d_model=64, kernel_size=15),
            decoder=DecoderConfig(n_layers=1, n_head=2, d_model=64),
        )
        model = Model(config)
        model._detokenize = lambda ids: "x"
        audio = mx.random.normal((40 * 16000,)) * 0.1
        with mock.patch.object(
            model, "_extract_fbank", wraps=model._extract_fbank
        ) as fbank:
            model.generate(audio, beam_size=1, max_len=1, chunk_duration=1200)
            self.assertEqual(fbank.call_count, 1)
            fbank.reset_mock()
            out = model.generate(audio, beam_size=1, max_len=1, chunk_duration=10)
            self.assertGreaterEqual(fbank.call_count, 3)
        self.assertEqual(len(out.segments), fbank.call_count)


if __name__ == "__main__":
    unittest.main()
