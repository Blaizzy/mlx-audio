import unittest

import mlx.core as mx

from ..models.s3 import S3TokenizerV2
from ..models.s3.utils import log_mel_spectrogram


class TestS3TokenizerV2(unittest.TestCase):
    """Test S3TokenizerV2 model encoding and decoding."""

    def test_s3_tokenizer_v2(self):
        audio = mx.zeros((160_000))
        mel = log_mel_spectrogram(audio)

        model = S3TokenizerV2("speech_tokenizer_v2_25hz")

        mel_batch = mel[None, ...]  # (1, n_mels, T)
        mel_len = mx.array([mel.shape[1]], dtype=mx.int32)

        codes, code_lens = model(mel_batch, mel_len)
        self.assertEqual(codes.shape, (1, 251))

        codes = codes[0, : code_lens[0].item()]
        self.assertEqual(codes.shape, (251,))
