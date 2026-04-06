import unittest

import mlx.core as mx
import numpy as np

from mlx_audio.tts.models.voxtral_tts.audio_tokenizer import WeightNormConv


class TestVoxtralTTSAudioTokenizer(unittest.TestCase):
    def test_replicate_padding_matches_reference(self):
        conv = WeightNormConv(1, 1, 4, pad_mode="replicate")
        x = mx.array([[[0.0], [1.0], [2.0], [3.0], [4.0]]])

        padded = conv._pad_1d(x, padding_left=2, padding_right=1)

        np.testing.assert_array_equal(
            np.array(padded[0, :, 0]),
            np.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0]),
        )

    def test_reflect_padding_handles_short_inputs_like_reference(self):
        conv = WeightNormConv(1, 1, 7, pad_mode="reflect")
        x = mx.array([[[0.0], [1.0], [2.0]]])

        padded = conv._pad_1d(x, padding_left=3, padding_right=0)

        np.testing.assert_array_equal(
            np.array(padded[0, :, 0]),
            np.array([0.0, 2.0, 1.0, 0.0, 1.0, 2.0]),
        )

    def test_stride_aware_causal_conv_matches_reference_padding_math(self):
        conv = WeightNormConv(1, 1, 4, pad_mode="replicate")
        conv.parametrizations["weight"]["original0"] = mx.ones((1, 1, 1))
        conv.parametrizations["weight"]["original1"] = mx.array(
            [[[1.0, 0.0, 0.0, 0.0]]]
        )

        x = mx.array([[[0.0], [1.0], [2.0], [3.0], [4.0]]])
        out = conv._conv_1d(x, conv._get_weight(), stride=2)

        np.testing.assert_array_equal(
            np.array(out[0, :, 0]),
            np.array([0.0, 0.0, 2.0]),
        )


if __name__ == "__main__":
    unittest.main()
