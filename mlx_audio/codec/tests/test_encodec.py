import unittest

import mlx.core as mx

from ..models.encodec import Encodec, EncodecConfig

config = EncodecConfig(
    audio_channels=1,
    chunk_length_s=None,
    codebook_dim=128,
    codebook_size=1024,
    compress=2,
    dilation_growth_rate=2,
    hidden_size=128,
    kernel_size=7,
    last_kernel_size=7,
    model_type="encodec",
    norm_type="weight_norm",
    normalize=False,
    num_filters=32,
    num_lstm_layers=2,
    num_residual_layers=1,
    overlap=None,
    pad_mode="reflect",
    residual_kernel_size=3,
    sampling_rate=24000,
    target_bandwidths=[1.5, 3.0, 6.0, 12.0, 24.0],
    trim_right_ratio=1.0,
    upsampling_ratios=[8, 5, 4, 2],
    use_causal_conv=True,
)


class TestEncodec(unittest.TestCase):
    """Test EnCodec model encoding and decoding."""

    def test_encodec_24khz(self):
        model = Encodec(config)

        audio = mx.zeros((1, 120_000, 1))

        # default bandwidth
        (codes, scales) = model.encode(audio)
        self.assertEqual(codes.shape, (1, 1, 2, 375))

        audio_out = model.decode(codes, scales)
        self.assertEqual(audio_out.shape, (1, 120_000, 1))

        # 6kbps bandwidth
        (codes, scales) = model.encode(audio, bandwidth=6)
        self.assertEqual(codes.shape, (1, 1, 8, 375))

        audio_out = model.decode(codes, scales)
        self.assertEqual(audio_out.shape, (1, 120_000, 1))


if __name__ == "__main__":
    unittest.main()
