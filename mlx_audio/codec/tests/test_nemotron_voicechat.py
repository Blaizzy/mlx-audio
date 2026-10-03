import unittest

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_audio.codec.models.nemotron_voicechat import (
    CausalConv1dCache,
    NemotronVoiceChatCodec,
    NemotronVoiceChatCodecConfig,
)


def tiny_config() -> NemotronVoiceChatCodecConfig:
    return NemotronVoiceChatCodecConfig(
        sample_rate=160,
        base_channels=4,
        channel_multipliers=(1, 2, 4),
        downsample_rates=(2, 2, 2),
        blocks_per_stage=3,
        block_kernel_size=3,
        latent_dim=8,
        n_fft=4,
        hop_length=2,
        num_quantizers=2,
        codebook_size=16,
    )


class TestNemotronVoiceChatCodec(unittest.TestCase):
    def test_config_uses_all_rfft_bins(self):
        config = NemotronVoiceChatCodecConfig()
        self.assertEqual(config.stft_channels, 18)
        self.assertEqual(config.waveform_to_token_ratio, 1764)
        self.assertEqual(config.frame_rate, 12.5)

    def test_tiny_encode_decode_contract(self):
        model = NemotronVoiceChatCodec(tiny_config())
        waveform = mx.zeros((1, 1, 64), dtype=mx.float32)
        codes = model.encode(waveform)
        decoded = model.decode(codes)
        self.assertEqual(codes.shape[0:2], (1, 2))
        self.assertEqual(decoded.shape[0:2], (1, 1))
        self.assertEqual(decoded.shape[-1], codes.shape[-1] * 16)
        self.assertTrue(bool(mx.all(mx.isfinite(decoded))))

    def test_streaming_decode_matches_offline_and_has_fixed_chunks(self):
        model = NemotronVoiceChatCodec(tiny_config())
        codes = mx.array([[[1, 2, 3], [4, 5, 6]]], dtype=mx.int32)
        offline = model.decode(codes)

        cache = CausalConv1dCache()
        chunks = [
            model.decode_step(codes[:, :, index : index + 1], cache)
            for index in range(codes.shape[-1])
        ]
        streamed = mx.concatenate(chunks, axis=-1)
        mx.eval(offline, streamed)

        self.assertTrue(all(chunk.shape[-1] == 16 for chunk in chunks))
        self.assertTrue(mx.allclose(streamed, offline, rtol=1e-4, atol=1e-4))

    def test_sanitize_convolution_layouts_and_strict_load(self):
        model = NemotronVoiceChatCodec(tiny_config())
        runtime = dict(tree_flatten(model.parameters()))
        source = {}
        for key, value in runtime.items():
            if key.endswith(".weight") and value.ndim == 3:
                if key.startswith("decoder.layers."):
                    layer_index = int(key.split(".")[2])
                    if layer_index in (0, 4, 8) and "dwconv" not in key:
                        value = value.transpose(2, 0, 1)
                    else:
                        value = value.transpose(0, 2, 1)
                else:
                    value = value.transpose(0, 2, 1)
            source[f"tts_model.audio_codec.{key}"] = value

        converted = model.sanitize(source, prefix="tts_model.audio_codec")
        self.assertEqual(set(converted), set(runtime))
        self.assertIn("prvq.variance_list.0.variance", converted)
        for key in runtime:
            self.assertEqual(converted[key].shape, runtime[key].shape)
        model.load_weights(list(converted.items()), strict=True)


if __name__ == "__main__":
    unittest.main()
