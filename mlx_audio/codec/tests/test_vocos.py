import unittest

import mlx.core as mx

from ..models.vocos import Vocos
from ..models.vocos.mel import log_mel_spectrogram

config_mel = {
    "feature_extractor": {
        "class_path": "vocos.feature_extractors.MelSpectrogramFeatures",
        "init_args": {
            "sample_rate": 24000,
            "n_fft": 1024,
            "hop_length": 256,
            "n_mels": 100,
        },
    },
    "backbone": {
        "class_path": "vocos.models.VocosBackbone",
        "init_args": {
            "input_channels": 100,
            "dim": 512,
            "intermediate_dim": 1536,
            "num_layers": 8,
        },
    },
    "head": {
        "class_path": "vocos.heads.ISTFTHead",
        "init_args": {"dim": 512, "n_fft": 1024, "hop_length": 256},
    },
}

config_encodec = {
    "feature_extractor": {
        "class_path": "vocos.feature_extractors.EncodecFeatures",
        "init_args": {
            "encodec_model": "encodec_24khz",
            "bandwidths": [1.5, 3.0, 6.0, 12.0, 24.0],
        },
    },
    "backbone": {
        "class_path": "vocos.models.VocosBackbone",
        "init_args": {
            "input_channels": 128,
            "dim": 384,
            "intermediate_dim": 1152,
            "num_layers": 8,
            "adanorm_num_embeddings": 4,
        },
    },
    "head": {
        "class_path": "vocos.heads.ISTFTHead",
        "init_args": {"dim": 384, "n_fft": 1280, "hop_length": 320, "padding": "same"},
    },
}


class TestVocos(unittest.TestCase):
    """Test Vocos model encoding and decoding."""

    def test_vocos_24khz(self):
        audio = mx.zeros((120_000))

        model = Vocos.from_hparams(config_mel)

        # reconstruct from mel spec
        reconstructed_audio = model(audio)
        self.assertEqual(reconstructed_audio.shape, (119552,))

        # decode from mel spec
        mel_spec = log_mel_spectrogram(audio)
        decoded = model.decode(mel_spec)
        self.assertEqual(decoded.shape, (119552,))

        model = Vocos.from_hparams(config_encodec)

        # reconstruct from encodec codes
        bandwidth_id = [3, 3, 3, 3]  # 24kbps
        reconstructed_audio = model(
            audio, bandwidth_id=mx.array(bandwidth_id)[None, ...]
        )
        self.assertEqual(reconstructed_audio.shape, (119680,))

        # decode with encodec codes
        codes = model.get_encodec_codes(audio, bandwidth_id=bandwidth_id)
        decoded = model.decode_from_codes(
            codes, bandwidth_id=mx.array(bandwidth_id)[None, ...]
        )
        self.assertEqual(decoded.shape, (119680,))


if __name__ == "__main__":
    unittest.main()
