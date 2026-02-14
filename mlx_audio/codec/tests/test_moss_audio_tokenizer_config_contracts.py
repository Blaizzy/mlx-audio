import unittest
from pathlib import Path

from mlx_audio.codec.models.moss_audio_tokenizer.config import (
    CANONICAL_MODEL_TYPE,
    load_moss_audio_tokenizer_config,
)


def _repo_root() -> Path:
    # .../mlx_audio/codec/tests -> project root
    return Path(__file__).resolve().parents[3]


class TestMossAudioTokenizerPhase0Config(unittest.TestCase):
    def test_load_upstream_config_contract(self):
        config_path = _repo_root() / "REFERENCE" / "MOSS-Audio-Tokenizer" / "config.json"
        if not config_path.exists():
            self.skipTest(f"Missing reference config at {config_path}")
        config = load_moss_audio_tokenizer_config(config_path)
        self.assertEqual(config.model_type, CANONICAL_MODEL_TYPE)
        self.assertEqual(config.sampling_rate, 24000)
        self.assertEqual(config.downsample_rate, 1920)
        self.assertEqual(config.frame_rate, 12.5)
        self.assertEqual(config.quantizer.num_quantizers, 32)
        self.assertEqual(config.quantizer.codebook_size, 1024)
        self.assertEqual(config.quantizer.quantizer_type, "rlfq")

    def test_patch_products_match_downsample_rate(self):
        config_path = _repo_root() / "REFERENCE" / "MOSS-Audio-Tokenizer" / "config.json"
        if not config_path.exists():
            self.skipTest(f"Missing reference config at {config_path}")
        config = load_moss_audio_tokenizer_config(config_path)
        self.assertEqual(config.encoder_patch_product, 1920)
        self.assertEqual(config.decoder_patch_product, 1920)
        self.assertTrue(config.patch_alignment_is_valid())


if __name__ == "__main__":
    unittest.main()
