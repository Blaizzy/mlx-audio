import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from mlx_audio.codec.models import MossAudioTokenizer as ExportedMossAudioTokenizer
from mlx_audio.codec.models.moss_audio_tokenizer import (
    AudioTokenizerConfig,
    MossAudioTokenizer,
)


def tiny_config_dict():
    return {
        "sample_rate": 24000,
        "sampling_rate": 24000,
        "downsample_rate": 1,
        "number_channels": 1,
        "enable_channel_interleave": True,
        "encoder_kwargs": [],
        "decoder_kwargs": [],
        "quantizer_type": "rlfq",
        "quantizer_kwargs": {
            "input_dim": 1,
            "rvq_dim": 1,
            "output_dim": 1,
            "num_quantizers": 2,
            "codebook_size": 4,
            "codebook_dim": 1,
        },
    }


def write_tiny_tokenizer(path: Path) -> None:
    config = tiny_config_dict()
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(config))

    model = MossAudioTokenizer(AudioTokenizerConfig.from_dict(config))
    weights = {}
    for key, value in tree_flatten(model.parameters()):
        if "original0" in key or "original1" in key:
            weights[key] = mx.ones(value.shape, dtype=value.dtype)
        elif "codebook.weight" in key:
            weights[key] = (
                mx.arange(value.size, dtype=mx.float32).reshape(value.shape) + 1
            ) / 10
        else:
            weights[key] = mx.zeros(value.shape, dtype=value.dtype)
    mx.save_safetensors(str(path / "model.safetensors"), weights)


class TestMossAudioTokenizer(unittest.TestCase):
    def test_parent_config_defaults_to_mono(self):
        config = AudioTokenizerConfig.from_dict(
            {
                "sampling_rate": 24000,
                "downsample_rate": 1920,
                "encoder_kwargs": [],
                "decoder_kwargs": [],
                "quantizer_kwargs": {"num_quantizers": 32},
            }
        )

        self.assertEqual(config.number_channels, 1)
        self.assertEqual(config.sampling_rate, 24000)

    def test_import_from_codec_models(self):
        self.assertIs(ExportedMossAudioTokenizer, MossAudioTokenizer)

    def test_from_pretrained_tiny_tokenizer_encode_decode(self):
        with TemporaryDirectory() as tmpdir:
            write_tiny_tokenizer(Path(tmpdir))

            tokenizer = MossAudioTokenizer.from_pretrained(tmpdir)

            self.assertEqual(tokenizer.sample_rate, 24000)
            self.assertEqual(tokenizer.num_quantizers, 2)

            codes = tokenizer.encode_audio(
                mx.zeros((4, 1), dtype=mx.float32),
                sample_rate=24000,
            )
            decoded = tokenizer.decode_audio_codes(mx.zeros((3, 2), dtype=mx.int32))

            self.assertEqual(codes.shape, (4, 2))
            self.assertEqual(decoded.shape, (3, 1))
            self.assertTrue(np.isfinite(np.asarray(decoded)).all())

    def test_from_model_dir_prefers_nested_audio_tokenizer(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config.json").write_text(
                json.dumps({"model_type": "moss_tts_nano", "gpt2_config": {}})
            )
            (root / "model.safetensors").write_bytes(b"")
            write_tiny_tokenizer(root / "audio_tokenizer")

            tokenizer = MossAudioTokenizer.from_model_dir(root)

            self.assertEqual(tokenizer.sample_rate, 24000)
            self.assertEqual(tokenizer.channels, 1)


if __name__ == "__main__":
    unittest.main()
