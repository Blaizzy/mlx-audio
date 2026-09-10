# Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

"""Offline tests for the Dia TTS model.

These tests run on a tiny randomly-initialized model so no network access or
pretrained weights are required. Generation with an audio prompt (ref_audio)
has additional coverage for the KV cache sizing contract in the PR that
introduced it; the ``TestKVCacheCapacity`` test below documents that contract.
"""

import unittest
from unittest.mock import patch

import mlx.core as mx

from mlx_audio.tts.models.dia.config import (
    DataConfig,
    DecoderConfig,
    EncoderConfig,
    ModelConfig,
    TrainingConfig,
)
from mlx_audio.tts.models.dia.dia import Model


def _tiny_config_dict() -> dict:
    return {
        "version": "1.0",
        "model": {
            "encoder": {
                "n_layer": 1,
                "n_embd": 16,
                "n_hidden": 32,
                "n_head": 2,
                "head_dim": 8,
            },
            "decoder": {
                "n_layer": 1,
                "n_embd": 16,
                "n_hidden": 32,
                "gqa_query_heads": 2,
                "kv_heads": 1,
                "gqa_head_dim": 8,
                "cross_query_heads": 2,
                "cross_head_dim": 8,
            },
            "src_vocab_size": 128,
            "tgt_vocab_size": 1028,
            "sample_rate": 44100,
        },
        "training": {"dtype": "float32", "logits_dot_in_fp32": False},
        "data": {
            "text_length": 128,
            "audio_length": 256,
            "channels": 9,
            "audio_eos_value": 1024,
            "audio_pad_value": 1025,
            "audio_bos_value": 1026,
            "delay_pattern": [0, 8, 9, 10, 11, 12, 13, 14, 15],
        },
    }


class FakeQuantizer:
    def from_codes(self, audio_codes):
        codes = mx.array(audio_codes, dtype=mx.float32)
        return (codes,)


class FakeDAC:
    """Stands in for the DAC codec so tests never touch the network."""

    def __init__(self):
        self.quantizer = FakeQuantizer()

    @classmethod
    def from_pretrained(cls, repo_id):
        return cls()

    def preprocess(self, input_values, sample_rate):
        return input_values

    def encode(self, audio_data, n_quantizers=None):
        # One frame of valid codebook indices per channel: shape (1, C, T).
        frame = mx.zeros((1, 9, 8), dtype=mx.int32)
        return None, frame, None, None, None

    def decode(self, audio_values):
        return mx.zeros((1, 1, 64), dtype=mx.float32)


def _make_model() -> Model:
    with patch("mlx_audio.tts.models.dia.dia.DAC", FakeDAC):
        return Model(_tiny_config_dict())


class TestDiaConfigParsing(unittest.TestCase):
    def test_tiny_config_loads(self):
        config_dict = _tiny_config_dict()
        model = _make_model()
        self.assertEqual(model.config.model.encoder.n_layer, 1)
        self.assertEqual(model.config.data.channels, 9)
        self.assertIsInstance(config_dict["model"]["decoder"]["kv_heads"], int)

    def test_model_config_dataclasses_roundtrip(self):
        encoder = EncoderConfig(n_layer=1, n_embd=16, n_hidden=32, n_head=2, head_dim=8)
        decoder = DecoderConfig(
            n_layer=1,
            n_embd=16,
            n_hidden=32,
            gqa_query_heads=2,
            kv_heads=1,
            gqa_head_dim=8,
            cross_query_heads=2,
            cross_head_dim=8,
        )
        model_config = ModelConfig(encoder=encoder, decoder=decoder)
        training = TrainingConfig()
        data = DataConfig(text_length=128, audio_length=256)
        # Lengths are rounded up to multiples of 128.
        self.assertEqual(data.audio_length % 128, 0)
        self.assertEqual(data.text_length % 128, 0)
        self.assertEqual(training.dtype, "bfloat16")
        self.assertIsNotNone(model_config)


class TestDiaGeneration(unittest.TestCase):
    def test_generation_without_ref_audio_completes(self):
        model = _make_model()
        result = next(
            model.generate("Hello world", max_tokens=32, verbose=False),
        )
        self.assertIsInstance(result.audio, mx.array)
        self.assertGreater(result.audio.shape[0], 0)
        self.assertEqual(result.sample_rate, 44100)

    def test_kv_cache_rejects_writes_beyond_capacity(self):
        # Documents the failure mode the ref_audio path used to hit when the
        # cache was sized only for the generated tokens.
        from mlx_audio.tts.models.dia.layers import KVCache

        cache = KVCache(num_heads=2, max_len=3, head_dim=4)
        cache.prefill_kv(
            mx.zeros((2, 2, 2, 4)), mx.zeros((2, 2, 2, 4))
        )  # prefill fills slots 0..1
        cache.update_and_fetch(
            mx.zeros((2, 2, 1, 4)), mx.zeros((2, 2, 1, 4))
        )  # writes slot 2
        with self.assertRaises(AssertionError):
            cache.update_and_fetch(mx.zeros((2, 2, 1, 4)), mx.zeros((2, 2, 1, 4)))


if __name__ == "__main__":
    unittest.main()
