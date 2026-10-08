# Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

"""Regression test: generation with an audio prompt must not overrun the KV cache.

The decoder self-attention caches are sized at allocation time. When a
ref_audio prompt is prefilled before decoding, those prefill tokens consume
cache slots first, so the cache length must be ``prefill_len + max_tokens``.
Sizing it to ``max_tokens`` alone crashes mid-generation with
``assert self.current_idx < self.max_len`` in ``KVCache.update_and_fetch``.

Runs on a tiny randomly-initialized model; no network or pretrained weights.
"""

import unittest
from unittest.mock import patch

import mlx.core as mx

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
    """Stands in for the DAC codec so the test never touches the network."""

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


class TestRefAudioKVCacheSizing(unittest.TestCase):
    def test_generation_with_ref_audio_does_not_overrun_kv_cache(self):
        with patch("mlx_audio.tts.models.dia.dia.DAC", FakeDAC):
            model = Model(_tiny_config_dict())
        ref_audio = mx.zeros((4410,), dtype=mx.float32)
        result = list(
            model.generate(
                "[S1] Hello there",
                max_tokens=32,
                ref_audio=ref_audio,
                verbose=False,
            )
        )
        self.assertEqual(len(result), 1)
        self.assertGreater(result[0].samples, 0)


if __name__ == "__main__":
    unittest.main()
