import unittest
from unittest.mock import patch

import mlx.core as mx
import numpy as np

from mlx_audio.lm.models.cache import KVCache
from mlx_audio.tts.models.dia import dia
from mlx_audio.tts.models.dia.config import DiaConfig
from mlx_audio.tts.models.dia.layers import Decoder, DenseGeneral, DiaModel


def _tiny_config():
    return {
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
        },
        "training": {"dtype": "float32"},
        "data": {"text_length": 128, "audio_length": 256},
    }


class _FakeDAC:
    frames = 8

    @classmethod
    def from_pretrained(cls, repo_id):
        return cls()

    def preprocess(self, audio, sample_rate):
        return audio

    def encode(self, audio, n_quantizers=None):
        return None, mx.zeros((1, 9, self.frames), dtype=mx.int32), None, None, None


def _randomize_projections(model):
    for _, module in model.named_modules():
        if isinstance(module, DenseGeneral):
            module.weight = (mx.random.normal(module.weight.shape) * 0.1).astype(
                module.weight.dtype
            )


class TestDiaDecoderCache(unittest.TestCase):
    def test_cached_logits_match_full_causal_forward_across_growth(self):
        self._check_cached_logits("float32", 2e-3)

    def test_bfloat16_cached_logits_match_full_causal_forward(self):
        self._check_cached_logits("bfloat16", 0.02)

    def _check_cached_logits(self, dtype, tolerance):
        mx.random.seed(0)
        config = _tiny_config()
        config["training"]["dtype"] = dtype
        config["model"]["weight_dtype"] = dtype
        model = DiaModel(DiaConfig.load_dict(config))
        _randomize_projections(model)
        source = mx.array([[1, 2, 3], [4, 5, 6]])
        src_positions = mx.broadcast_to(mx.arange(3)[None], (2, 3))
        tokens = mx.arange(2 * 13 * 9).reshape(2, 13, 9) % 1024
        positions = mx.broadcast_to(mx.arange(13)[None], (2, 13))
        causal = mx.tril(mx.ones((13, 13), dtype=mx.bool_))[None, None]
        expected = model(
            source,
            tokens,
            src_positions=src_positions,
            tgt_positions=positions,
            dec_self_attn_mask=causal,
            enable_dropout=False,
        )
        encoder_out = model.encoder(source, src_positions=src_positions)

        for prefill_length in (1, 3, 9):
            with self.subTest(prefill_length=prefill_length):
                cache = KVCache()
                cache.step = 4
                cross = model.decoder.precompute_cross_attention_kv(
                    encoder_out, src_positions
                )
                fixed_k, fixed_v = cross[0]
                outputs = [
                    model.decoder(
                        tokens[:, :prefill_length],
                        encoder_out,
                        positions[:, :prefill_length],
                        src_positions,
                        True,
                        causal[:, :, :prefill_length, :prefill_length],
                        None,
                        [cache],
                        cross,
                    )
                ]
                capacity = cache.keys.shape[2]
                for position in range(prefill_length, 13):
                    self.assertEqual(cache.offset, position)
                    outputs.append(
                        model.decoder.decode_step(
                            tokens[:, position : position + 1],
                            positions[:, position : position + 1],
                            encoder_out,
                            None,
                            None,
                            [cache],
                            cross,
                        )
                    )
                actual = mx.concatenate(outputs, axis=1)
                self.assertGreater(mx.std(expected).item(), 0.01)
                self.assertTrue(mx.allclose(actual, expected, atol=tolerance).item())
                self.assertEqual(cache.offset, 13)
                self.assertGreater(cache.keys.shape[2], capacity)
                self.assertEqual(cache.keys.dtype, getattr(mx, dtype))
                self.assertIs(cross[0].k, fixed_k)
                self.assertIs(cross[0].v, fixed_v)
                self.assertEqual(fixed_k.shape[2], source.shape[1])


class TestDiaGenerationCache(unittest.TestCase):
    def _run(self, frames, budget, eos=False, public=False, cache_step=8):
        mx.random.seed(1)
        with patch.object(dia, "DAC", _FakeDAC):
            model = dia.Model(_tiny_config())
        _randomize_projections(model.model)
        model.dac_model.frames = frames or 8
        state = {
            "decode": [],
            "samples": [],
            "prefill": [],
            "prefill_capacity": [],
            "caches": [],
        }
        decoder = model.model.decoder
        decode_step = decoder.decode_step
        forward = Decoder.__call__

        def make_cache():
            cache = KVCache()
            cache.step = cache_step
            state["caches"].append(cache)
            return cache

        def record_prefill(instance, *args, **kwargs):
            logits = forward(instance, *args, **kwargs)
            state["prefill"].append(logits[:, -1])
            state["prefill_capacity"].append(
                kwargs["self_attention_cache"][0].keys.shape[2]
            )
            return logits

        def record_decode(**kwargs):
            cache = kwargs["self_attention_cache"][0]
            position = int(kwargs["tgt_pos_Bx1"][0, 0])
            state["decode"].append((position, cache.offset))
            return decode_step(**kwargs)

        def sample(logits, **kwargs):
            mx.eval(logits)
            state["samples"].append(logits)
            value = 1024 if eos and len(state["samples"]) == 1 else 0
            return mx.full((9,), value, dtype=mx.int32)

        decoder.decode_step = record_decode
        reference = None if frames is None else mx.zeros((4410,), dtype=mx.float32)
        with (
            patch.object(dia, "KVCache", side_effect=make_cache),
            patch.object(Decoder, "__call__", record_prefill),
            patch.object(dia, "_sample_next_token", side_effect=sample),
            patch.object(dia, "codebook_to_audio", return_value=np.zeros(64)),
            patch.object(
                decoder,
                "precompute_cross_attention_kv",
                wraps=decoder.precompute_cross_attention_kv,
            ) as cross,
        ):
            if public:
                results = list(
                    model.generate("[S1] Hello", max_tokens=budget, ref_audio=reference)
                )
                state["tokens"] = results[0].token_count
                self.assertEqual(results[0].samples, 64)
            else:
                _, state["tokens"] = model._generate(
                    "[S1] Hello", max_tokens=budget, ref_audio=reference, cfg_scale=0
                )
            self.assertEqual(cross.call_count, 1)
        return state

    def test_reference_prefill_is_not_decoded_twice(self):
        for frames in (8, 40):
            with self.subTest(frames=frames):
                state = self._run(frames, 32)
                prefill = frames + 1
                self.assertEqual(len(state["samples"]), 32)
                self.assertEqual(state["tokens"], 32)
                self.assertEqual(
                    state["decode"], [(i, i) for i in range(prefill, prefill + 31)]
                )
                self.assertEqual(state["caches"][0].offset, prefill + 31)
                self.assertTrue(
                    mx.array_equal(
                        state["samples"][0][:, :1025],
                        state["prefill"][0][1, :, :1025],
                    ).item()
                )
                self.assertTrue(mx.isneginf(state["samples"][0][:, 1025:]).all().item())

    def test_single_generated_token_uses_prefill_logits(self):
        state = self._run(8, 1)
        self.assertEqual(state["decode"], [])
        self.assertEqual(len(state["samples"]), 1)
        self.assertEqual(state["tokens"], 1)
        self.assertEqual(state["caches"][0].offset, 9)

    def test_generation_grows_past_default_cache_block(self):
        state = self._run(248, 32, cache_step=KVCache.step)
        self.assertEqual(state["prefill_capacity"], [256])
        self.assertEqual(state["caches"][0].keys.shape[2], 512)
        self.assertEqual(state["caches"][0].offset, 280)
        self.assertEqual(state["tokens"], 32)

    def test_no_reference_decodes_bos_then_generated_tokens(self):
        state = self._run(None, 32)
        self.assertEqual(state["prefill"], [])
        self.assertEqual(state["decode"], [(i, i) for i in range(32)])
        self.assertEqual(state["caches"][0].offset, 32)
        self.assertEqual(state["tokens"], 32)

    def test_eos_stops_within_budget_after_codec_tail(self):
        for frames in (None, 8):
            with self.subTest(frames=frames):
                state = self._run(frames, 64, eos=True)
                self.assertEqual(len(state["samples"]), 30)
                self.assertEqual(state["tokens"], 30)
                self.assertLess(state["tokens"], 64)
                self.assertTrue(
                    all(position == offset for position, offset in state["decode"])
                )

    def test_public_generation_with_reference_audio(self):
        state = self._run(40, 32, public=True)
        self.assertEqual(state["tokens"], 32)
        self.assertEqual(state["caches"][0].offset, 72)


if __name__ == "__main__":
    unittest.main()
