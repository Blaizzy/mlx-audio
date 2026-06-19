from __future__ import annotations

import unittest

import mlx.core as mx
import numpy as np

from mlx_audio.research.zonos2_youth.adapter import (
    AdapterManifest,
    LoRASpec,
    LoRAWeights,
    assert_value_slice_only,
    load_lora_weights,
    save_lora_weights,
)
from mlx_audio.research.zonos2_youth.bandwidth import (
    bandwidth_tier,
    codebook_policy_for_tier,
    estimate_effective_bandwidth,
)
from mlx_audio.research.zonos2_youth.eval import GenerationRecord, anti_studio_score, hash_text
from mlx_audio.research.zonos2_youth.schema import (
    validate_adapter_manifest,
    validate_generation_record,
)


class TestYouthNaturalAdapter(unittest.TestCase):
    def test_exact_zero_lora_has_zero_contribution_and_strength_zero_parity(self):
        spec = LoRASpec(name="attention.wq", base_weight_shape=(3, 4), rank=2)
        lora = LoRAWeights.exact_zero(spec)
        x = mx.ones((2, 4))
        self.assertTrue(bool(mx.allclose(lora.contribution(x), mx.zeros((2, 3)))))
        weight = mx.arange(12, dtype=mx.float32).reshape(3, 4)
        self.assertTrue(bool(mx.allclose(lora.apply_to_weight(weight, strength=0.0), weight)))

    def test_value_slice_lora_does_not_alter_key_slice(self):
        spec = LoRASpec(
            name="attention.wkv.value",
            base_weight_shape=(2, 3, 4),
            rank=2,
            target="chunked_value",
            value_slice=(0, 3),
        )
        lora = LoRAWeights.exact_zero(spec)
        lora.a = mx.ones_like(lora.a)
        lora.b = mx.ones_like(lora.b)
        original = mx.arange(24, dtype=mx.float32).reshape(2, 3, 4)
        merged = lora.apply_to_weight(original, strength=1.0)
        self.assertTrue(bool(mx.allclose(original[0], merged[0])))
        self.assertFalse(bool(mx.allclose(original[1], merged[1])))
        assert_value_slice_only(original, merged, (0, 3))

    def test_adapter_manifest_validates(self):
        manifest = AdapterManifest(
            base_checkpoint_hash="abc",
            target_modules=[{"name": "layers.0.attention.wq", "rank": 8}],
            lineage={"dataset_snapshots": ["synthetic"]},
        )
        validate_adapter_manifest(manifest.to_dict())

    def test_lora_safetensors_save_reload(self):
        import tempfile
        from pathlib import Path

        spec = LoRASpec(name="attention.wo", base_weight_shape=(3, 4), rank=2)
        lora = LoRAWeights.exact_zero(spec)
        lora.a = mx.ones_like(lora.a)
        lora.b = mx.ones_like(lora.b)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "adapter.safetensors"
            save_lora_weights(path, [lora])
            loaded = load_lora_weights(path, [spec])[0]
            self.assertTrue(bool(mx.allclose(loaded.a, lora.a)))
            self.assertTrue(bool(mx.allclose(loaded.b, lora.b)))


class TestYouthNaturalBandwidth(unittest.TestCase):
    def test_bandwidth_estimator_detects_tone_frequency(self):
        sr = 16000
        t = np.arange(sr) / sr
        tone = np.sin(2 * np.pi * 3000 * t)
        bw = estimate_effective_bandwidth(tone, sr)
        self.assertGreater(bw, 2900)
        self.assertLess(bw, 3100)
        self.assertEqual(bandwidth_tier(3500), "narrowband_low")
        self.assertEqual(bandwidth_tier(7000), "narrowband_16k")

    def test_narrowband_policy_avoids_unproven_codebook_downweight(self):
        policy = codebook_policy_for_tier("narrowband_16k")
        self.assertEqual(policy["policy"], "all_codebooks_with_anchor_kl")
        self.assertEqual(policy["ce_weight"], [1.0] * 9)


class TestYouthNaturalEvaluation(unittest.TestCase):
    def test_generation_record_validates_future_rl_fields(self):
        record = GenerationRecord(
            generation_id="g1",
            prompt_hash=hash_text("hello"),
            provided_age_band="teen",
            voice_profile_id="spk_x",
            reference_hashes=["refhash"],
            rights_lane="permissive_release",
            base_hash="base",
            adapter_hash="adapter",
            speaker_encoder_hash="speaker",
            dac_hash="dac",
            code_commit="commit",
            adapter_strength=0.75,
            sampling={"temperature": 1.15},
            seed=7,
            audio_hash="audio",
            local_audio_path="local/only.wav",
            transcript="hello",
            checkpoint_stage="youth-stage-best",
            future_preference_eligible=True,
        )
        validate_generation_record(record.to_dict())

    def test_anti_studio_score_penalizes_bandwidth_loss(self):
        result = anti_studio_score(
            {
                "pause_distance": 1.0,
                "f0_delta_distance": 1.0,
                "energy_delta_distance": 1.0,
                "rate_variability_distance": 1.0,
                "bandwidth_hz": 16000.0,
            },
            {
                "pause_distance": 0.5,
                "f0_delta_distance": 0.5,
                "energy_delta_distance": 1.0,
                "rate_variability_distance": 1.0,
                "bandwidth_hz": 8000.0,
            },
        )
        self.assertGreater(result["pause_distance"], 0)
        self.assertGreater(result["bandwidth_penalty"], 0)
        self.assertLess(result["score"], 0)


if __name__ == "__main__":
    unittest.main()
