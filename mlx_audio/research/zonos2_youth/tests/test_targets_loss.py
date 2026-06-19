from __future__ import annotations

import unittest

import mlx.core as mx

from mlx_audio.research.zonos2_youth.targets import (
    INVALID_TARGET_ID,
    build_teacher_forced_batch,
    masked_cross_entropy,
    unshear_targets,
)
from mlx_audio.tts.models.zonos2.prompt import shear


class TestYouthNaturalTargetsAndLoss(unittest.TestCase):
    def test_teacher_forced_batch_predicts_first_target_from_last_prompt_row(self):
        prompt = mx.array([[9, 9, 519], [8, 8, 519]], dtype=mx.int32)
        codes = mx.array([[1, 2], [3, 4], [5, 6]], dtype=mx.int32)
        batch = build_teacher_forced_batch(
            prompt,
            codes,
            audio_pad_id=99,
            text_vocab=519,
        )
        self.assertEqual(batch.input_ids.shape, (4, 3))
        self.assertEqual(batch.targets.shape, (4, 2))
        self.assertEqual(batch.prompt_len, 2)
        self.assertEqual(batch.target_frames, 3)
        self.assertEqual(int(batch.targets[0, 0]), INVALID_TARGET_ID)
        expected_delayed = shear(codes, 99)
        self.assertTrue(bool(mx.all(batch.targets[1:] == expected_delayed)))
        self.assertTrue(
            bool(
                mx.all(
                    unshear_targets(expected_delayed, audio_pad_id=99)
                    == mx.array([[1, 2], [3, 4], [5, 99]], dtype=mx.int32)
                )
            )
        )

    def test_masked_cross_entropy_ignores_prompt_pad_and_invalid_cells(self):
        logits = mx.zeros((3, 2, 5), dtype=mx.float32)
        logits = logits.at[1, 0, 1].add(5.0)
        logits = logits.at[2, 0, 3].add(5.0)
        targets = mx.array(
            [
                [INVALID_TARGET_ID, INVALID_TARGET_ID],
                [1, 99],
                [3, 99],
            ],
            dtype=mx.int32,
        )
        mask = (targets != INVALID_TARGET_ID) & (targets != 99)
        metrics = masked_cross_entropy(logits, targets, mask)
        self.assertEqual(int(metrics["valid_token_count"]), 2)
        self.assertGreater(float(metrics["accuracy"]), 0.99)
        self.assertEqual(metrics["per_codebook_loss"].shape, (2,))


if __name__ == "__main__":
    unittest.main()
