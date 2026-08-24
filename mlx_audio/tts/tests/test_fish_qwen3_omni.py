# Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

"""Guard the Fish/Qwen3-Omni quantization exclusion policy.

Quantizing ``nn.Embedding`` modules (one of which doubles as the tied output
head) or the ``fast_`` residual path produces broken converted checkpoints,
so the model defines a ``model_quant_predicate``. These tests keep that
policy from silently regressing.
"""

import unittest

import mlx.nn as nn


class TestFishSpeechModelQuantPredicate(unittest.TestCase):
    def setUp(self):
        from mlx_audio.tts.models.fish_qwen3_omni.fish_speech import Model

        self.predicate = Model.model_quant_predicate

    def test_embeddings_are_excluded(self):
        emb = nn.Embedding(128, 64)
        self.assertFalse(self.predicate("model.embeddings", emb))
        self.assertFalse(self.predicate("model.codebook_embeddings", emb))

    def test_fast_path_is_excluded(self):
        linear = nn.Linear(64, 64)
        self.assertFalse(self.predicate("model.fast_layers.0", linear))
        self.assertFalse(self.predicate("model.fast_output", linear))
        self.assertFalse(self.predicate("model.fast_project_in", linear))

    def test_regular_linear_layers_are_included(self):
        linear = nn.Linear(64, 64)
        self.assertTrue(self.predicate("model.layers.0.self_attn.q_proj", linear))
        self.assertTrue(self.predicate("model.text_encoder.block.0.mlp", linear))


if __name__ == "__main__":
    unittest.main()
