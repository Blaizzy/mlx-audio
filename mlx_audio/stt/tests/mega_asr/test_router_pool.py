from __future__ import annotations

import mlx.core as mx
import numpy as np


def test_attention_pooling_and_classifier_output_shape():
    from mlx_audio.stt.models.mega_asr.router import AttentionPooling, ClassifierHead

    hidden_states = mx.random.normal((1, 10, 256))

    pooled = AttentionPooling()(hidden_states)
    logits = mx.squeeze(ClassifierHead()(pooled), axis=0)

    assert pooled.shape == (1, 256)
    assert logits.shape == (2,)
    assert np.isfinite(np.array(logits)).all()
