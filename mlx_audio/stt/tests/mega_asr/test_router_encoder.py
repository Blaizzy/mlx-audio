from __future__ import annotations

import mlx.core as mx
import numpy as np


def test_transformer_encoder_shape_and_finiteness():
    from mlx_audio.stt.models.mega_asr.router import TransformerEncoder

    hidden_states = mx.random.normal((1, 10, 256))

    encoded = TransformerEncoder()(hidden_states)

    assert encoded.shape == (1, 10, 256)
    assert np.isfinite(np.array(encoded)).all()
