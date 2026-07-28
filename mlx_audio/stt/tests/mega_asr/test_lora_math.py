from __future__ import annotations

import mlx.core as mx
import numpy as np

from mlx_audio.stt.models.mega_asr.convert_lora import LoraModule


def test_materialize_delta_equals_scaled_BA():
    from mlx_audio.stt.models.mega_asr.lora import materialize_delta

    A = mx.random.normal((8, 16))
    B = mx.random.normal((32, 8))
    scaling = 1.5

    delta = materialize_delta({"A": A, "B": B, "scaling": scaling})

    exp = scaling * (np.array(B) @ np.array(A))
    got = np.array(delta)

    assert got.shape == (32, 16)
    # Metal float32 matmul uses different accumulation than NumPy's CPU path.
    np.testing.assert_allclose(got, exp, rtol=1e-3, atol=1e-2)


def test_materialize_delta_shape_is_out_by_in_matching_linear_weight():
    from mlx_audio.stt.models.mega_asr.lora import materialize_delta

    A = mx.random.normal((4, 10))
    B = mx.random.normal((20, 4))

    delta = materialize_delta({"A": A, "B": B, "scaling": 1.0})

    assert delta.shape == (20, 10)


def test_materialize_delta_upcasts_low_precision_factors_to_fp32():
    from mlx_audio.stt.models.mega_asr.lora import materialize_delta

    module: LoraModule = {
        "A": mx.random.normal((4, 10)).astype(mx.bfloat16),
        "B": mx.random.normal((20, 4)).astype(mx.float16),
        "scaling": 0.75,
    }

    delta = materialize_delta(module)
    expected = 0.75 * (
        np.array(module["B"].astype(mx.float32))
        @ np.array(module["A"].astype(mx.float32))
    )

    assert delta.dtype == mx.float32
    assert np.allclose(np.array(delta), expected, atol=1e-3)
