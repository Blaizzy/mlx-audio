"""Small actual-MLX tests; no downloaded weights, reference audio or devices.

These intentionally exercise the device runtime. CPU-only CI should run the
separate NumPy contract file rather than this file.
"""

import mlx.core as mx
import numpy as np
import pytest

from mlx_audio.codec.models.stepaudio2.decoder_dit import Attention, DiT
from mlx_audio.codec.models.stepaudio2.flow import CausalMaskedDiffWithXvec
from mlx_audio.codec.models.stepaudio2.flow_matching import CausalConditionalCFM
from mlx_audio.codec.models.stepaudio2.streaming import (
    CodecError,
    _dit_attention,
    decode_chunk,
    encoder_chunk,
    hamming_window,
    prepare_stream,
    reset_stream,
)
from mlx_audio.codec.models.stepaudio2.upsample_encoder_v2 import (
    UpsampleConformerEncoderV2,
)


def tiny_encoder():
    model = UpsampleConformerEncoderV2(
        input_size=8,
        output_size=8,
        num_blocks=1,
        num_up_blocks=1,
        attention_heads=2,
        linear_units=16,
        dropout_rate=0,
        positional_dropout_rate=0,
        attention_dropout_rate=0,
    )
    model.eval()
    mx.eval(model.parameters())
    return model


def test_actual_conformer_chunk_lengths_history_and_unchanged_input_cache():
    import mlx.nn as nn

    mx.random.seed(42)
    model = tiny_encoder()
    values = mx.random.normal((1, 56, 8))
    first, cnn, att = encoder_chunk(model, values[:, :28], False, None, None, mx, nn)
    mx.eval(first, cnn, att)
    saved_cnn, saved_att = np.array(cnn), np.array(att)
    second, next_cnn, next_att = encoder_chunk(
        model, values[:, 25:53], False, cnn, att, mx, nn
    )
    final, _, final_att = encoder_chunk(
        model, values[:, 50:53], True, next_cnn, next_att, mx, nn
    )
    mx.eval(second, final, final_att)
    assert first.shape == second.shape == (1, 50, 8)
    assert final.shape == (1, 6, 8)
    assert cnn.shape == (1, 8, 6)
    assert att.shape == (2, 1, 2, 50, 8)
    assert next_att.shape[3] == 100 and final_att.shape[3] == 106
    assert bool(mx.all(mx.isfinite(final)).item())
    np.testing.assert_array_equal(np.array(cnn), saved_cnn)
    np.testing.assert_array_equal(np.array(att), saved_att)


class SyntheticVocoder:
    """Deterministic MLX waveform for real-flow state tests, not HiFT parity."""

    def __call__(self, mel, source):
        wave = mx.repeat(mx.mean(mel, axis=1), 480, axis=1)
        return wave, mx.expand_dims(wave, 1)


@pytest.mark.parametrize("tail", [3, 17, 25])
def test_actual_ten_step_flow_cache_and_reset_with_synthetic_vocoder(tail):
    mx.random.seed(42)
    estimator = DiT(
        in_channels=320,
        out_channels=80,
        hidden_size=8,
        depth=1,
        num_heads=2,
        head_dim=4,
        mlp_ratio=2,
    )
    solver = CausalConditionalCFM(estimator)
    solver.rand_noise = mx.random.normal((1, 80, 512))
    noise = solver.rand_noise
    flow = CausalMaskedDiffWithXvec(
        input_size=8, output_size=80, encoder=tiny_encoder(), decoder=solver
    )
    flow.eval()
    mx.eval(flow.parameters())
    reference = (
        mx.array([list(range(8))], dtype=mx.int32),
        mx.array([8], dtype=mx.int32),
        mx.ones((1, 192), dtype=mx.float32),
        mx.zeros((1, 16, 80), dtype=mx.float32),
        mx.array([16], dtype=mx.int32),
    )
    base = prepare_stream(flow, reference)
    state = reset_stream(base)
    window = hamming_window()
    hift = SyntheticVocoder()
    codes = [4218] * 3 + list(range(25))
    first = decode_chunk(flow, hift, codes, state, reference[2], window)
    second = decode_chunk(
        flow, hift, codes[-3:] + list(range(25, 50)), first.state, reference[2], window
    )
    final = decode_chunk(
        flow,
        hift,
        list(range(tail)),
        second.state,
        reference[2],
        window,
        last_chunk=True,
    )
    again = decode_chunk(flow, hift, codes, reset_stream(base), reference[2], window)
    assert first.waveform.shape == second.waveform.shape == (1, 24000)
    assert final.waveform.shape == (1, (8 + 2 * tail) * 480)
    assert first.state.flow_cache["estimator_att_cache"].shape[:2] == (10, 1)
    assert final.state.flow_cache["estimator_att_cache"].shape[4] == 116
    assert final.state.flow_cache["conformer_att_cache"].shape[3] == 116
    assert final.state.finalized and base.chunks == 0
    assert solver.rand_noise is noise
    np.testing.assert_array_equal(np.array(first.waveform), np.array(again.waveform))
    for name, value in base.flow_cache.items():
        assert value is not state.flow_cache[name]
    with pytest.raises(CodecError, match="open stream"):
        decode_chunk(
            flow, hift, [1, 2, 3], final.state, reference[2], window, last_chunk=True
        )


@pytest.mark.parametrize("history", [0, 7])
def test_actual_waveform_sdpa_matches_eager_cache_and_fp32_outputs(history):
    mx.random.seed(42)
    layer = Attention(dim=512, num_heads=8, head_dim=64, qk_norm=True)
    layer.eval()
    x = mx.random.normal((2, 5, 512))
    cache = None if not history else mx.random.normal((2, 8, history, 128))
    eager, eager_cache = _dit_attention(layer, x, cache, mx, "eager")
    fused, fused_cache = _dit_attention(layer, x, cache, mx, "sdpa")
    mx.eval(eager, eager_cache, fused, fused_cache)
    assert eager.shape == fused.shape == x.shape and fused.dtype == mx.float32
    np.testing.assert_array_equal(np.array(eager_cache), np.array(fused_cache))
    np.testing.assert_allclose(np.array(eager), np.array(fused), rtol=1e-4, atol=2e-5)
