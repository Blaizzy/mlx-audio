"""Regression tests for Ming sliding-attention masks and activation dtypes."""

import mlx.core as mx
import pytest

from mlx_audio.lm.models.cache import KVCache
from mlx_audio.lm.models.qwen2 import ModelArgs as Qwen2ModelArgs
from mlx_audio.tts.models.bailingmm.bailingmm import MingQwen2Attention

# The mask uses a finite negative sentinel that overflows to -inf in
# low-precision dtypes, so compare against a threshold instead of -inf.
MASKED = -1e30


def _build(mask, q_len, kv_len, sliding_window, dtype, cache_offset=0):
    return MingQwen2Attention._build_sliding_additive_mask(
        mask=mask,
        q_len=q_len,
        kv_len=kv_len,
        cache_offset=cache_offset,
        sliding_window=sliding_window,
        dtype=dtype,
    )


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
def test_mask_dtype_follows_query_dtype(dtype):
    """The mask must be built in the dtype of the query activations."""
    mask = _build("causal", 4, 4, 2, dtype)
    assert mask.dtype == dtype


def test_sliding_window_keeps_causal_structure():
    """A window covering the sequence must preserve causal attention."""
    mask = _build("causal", 4, 4, 4, mx.bfloat16)
    values = mask[0, 0].tolist()

    for row in range(4):
        for col in range(4):
            if col <= row:
                assert values[row][col] == 0.0, f"({row},{col}) should be visible"
            else:
                assert values[row][col] < MASKED, f"({row},{col}) should be masked"


def test_enabled_sliding_window_still_applies():
    """A positive sliding window keeps restricting the visible range."""
    mask = _build("causal", 4, 4, 2, mx.float32)
    values = mask[0, 0].tolist()

    # With window=2, row 3 can only see columns 2 and 3.
    assert values[3][0] < MASKED
    assert values[3][1] < MASKED
    assert values[3][2] == 0.0
    assert values[3][3] == 0.0


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
def test_sliding_attention_forward_preserves_dtype(dtype):
    """Exercise the query-dtype handoff during prefill and cached decoding."""
    args = Qwen2ModelArgs(
        model_type="qwen2",
        hidden_size=32,
        num_hidden_layers=1,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        rms_norm_eps=1e-6,
        vocab_size=1,
    )
    attention = MingQwen2Attention(args, sliding_window=2)
    attention.set_dtype(dtype)
    cache = KVCache()

    for q_len in (4, 1):
        x = mx.ones((1, q_len, args.hidden_size), dtype=dtype)
        output = attention(x, mask="causal" if q_len > 1 else None, cache=cache)
        assert output.dtype == dtype
        assert mx.all(mx.isfinite(output)).item()
