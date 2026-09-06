import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_audio.lm.models.ssm import ssm_update


def sequential_reference(x, A_log, B, C, D, dt, dt_bias, time_step_limit):
    dt = dt.astype(mx.float32)
    dt = nn.softplus(dt + dt_bias)
    dt = mx.clip(dt, time_step_limit[0], time_step_limit[1])
    decay = mx.exp(-mx.exp(A_log) * dt)
    heads_per_group = x.shape[2] // B.shape[2]
    B = mx.repeat(B, heads_per_group, axis=2).astype(mx.float32)
    C = mx.repeat(C, heads_per_group, axis=2).astype(mx.float32)
    x_float = x.astype(mx.float32)
    state = mx.zeros((*x.shape[:1], *x.shape[2:], B.shape[-1]))
    outputs = []

    for token in range(x.shape[1]):
        input_scale = x_float[:, token] * dt[:, token, :, None]
        state = decay[:, token, :, None, None] * state + (
            input_scale[..., None] * B[:, token, :, None, :]
        )
        output = mx.sum(state * C[:, token, :, None, :], axis=-1)
        output += x_float[:, token] * D.astype(mx.float32).reshape(1, -1, 1)
        outputs.append(output.astype(x.dtype))

    return mx.stack(outputs, axis=1), state


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_initial_scan_matches_sequential_recurrence(dtype):
    mx.random.seed(7)
    shape = (2, 17, 4, 8)
    x = (0.5 * mx.random.normal(shape)).astype(dtype)
    A_log = mx.log(mx.arange(1, shape[2] + 1, dtype=mx.float32))
    B = (0.1 * mx.random.normal((2, shape[1], 2, 32))).astype(dtype)
    C = (0.1 * mx.random.normal((2, shape[1], 2, 32))).astype(dtype)
    D = mx.ones((shape[2],), dtype=dtype)
    dt = mx.random.normal(shape[:3], dtype=dtype)
    dt_bias = mx.random.normal((shape[2],), dtype=mx.float32)
    time_step_limit = (0.001, 100.0)

    got = ssm_update(x, A_log, B, C, D, dt, dt_bias, None, time_step_limit)
    want = sequential_reference(x, A_log, B, C, D, dt, dt_bias, time_step_limit)
    mx.eval(*got, *want)

    assert got[0].dtype == x.dtype
    assert got[1].dtype == mx.float32
    output_tolerance = 1e-6 if dtype == mx.float32 else 0.004
    assert (
        mx.max(mx.abs(got[0].astype(mx.float32) - want[0])).item() <= output_tolerance
    )
    assert mx.max(mx.abs(got[1] - want[1])).item() <= 1e-6


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_initial_scan_falls_back_for_unsupported_inputs(monkeypatch):
    from mlx_audio.lm.models import ssm

    x = mx.zeros((1, 2, 2, 4))
    A_log = mx.zeros((2,))
    B = mx.zeros((1, 2, 1, 8))
    C = mx.zeros((1, 2, 1, 8))
    D = mx.ones((2,))
    dt = mx.zeros((1, 2, 2))
    dt_bias = mx.zeros((2,))

    monkeypatch.setattr(
        ssm,
        "ssm_initial_kernel",
        lambda *args, **kwargs: pytest.fail("initial kernel should not run"),
    )

    output, state = ssm.ssm_update(x, A_log, B, C, D, dt, dt_bias)
    mx.eval(output, state)
    assert output.shape == x.shape
