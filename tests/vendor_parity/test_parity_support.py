import mlx.core as mx
import pytest
from conftest import assert_exactly_equal

upstream_rope = pytest.importorskip("mlx_lm.models.rope_utils")
upstream_switch = pytest.importorskip("mlx_lm.models.switch_layers")
upstream_act = pytest.importorskip("mlx_lm.models.activations")

from mlx_audio.lm.models import activations as vendored_act
from mlx_audio.lm.models import rope_utils as vendored_rope
from mlx_audio.lm.models import switch_layers as vendored_switch

DIMS, HEADS, SEQ = 64, 2, 12

SCALINGS = [
    None,
    {"rope_type": "linear", "factor": 2.0},
    {
        "rope_type": "llama3",
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
    },
    {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 4096},
]


@pytest.mark.parametrize("scaling", SCALINGS)
@pytest.mark.parametrize("traditional", [False, True])
@pytest.mark.parametrize("offset", [0, 5])
def test_rope_matches_upstream(scaling, traditional, offset):
    kwargs = dict(
        dims=DIMS, base=10000.0, traditional=traditional, scaling_config=scaling
    )
    v = vendored_rope.initialize_rope(**kwargs)
    u = upstream_rope.initialize_rope(**kwargs)
    mx.random.seed(0)
    x = mx.random.normal((1, HEADS, SEQ, DIMS))
    assert_exactly_equal(v(x, offset=offset), u(x, offset=offset))


def test_swiglu_matches_upstream():
    mx.random.seed(0)
    gate, x = mx.random.normal((2, 16)), mx.random.normal((2, 16))
    assert_exactly_equal(vendored_act.swiglu(gate, x), upstream_act.swiglu(gate, x))


@pytest.mark.parametrize("cls", ["SwitchLinear", "SwitchGLU"])
def test_switch_layers_match_upstream(cls):
    n_experts, in_dims, out_dims = 4, 16, 32
    if cls == "SwitchLinear":
        v = vendored_switch.SwitchLinear(in_dims, out_dims, n_experts)
        u = upstream_switch.SwitchLinear(in_dims, out_dims, n_experts)
    else:
        v = vendored_switch.SwitchGLU(in_dims, out_dims, n_experts)
        u = upstream_switch.SwitchGLU(in_dims, out_dims, n_experts)
    u.update(v.parameters())
    mx.random.seed(0)
    x = mx.random.normal((2, 1, in_dims))
    idx = mx.array([[0, 2], [1, 3]])
    assert_exactly_equal(v(x, idx), u(x, idx))


def test_quantized_switch_linear_matches_upstream():
    v = vendored_switch.SwitchLinear(64, 32, 4)
    u = upstream_switch.SwitchLinear(64, 32, 4)
    u.update(v.parameters())
    vq, uq = v.to_quantized(group_size=32), u.to_quantized(group_size=32)
    mx.random.seed(0)
    x = mx.random.normal((2, 1, 64))
    idx = mx.array([[0, 2], [1, 3]])
    assert_exactly_equal(vq(x, idx), uq(x, idx))
