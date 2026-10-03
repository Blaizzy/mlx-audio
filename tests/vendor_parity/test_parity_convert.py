import json

import mlx.core as mx
import mlx.nn as nn
import pytest
from conftest import assert_exactly_equal

upstream_utils = pytest.importorskip("mlx_lm.utils")
upstream_convert = pytest.importorskip("mlx_lm.convert")

from mlx_audio.lm import convert as vendored


class Tiny(nn.Module):
    def __init__(self, layers=8):
        super().__init__()
        self.layers = [
            nn.Sequential(nn.Linear(64, 64), nn.Linear(64, 64)) for _ in range(layers)
        ]
        self.lm_head = nn.Linear(64, 128)

    def __call__(self, x):
        return x


class Deep(nn.Module):
    """Named so paths contain down_proj / v_proj / lm_head like a real LM."""

    def __init__(self, layers=8):
        super().__init__()
        self.layers = [Deep._Block() for _ in range(layers)]
        self.lm_head = nn.Linear(64, 128)

    class _Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.v_proj = nn.Linear(64, 64)
            self.down_proj = nn.Linear(64, 64)

    def __call__(self, x):
        return x


def test_quantize_model_matches_upstream():
    v, u = Tiny(), Tiny()
    mx.eval(v.parameters())
    u.update(v.parameters())
    cfg = {"model_type": "tiny"}
    vm, vc = vendored.quantize_model(v, cfg, 64, 4)
    um, uc = upstream_utils.quantize_model(u, cfg, 64, 4)
    assert vc == uc
    from mlx.utils import tree_flatten

    vw, uw = dict(tree_flatten(vm.parameters())), dict(tree_flatten(um.parameters()))
    assert vw.keys() == uw.keys()
    for k in vw:
        assert_exactly_equal(vw[k], uw[k])


@pytest.mark.parametrize("recipe", ["mixed_2_6", "mixed_3_4", "mixed_3_6", "mixed_4_6"])
@pytest.mark.parametrize("layers", [4, 8, 32])
def test_mixed_quant_predicate_matches_upstream(recipe, layers):
    model = Deep(layers)
    v = vendored.mixed_quant_predicate_builder(recipe, model)
    u = upstream_convert.mixed_quant_predicate_builder(recipe, model)
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        assert v(name, module) == u(name, module), name


def test_save_model_matches_upstream(tmp_path):
    v, u = Tiny(), Tiny()
    mx.eval(v.parameters())
    u.update(v.parameters())
    mx.eval(u.parameters())

    vendored.save_model(tmp_path / "v", v)
    upstream_utils.save_model(tmp_path / "u", u)

    vi = json.loads((tmp_path / "v" / "model.safetensors.index.json").read_text())
    ui = json.loads((tmp_path / "u" / "model.safetensors.index.json").read_text())
    assert vi["weight_map"] == ui["weight_map"]
    assert vi["metadata"]["total_size"] == ui["metadata"]["total_size"]

    vw = mx.load(str(tmp_path / "v" / "model.safetensors"))
    uw = mx.load(str(tmp_path / "u" / "model.safetensors"))
    assert vw.keys() == uw.keys()
    for k in vw:
        assert_exactly_equal(vw[k], uw[k])


def test_save_model_shards_like_upstream():
    weights = {f"w{i}": mx.zeros((1024, 1024)) for i in range(4)}
    got = vendored.make_shards(weights, max_file_size_gb=1)
    want = upstream_utils.make_shards(weights, max_file_size_gb=1)
    assert [sorted(s) for s in got] == [sorted(s) for s in want]


def test_save_config_matches_upstream(tmp_path):
    cfg = {
        "b": 2,
        "a": 1,
        "_name_or_path": "x",
        "vision_config": {"y": 1},
        "quantization": {"group_size": 64, "bits": 4},
    }
    vendored.save_config(dict(cfg), tmp_path / "v.json")
    upstream_utils.save_config(dict(cfg), tmp_path / "u.json")
    assert json.loads((tmp_path / "v.json").read_text()) == json.loads(
        (tmp_path / "u.json").read_text()
    )
