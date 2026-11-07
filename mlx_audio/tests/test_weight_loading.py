import json
import logging
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_audio import utils


class TinyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = nn.Linear(4, 4)
        self.decoder = nn.Linear(4, 4)


def full_checkpoint(prefix=""):
    return {
        f"{prefix}{layer}.{name}": mx.ones(shape)
        for layer in ("encoder", "decoder")
        for name, shape in (("weight", (4, 4)), ("bias", (4,)))
    }


@pytest.fixture
def load(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "tiny"}))

    def load(weights, model_class=TinyModel, **kwargs):
        package = SimpleNamespace(Model=model_class)
        monkeypatch.setattr(utils.importlib, "import_module", lambda path: package)
        monkeypatch.setattr(utils, "load_weights", lambda path: dict(weights))
        return utils.base_load_model(
            tmp_path, category="stt", model_remapping={}, **kwargs
        )

    return load


def unloaded_warnings(caplog):
    return [r.getMessage() for r in caplog.records if "not loaded" in r.getMessage()]


def test_warns_when_checkpoint_names_do_not_match(load, caplog):
    weights = full_checkpoint()
    for name in ("decoder.weight", "decoder.bias"):
        weights[f"model.{name}"] = weights.pop(name)

    with caplog.at_level(logging.WARNING, logger="mlx_audio.utils"):
        model = load(weights)

    assert mx.array_equal(model.encoder.weight, mx.ones((4, 4)))
    messages = unloaded_warnings(caplog)
    assert len(messages) == 1
    message = messages[0]
    assert message.startswith("2 of 4 parameters were not loaded")
    assert "decoder.bias, decoder.weight" in message


def test_checks_models_that_wrap_a_module(load, caplog):
    # Like qwen3_asr.Model: delegates to an nn.Module without subclassing it.
    class Wrapper:
        def __init__(self, config):
            self._model = TinyModel(config)

        def parameters(self):
            return self._model.parameters()

        def load_weights(self, weights, strict=False):
            return self._model.load_weights(weights, strict=strict)

        def eval(self):
            return self._model.eval()

    weights = full_checkpoint()
    del weights["encoder.bias"]

    with caplog.at_level(logging.WARNING, logger="mlx_audio.utils"):
        load(weights, model_class=Wrapper)

    messages = unloaded_warnings(caplog)
    assert len(messages) == 1
    assert "encoder.bias" in messages[0]


def test_warns_when_model_load_weights_drops_mismatched_shapes(load, caplog):
    # Like sam_audio.SAMAudio: the override skips weights whose shapes differ
    # from the config-built model, so the names alone all match.
    class ShapeCheckingModel(TinyModel):
        def load_weights(self, weights, strict=False):
            shapes = {k: v.shape for k, v in nn.utils.tree_flatten(self.parameters())}
            kept = [(k, v) for k, v in weights if shapes.get(k) == v.shape]
            return super().load_weights(kept, strict=strict)

    weights = full_checkpoint()
    weights["decoder.weight"] = mx.ones((8, 4))

    with caplog.at_level(logging.WARNING, logger="mlx_audio.utils"):
        load(weights, model_class=ShapeCheckingModel)

    messages = unloaded_warnings(caplog)
    assert len(messages) == 1
    assert "1 of 4 parameters" in messages[0]
    assert "decoder.weight" in messages[0]


def test_complete_checkpoint_loads_quietly(load, caplog):
    with caplog.at_level(logging.WARNING, logger="mlx_audio.utils"):
        load(full_checkpoint())

    assert unloaded_warnings(caplog) == []


def test_parameters_filled_by_post_load_hook_count_as_loaded(load, caplog):
    class HookedModel(TinyModel):
        @staticmethod
        def post_load_hook(model, path):
            model.decoder.update({"weight": mx.ones((4, 4)), "bias": mx.ones((4,))})
            return model

    weights = full_checkpoint()
    del weights["decoder.weight"], weights["decoder.bias"]

    with caplog.at_level(logging.WARNING, logger="mlx_audio.utils"):
        load(weights, model_class=HookedModel)

    assert unloaded_warnings(caplog) == []


def test_strict_loading_still_raises(load):
    weights = full_checkpoint()
    del weights["decoder.bias"]

    with pytest.raises(ValueError, match="Missing 1 parameters"):
        load(weights, strict=True)
