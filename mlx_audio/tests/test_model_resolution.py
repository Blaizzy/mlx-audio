from types import SimpleNamespace

import pytest

from mlx_audio import utils


@pytest.mark.parametrize(
    "model_type,repo,category",
    [
        ("gradium_phonon_tts", "gradium/phonon-40m-es", "tts"),
        ("pocket_tts", "gradium/phonon-40m-es", "tts"),
        ("pocket_tts", "nativ-community/phonon", "tts"),
        ("phonon", "FermionResearch/Phonon-1", "stt"),
    ],
)
def test_phonon_category_routing(model_type, repo, category):
    assert (
        utils.get_model_category(model_type, utils.get_model_name_parts(repo))
        == category
    )


@pytest.mark.parametrize(
    "category,model_type,canonical",
    [
        ("stt", "granite_speech_plus", "granite_speech"),
        ("tts", "csm", "sesame"),
        ("sts", "mossformer2", "mossformer2_se"),
    ],
)
@pytest.mark.parametrize("name", [None, [], ["checkpoint"], ["whisper"]])
def test_config_alias_takes_precedence_over_name(
    monkeypatch, category, model_type, canonical, name
):
    module = SimpleNamespace()
    imported = []

    def import_module(path):
        imported.append(path)
        return module

    monkeypatch.setattr(utils.importlib, "import_module", import_module)
    assert utils.get_model_class(
        model_type, name, category, {model_type: canonical}
    ) == (module, canonical)
    assert imported == [f"mlx_audio.{category}.models.{canonical}"]


@pytest.mark.parametrize(
    "model_type,name,mapping,expected",
    [
        ("qwen2", ["mimo"], {"mimo": "mimo_v2_asr"}, "mimo_v2_asr"),
        ("unknown", ["whisper"], {}, "whisper"),
        (
            "whisper",
            ["granite_speech"],
            {"whisper": "whisper", "granite_speech": "granite_speech"},
            "whisper",
        ),
    ],
)
def test_model_name_fallback_preserves_known_config(
    monkeypatch, model_type, name, mapping, expected
):
    monkeypatch.setattr(utils.importlib, "import_module", lambda path: path)
    assert utils.get_model_class(model_type, name, "stt", mapping) == (
        f"mlx_audio.stt.models.{expected}",
        expected,
    )


def test_checkpoint_preparation_precedes_config_and_weight_loading(
    tmp_path, monkeypatch
):
    import json

    source = tmp_path / "snapshot"
    prepared = tmp_path / "prepared"
    source.mkdir()
    prepared.mkdir()
    (source / "config.json").write_text('{"model_type": "packed"}')
    (prepared / "config.json").write_text('{"model_type": "runtime", "ready": true}')
    calls = []

    def download(repo, **kwargs):
        assert repo == "example/checkpoint"
        assert kwargs["revision"] == "pinned"
        assert kwargs["force_download"] is True
        return source

    def prepare(path, **kwargs):
        assert path == source
        assert kwargs == {
            "source": "example/checkpoint",
            "revision": "pinned",
            "force_download": True,
        }
        calls.append("prepare")
        return prepared

    def config_from_dict(config):
        assert config["ready"]
        assert config["model_path"] == str(prepared)
        calls.append("config")
        return config

    class RuntimeModel:
        def __init__(self, config):
            self.config = config
            calls.append("model")

        def load_weights(self, weights, strict):
            assert weights == [("weight", "loaded")]
            assert strict
            calls.append("weights")

        def eval(self):
            calls.append("eval")

        @staticmethod
        def post_load_hook(model, path):
            assert path == prepared
            calls.append("post_load")
            return model

    def load_weights(path):
        assert path == prepared
        return {"weight": "loaded"}

    modules = {
        "mlx_audio.stt.models.packed": SimpleNamespace(prepare_model_path=prepare),
        "mlx_audio.stt.models.runtime": SimpleNamespace(
            Model=RuntimeModel,
            ModelConfig=SimpleNamespace(from_dict=config_from_dict),
        ),
    }
    monkeypatch.setattr(utils, "get_model_path", download)
    monkeypatch.setattr(utils.importlib, "import_module", modules.__getitem__)
    monkeypatch.setattr(utils, "load_weights", load_weights)
    monkeypatch.setattr(utils, "apply_quantization", lambda *args: None)
    model = utils.base_load_model(
        "example/checkpoint",
        category="stt",
        model_remapping={},
        lazy=True,
        strict=True,
        revision="pinned",
        force_download=True,
    )
    assert isinstance(model, RuntimeModel)
    assert json.loads((source / "config.json").read_text()) == {"model_type": "packed"}
    assert calls == ["prepare", "config", "model", "weights", "eval", "post_load"]
