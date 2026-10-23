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
