"""Tests for mlx_audio.utils helpers."""

from mlx_audio.utils import is_valid_module_name


def test_is_valid_module_name_accepts_plain_identifiers():
    assert is_valid_module_name("parakeet")
    assert is_valid_module_name("whisper")
    assert is_valid_module_name("parakeet_tdt")
    assert is_valid_module_name("_private")
    assert is_valid_module_name("model2")


def test_is_valid_module_name_rejects_dots():
    # Regression: candidates synthesized by get_model_name_parts can
    # contain '.' (e.g. "parakeet_tdt_0.6b" built from "parakeet-tdt-0.6b-v3").
    # importlib.util.find_spec would otherwise interpret the dot as a package
    # separator and crash with ModuleNotFoundError on the synthetic parent.
    assert not is_valid_module_name("parakeet_tdt_0.6b")
    assert not is_valid_module_name("whisper.large.v3")
    assert not is_valid_module_name("a.b")


def test_is_valid_module_name_rejects_other_non_identifiers():
    assert not is_valid_module_name("")
    assert not is_valid_module_name("2model")        # leading digit
    assert not is_valid_module_name("model-name")    # hyphen
    assert not is_valid_module_name("model name")    # space
    assert not is_valid_module_name(None)            # type: ignore[arg-type]
