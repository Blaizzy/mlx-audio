"""CPU-safe CLI and registry coverage for Breeze TTS 2."""

import pytest

from mlx_audio.registry import SUPPORTED_MODEL_TYPES
from mlx_audio.tts.models.breeze_tts import Model
from mlx_audio.tts.utils import MODEL_REMAPPING, get_model_and_args
from mlx_audio.utils import get_model_category, get_model_name_parts


@pytest.mark.parametrize("alias", ["breeze", "breeze-tts", "breeze_tts"])
def test_breeze_registry_aliases_resolve_to_one_module(alias):
    module, model_type = get_model_and_args(
        alias, get_model_name_parts("BreezeBlue/breeze-tts-2")
    )

    assert model_type == "breeze_tts"
    assert module.Model is Model
    assert MODEL_REMAPPING[alias] == "breeze_tts"


def test_breeze_registry_advertises_all_aliases():
    assert {"breeze", "breeze-tts", "breeze_tts"} <= SUPPORTED_MODEL_TYPES["tts"]


def test_breeze_publisher_repository_is_classified_as_tts():
    parts = get_model_name_parts("BreezeBlue/breeze-tts-2")

    assert get_model_category("breeze", parts) == "tts"
