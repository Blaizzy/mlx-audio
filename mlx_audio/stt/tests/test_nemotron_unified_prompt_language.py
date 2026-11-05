"""Prompt language resolution against NeMo, for every language string.

The fixture holds the checkpoint's prompt_dictionary (nvidia/nemotron-labs-unified-asr-0.6b
@023ce907) and, for each key, bare code and case variant, the index NeMo's
``resolve_prompt_index`` returns, or null where it raises (unsupported or ambiguous).
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mlx_audio.stt.models.nemotron_asr import Model

FIXTURE = json.loads(
    (
        Path(__file__).parent / "data" / "nemotron_unified_prompt_languages.json"
    ).read_text()
)
MODEL = SimpleNamespace(
    prompt_dictionary=FIXTURE["prompt_dictionary"], default_language="auto"
)


def resolve(language):
    return Model._resolve_prompt_index(MODEL, language)


@pytest.mark.parametrize("language,expected", sorted(FIXTURE["nemo"].items()))
def test_resolves_like_nemo(language, expected):
    if expected is None:
        with pytest.raises(ValueError):
            resolve(language)
    else:
        assert resolve(language) == expected


def test_bare_codes_and_locales_differ_where_nemo_says():
    # es -> 3 but es-ES -> 2; pt -> 13 (pt-PT) but pt-BR -> 12; en-GB is its own prompt.
    assert [
        resolve(k) for k in ("es", "es-ES", "pt", "pt-PT", "pt-BR", "en", "en-GB")
    ] == [3, 2, 13, 13, 12, 0, 1]
    assert resolve(None) == resolve("auto") == 127
