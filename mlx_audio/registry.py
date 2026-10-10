"""Lightweight, import-free classification of the audio models mlx-audio supports.

Every model architecture lives under ``mlx_audio/<kind>/models/<family>`` (``tts``,
``stt``, ``sts``, ``codec``, …), and each ``<kind>/utils.py`` carries a
``MODEL_REMAPPING`` that aliases config ``model_type`` values onto those families.
This module surfaces that organisation as data so callers can ask "what kind of
audio model is this?" without importing mlx or any model code.
"""

from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parent

# User-facing voice kinds first so an ambiguous type resolves to the kind a person
# would actually pick; remaining kinds (codec, …) follow.
_PREFERRED_ORDER = ("tts", "stt", "sts")

# Generic backbones that also ship as audio sub-models. A bare ``model_type`` of
# these is far more likely a plain LLM, so they only classify as audio via an
# explicit, audio-specific remapping alias (e.g. ``qwen3_tts``), never by name.
_AMBIGUOUS_FAMILIES = frozenset({"llama", "qwen3", "dense"})


def model_type_from_config(config: dict) -> Optional[str]:
    """Resolve audio architectures whose upstream config names a generic LLM."""
    model_type = config.get("model_type") or config.get("architecture")
    if model_type in {None, "qwen2", "mimo_audio"} and (
        "MiMoAudioModel" in (config.get("architectures") or ())
        or {
            "speech_vocab_size",
            "speech_zeroemb_idx",
            "input_local_layers",
            "local_layers",
            "group_size",
            "audio_channels",
            "delay_pattern",
        }.issubset(config)
    ):
        return "mimo_audio"
    return model_type


@lru_cache(maxsize=None)
def kinds() -> Tuple[str, ...]:
    """All model kinds that ship a ``models/`` directory, voice kinds first."""
    discovered = {
        path.name
        for path in _ROOT.iterdir()
        if path.is_dir()
        and not path.name.startswith("__")
        and (path / "models").is_dir()
    }
    preferred = [kind for kind in _PREFERRED_ORDER if kind in discovered]
    rest = sorted(discovered - set(preferred))
    return tuple(preferred + rest)


@lru_cache(maxsize=None)
def _families(kind: str) -> FrozenSet[str]:
    models_dir = _ROOT / kind / "models"
    if not models_dir.is_dir():
        return frozenset()
    return frozenset(
        path.name
        for path in models_dir.iterdir()
        if path.is_dir() and not path.name.startswith("__")
    )


@lru_cache(maxsize=None)
def _remapping(kind: str) -> Dict[str, str]:
    source = _ROOT / kind / "utils.py"
    if not source.is_file():
        return {}
    try:
        tree = ast.parse(source.read_text())
    except (OSError, SyntaxError):
        return {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "MODEL_REMAPPING"
            for target in node.targets
        ):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, SyntaxError):
            return {}
        return {str(k).lower(): str(v).lower() for k, v in value.items()}
    return {}


@lru_cache(maxsize=None)
def supported_model_types(kind: str) -> FrozenSet[str]:
    """Every ``model_type`` the given kind accepts: families and remapping aliases."""
    remap = _remapping(kind)
    return _families(kind) | frozenset(remap) | frozenset(remap.values())


def _segments(name: str) -> List[str]:
    lowered = name.lower()
    for separator in "/\\-_. ":
        lowered = lowered.replace(separator, " ")
    return [segment for segment in lowered.split() if segment]


@lru_cache(maxsize=None)
def _matchable_families(kind: str) -> FrozenSet[str]:
    return _families(kind) - _AMBIGUOUS_FAMILIES


def _matches(kind: str, model_type: str, segments: List[str]) -> bool:
    families = _matchable_families(kind)
    remap = _remapping(kind)
    if model_type and remap.get(model_type, model_type) in families:
        return True
    return any(
        remap.get(segment, segment) in families or segment in remap
        for segment in segments
    )


def classify_model(model_type: str, model_name: str = "") -> Optional[str]:
    """Resolve a config ``model_type`` (and optional repo name) to its audio kind.

    Returns a kind such as ``"tts"``, ``"stt"`` or ``"sts"``, or ``None`` when the
    model is not a supported audio model. Mirrors the resolution mlx-audio uses when
    loading: exact ``model_type``, then a single family match, then repo-name hints.
    """
    model_type = (model_type or "").strip().lower()
    available = kinds()

    for kind in available:
        if model_type and model_type in _remapping(kind):
            return kind

    folder_hits = [
        kind
        for kind in available
        if model_type and model_type in _matchable_families(kind)
    ]
    if len(folder_hits) == 1:
        return folder_hits[0]

    segments = _segments(model_name)
    for kind in available:
        if _matches(kind, model_type, segments):
            return kind

    return None


def is_supported_model(model_type: str, model_name: str = "") -> bool:
    """Whether mlx-audio can serve the given model as any audio kind."""
    return classify_model(model_type, model_name) is not None


#: ``{kind: frozenset(model_types)}`` for every discovered kind.
SUPPORTED_MODEL_TYPES: Dict[str, FrozenSet[str]] = {
    kind: supported_model_types(kind) for kind in kinds()
}
