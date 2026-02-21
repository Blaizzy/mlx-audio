"""Sampling presets for MOSS-TTS family variants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

MOSS_TTS_RUNTIME = "moss_tts"
MOSS_TTS_REALTIME_RUNTIME = "moss_tts_realtime"


@dataclass(frozen=True)
class MossSamplingPreset:
    name: str
    runtime: str
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    repetition_window: Optional[int] = None


_PRESET_CATALOG: Dict[str, MossSamplingPreset] = {
    "moss_tts": MossSamplingPreset(
        name="moss_tts",
        runtime=MOSS_TTS_RUNTIME,
        temperature=1.7,
        top_p=0.8,
        top_k=25,
        repetition_penalty=1.0,
    ),
    "moss_tts_local": MossSamplingPreset(
        name="moss_tts_local",
        runtime=MOSS_TTS_RUNTIME,
        temperature=1.0,
        top_p=0.95,
        top_k=50,
        repetition_penalty=1.1,
    ),
    "ttsd": MossSamplingPreset(
        name="ttsd",
        runtime=MOSS_TTS_RUNTIME,
        temperature=1.1,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.1,
    ),
    "voice_generator": MossSamplingPreset(
        name="voice_generator",
        runtime=MOSS_TTS_RUNTIME,
        temperature=1.5,
        top_p=0.6,
        top_k=50,
        repetition_penalty=1.1,
    ),
    "soundeffect": MossSamplingPreset(
        name="soundeffect",
        runtime=MOSS_TTS_RUNTIME,
        temperature=1.5,
        top_p=0.6,
        top_k=50,
        repetition_penalty=1.2,
    ),
    "realtime": MossSamplingPreset(
        name="realtime",
        runtime=MOSS_TTS_REALTIME_RUNTIME,
        temperature=0.8,
        top_p=0.6,
        top_k=30,
        repetition_penalty=1.1,
        repetition_window=50,
    ),
}

_PRESET_ALIASES: Dict[str, str] = {
    "delay": "moss_tts",
    "moss_tts_delay": "moss_tts",
    "moss_ttsd": "ttsd",
    "moss_voice_generator": "voice_generator",
    "voicegenerator": "voice_generator",
    "moss_soundeffect": "soundeffect",
    "moss_sound_effect": "soundeffect",
    "sound_effect": "soundeffect",
    "moss_tts_realtime": "realtime",
}


def _normalize_preset_name(preset: str) -> str:
    return str(preset).strip().lower().replace("-", "_")


def available_presets_for_runtime(runtime: str) -> Tuple[str, ...]:
    return tuple(
        preset.name for preset in _PRESET_CATALOG.values() if preset.runtime == runtime
    )


def resolve_sampling_preset(
    preset: Optional[str],
    *,
    runtime: str,
) -> Optional[MossSamplingPreset]:
    if preset is None:
        return None
    normalized_name = _normalize_preset_name(preset)
    canonical_name = _PRESET_ALIASES.get(normalized_name, normalized_name)
    resolved = _PRESET_CATALOG.get(canonical_name)
    if resolved is None:
        known = sorted(_PRESET_CATALOG.keys())
        raise ValueError(f"Unsupported preset '{preset}'. Expected one of {known}")
    if resolved.runtime != runtime:
        expected = sorted(available_presets_for_runtime(runtime))
        raise ValueError(
            f"Preset '{preset}' is not valid for runtime '{runtime}'. "
            f"Expected one of {expected}"
        )
    return resolved


__all__ = [
    "MOSS_TTS_REALTIME_RUNTIME",
    "MOSS_TTS_RUNTIME",
    "MossSamplingPreset",
    "available_presets_for_runtime",
    "resolve_sampling_preset",
]
