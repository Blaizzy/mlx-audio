from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .schema import validate_generation_record


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class GenerationRecord:
    generation_id: str
    prompt_hash: str
    provided_age_band: str
    voice_profile_id: str
    reference_hashes: list[str]
    rights_lane: str
    base_hash: str
    adapter_hash: str | None
    speaker_encoder_hash: str | None
    dac_hash: str | None
    code_commit: str
    adapter_strength: float
    sampling: dict[str, Any]
    seed: int
    audio_hash: str
    local_audio_path: str
    transcript: str
    alignment: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    checkpoint_stage: str = "studio"
    future_preference_eligible: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        validate_generation_record(data)
        return data


def write_generation_records(path: str | Path, rows: list[GenerationRecord]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row.to_dict(), sort_keys=True) + "\n")


def anti_studio_score(studio: dict[str, float], candidate: dict[str, float]) -> dict[str, float]:
    keys = [
        "pause_distance",
        "f0_delta_distance",
        "energy_delta_distance",
        "rate_variability_distance",
    ]
    improvements = {}
    for key in keys:
        base = float(studio.get(key, 0.0))
        cand = float(candidate.get(key, base))
        improvements[key] = 0.0 if base == 0 else (base - cand) / abs(base)
    penalties = {
        "noise_penalty": max(0.0, float(candidate.get("noise_db_increase", 0.0))),
        "bandwidth_penalty": max(0.0, float(studio.get("bandwidth_hz", 0.0)) - float(candidate.get("bandwidth_hz", 0.0))),
        "clipping_penalty": max(0.0, float(candidate.get("clipping", 0.0))),
    }
    score = sum(improvements.values()) / len(improvements) - sum(penalties.values())
    return {"score": score, **improvements, **penalties}

