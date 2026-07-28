from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class STTEvalSample:
    utt_id: str
    audio_path: Path
    reference_text: str
    source_path: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
