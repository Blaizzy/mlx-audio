from dataclasses import dataclass
from typing import List


@dataclass
class STTOutput:
    text: str
    segments: List[dict] = None
    language: str = None
