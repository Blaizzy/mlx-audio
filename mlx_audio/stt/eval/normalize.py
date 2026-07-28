"""Text normalization for STT WER evaluation."""

from __future__ import annotations

import re
import unicodedata

_WHITESPACE_RE = re.compile(r"\s+")
_APOSTROPHES = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201b": "'",
    "\u2032": "'",
}


def normalize_for_wer(text: str) -> str:
    """Normalize English text for Seed-TTS-style WER scoring.

    This intentionally keeps normalization small: lowercase, remove punctuation,
    preserve apostrophes in contractions, and collapse whitespace.
    """
    if not text:
        return ""

    for src, dst in _APOSTROPHES.items():
        text = text.replace(src, dst)

    chars: list[str] = []
    for char in text:
        if char == "'":
            chars.append(char)
            continue
        if unicodedata.category(char).startswith("P"):
            continue
        chars.append(char)

    normalized = "".join(chars).lower()
    return _WHITESPACE_RE.sub(" ", normalized).strip()
