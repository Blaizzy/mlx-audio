"""Word-safe multilingual text chunking for long-text TTS generation.

The primary split is sentence-level; a
length-cap *fallback hierarchy* (clause -> word -> grapheme) guarantees no chunk
exceeds ``max_chars`` while NEVER breaking mid-word or mid-codepoint/grapheme.
"""

from __future__ import annotations

import re
import unicodedata

import numpy as np

# Split AFTER a terminal mark, consuming trailing whitespace (the mark stays with the
# preceding chunk). Latin .!?  CJK 。！？  Devanagari danda ।॥.
_SENTENCE_RE = re.compile(r"(?<=[.!?。！？।॥])\s*")
# Clause marks: Latin , ; : —  +  CJK/full-width 、 ， ； ：
_CLAUSE_RE = re.compile(r"(?<=[,;:—、，；：])\s*")

_MAX_CHARS_DEFAULT = 240  # Latin / unknown
_MAX_CHARS_CJK = 120  # CJK/Kana/Hangul pack ~2x model-tokens per char
_MAX_CHARS_DEVANAGARI = 160

_WS_RE = re.compile(r"\s")


def _detect_script(text: str) -> str:
    for ch in text:
        o = ord(ch)
        if 0x0900 <= o <= 0x097F:
            return "devanagari"
        if (
            (0x4E00 <= o <= 0x9FFF)
            or (0x3040 <= o <= 0x30FF)
            or (0xAC00 <= o <= 0xD7AF)
        ):
            return "cjk"
    return "latin"


def resolve_max_chars(text: str, *, language: str | None = None) -> int:
    """Script-aware safety cap (characters). Sentence split is the primary mechanism;
    this only bounds unusually long run-on chunks."""
    code = (language or "").upper()
    if code in ("ZH", "JA", "YUE", "KO"):
        return _MAX_CHARS_CJK
    if code == "HI":
        return _MAX_CHARS_DEVANAGARI
    if code:
        return _MAX_CHARS_DEFAULT
    return {"cjk": _MAX_CHARS_CJK, "devanagari": _MAX_CHARS_DEVANAGARI}.get(
        _detect_script(text), _MAX_CHARS_DEFAULT
    )


def _split_keep(text: str, regex: re.Pattern) -> list[str]:
    return [p for p in (s.strip() for s in regex.split(text)) if p]


def _pack_words(s: str, max_chars: int) -> list[str]:
    out, cur = [], ""
    for word in s.split():
        if not cur:
            cur = word
        elif len(cur) + 1 + len(word) <= max_chars:
            cur += " " + word
        else:
            out.append(cur)
            cur = word
        if len(cur) > max_chars and " " not in cur:
            if _WS_RE.search(cur) is None and _detect_script(cur) == "cjk":
                out.extend(_pack_graphemes(cur, max_chars))
                cur = ""
    if cur:
        out.append(cur)
    return out


def _pack_graphemes(s: str, max_chars: int) -> list[str]:
    """Pack codepoints up to max_chars, never starting a chunk on a combining mark
    (Unicode category M*) — i.e. keep base+combining clusters intact."""
    out, cur = [], ""
    for ch in s:
        if len(cur) >= max_chars and not unicodedata.category(ch).startswith("M"):
            out.append(cur)
            cur = ch
        else:
            cur += ch
    if cur:
        out.append(cur)
    return out


def _cap(piece: str, max_chars: int) -> list[str]:
    piece = piece.strip()
    if not piece:
        return []
    if len(piece) <= max_chars:
        return [piece]
    out: list[str] = []
    for clause in _split_keep(piece, _CLAUSE_RE):
        if len(clause) <= max_chars:
            out.append(clause)
        elif _WS_RE.search(clause) is not None:
            out.extend(_pack_words(clause, max_chars))
        else:
            out.extend(_pack_graphemes(clause, max_chars))
    return out


def split_for_generation(
    text: str, *, max_chars: int, language: str | None = None
) -> list[str]:
    """Split ``text`` into TTS-sized chunks: sentence-first, then a word-safe
    length-cap fallback (clause -> word -> grapheme). Returns non-empty stripped chunks.
    """
    text = text.strip()
    if not text:
        return []
    chunks: list[str] = []
    for sent in _split_keep(text, _SENTENCE_RE):
        chunks.extend(_cap(sent, max_chars))
    return [c for c in (c.strip() for c in chunks) if c]


def assemble_chunks(
    wavs: list[np.ndarray], sample_rate: int, gap_ms: int
) -> np.ndarray:
    """Concatenate per-chunk mono waveforms with ``gap_ms`` silence BETWEEN chunks
    (not after the last). Returns float32 1-D."""
    clean = [np.asarray(w, dtype=np.float32).ravel() for w in wavs]
    clean = [w for w in clean if w.size]
    if not clean:
        return np.zeros(0, dtype=np.float32)
    gap = np.zeros(int(sample_rate * gap_ms / 1000), dtype=np.float32)
    pieces: list[np.ndarray] = []
    for i, w in enumerate(clean):
        if i:
            pieces.append(gap)
        pieces.append(w)
    return np.concatenate(pieces)


# --- degenerate-chunk health check (used by generate_long's retry guard) ---
_MIN_SILENCE_STD = 0.01  # below this RMS std -> silent/dead chunk
_MIN_SEC_PER_WORD = 0.18  # truncation floor for spaced scripts (real speech ~0.3-0.5)
_MIN_SEC_PER_CHAR_CJK = 0.12  # per-char floor for no-space scripts (CJK/Kana/Hangul)


def _count_words(text: str) -> int:
    return len([w for w in _WS_RE.split(text.strip()) if w])


def chunk_health(
    audio: np.ndarray, text: str, sample_rate: int, *, language: str | None = None
) -> bool:
    """Cheap, dependency-free heuristic: is this chunk's audio non-degenerate for ``text``?

    Returns False for empty/non-finite/silent audio, or audio far too short for the text
    (clear truncation). Does NOT detect same-length hallucination -- that needs an ASR
    validator (see generate_long(validator=...)). Never raises on odd input.
    """
    a = np.asarray(audio).ravel()
    if a.size == 0 or not np.all(np.isfinite(a)):
        return False
    if float(np.std(a)) <= _MIN_SILENCE_STD:
        return False
    t = (text or "").strip()
    if not t:
        return True
    audio_s = a.size / float(sample_rate)
    if _detect_script(t) == "cjk":
        n_chars = sum(
            1
            for c in t
            if not c.isspace() and not unicodedata.category(c).startswith("P")
        )
        floor = _MIN_SEC_PER_CHAR_CJK * max(n_chars, 1)
    else:
        floor = _MIN_SEC_PER_WORD * max(_count_words(t), 1)
    return audio_s >= floor
