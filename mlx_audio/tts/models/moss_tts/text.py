from __future__ import annotations

import re
import unicodedata

_CJK_CHARS = r"\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff"
_CJK = f"[{_CJK_CHARS}]"
_PROT = r"___PROT\d+___"

_URL_RE = re.compile(
    r"https?://[^\s\u3000\uff0c\u3002\uff01\uff1f\uff1b\u3001\uff09\u3011\u300b\u3009\u300d\u300f]+"
)
_EMAIL_RE = re.compile(
    r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![\w.-])"
)
_MENTION_RE = re.compile(r"(?<![A-Za-z0-9_])@[A-Za-z0-9_]{1,32}")
_REDDIT_RE = re.compile(r"(?<![A-Za-z0-9_])(?:u|r)/[A-Za-z0-9_]+")
_HASHTAG_RE = re.compile(r"(?<![A-Za-z0-9_])#(?!\s)[^\s#]+")
_DOT_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_])\.(?=[A-Za-z0-9._-]*[A-Za-z0-9])[A-Za-z0-9._-]+"
)
_FILELIKE_RE = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"(?=[A-Za-z0-9._/+:-]*[A-Za-z])"
    r"(?=[A-Za-z0-9._/+:-]*[._/+:-])"
    r"[A-Za-z0-9](?:[A-Za-z0-9._/+:-]*[A-Za-z0-9])?"
    r"(?![A-Za-z0-9_])"
)
_LATINISH = rf"(?:{_PROT}|(?=[A-Za-z0-9._/+:-]*[A-Za-z])[A-Za-z0-9][A-Za-z0-9._/+:-]*)"
_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200d\ufeff]")


def normalize_tts_text(text: str | None) -> str | None:
    """Normalize MOSS-TTS prompt text using the upstream v1.5 cleanup rules."""
    if text is None:
        return None

    text = _base_cleanup(str(text))
    text = _normalize_markdown_and_lines(text)
    text, protected = _protect_spans(text)
    text = _normalize_spaces(text)
    text = _normalize_structural_punctuation(text)
    text = _normalize_repeated_punctuation(text)
    text = _normalize_spaces(text)
    text = _restore_spans(text, protected)
    return text.strip()


def _base_cleanup(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u3000", " ")
    text = _ZERO_WIDTH_RE.sub("", text)

    cleaned = []
    for char in text:
        category = unicodedata.category(char)
        if char in "\n\t " or not category.startswith("C"):
            cleaned.append(char)
    return "".join(cleaned)


def _normalize_markdown_and_lines(text: str) -> str:
    text = re.sub(r"\[([^\[\]]+?)\]\((https?://[^)\s]+)\)", r"\1 \2", text)

    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        line = re.sub(r"^#{1,6}\s+", "", line)
        line = re.sub(r"^>\s+", "", line)
        line = re.sub(r"^[-*+]\s+", "", line)
        line = re.sub(r"^\d+[.)]\s+", "", line)
        lines.append(line)

    return "\u3002".join(lines) if lines else ""


def _protect_spans(text: str) -> tuple[str, list[str]]:
    protected: list[str] = []

    def replace(match: re.Match[str]) -> str:
        index = len(protected)
        protected.append(match.group(0))
        return f"___PROT{index}___"

    for pattern in (
        _URL_RE,
        _EMAIL_RE,
        _MENTION_RE,
        _REDDIT_RE,
        _HASHTAG_RE,
        _DOT_TOKEN_RE,
        _FILELIKE_RE,
    ):
        text = pattern.sub(replace, text)

    return text, protected


def _restore_spans(text: str, protected: list[str]) -> str:
    for index, original in enumerate(protected):
        text = text.replace(f"___PROT{index}___", original)
    return text


def _normalize_spaces(text: str) -> str:
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(rf"({_CJK})\s+(?={_CJK})", r"\1", text)
    text = re.sub(rf"({_CJK})\s+(?=\d)", r"\1", text)
    text = re.sub(rf"(\d)\s+(?={_CJK})", r"\1", text)
    text = re.sub(rf"({_CJK})(?=({_LATINISH}))", r"\1 ", text)
    text = re.sub(rf"(({_LATINISH}))(?={_CJK})", r"\1 ", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(
        r"\s+([\uff0c\u3002\uff01\uff1f\uff1b\uff1a\u3001\u201d\u2019\u300d\u300f\u3011\uff09\u300b])",
        r"\1",
        text,
    )
    text = re.sub(r"([\uff08\u3010\u300c\u300e\u300a\u201c\u2018])\s+", r"\1", text)
    text = re.sub(r"([\uff0c\u3002\uff01\uff1f\uff1b\uff1a\u3001])\s*", r"\1", text)
    text = re.sub(r"\s+([,.;!?])", r"\1", text)
    return re.sub(r" {2,}", " ", text).strip()


def _normalize_structural_punctuation(text: str) -> str:
    for _ in range(2):
        text = re.sub(
            r"(^|[\u3002\uff01\uff1f!?\uff1b;]\s*)[\u3010\u3016\u300e\u300c]([^\u3011\u3017\u300f\u300d]+)[\u3011\u3017\u300f\u300d]\s*",
            "\\1\\2\u3002",
            text,
        )

    text = re.sub(
        r"(^|[\u3002\uff01\uff1f!?\uff1b;]\s*)\u300a([^\u300b]+)\u300b(?=\s*(?:___PROT\d+___|[\u2014\u2013\u2015-]{2,}|$|[\u3002\uff01\uff1f!?\uff1b;,\uff0c]))",
        r"\1\2",
        text,
    )
    text = re.sub(
        r"\s*(?:<[-=]+>|[-=]+>|<[-=]+|[\u2192\u2190\u2194\u21d2\u21d0\u21d4\u27f6\u27f5\u27f7\u27f9\u27f8\u27fa\u21a6\u21a4\u21aa\u21a9])\s*",
        "\uff0c",
        text,
    )
    return re.sub(r"\s*(?:\u2014|\u2013|\u2015|-){2,}\s*", "\u3002", text)


def _normalize_repeated_punctuation(text: str) -> str:
    text = re.sub(r"(?:\.{3,}|\u2026{2,}|\u2026\u2026+)", "\u3002", text)
    text = re.sub(r"[\u3002\uff0e]{2,}", "\u3002", text)
    text = re.sub(r"[\uff0c,]{2,}", "\uff0c", text)
    text = re.sub(r"[!\uff01]{2,}", "\uff01", text)
    text = re.sub(r"[?\uff1f]{2,}", "\uff1f", text)

    def collapse_mixed(match: re.Match[str]) -> str:
        value = match.group(0)
        has_question = any(char in value for char in "?\uff1f")
        has_exclaim = any(char in value for char in "!\uff01")
        if has_question and has_exclaim:
            return "\uff1f\uff01"
        return "\uff1f" if has_question else "\uff01"

    return re.sub(r"[!?\uff01\uff1f]{2,}", collapse_mixed, text)
