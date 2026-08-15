"""Assemble the MiniMax Music 3 caption-and-lyrics prompt.

Adapted and modified from mikolaj92/minimax-music3-mlx under Apache-2.0.
See LICENSE and NOTICE.
"""

from __future__ import annotations

import re

_SPECIAL_TAG_RE = re.compile(r"<\|([^|]*)\|>")
_LEADING_TAGS_RE = re.compile(r"^[ \t]*((?:\[[^\]]+\][ \t]*)+)")


def clean_caption(caption: str) -> str:
    def rewrite_tag(match: re.Match) -> str:
        parts = match.group(1).strip().split(None, 1)
        return f"{parts[0]} is {parts[1]}" if len(parts) == 2 else parts[0]

    text = _SPECIAL_TAG_RE.sub(rewrite_tag, caption)
    lines = []
    for line in text.splitlines():
        line = re.sub(r"^\s{0,3}#{1,6}\s+", "", line)
        line = re.sub(r"^\s*[*+-]\s+", "", line)
        line = re.sub(r"\*\*([^*]+)\*\*", r"\1", line)
        line = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", line)
        lines.append(line.rstrip())
    text = "\n".join(lines).replace("• ", "").replace("    ", "")
    text = re.sub(r"^\s*[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    return re.sub(r"\n{2,}", "\n", text)


def normalize_lyrics(lyrics: str) -> str:
    output = []
    for line in lyrics.splitlines():
        match = _LEADING_TAGS_RE.match(line)
        output.append(match.group(1).strip() if match else line)
    text = "\n".join(output)
    text = text.replace("] ", "]\n").replace(" [", "\n[").replace(" ^ ", "\n")
    text = re.sub(r"\[([^\]]+)\]", lambda match: f"[{match.group(1).lower()}]", text)
    return f"[start]\n{text}"


def assemble_prompt(caption: str, lyrics: str) -> str:
    return (
        f"<|im_start|><|caption_start|>{clean_caption(caption)}<|caption_end|>"
        f"<|lyrics_start|>{normalize_lyrics(lyrics)}<|lyrics_end|>"
        "<|im_end|><|audio_start|>"
    )
