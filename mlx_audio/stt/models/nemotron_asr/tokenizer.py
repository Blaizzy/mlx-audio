"""Vocabulary helpers for the Nemotron multilingual BPE tokenizer.

The vocabulary is the flat list of SentencePiece pieces from the ``.nemo``. The
first token is ``<unk>`` and the model can emit a leading language tag such as
``<en-US>`` (especially in ``auto`` mode); ``strip_lang_tags`` removes these from
the final text.
"""

import re

_LANG_TAG_RE = re.compile(r"^<[a-z]{2,3}-[A-Za-z]{2,4}>$")
_OTHER_SPECIAL = {"<unk>", "<pad>", "<s>", "</s>"}


def is_lang_tag(piece: str) -> bool:
    return bool(_LANG_TAG_RE.match(piece))


def is_special_piece(piece: str) -> bool:
    return piece in _OTHER_SPECIAL or is_lang_tag(piece)


def is_special_token(token_id: int, vocabulary: list[str]) -> bool:
    if token_id < 0 or token_id >= len(vocabulary):
        return False
    return is_special_piece(vocabulary[token_id])


def piece_to_text(piece: str) -> str:
    return piece.replace("▁", " ")  # ▁ -> space


def decode(
    tokens: list[int], vocabulary: list[str], strip_lang_tags: bool = True
) -> str:
    parts: list[str] = []
    for token in tokens:
        if token < 0 or token >= len(vocabulary):
            continue
        piece = vocabulary[token]
        if piece in _OTHER_SPECIAL:
            continue
        if strip_lang_tags and is_lang_tag(piece):
            continue
        parts.append(piece_to_text(piece))
    return "".join(parts)


def detected_language(tokens: list[int], vocabulary: list[str]) -> str | None:
    """Return the first emitted language tag (e.g. ``en-US``) if any."""
    for token in tokens:
        if 0 <= token < len(vocabulary) and is_lang_tag(vocabulary[token]):
            return vocabulary[token][1:-1]
    return None
