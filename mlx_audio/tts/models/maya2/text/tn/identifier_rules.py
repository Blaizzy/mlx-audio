# Vendored from MayaResearch/maya2-native (https://github.com/MayaResearch/maya2-native), Copyright Maya Research,
# licensed under the Apache License 2.0 (see LICENSE in this directory). Modified for mlx-audio: reformatted with black.
"""Opt-in identifier rendering; no model calls or prose transliteration.

Uppercase dictionary components prescribe character spelling. In particular,
AXIS is spelled as an identifier rather than guessed as the word 'access'.
Ordinary words stay words; existing locale letter/symbol tables supply sounds.
"""

import json
import re
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def vocabulary():
    return json.loads(
        Path(__file__).with_name("identifier_words.json").read_text(encoding="utf-8")
    )


def spell_uri(part, lang, spell_digits):
    data = vocabulary()

    def word(token, expand=True):
        if expand and token.lower() in data["brands"]:
            return " ".join(word(p, False) for p in data["brands"][token.lower()])
        if (
            len(token) == 1
            or token.isupper()
            or token.upper() in data["acronyms"]
            or token.upper() in lang.ABBREV_UNITS
        ):
            return " ".join(lang.letter(c) for c in token)
        pieces = re.findall(r"[A-Z]+(?=[A-Z][a-z]|$)|[A-Z]?[a-z]+|[A-Z]+", token)
        if len(pieces) > 1:
            return " ".join(word(p, False) for p in pieces)
        return token

    symbols = {
        "@": "at",
        ".": "dot",
        "_": "underscore",
        "-": "minus",
        "/": "slash",
        "+": "plus",
        "=": "equals",
        "&": "amp",
        "#": "hash",
        "%": "percent",
    }
    out = []
    # Retain non-ASCII runs intact, including Indic combining marks.
    for token in re.findall(
        r"[A-Za-z]+|[0-9]+|[@._/\-+?=&#%:~]|[^A-Za-z0-9@._/\-+?=&#%:~]+", part
    ):
        if token.isascii() and token.isalpha():
            out.append(word(token))
        elif token.isdecimal():
            out.append(spell_digits(token, comma=False))
        elif token in symbols:
            out.append(lang.SYM[symbols[token]])
        else:
            out.append(
                {"?": "question mark", ":": "colon", "~": "tilde"}.get(token, token)
            )
    return " ".join(p for p in out if p.strip())


def group_digits(digits, lang, sizes=None):
    """Spaces inside groups, comma pause between groups; every digit survives."""
    if sizes is None:
        if len(digits) == 16:
            sizes = [4] * 4
        elif len(digits) == 10:
            sizes = [3, 3, 4]
        else:
            sizes = [4] * (len(digits) // 4) + [len(digits) % 4]
    out, start = [], 0
    for size in sizes:
        if size:
            out.append(
                " ".join(lang.digit(int(c)) for c in digits[start : start + size])
            )
            start += size
    assert start == len(digits)
    return ", ".join(out)


# These shapes identify card/telephone numbers even without English keywords.
# Dates and currency retain higher matcher priority in the normalizer.
LONG_RUN = re.compile(
    r"(?<![A-Za-z0-9.,])(?:\+\d{1,3}[ -]+|91[ -]+)?"
    r"(?:\d{4}(?:[ -]\d{4}){3}|\d{5}[ -]\d{5}|\d{3}[ -]\d{3}[ -]\d{4}|\d{4}[ -]\d{7}|\d{8,})"
    r"(?![A-Za-z0-9]|[.,]\d)"
)


def long_runs(text, lang):
    for match in LONG_RUN.finditer(text):
        token = match.group().lstrip("+")
        groups = re.split(r"[ -]", token)
        digits = "".join(groups)
        sizes = [len(g) for g in groups] if len(groups) > 1 else None
        spoken = group_digits(digits, lang, sizes)
        if match.group().startswith("+"):
            spoken = lang.SYM["plus"] + " " + spoken
        yield match.start(), match.end(), spoken
