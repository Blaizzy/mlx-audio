# Vendored from MayaResearch/maya2-native (https://github.com/MayaResearch/maya2-native), Copyright Maya Research,
# licensed under the Apache License 2.0 (see LICENSE in this directory). Modified for mlx-audio: reformatted with black.
"""Deterministic, CPU-only locale inference and output script containment.

Count letters/marks, not shared danda punctuation or decimal digits. Devanagari
cannot distinguish Hindi/Marathi: auto uses Hindi; explicit Marathi is preferred.
"""

import unicodedata
from collections import Counter

BLOCKS = (
    (0x0900, 0x097F, "hi"),
    (0x0980, 0x09FF, "bn"),
    (0x0A00, 0x0A7F, "pa"),
    (0x0A80, 0x0AFF, "gu"),
    (0x0B00, 0x0B7F, "or"),
    (0x0B80, 0x0BFF, "ta"),
    (0x0C00, 0x0C7F, "te"),
    (0x0C80, 0x0CFF, "kn"),
    (0x0D00, 0x0D7F, "ml"),
)


def script_counts(text):
    counts = Counter()
    for c in text:
        if unicodedata.category(c)[0] not in "LM":
            continue
        if unicodedata.name(c, "").startswith("LATIN"):
            counts["en-IN"] += 1
            continue
        for low, high, code in BLOCKS:
            if low <= ord(c) <= high:
                counts[code] += 1
                break
    return counts


def detect_language(text):
    counts = script_counts(text)
    code = max(counts, key=counts.get) if counts else "en-IN"
    if code == "bn" and any(c in text for c in "ৰৱ"):
        return "as"
    return code


def guard_output(original, result, locale):
    allowed = set(script_counts(original))
    # A numeric/symbol-only input supplies no script; use the default locale.
    if not allowed or set(script_counts(result)) <= allowed:
        return result
    # Fail conservatively: preserve prose and read the original digits in the
    # chosen input script; never return a generated foreign-script expansion.
    if __package__:
        from .normalize import language_table
    else:
        from normalize import language_table
    locale = language_table(detect_language(original))
    result = "".join(
        " " + locale.digit(unicodedata.decimal(c)) + " " if c.isdecimal() else c
        for c in original
    )
    return (
        " ".join(result.split()) if set(script_counts(result)) <= allowed else original
    )
