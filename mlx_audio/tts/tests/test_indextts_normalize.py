"""Tests for the IndexTTS text normalizer character map.

The module is imported by file path so these tests do not pull in the heavy
``mlx_audio.tts.models.indextts`` package ``__init__`` (which imports model
weights / sentencepiece).
"""

import importlib.util
from pathlib import Path

import pytest

_NORMALIZE_PATH = (
    Path(__file__).resolve().parents[2] / "tts" / "models" / "indextts" / "normalize.py"
)
_spec = importlib.util.spec_from_file_location(
    "indextts_normalize_under_test", _NORMALIZE_PATH
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

CHAR_MAP = _mod.CHAR_MAP
normalize = _mod.normalize


def test_curly_double_quotes_are_mapped_to_apostrophe():
    # U+201C / U+201D must fold to a straight apostrophe like every other
    # quote/bracket variant in CHAR_MAP. They were previously absent because
    # the source line `""": "'", """: "'",` parsed as a triple-quoted string
    # literal, creating one bogus 7-char key and zero curly-quote entries.
    assert CHAR_MAP.get("“") == "'"
    assert CHAR_MAP.get("”") == "'"


def test_no_spurious_multichar_quote_key():
    # The accidental triple-quote produced a bogus key `: "'", ` (7 chars) that
    # could swallow unrelated text. The longest legitimate keys are the 3-char
    # dot/comma runs, so no key longer than that should exist.
    long_keys = {k for k in CHAR_MAP if len(k) > 3}
    assert long_keys == set()
    # And the specific bogus key must be gone.
    assert ': "\'", ' not in CHAR_MAP


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("She said “best” option", "She said 'best' option"),
        ("“quoted”", "'quoted'"),
    ],
)
def test_normalize_folds_curly_quotes(text, expected):
    assert normalize(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("It costs $2.50 today.", "It costs two dollars fifty cents today."),
        (
            "Pay $1,299.99 now.",
            "Pay one thousand two hundred ninety nine dollars ninety nine cents now.",
        ),
        ("It costs $0.99 each.", "It costs ninety nine cents each."),
        ("Only $1 left.", "Only one dollar left."),
        ("It costs $5.", "It costs five dollars."),
        ("I have $5 10 apples.", "I have five dollars ten apples."),
    ],
)
def test_dollar_amounts_read_cents_separately(text, expected):
    assert normalize(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "Prices: $5,$10,$20.",
            "Prices, five dollars,ten dollars,twenty dollars.",
        ),
        ("Pay $5,then leave.", "Pay five dollars,then leave."),
        ("Pay $5, or $10.", "Pay five dollars, or ten dollars."),
        (
            "Prices: $1,000,$2,500.25.",
            "Prices, one thousand dollars,"
            "two thousand five hundred dollars twenty five cents.",
        ),
    ],
)
def test_dollar_amounts_preserve_following_commas(text, expected):
    assert normalize(text) == expected
