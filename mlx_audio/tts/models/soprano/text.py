"""
Text normalization utilities for Soprano TTS.
Adapted from https://github.com/ekwek1/soprano
"""

import re
import unicodedata

# Simple number to words conversion (no external deps)
ONES = [
    "",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
]
TENS = [
    "",
    "",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
]
ORDINALS = {
    1: "first",
    2: "second",
    3: "third",
    4: "fourth",
    5: "fifth",
    6: "sixth",
    7: "seventh",
    8: "eighth",
    9: "ninth",
    10: "tenth",
    11: "eleventh",
    12: "twelfth",
    13: "thirteenth",
    14: "fourteenth",
    15: "fifteenth",
    16: "sixteenth",
    17: "seventeenth",
    18: "eighteenth",
    19: "nineteenth",
    20: "twentieth",
    30: "thirtieth",
    40: "fortieth",
    50: "fiftieth",
    60: "sixtieth",
    70: "seventieth",
    80: "eightieth",
    90: "ninetieth",
}


def _num_to_words(n: int) -> str:
    """Convert integer to words."""
    if n < 0:
        return "minus " + _num_to_words(-n)
    if n == 0:
        return "zero"
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else " " + ONES[n % 10])
    if n < 1000:
        return (
            ONES[n // 100]
            + " hundred"
            + ("" if n % 100 == 0 else " " + _num_to_words(n % 100))
        )
    if n < 1000000:
        return (
            _num_to_words(n // 1000)
            + " thousand"
            + ("" if n % 1000 == 0 else " " + _num_to_words(n % 1000))
        )
    if n < 1000000000:
        return (
            _num_to_words(n // 1000000)
            + " million"
            + ("" if n % 1000000 == 0 else " " + _num_to_words(n % 1000000))
        )
    return (
        _num_to_words(n // 1000000000)
        + " billion"
        + ("" if n % 1000000000 == 0 else " " + _num_to_words(n % 1000000000))
    )


def _ordinal_to_words(n: int) -> str:
    """Convert ordinal number to words."""
    if n in ORDINALS:
        return ORDINALS[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        if ones == 0:
            return ORDINALS.get(n, TENS[tens] + "th")
        return TENS[tens] + " " + ORDINALS.get(ones, ONES[ones] + "th")
    base = _num_to_words(n)
    if base.endswith("y"):
        return base[:-1] + "ieth"
    return base + "th"


# Abbreviations
_abbreviations = [
    (re.compile(r"\b%s\." % x[0], re.IGNORECASE), x[1])
    for x in [
        ("mrs", "misuss"),
        ("ms", "miss"),
        ("mr", "mister"),
        ("dr", "doctor"),
        ("st", "saint"),
        ("co", "company"),
        ("jr", "junior"),
        ("maj", "major"),
        ("gen", "general"),
        ("drs", "doctors"),
        ("rev", "reverend"),
        ("lt", "lieutenant"),
        ("hon", "honorable"),
        ("sgt", "sergeant"),
        ("capt", "captain"),
        ("esq", "esquire"),
        ("ltd", "limited"),
        ("col", "colonel"),
        ("ft", "fort"),
    ]
]

_cased_abbreviations = [
    (re.compile(r"\b%s\b" % x[0]), x[1])
    for x in [
        ("TTS", "text to speech"),
        ("Hz", "hertz"),
        ("kHz", "kilohertz"),
        ("KBs", "kilobytes"),
        ("KB", "kilobyte"),
        ("MBs", "megabytes"),
        ("MB", "megabyte"),
        ("GBs", "gigabytes"),
        ("GB", "gigabyte"),
        ("TBs", "terabytes"),
        ("TB", "terabyte"),
        ("APIs", "a p i's"),
        ("API", "a p i"),
        ("CLIs", "c l i's"),
        ("CLI", "c l i"),
        ("CPUs", "c p u's"),
        ("CPU", "c p u"),
        ("GPUs", "g p u's"),
        ("GPU", "g p u"),
        ("Ave", "avenue"),
        ("etc", "etcetera"),
    ]
]


def expand_abbreviations(text: str) -> str:
    """Expand common abbreviations."""
    for regex, replacement in _abbreviations + _cased_abbreviations:
        text = re.sub(regex, replacement, text)
    return text


# Number patterns
_num_prefix_re = re.compile(r"#\d")
_num_suffix_re = re.compile(r"\d(K|M|B|T)", re.IGNORECASE)
_comma_number_re = re.compile(r"(\d[\d,]+\d)")
_dollars_re = re.compile(r"\$([\d.,]*\d+)")
_ordinal_re = re.compile(r"\d+(st|nd|rd|th)")
_number_re = re.compile(r"\d+")


def _expand_num_prefix(m):
    return f"number {m.group(0)[1]}"


def _expand_num_suffix(m):
    match = m.group(0)
    suffix = match[1].upper()
    suffixes = {"K": "thousand", "M": "million", "B": "billion", "T": "trillion"}
    return f"{match[0]} {suffixes.get(suffix, '')}"


def _remove_commas(m):
    return m.group(1).replace(",", "")


def _expand_dollars(m):
    match = m.group(1).replace(",", "")
    parts = match.split(".")
    if len(parts) > 2:
        return match + " dollars"
    dollars = int(parts[0]) if parts[0] else 0
    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    if dollars and cents:
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        cent_unit = "cent" if cents == 1 else "cents"
        return f"{_num_to_words(dollars)} {dollar_unit}, {_num_to_words(cents)} {cent_unit}"
    if dollars:
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        return f"{_num_to_words(dollars)} {dollar_unit}"
    if cents:
        cent_unit = "cent" if cents == 1 else "cents"
        return f"{_num_to_words(cents)} {cent_unit}"
    return "zero dollars"


def _expand_ordinal(m):
    num = int(re.sub(r"(st|nd|rd|th)$", "", m.group(0)))
    return _ordinal_to_words(num)


def _expand_number(m):
    num = int(m.group(0))
    if 1000 < num < 3000:
        if num == 2000:
            return "two thousand"
        if 2000 < num < 2010:
            return "two thousand " + _num_to_words(num % 100)
        if num % 100 == 0:
            return _num_to_words(num // 100) + " hundred"
        # Year-like pronunciation
        first = num // 100
        second = num % 100
        if second < 10:
            return _num_to_words(first) + " oh " + _num_to_words(second)
        return _num_to_words(first) + " " + _num_to_words(second)
    return _num_to_words(num)


def normalize_numbers(text: str) -> str:
    """Normalize numbers in text."""
    text = re.sub(_num_prefix_re, _expand_num_prefix, text)
    text = re.sub(_num_suffix_re, _expand_num_suffix, text)
    text = re.sub(_comma_number_re, _remove_commas, text)
    text = re.sub(_dollars_re, _expand_dollars, text)
    text = re.sub(_ordinal_re, _expand_ordinal, text)
    text = re.sub(_number_re, _expand_number, text)
    return text


# Special characters
_special_characters = [
    (re.compile(x[0]), x[1])
    for x in [
        ("@", " at "),
        ("&", " and "),
        ("%", " percent "),
        (":", "."),
        (";", ","),
        (r"\+", " plus "),
        (r"\\", " backslash "),
        ("~", " about "),
        ("<", " less than "),
        (">", " greater than "),
        ("=", " equals "),
        ("/", " slash "),
        ("_", " "),
    ]
]


def expand_special_characters(text: str) -> str:
    """Expand special characters to words."""
    for regex, replacement in _special_characters:
        text = re.sub(regex, replacement, text)
    return text


def lowercase(text: str) -> str:
    """Convert text to lowercase."""
    return text.lower()


def convert_to_ascii(text: str) -> str:
    """Convert unicode to ASCII using NFKD normalization."""
    # Normalize unicode characters and encode to ASCII
    normalized = unicodedata.normalize("NFKD", text)
    return normalized.encode("ascii", "ignore").decode("ascii")


def remove_unknown_characters(text: str) -> str:
    """Remove characters not recognized by the model."""
    text = re.sub(r"[^A-Za-z !\$%&'\*\+,\-./0123456789<>\?_]", "", text)
    text = re.sub(r"[<>/_+]", "", text)
    return text


def collapse_whitespace(text: str) -> str:
    """Collapse multiple whitespace to single space."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r" ([.?!,])", r"\1", text)
    return text.strip()


def dedup_punctuation(text: str) -> str:
    """Remove duplicate punctuation."""
    text = re.sub(r"\.\.\.+", "...", text)
    text = re.sub(r",+", ",", text)
    text = re.sub(r"[.,]*\.[.,]*", ".", text)
    text = re.sub(r"[.,!]*![.,!]*", "!", text)
    text = re.sub(r"[.,!?]*\?[.,!?]*", "?", text)
    return text


def clean_text(text: str) -> str:
    """Clean and normalize text for TTS.

    Args:
        text: Input text to clean.

    Returns:
        Cleaned and normalized text.
    """
    text = convert_to_ascii(text)
    text = normalize_numbers(text)
    text = expand_abbreviations(text)
    text = expand_special_characters(text)
    text = lowercase(text)
    text = remove_unknown_characters(text)
    text = collapse_whitespace(text)
    text = dedup_punctuation(text)
    return text
