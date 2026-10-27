# Vendored from MayaResearch/maya2-native (https://github.com/MayaResearch/maya2-native), Copyright Maya Research,
# licensed under the Apache License 2.0 (see LICENSE in this directory). Modified for mlx-audio: reformatted with black.
"""Dependency-free, reviewable TTS normalization.

Indian English and ten Indic language tables share one corrected parser.
The selected locale controls numbers, currency, dates, units and identifiers;
it does not translate prose or choose/change the model conditioning tag.
Hindi speech output folds nukta and ends standalone है with a danda outside
explicit question sentences (ending in ?).
These are intentional pronunciation/pause workarounds, not spelling corrections.
Invalid/ambiguous tokens fall back to digit and separator words.
FR3_TN_STRICT=1 retains the strict audit contract.
See docs/NORMALIZATION_REVIEW.md for the contract and remaining listening gates.
"""

import argparse
import json
import os

# -*- coding: utf-8 -*-
# The former generator/indic_norm package is absent from this repository.
# This tracked file is the source of truth; protect edits with tests/test_normalize.py.
import re
import sys
import unicodedata
from datetime import date
from functools import lru_cache

# Keep CLI execution and the imported table module on the same LangBase class.
if __name__ == "__main__":
    sys.modules.setdefault("normalize", sys.modules[__name__])


LANGUAGE_NAMES = {
    "en-IN": "Indian English",
    "hi": "Hindi",
    "te": "Telugu",
    "ta": "Tamil",
    "bn": "Bengali",
    "as": "Assamese",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mr": "Marathi",
    "or": "Odia",
    "pa": "Punjabi",
}
_LANGUAGE_ALIASES = {
    alias.lower(): code
    for code, name in LANGUAGE_NAMES.items()
    for alias in (code, name, code if code == "en-IN" else code + "-IN")
}
_LANGUAGE_ALIASES.update(
    {
        "en": "en-IN",
        "english": "en-IN",
        "en-us": "en-IN",
        "en-gb": "en-IN",
        "indian-english": "en-IN",
        "oriya": "or",
        "od": "or",
    }
)

# Input aliases, not translations of arbitrary prose. Keep identifiers digitwise
# even when callers write OTP in native script (including Indian English mode).
_OTP_INPUT_WORDS = {
    "ओटीपी",
    "ఓటీపీ",
    "ఓటిపి",
    "ஓடிபி",
    "ஓடீபி",
    "ওটিপি",
    "ઓટીપી",
    "ಒಟಿಪಿ",
    "ಓಟಿಪಿ",
    "ഒടിപി",
    "ഓടിപി",
    "ଓଟିପି",
    "ਓਟੀਪੀ",
}
_PAISA_INPUT_WORDS = {
    "paise",
    "paisa",
    "पैसा",
    "पैसे",
    "పైసా",
    "పైసలు",
    "பைசா",
    "পয়সা",
    "પૈસા",
    "ಪೈಸೆ",
    "പൈസ",
    "ପଇସା",
    "ਪੈਸਾ",
    "ਪੈਸੇ",
}


def resolve_language(language):
    if not isinstance(language, str) or language.lower() not in _LANGUAGE_ALIASES:
        raise NormalizationError(
            "unsupported_normalization_language",
            "Choose one of the documented normalization languages.",
        )
    return _LANGUAGE_ALIASES[language.lower()]


@lru_cache(maxsize=12)
def language_table(language):
    code = resolve_language(language)
    if code == "en-IN":
        return LANG
    if __package__:
        from .indic_languages import LANGUAGES
    else:
        from indic_languages import LANGUAGES
    return LANGUAGES[code]


class NormalizationError(ValueError):
    """A safe-to-return error code. Never includes the caller's financial text."""

    def __init__(self, code, message, span=None, source=None):
        self.span, self.source = span, source
        self.code = code
        super().__init__(message)


_TAMIL_MONTHS = [
    "ஜனவரி",
    "பிப்ரவரி",
    "மார்ச்",
    "ஏப்ரல்",
    "மே",
    "ஜூன்",
    "ஜூலை",
    "ஆகஸ்ட்",
    "செப்டம்பர்",
    "அக்டோபர்",
    "நவம்பர்",
    "டிசம்பர்",
]
# Calendar days only. Native-speaker/audio review is required before rollout.
_TAMIL_ORDINALS = [
    "",
    "முதலாம்",
    "இரண்டாம்",
    "மூன்றாம்",
    "நான்காம்",
    "ஐந்தாம்",
    "ஆறாம்",
    "ஏழாம்",
    "எட்டாம்",
    "ஒன்பதாம்",
    "பத்தாம்",
    "பதினொன்றாம்",
    "பன்னிரண்டாம்",
    "பதின்மூன்றாம்",
    "பதினான்காம்",
    "பதினைந்தாம்",
    "பதினாறாம்",
    "பதினேழாம்",
    "பதினெட்டாம்",
    "பத்தொன்பதாம்",
    "இருபதாம்",
    "இருபத்தொன்றாம்",
    "இருபத்திரண்டாம்",
    "இருபத்துமூன்றாம்",
    "இருபத்துநான்காம்",
    "இருபத்தைந்தாம்",
    "இருபத்தாறாம்",
    "இருபத்தேழாம்",
    "இருபத்தெட்டாம்",
    "இருபத்தொன்பதாம்",
    "முப்பதாம்",
    "முப்பத்தொன்றாம்",
]


def _ascii_digits(text):
    # str.isdigit also accepts superscripts. Only Unicode DECIMAL digits belong
    # here; keep math superscripts for the math matcher.
    return "".join(str(unicodedata.decimal(c)) if c.isdecimal() else c for c in text)


# Hindi speech compatibility, not a general Unicode diacritic remover. Apply to
# FINAL output because number/date/letter/URL expansions can introduce nukta.
# Keep every other script, accent, vowel sign and nasal mark untouched.
_HINDI_NUKTA_FOLD = str.maketrans(
    {
        "\u093c": None,
        "\u0929": "\u0928",
        "\u0931": "\u0930",
        "\u0934": "\u0933",
        "\u0958": "\u0915",
        "\u0959": "\u0916",
        "\u095a": "\u0917",
        "\u095b": "\u091c",
        "\u095c": "\u0921",
        "\u095d": "\u0922",
        "\u095e": "\u092b",
        "\u095f": "\u092f",
    }
)


def _calendar_check(day, month, year=2000, span=None):
    # A missing year can include Feb 29; never invent a year in the output.
    try:
        date(year, month, day)
    except ValueError:
        raise NormalizationError(
            "invalid_date", "Invalid calendar date; correct the input.", span=span
        ) from None


def _sentence_before(text, start):
    return re.split(r"[.!?;।॥\n]", text[max(0, start - 100) : start])[-1]


def _sentence_after(text, end):
    return re.split(r"[.!?;।॥\n]", text[end : end + 100])[0]


def _valid_grouping(number):
    integer = number.lstrip("+-−").split(".")[0]
    return "," not in integer or bool(
        re.fullmatch(
            r"[0-9]{1,3}(?:,[0-9]{3})+|[0-9]{1,2}(?:,[0-9]{2})*,[0-9]{3}", integer
        )
    )


def _month_lookup(locale):
    return {
        **_ENGLISH_MONTHS,
        **{name.lower(): i + 1 for i, name in enumerate(locale.MONTHS)},
        **{name.lower(): i for name, i in locale.MONTH_ALIASES.items()},
    }


def validate_text(
    text, *, date_order="DMY", two_digit_year_base=None, language="en-IN"
):
    """Validate before destructive cleanup. Dates default to explicit DMY policy.

    Two-digit years require a caller-supplied century (1900 or 2000). Short dates
    in date context must be spelled out; ordinary fractions remain fractions.
    Plain text only: SSML/templates are rejected, not stripped and spoken.
    """
    if not isinstance(text, str):
        raise NormalizationError("invalid_text", "text must be a string.")
    if len(text) > 20000:
        raise NormalizationError(
            "text_too_long", "text exceeds the 20000 character limit."
        )
    if date_order not in ("DMY", "MDY"):
        raise NormalizationError("invalid_date_order", "date_order must be DMY or MDY.")
    if two_digit_year_base is not None and (
        type(two_digit_year_base) is not int or two_digit_year_base not in (1900, 2000)
    ):
        raise NormalizationError(
            "invalid_century", "two_digit_year_base must be 1900 or 2000."
        )
    locale = language_table(language)
    t = _ascii_digits(text)
    if bad := re.search(r"<\s*(?:/?[A-Za-z_]|[!?])|[{}]|\$\{", t):
        raise NormalizationError(
            "markup_or_placeholder",
            "Resolve templates and send plain text, not markup/SSML.",
            span=bad.span(),
        )
    for i, c in enumerate(t):
        if (
            unicodedata.category(c) == "Cc" and c not in "\n\r\t"
        ) or unicodedata.category(c) == "Co":
            raise NormalizationError(
                "control_character",
                "Unsupported control character in text.",
                span=(i, i + 1),
            )
    # URI numeric components are identifiers, not dates, amounts or clocks.
    t = re.sub(
        r"https?://\S+|www\.\S+|\b[\w.%+\-]+@[\w.\-]+\.[A-Za-z]{2,}",
        lambda m: " " * len(m[0]),
        t,
    )
    if bad := re.search(r"[0-9]{31,}", t):
        raise NormalizationError(
            "number_too_long",
            "Numeric spans are limited to 30 digits.",
            span=bad.span(),
        )
    if bad := re.search(r"\d{4}-\d{2}-\d{2}T\d", t, re.I):
        raise NormalizationError(
            "unsupported_timestamp",
            "Spell out the date, time and timezone separately.",
            span=bad.span(),
        )
    for m in re.finditer(r"(?<![\w.])[+-]?[0-9][0-9,]*(?:\.[0-9]+)?", t):
        if len(re.sub(r"\D", "", m[0])) > 30:
            raise NormalizationError(
                "number_too_long",
                "Numeric spans are limited to 30 digits.",
                span=m.span(),
            )
        if not _valid_grouping(m[0].rstrip(",")):
            raise NormalizationError(
                "invalid_grouping",
                "Invalid number grouping; send ungrouped digits or standard grouping.",
                span=m.span(),
            )
    for m in re.finditer(r"(?<![\w\d])([0-9]{4})-([0-9]{1,2})-([0-9]{1,2})(?!\d)", t):
        _calendar_check(int(m[3]), int(m[2]), int(m[1]), span=m.span())
    for m in re.finditer(
        r"(?<![\w\d-])([0-9]{1,2})([/.-])([0-9]{1,2})\2([0-9]{2,4})(?!\d)", t
    ):
        y = int(m[4])
        if len(m[4]) != 4:
            if len(m[4]) != 2 or two_digit_year_base is None:
                raise NormalizationError(
                    "ambiguous_year",
                    "Use a four-digit year or explicitly configure the century.",
                    span=m.span(),
                )
            y += two_digit_year_base
        a, b = int(m[1]), int(m[3])
        _calendar_check(
            a if date_order == "DMY" else b,
            b if date_order == "DMY" else a,
            y,
            span=m.span(),
        )
    months = {
        **{name: i + 1 for i, name in enumerate(_TAMIL_MONTHS)},
        **_month_lookup(locale),
    }
    month_re = "|".join(sorted(map(re.escape, months), key=len, reverse=True))
    ordinal_re = "|".join(
        sorted(
            map(
                re.escape,
                set(locale.ORDINAL_INPUT_SUFFIXES)
                | {"st", "nd", "rd", "th", "ஆம்", "ம்"},
            ),
            key=len,
            reverse=True,
        )
    )
    for pattern, reverse in [
        (
            rf"(?<!\w)([0-9]{{1,2}})(?:-?(?:{ordinal_re}))?[ -]+({month_re})(?!\w)(?:[ ,\-]+([0-9]{{4}}))?",
            False,
        ),
        (
            rf"(?<!\w)({month_re})[ -]+([0-9]{{1,2}})(?:-?(?:{ordinal_re}))?(?!\d)(?:,?\s+([0-9]{{4}}))?",
            True,
        ),
    ]:
        for m in re.finditer(pattern, t, re.I):
            day, month = (m[2], m[1]) if reverse else (m[1], m[2])
            _calendar_check(
                int(day),
                months[month.lower()],
                int(m[3]) if m[3] else 2000,
                span=m.span(),
            )
    tamil_month_re = "|".join(_TAMIL_MONTHS)
    for m in re.finditer(
        rf"({tamil_month_re})\s+(\d{{1,2}})(?:-?(?:ஆம்|ம்))?(?:\s*தேதி)?(?:\s+(\d{{4}}))?",
        t,
    ):
        _calendar_check(
            int(m[2]), months[m[1]], int(m[3]) if m[3] else 2000, span=m.span()
        )
    if bad := re.search(
        r"\b(?:on|by|before|after|date|due)\s+[0-9]{1,2}/[0-9]{1,2}(?![/\d])", t, re.I
    ):
        raise NormalizationError(
            "ambiguous_short_date",
            "Spell out the month and year for a short date.",
            span=bad.span(),
        )
    if bad := re.search(r"தேதி\s+[0-9]{1,2}\.\s*[0-9]{4}", t):
        raise NormalizationError(
            "conflicting_date",
            "Conflicting date components; confirm the intended date.",
            span=bad.span(),
        )
    rupee_words = [
        word for word, key in locale.CURRENCY_WORDS.items() if key == "rupee"
    ]
    rupee_aliases = "|".join(
        sorted(map(re.escape, rupee_words + ["INR", "Rs"]), key=len, reverse=True)
    )
    rupee_prefix = rf"(?:₹|(?<!\w)(?:{rupee_aliases})\.?)"
    minor_units = "|".join(
        sorted(map(re.escape, _PAISA_INPUT_WORDS), key=len, reverse=True)
    )
    if bad := re.search(
        rf"{rupee_prefix}\s*[+-]?[0-9][0-9,.]*\s*(?:{minor_units})(?=$|[\s.,!?;।॥-])",
        t,
        re.I,
    ):
        raise NormalizationError(
            "conflicting_currency_unit",
            "Rupee prefix conflicts with a paise suffix; confirm the unit.",
            span=bad.span(),
        )
    if bad := re.search(
        rf"(?:{rupee_prefix}|[$€£]|\b(?:USD|EUR|GBP))\s*[+-]?[0-9][0-9,.]*\s*%", t, re.I
    ):
        raise NormalizationError(
            "conflicting_currency_unit",
            "A value cannot have both a currency prefix and percent suffix.",
            span=bad.span(),
        )
    for m in re.finditer(r"(?<![\w:])([0-9]{1,3}):([0-9]{2}):([0-9]{2})(?![\d:])", t):
        if int(m[2]) > 59 or int(m[3]) > 59:
            raise NormalizationError(
                "invalid_duration",
                "Duration minutes and seconds must be below 60.",
                span=m.span(),
            )
    no_durations = re.sub(r"[0-9]+:[0-9]{2}:[0-9]{2}", lambda m: " " * len(m[0]), t)
    for m in re.finditer(
        r"(?<![\w:])([0-9]{1,2})(?:[:.]([0-9]{2}))?\s*(a\.?m\.?|p\.?m\.?)(?!\w)|(?<![\w:])([0-9]{1,2}):([0-9]{2})(?![\d:])",
        no_durations,
        re.I,
    ):
        if m[1]:
            valid = 1 <= int(m[1]) <= 12 and int(m[2] or 0) < 60
        else:
            # Explicit ratio context is not a clock.
            if re.search(
                r"\b(?:ratio|odds|proportion)\b", _sentence_before(t, m.start()), re.I
            ):
                continue
            valid = int(m[4]) < 24 and int(m[5]) < 60
        if not valid:
            raise NormalizationError(
                "invalid_time",
                "Invalid clock time; use 00:00-23:59 or 1-12 with am/pm.",
                span=m.span(),
            )


class LangBase:
    # ---- identity -------------------------------------------------------
    code = ""  # ISO 639-1: hi bn ta te mr gu kn ml pa or
    name = ""
    native_digits = ""  # the 10 native digit glyphs "०१२३४५६७८९" etc.

    # ---- numbers --------------------------------------------------------
    ONES = []  # words for 0..99 (100 entries, native script)
    HUNDRED = ""  # word used in "X hundred" composition (Indo-Aryan)
    THOUSAND = ""
    LAKH = ""
    CRORE = ""

    # ---- letters (English letter names written in native script) --------
    LETTERS = {}  # {"A": "...", ..., "Z": "..."}

    # ---- calendar -------------------------------------------------------
    MONTHS = []  # 12 native month names (Jan..Dec, Gregorian)
    MONTH_ALIASES = {}  # accepted alternate input spelling -> month 1..12

    # ---- symbols / unit words -------------------------------------------
    # Values are native-script spoken words. Keys are fixed; the engine
    # relies on every key existing.
    SYM = {
        "rupee": "",
        "dollar": "",
        "euro": "",
        "pound": "",
        "cent": "",
        "percent": "",  # % symbol
        "point": "",  # decimal point word (Hindi: दशमलव)
        "minus": "",
        "plus": "",
        "equals": "",
        "times": "",
        "divide": "",
        "squared": "",  # ² (Hindi: का वर्ग — appended after operand)
        "sqrt": "",  # √ (spoken before operand)
        "lt": "",
        "gt": "",  # < >
        "plus_minus": "",  # ±
        "at": "",  # @  (Hindi: एट द रेट)
        "underscore": "",
        "dot": "",  # . in emails/URLs
        "slash": "",  # / in URLs
        "hash": "",  # #
        "amp": "",  # &
        "half": "",
        "quarter": "",
        "three_quarters": "",
        "degree": "",
        "celsius": "",
        "fahrenheit": "",
        "minute": "",  # time minutes word
    }

    # ---- templates (use {placeholders}) ----------------------------------
    TIME_ON_HOUR = ""  # "{h} बजे"
    TIME_HM = ""  # "{h} बजकर {m} मिनट"
    RANGE = ""  # "{a} से {b}"          (10–20)
    RANGE_TO = ""  # "{a} से {b} तक"       (time ranges)
    RATIO = ""  # "{a} अनुपात {b}"
    POWER = ""  # "{base} की घात {exp}"
    FRACTION = ""  # "{num} बटा {den}"     (2/7)
    DECADE_SHORT = ""  # "{tens} के"           (90s)
    DECADE_FULL = ""  # "{tens} का दशक"       (1980s)

    # Period words emitted before the time (order: period first, then time).
    PERIODS = {"morning": "", "afternoon": "", "evening": "", "night": ""}

    # ---- source-text detection vocab (native words the engine must
    #      recognise in the INPUT). Include all common variants/spellings.
    TIME_MARKERS = []  # e.g. ["बजे", "बजकर"]
    # markers that carry a case suffix which must be re-attached to the
    # rendered time ("10 గంటలకు" -> "పది గంటలకు", not "పది గంటలు")
    MARKER_SUFFIXES = {}  # e.g. {"గంటలకు": "కు"}
    PERIOD_DETECT = {}  # native word -> "morning"/"afternoon"/"evening"/"night"
    YEAR_WORDS = []  # e.g. ["साल", "वर्ष", "सन्"]
    ID_KEYWORDS = []  # native keywords marking identifiers: नंबर, खाता ...
    ID_AFTER_WORDS = []  # words AFTER a digit-run marking it as an
    # identifier ("7104 తో ఎండ్ అయ్యే number")
    LABEL_WORDS = []  # native label words read as cardinal: कमरा, अनुच्छेद ...
    MINUTE_WORDS = []  # native "minute" words in input, e.g. ["मिनट"]
    SCALE_WORDS = {}  # native scale word in input -> "crore"/"lakh"/"thousand"
    CURRENCY_WORDS = {}  # native currency words in input -> canonical key
    # e.g. {"रुपये": "rupee", "रु": "rupee"}

    # ---- ordinals ---------------------------------------------------------
    ORDINAL_EXCEPTIONS = {}  # n -> word (Hindi: {1: "पहला", 2: "दूसरा", ...})
    ORDINAL_SUFFIX = ""  # default suffix appended to cardinal (Hindi: वाँ)
    # Suffixes seen in input that should be preserved on output when possible
    ORDINAL_INPUT_SUFFIXES = []  # e.g. ["वीं", "वाँ", "वें", "वां"]

    # ======================================================================
    # Composition methods — Indo-Aryan defaults; Dravidian modules override.
    # ======================================================================

    def digit(self, d: int) -> str:
        return self.ONES[d]

    def two_digit(self, n: int) -> str:
        return self.ONES[n]

    def hundreds(self, h: int, rest: int) -> str:
        """h in 1..9, rest 0..99 -> 'X sau [rest]'."""
        out = f"{self.ONES[h]} {self.HUNDRED}"
        if rest:
            out += f" {self.two_digit(rest)}"
        return out

    def under_thousand(self, n: int) -> str:
        if n < 100:
            return self.two_digit(n)
        return self.hundreds(n // 100, n % 100)

    def cardinal(self, n: int) -> str:
        """Indian-system cardinal. Override in Dravidian modules."""
        if n < 0:
            return f"{self.SYM['minus']} {self.cardinal(-n)}"
        if n < 1000:
            return self.under_thousand(n)
        parts = []
        crore, n = divmod(n, 10**7)
        lakh, n = divmod(n, 10**5)
        thousand, n = divmod(n, 1000)
        if crore:
            parts.append(f"{self.cardinal(crore)} {self.CRORE}")
        if lakh:
            parts.append(f"{self.two_digit(lakh)} {self.LAKH}")
        if thousand:
            parts.append(f"{self.two_digit(thousand)} {self.THOUSAND}")
        if n:
            parts.append(self.under_thousand(n))
        return " ".join(parts)

    def year(self, y: int) -> str:
        """Year-style reading. Indo-Aryan: 1947 -> 'unnis sau saintalis'.
        Dravidian modules usually override to full cardinal."""
        if 1100 <= y <= 1999:
            hi, lo = divmod(y, 100)
            out = f"{self.two_digit(hi)} {self.HUNDRED}"
            if lo:
                out += f" {self.two_digit(lo)}"
            return out
        return self.cardinal(y)

    def letter(self, c: str) -> str:
        return self.LETTERS.get(c.upper(), c)

    def month(self, i: int) -> str:
        """i is 1..12"""
        return self.MONTHS[i - 1]

    def time_phrase(self, h: int, m: int, period_key=None) -> str:
        # _resolve_time may deliberately hand back a 24-hour hour (languages
        # that transcribe am/pm keep "14:30" as "fourteen thirty"); only fold
        # into 12-hour when it has not already been decided upstream.
        if self.AMPM_WORDS and h > 12:
            h12 = h
        else:
            h12 = h % 12
            if h12 == 0:
                h12 = 12
        if m == 0:
            core = self.TIME_ON_HOUR.format(h=self.two_digit(h12))
        else:
            core = self.TIME_HM.format(h=self.two_digit(h12), m=self.two_digit(m))
        if period_key:
            return f"{self.PERIODS[period_key]} {core}"
        return core

    def ordinal(self, n: int, src_suffix: str = "") -> str:
        if n in self.ORDINAL_EXCEPTIONS and not src_suffix:
            return self.ORDINAL_EXCEPTIONS[n]
        return self._ordinal_compose(n, src_suffix)

    def _ordinal_compose(self, n: int, src_suffix: str) -> str:
        """Default: cardinal + suffix. Languages with stem changes override."""
        base = self.ORDINAL_EXCEPTIONS.get(n)
        if base and not src_suffix:
            return base
        suffix = (
            src_suffix
            if src_suffix in self.ORDINAL_INPUT_SUFFIXES
            else self.ORDINAL_SUFFIX
        )
        return f"{self.cardinal(n)}{suffix}"

    def decimal(self, int_part: int, frac_digits: str) -> str:
        frac = " ".join(self.digit(int(d)) for d in frac_digits)
        return f"{self.cardinal(int_part)} {self.SYM['point']} {frac}"

    def decade(self, tens: int, full: bool) -> str:
        """tens: e.g. 90 for '90s', 80 for '1980s'."""
        tpl = self.DECADE_FULL if full else self.DECADE_SHORT
        return tpl.format(tens=self.two_digit(tens))

    # ------------------------------------------------------------------
    # Grammar hooks — no-ops by default; languages with case/sandhi
    # morphology (Telugu, ...) override these.
    # ------------------------------------------------------------------

    # unit initialisms that read as full native words instead of letters
    # (e.g. Telugu KM -> కిలోమీటర్లు)
    UNIT_WORDS = {}

    # written abbreviations that are not speakable words — expanded to native
    # words wherever they appear (mins -> నిమిషాలు). Keys uppercased.
    ABBREV_UNITS = {}

    # spoken word for the "No." / "no." abbreviation before digits
    NUMBER_WORD = ""

    # When non-empty the language transcribes the am/pm token literally and the
    # engine stops INFERRING a period from the hour. Indo-Aryan languages say
    # the period naturally ("shaam paanch baje"), so they leave this empty;
    # English must not have "in the evening" inserted into text that only said
    # "pm". Keys are lowercase, dots stripped.
    AMPM_WORDS = {}
    COUNT_ONE = None
    SINGULAR_NOUNS = {}

    def join_amount(self, words: str, noun: str) -> str:
        """Join number words with a following noun we insert (currency,
        scale, unit). Languages may apply oblique/singular morphology."""
        if words == self.ONES[1]:
            return f"{self.COUNT_ONE or words} {self.SINGULAR_NOUNS.get(noun, noun)}"
        return f"{words} {noun}"

    def scale_noun(self, canon: str, numstr: str) -> str:
        """Scale word used in '₹3 crore'-type phrases; languages may
        pluralize when the number is not 1."""
        return {"crore": self.CRORE, "lakh": self.LAKH, "thousand": self.THOUSAND}.get(
            canon, canon
        )

    def attach_postposition(self, rep: str, tok: str):
        """Called when a native token `tok` immediately follows a rendered
        span `rep`. Return the merged string (consuming the token), or None
        to leave both untouched."""
        return None


# ----------------------------------------------------------------------------
# Static tables
# ----------------------------------------------------------------------------

# All Indic decimal digit blocks -> ASCII
_INDIC_DIGIT_BLOCKS = [
    0x0966,  # Devanagari
    0x09E6,  # Bengali
    0x0A66,  # Gurmukhi
    0x0AE6,  # Gujarati
    0x0B66,  # Odia
    0x0BE6,  # Tamil
    0x0C66,  # Telugu
    0x0CE6,  # Kannada
    0x0D66,  # Malayalam
]
_DIGIT_TRANS = {}
for _base in _INDIC_DIGIT_BLOCKS:
    for _i in range(10):
        _DIGIT_TRANS[_base + _i] = ord("0") + _i

# Emoji / decorative ranges to strip entirely
_EMOJI_RE = re.compile(
    "["
    "\U0001f300-\U0001faff"  # symbols & pictographs, emoticons, transport...
    "\U00002700-\U000027bf"  # dingbats
    "\U00002600-\U000026ff"  # misc symbols
    "\U0001f1e6-\U0001f1ff"  # flags
    "\U0000fe00-\U0000fe0f"  # variation selectors
    "\U0001f000-\U0001f0ff"
    "\U00002b00-\U00002bff"  # arrows/stars (⭐)
    "\U00002300-\U000023ff"  # misc technical (⌛ ⏳ ⏬)
    "\U00002190-\U000021ff"  # arrows (→)
    "\U000025a0-\U000025ff"  # geometric (● ▶)
    "\U00002022"  # bullet •
    "\U00002605\U00002606"  # ★ ☆
    "\U00002122\U000000a9\U000000ae"  # ™ © ®
    # NOTE: ZWNJ/ZWJ (200C/200D) are deliberately NOT stripped — they are
    # meaningful orthography in Telugu/Malayalam (ఆన్‌లైన్). Only the truly
    # invisible ZWSP/BOM are removed, without inserting a space.
    "`~|\\\\{}\\[\\]«»‹›„‚•·¤§†‡"
    "]+"
)
_ZW_STRIP = {0x200B: None, 0x200E: None, 0x200F: None, 0xFEFF: None}

_ENGLISH_MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

# Established acronyms that stay in Latin script (never spelled out unless
# they are part of an alphanumeric identifier).
_ACRONYM_WHITELIST = {
    "UPI",
    "EMI",
    "API",
    "MRI",
    "GST",
    "SEBI",
    "MCX",
    "IGI",
    "HDFC",
    "ISRO",
    "IMD",
    "SBI",
    "RBI",
    "ICICI",
    "IRCTC",
    "ATM",
    "SIM",
    "OTP",
    "PIN",
    "CEO",
    "CFO",
    "CTO",
    "IPL",
    "BCCI",
    "DNA",
    "RNA",
    "WHO",
    "NASA",
    "PDF",
    "GPS",
    "LED",
    "LCD",
    "USB",
    "WIFI",
    "AI",
    "IT",
    "TV",
    "FM",
    "AC",
    "EV",
    "CBI",
    "NIA",
    "ED",
    "GDP",
    "CPI",
    "FII",
    "DII",
    "IPO",
    "NSE",
    "BSE",
    "KYC",
    "NEFT",
    "RTGS",
    "IMPS",
    "IFSC",
    "CVV",
    "PAN",
    "UID",
    "NRI",
    "MLA",
    "MP",
    "PM",
    "CM",
    "DM",
    "SP",
    "IAS",
    "IPS",
    "NEET",
    "JEE",
    "UPSC",
    "CBSE",
    "ICSE",
    "BA",
    "MA",
    "MBA",
    "MBBS",
    "AIIMS",
    "ISKCON",
    "DLF",
    "TCS",
    "LTIMINDTREE",
    "ONGC",
    "NTPC",
    "BHEL",
    "SAIL",
    "GAIL",
    "LIC",
    "RAM",
    "ROM",
    "CPU",
    "GPU",
    "SSD",
    "HDD",
    "GPA",
    "SUV",
    "SLA",
    "GDP",
    "UIDAI",
    "FASTAG",
    "GSTIN",
    "OK",
    "US",
    "UK",
    "UAE",
    "USA",
    "EU",
    "UN",
    "PSU",
    "FIR",
    "RTI",
    "PIL",
    "GPRS",
    "HD",
    "UHD",
    "CCTV",
    "AQI",
    "UV",
}

# Units read letter-by-letter (initialisms). Keys uppercased for lookup.
_UNIT_INITIALISMS = {
    "GB",
    "MB",
    "KB",
    "TB",
    "PB",
    "MAH",
    "MPH",
    "KMPH",
    "MHZ",
    "GHZ",
    "KHZ",
    "HZ",
    "KWH",
    "RPM",
    "HP",
    "BHP",
    "CC",
    "DPI",
    "PPI",
    "FPS",
    "MBPS",
    "GBPS",
    "KBPS",
    "PSI",
    "BPM",
    "V",
    "KV",
    "W",
    "KW",
    "MW",
    "MA",
    "AH",
    "WH",
    "DB",
    "PPM",
    "ML",
    "MG",
    "KG",
    "KM",
    "CM",
    "MM",
    "GM",
    "MCG",
    "IU",
}

# English "big number" scale words that stay in Latin (5 million -> पाँच million)
_EN_SCALES = {"million", "billion", "trillion", "mn", "bn", "millions", "billions"}

_ID_KEYWORDS_EN = {
    "phone",
    "mobile",
    "card",
    "account",
    "otp",
    "pin",
    "code",
    "order",
    "ref",
    "reference",
    "train",
    "flight",
    "ticket",
    "tracking",
    "license",
    "licence",
    "booking",
    "pnr",
    "upi",
    "imei",
    "registration",
    "number",
    "no",
    "ending",
    "khasra",
    "plot",
    "docket",
    "complaint",
    "case",
    "loan",
    "policy",
    "invoice",
    "receipt",
    "transaction",
    "txn",
    "utr",
    "id",
    "extension",
    "helpline",
    "hotline",
    "whatsapp",
    "aadhaar",
    "aadhar",
    "passport",
    "cvv",
    "gstin",
}

_LABEL_WORDS_EN = {
    "hall",
    "window",
    "counter",
    "room",
    "ward",
    "terminal",
    "zone",
    "gate",
    "section",
    "tower",
    "platform",
    "floor",
    "seat",
    "berth",
    "coach",
    "season",
    "episode",
    "chapter",
    "class",
    "grade",
    "level",
    "article",
    "position",
    "page",
    "question",
    "step",
    "stage",
    "round",
    "phase",
    "wing",
    "block",
    "lane",
    "screen",
    "stand",
    "row",
    "table",
    "lesson",
    "unit",
    "part",
    "volume",
    "rank",
    "standard",
    "semester",
    "waitlist",
}

_STREET_WORDS = {
    "मार्ग",
    "रोड",
    "नगर",
    "सड़क",
    "गली",
    "चौक",
    "road",
    "street",
    "lane",
    "avenue",
    "marg",
    "nagar",
    "chowk",
    "colony",
}

_CURRENCY_SYMBOLS = {
    "₹": "rupee",
    "$": "dollar",
    "€": "euro",
    "£": "pound",
    "Rs.": "rupee",
    "Rs": "rupee",
}

_MATH_WORDS_EN = {
    "squared",
    "cubed",
    "plus",
    "minus",
    "times",
    "equals",
    "power",
    "root",
    "over",
    "by",
    "divided",
    "into",
}


def _fmt(template, **kw):
    out = template
    for k, v in kw.items():
        out = out.replace("{" + k + "}", str(v))
    return out


class Normalizer:
    def __init__(self, lang, *, date_order="DMY", two_digit_year_base=None):
        self.lang = lang
        self.date_order = date_order
        self.two_digit_year_base = two_digit_year_base
        self._build_regexes()

    # ------------------------------------------------------------------
    # Regex construction (depends on language vocab)
    # ------------------------------------------------------------------
    def _build_regexes(self):
        L = self.lang

        def alt(words):
            return (
                "|".join(
                    re.escape(w) for w in sorted(set(words), key=len, reverse=True)
                )
                or r"(?!x)x"
            )

        self._period_re = alt(list(L.PERIOD_DETECT.keys()))
        self._marker_re = alt(L.TIME_MARKERS)
        self._month_lookup = _month_lookup(L)
        self._month_re = alt(self._month_lookup)
        self._scale_re = alt(
            list(L.SCALE_WORDS.keys())
            + [
                "crore",
                "crores",
                "Crore",
                "Crores",
                "Cr",
                "cr",
                "lakh",
                "lakhs",
                "Lakh",
                "Lakhs",
                "thousand",
            ]
            + sorted(_EN_SCALES)
        )
        self._ord_suffix_re = alt(
            list(L.ORDINAL_INPUT_SUFFIXES) + ["st", "nd", "rd", "th"]
        )
        native_currency = [
            word
            for word, key in L.CURRENCY_WORDS.items()
            if not word.isascii() and key in {"rupee", "dollar", "euro", "pound"}
        ]
        self._currency_re = r"₹|\$|€|£|Rs\.?|INR|USD|EUR|GBP"
        if native_currency:
            self._currency_re += "|(?:" + alt(native_currency) + r")\.?"
        self._id_keywords = (
            _ID_KEYWORDS_EN | _OTP_INPUT_WORDS | {w.lower() for w in L.ID_KEYWORDS}
        )
        self._label_words = _LABEL_WORDS_EN | {w.lower() for w in L.LABEL_WORDS}
        self._minute_re = alt(L.MINUTE_WORDS + ["minute", "minutes", "min"])
        # ratio detection keywords: english words + the language's own ratio
        # word extracted from its RATIO template ("{a} अनुपात {b}" -> अनुपात),
        # so "விகிதம் 9:15" reads as a ratio, not a time
        tpl_words = re.sub(r"\{[ab]\}", " ", L.RATIO or "").split()
        self._ratio_kws = {"ratio", "odds", "proportion", "अनुपात"} | {
            w.lower() for w in tpl_words
        }

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------
    def _month_index(self, token):
        return self._month_lookup.get(token.strip().lower())

    def _digits_spelled(self, digits, comma=True):
        if os.environ.get("FR3_TN_LONG_DIGITS", "0") == "1" and len(digits) >= 8:
            if __package__:
                from .identifier_rules import group_digits
            else:
                from identifier_rules import group_digits
            return group_digits(digits, self.lang)
        words = [self.lang.digit(int(d)) for d in digits]
        return ", ".join(words) if comma else " ".join(words)

    def _letters_spelled(self, letters, comma=False):
        words = [self.lang.letter(c) for c in letters]
        return ", ".join(words) if comma else " ".join(words)

    def _spell_alnum(self, token, comma):
        """Spell a code segment: letters by sound, digits one by one."""
        parts = []
        for ch in token:
            if ch.isdigit():
                parts.append(self.lang.digit(int(ch)))
            elif ch.isalpha() and ch.isascii():
                parts.append(self.lang.letter(ch))
            # other chars (hyphens) become separators, dropped
        return (", " if comma else " ").join(parts)

    def _spell_model_segment(self, seg):
        """WH-1000XM5 style segment: letter runs by sound, digit runs as
        cardinal when a clean round block, else digit-by-digit; spaces."""
        out = []
        for run in re.findall(r"[A-Za-z]+|\d+", seg):
            if run.isdigit():
                n = int(run)
                if len(run) == 1 or (n % 100 == 0 and n >= 100):
                    out.append(self.lang.cardinal(n))
                else:
                    out.append(self._digits_spelled(run, comma=False))
            else:
                out.append(self._letters_spelled(run))
        return " ".join(out)

    def _number_words(self, numstr):
        """Plain number string (commas allowed) -> cardinal / decimal words."""
        if numstr.startswith(("-", "+")):
            sign = self.lang.SYM["minus" if numstr[0] == "-" else "plus"]
            return f"{sign} {self._number_words(numstr[1:])}"
        numstr = numstr.replace(",", "")
        if "." in numstr:
            ip, fp = numstr.split(".", 1)
            return self.lang.decimal(int(ip or 0), fp)
        return self.lang.cardinal(int(numstr))

    def _keyword_before(self, text, start, keywords, window=3):
        before = _sentence_before(text, start).replace("\u200c", "")
        toks = re.findall(r"[\w#ऀ-෿']+", before.lower())
        for tok in toks[-window:]:
            tok = tok.strip("#'")
            if tok in keywords:
                return True
        return False

    def _period_for_hour24(self, h):
        if 4 <= h < 12:
            return "morning"
        if 12 <= h < 16:
            return "afternoon"
        if 16 <= h < 20:
            return "evening"
        return "night"

    def _time_words(self, h, m, period_key):
        if self.lang.code != "en" and (h == 0 or h > 12):
            # Keep the written 24-hour hour. The inherited language time methods
            # otherwise fold zero into twelve or fourteen into two.
            template = self.lang.TIME_HM if m else self.lang.TIME_ON_HOUR
            return template.format(h=self.lang.cardinal(h), m=self.lang.cardinal(m))
        return self.lang.time_phrase(h, m, period_key)

    # ------------------------------------------------------------------
    # Pre-clean
    # ------------------------------------------------------------------
    def _preclean(self, text):
        text = _ascii_digits(text)
        text = text.replace("−", "-")
        text = text.translate(_ZW_STRIP)
        # double quotation marks (incl. CSV-style "" and """ runs) carry no
        # sound — a voice model cannot speak them.
        text = re.sub(r'["“”„]+', "", text)
        # single quotes wrapping phrases ('Premium Home Decor') go too.
        # Kept: apostrophes inside words (D'Souza, John's) and quoted single
        # letters ('Y' -> letter name, handled by its own matcher).
        text = re.sub(r"(?<=[\s(।,—–-])['‘’](?![A-Za-z]['‘’])(?=\S)", "", text)
        text = re.sub(r"^['‘’](?![A-Za-z]['‘’])(?=\S)", "", text)
        text = re.sub(r"(?<=\S)(?<!['‘’][A-Za-z])['‘’](?=[\s,।.!?;:)]|$)", "", text)
        text = re.sub(r"#{2,}", " ", text)  # markdown header/### junk
        # ellipses (… / ... / ..) are a spoken pause -> read as a comma
        text = re.sub(r"\s*(?:…|‥|\.{2,})\s*", ", ", text)
        # parentheses are silent; the aside becomes a comma pause
        text = re.sub(r"\s*[()]\s*", ", ", text)
        # //comment// markers are markup junk ((?<!:) keeps https:// intact)
        text = re.sub(r"(?<!:)/{2,}", " ", text)
        # "No." / "no." before digits is the number abbreviation, not "no"
        if self.lang.NUMBER_WORD:
            text = re.sub(r"\b[Nn][Oo]s?\.\s*(?=\d)", self.lang.NUMBER_WORD + " ", text)
        # written-only unit abbreviations -> native words (mins -> నిమిషాలు);
        # guards keep code segments like MIN-22 intact
        for ab, word in self.lang.ABBREV_UNITS.items():
            # consume the abbreviation's own full stop when the sentence
            # continues ("Dr. Rao" -> "doctor Rao"), but keep it when the
            # abbreviation ends the sentence ("... approx." -> "approximately.")
            text = re.sub(
                rf"(?<![\w-]){ab}(?:\.(?=\s+\S))?(?![\w-])",
                word,
                text,
                flags=re.IGNORECASE,
            )
        text = re.sub(r"^[,\s]+", "", text)
        # Comparison operators are meaningful even without a left numeric operand.
        text = _EMOJI_RE.sub(" ", text)
        # en/em dash between numbers -> canonical range dash; others deleted
        text = re.sub(r"(?<=[\d])\s*[–—]\s*(?=[\d])", "–", text)
        # time-range dash: "11 AM – 3 PM" (dash not directly digit-adjacent)
        text = re.sub(r"\s+[–—]\s+(?=\d)", " – ", text)
        text = re.sub(r"[—]", " ", text)
        text = re.sub(r"(?<![\d\s])[–](?![\d\s])", " ", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text

    # ------------------------------------------------------------------
    # Matchers — each returns list of (start, end, replacement)
    # Priority = order in self._matchers (earlier wins).
    # ------------------------------------------------------------------
    def _gather(self, text):
        matches = []  # (priority, start, end, replacement)
        matchers = [
            self._m_email,
            self._m_url,
            self._m_handle,
            self._m_comparison,
            self._m_time_range,
            self._m_duration,
            self._m_time_verbalized,
            self._m_time,
            self._m_tamil_date,
            self._m_date,
            self._m_currency_range,
            self._m_currency,
            self._m_bare_rupee_suffix,
            self._m_long_digit_runs,
            self._m_amount_context,
            self._m_percent_range,
            self._m_percent,
            self._m_negative,
            self._m_designation,
            self._m_vehicle_plate,
            self._m_ordinal,
            self._m_code,
            self._m_id_digits,
            self._m_chem,
            self._m_math,
            self._m_num_range,
            self._m_dimensions,
            self._m_roman,
            self._m_unit_initialism,
            self._m_kmb_suffix,
            self._m_fraction,
            self._m_ratio,
            self._m_label,
            self._m_acronym_model,
            self._m_num_hyphen_word,
            self._m_decade,
            self._m_year,
            self._m_quoted_letter,
            self._m_short_mixed_token,
            self._m_decimal,
            self._m_int,
        ]
        for prio, fn in enumerate(matchers):
            for s, e, rep in fn(text):
                matches.append((prio, s, e, rep))
        return matches

    # --- email / url / handle -----------------------------------------
    def _spell_uri_part(self, part):
        """Word chunks stay; digits spelled; symbols to words."""
        if os.environ.get("FR3_TN_URL_SPELL", "0") == "1":
            if __package__:
                from .identifier_rules import spell_uri
            else:
                from identifier_rules import spell_uri
            return spell_uri(part, self.lang, self._digits_spelled)
        S = self.lang.SYM
        out = []
        for run in re.findall(r"[^\W\d_]+|\d+|[^\w\s]|_", part):
            if run.isdigit():
                out.append(self._digits_spelled(run, comma=False))
            elif run == ".":
                out.append(S["dot"])
            elif run == "_":
                out.append(S["underscore"])
            elif run == "@":
                out.append(S["at"])
            elif run == "/":
                out.append(S["slash"])
            elif run == "-":
                out.append("dash")
            elif run == "+":
                out.append(S["plus"])
            elif run in {"?", "=", "&", "#", "%", ":", "~"}:
                out.append(
                    {
                        "?": "question mark",
                        "=": "equals",
                        "&": "ampersand",
                        "#": "hash",
                        "%": "percent",
                        ":": "colon",
                        "~": "tilde",
                    }[run]
                )
            elif run.lower() == "www":
                out.append(self._letters_spelled("WWW"))
            elif len(run) == 1 or run.upper() in self.lang.ABBREV_UNITS:
                # Do not expand /Dr/ to /doctor/ on a second normalization pass.
                out.append(self._letters_spelled(run.upper()))
            else:
                out.append(run)
        return " ".join(w for w in out if w)

    def _m_email(self, text):
        for m in re.finditer(
            r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b", text
        ):
            yield m.start(), m.end(), self._spell_uri_part(m.group(0))

    def _m_url(self, text):
        pat = (
            r"https?://[^\s<>]+|www\.[^\s<>]+"
            # bare domains: maya.ai, irctc.co.in/booking
            r"|\b[A-Za-z][\w\-]*\.(?:com|org|net|ai|io|co|dev|app|edu|gov|in)"
            r"(?:\.(?:in|uk|us))?(?:[/?#][^\s<>]*)?(?![\w.])"
        )
        for m in re.finditer(pat, text):
            token = m.group(0).rstrip(".,!?;)")
            body = re.sub(r"^https?://", "", token)
            yield m.start(), m.start() + len(token), self._spell_uri_part(body)

    def _m_handle(self, text):
        # trailing \w so a sentence-final period after the handle is not
        # swallowed and spoken as "dot" ("@test_user." -> handle + full stop)
        pat = (
            r"(?<![\w.])@([A-Za-z](?:[\w.\-]*\w)?)"
            if os.environ.get("FR3_TN_URL_SPELL", "0") == "1"
            else r"(?<![\w.])@([A-Za-z](?:[\w.]*\w)?)"
        )
        for m in re.finditer(pat, text):
            yield m.start(), m.end(), f"{self.lang.SYM['at']} {self._spell_uri_part(m.group(1))}"

    # --- time -----------------------------------------------------------
    def _parse_time_match(self, m, text):
        """Shared logic. Returns (h12, mm, period_key)."""
        pass

    def _time_pattern(self):
        P = self._period_re
        MK = self._marker_re
        # (?<!\w) / (?!\w) around native-word alternations: "ગુજરાત" ends
        # with રાત (night) — without boundaries the period word matches
        # inside other words
        return re.compile(
            rf"(?:(?<!\w)({P})\s+)?"  # leading period word
            rf"(\d{{1,2}})"  # hour
            # a real time needs :mm, AM/PM or a time marker right after the
            # hour — otherwise bare numbers spuriously swallow period words
            rf"(?=:\d{{2}}"
            rf"|\.\d{{2}}\s*(?:AM|PM|am|pm|a\.m\.|p\.m\.)(?![A-Za-z])"  # 6.45 pm — dot form
            rf"|\s*(?:AM|PM|am|pm|a\.m\.|p\.m\.)(?![A-Za-z])|\s*(?:{MK})(?!\w))"
            rf"(?:[.:](\d{{2}}))?"  # :mm  or  .mm (gated by the lookahead)
            rf"(?:\s*(AM|PM|am|pm|a\.m\.|p\.m\.))?"  # am/pm
            rf"(?:\s*({MK})(?!\w))?"  # बजे / मणि ...
            rf"(?:\s+({P})(?!\w))?",
            re.I,  # trailing period word
        )

    def _resolve_time(self, lead_p, h, mm, ampm, marker, trail_p):
        L = self.lang
        h = int(h)
        m = int(mm) if mm is not None else 0
        period = None
        src_period = lead_p or trail_p
        if src_period:
            period = L.PERIOD_DETECT.get(src_period)
        if L.code != "en" and not ampm and not src_period:
            return h, m, None
        infer = not L.AMPM_WORDS  # languages that speak am/pm keep the token instead
        if ampm:
            a = ampm.lower().replace(".", "")
            h24 = h % 12 + (12 if a == "pm" else 0)
            if infer:
                period = period or self._period_for_hour24(h24)
        elif h > 12 or h == 0:
            # 24-hour clock
            if infer:
                period = period or self._period_for_hour24(h)
        if L.AMPM_WORDS and not ampm and (h > 12 or h == 0):
            return h, m, period  # "14:30" -> "fourteen thirty", not "two thirty"
        h12 = h % 12
        if h12 == 0:
            h12 = 12
        return h12, m, period

    def _m_duration(self, text):
        # H:MM:SS is a duration, not a 12-hour clock. Zero hours must not become 12.
        L = self.lang
        sec_word = L.SYM.get("second", "seconds")
        for m in re.finditer(r"\b(\d{1,3}):(\d{2}):(\d{2})\b", text):
            h, mm, ss = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if mm > 59 or ss > 59:
                continue
            hours = L.ABBREV_UNITS.get("HRS", "hours")
            minutes = L.SYM["minute"]
            parts = [
                L.join_amount(
                    L.cardinal(n),
                    unit if n != 1 or L.code != "en" else unit.rstrip("s"),
                )
                for n, unit in ((h, hours), (mm, minutes), (ss, sec_word))
                if n
            ]
            yield m.start(), m.end(), " ".join(parts) or L.join_amount(
                L.cardinal(0), sec_word
            )

    def _m_time_verbalized(self, text):
        # "9 बजकर 15 मिनट" — already half-verbalized with digits: rebuild the
        # canonical phrase instead of reading the pieces separately
        L = self.lang
        MIN = self._minute_re
        pat = re.compile(
            rf"\b(\d{{1,2}})\s*(?:{self._marker_re})\s+(\d{{1,2}})\s*({MIN})(?!\w)"
        )
        for m in pat.finditer(text):
            h, mm = int(m.group(1)), int(m.group(2))
            if h > 23 or mm > 59:
                continue
            rep = self._time_words(h, mm, None)
            sfx = L.MARKER_SUFFIXES.get(m.group(3) or "")
            if sfx:
                rep = L.attach_postposition(rep, sfx) or rep
            yield m.start(), m.end(), rep

    def _m_time(self, text):
        pat = self._time_pattern()
        for m in pat.finditer(text):
            lead_p, h, mm, ampm, marker, trail_p = m.groups()
            if h is None:
                continue
            # must look like a time: needs marker, am/pm, period word, or a
            # valid HH:MM
            hh = int(h)
            is_hhmm = mm is not None and int(mm) <= 59 and hh < 24
            if not (
                ampm or marker or ((lead_p or trail_p) and (is_hhmm or mm is None))
            ):
                if not (is_hhmm and mm is not None and hh > 12):
                    # bare h:mm with hh<=12: still a time if mm is 2-digit <=59
                    if not (
                        is_hhmm
                        and not self._keyword_before(
                            text, m.start(), self._ratio_kws, window=4
                        )
                    ):
                        continue
            if mm is None and not (ampm or marker):
                continue  # bare number is not a time
            if mm is not None and int(mm) > 59:
                continue
            if hh > 23:
                continue
            h12, mnt, period = self._resolve_time(lead_p, h, mm, ampm, marker, trail_p)
            src_period = lead_p or trail_p
            if src_period:
                rep = f"{src_period} {self.lang.time_phrase(h12, mnt, None)}"
            else:
                rep = self._time_words(h12, mnt, period)
            if ampm and self.lang.AMPM_WORDS:
                w = self.lang.AMPM_WORDS.get(ampm.lower().replace(".", ""))
                if w:
                    # a bare hour with am/pm reads "eight p m", never
                    # "eight o'clock p m" -- o'clock was not in the input
                    if mnt == 0:
                        rep = self.lang.two_digit(h12)
                    rep = f"{rep} {w}"
            sfx = self.lang.MARKER_SUFFIXES.get(marker or "")
            if sfx:
                rep = self.lang.attach_postposition(rep, sfx) or rep
            yield m.start(), m.end(), rep

    def _m_time_range(self, text):
        # 11 AM – 3 PM  /  9:00 – 17:00  / 11 AM to 3 PM stays "to"
        pat = self._time_pattern()
        rng = re.compile(rf"({pat.pattern})\s*[–\-]\s*({pat.pattern})", pat.flags)
        for m in rng.finditer(text):
            g = m.groups()
            left, right = g[0], g[7]
            gl, gr = g[1:7], g[8:14]
            if gl[1] is None or gr[1] is None:
                continue
            # each side must independently look like a time
            if not ((gl[3] or gl[4] or gl[2]) and (gr[3] or gr[4] or gr[2])):
                continue
            h1, m1, p1 = self._resolve_time(*gl)
            h2, m2, p2 = self._resolve_time(*gr)
            a = self._time_words(h1, m1, p1)
            b = self._time_words(h2, m2, p2)
            if self.lang.AMPM_WORDS:
                if gl[3]:
                    a = (
                        (self.lang.two_digit(h1) if m1 == 0 else a)
                        + " "
                        + self.lang.AMPM_WORDS[gl[3].lower().replace(".", "")]
                    )
                if gr[3]:
                    b = (
                        (self.lang.two_digit(h2) if m2 == 0 else b)
                        + " "
                        + self.lang.AMPM_WORDS[gr[3].lower().replace(".", "")]
                    )
            rep = _fmt(self.lang.RANGE_TO, a=a, b=b)
            tail = self.lang.RANGE_TO.split("{b}")[-1].strip()
            if tail and re.match(rf"\s*{re.escape(tail)}(?!\w)", text[m.end() :]):
                rep = rep[: -len(tail)].rstrip()  # source already says it
            yield m.start(), m.end(), rep

    # --- dates -----------------------------------------------------------
    def _m_tamil_date(self, text):
        if self.lang.code not in ("en", "ta"):
            return
        months = "|".join(_TAMIL_MONTHS)
        # Only calendar-day suffixes. Do not pretend all Tamil number morphology
        # can be repaired by concatenating an English cardinal with a suffix.
        pat = rf"(?<!\w)({months})\s+(\d{{1,2}})(?:-?(?:ஆம்|ம்))?(?![\dA-Za-z])(?:\s*தேதி)?"
        for m in re.finditer(pat, text):
            d = int(m[2])
            if 1 <= d <= 31:
                yield m.start(), m.end(), f"{m[1]} {_TAMIL_ORDINALS[d]} தேதி"
        for m in re.finditer(r"(?<!\w)(\d+)-?(?:ஆம்|ம்)(?![\w\u0B80-\u0BFF])", text):
            d = int(m[1])
            if not 1 <= d <= 31:
                if self.lang.code == "en":
                    raise NormalizationError(
                        "unsupported_tamil_ordinal",
                        "Select Tamil normalization for ordinals outside calendar days 1-31.",
                        span=m.span(),
                        source=text,
                    )
                yield m.start(), m.end(), self.lang.ordinal(d, "ஆம்")
            else:
                yield m.start(), m.end(), _TAMIL_ORDINALS[d]

    def _date_words(self, day, month, year=None, ordinal=False, month_first=False):
        L = self.lang
        if L.code == "ta":
            out = f"{L.month(month)} {_TAMIL_ORDINALS[day]} தேதி"
        else:
            suffix = (
                ordinal
                if isinstance(ordinal, str) and ordinal in L.ORDINAL_INPUT_SUFFIXES
                else ""
            )
            day_words = L.ordinal(day, suffix) if ordinal else L.cardinal(day)
            out = (
                f"{L.month(month)} {day_words}"
                if month_first and L.code != "en"
                else f"{day_words} {L.month(month)}"
            )
        return f"{out} {L.year(year)}" if year is not None else out

    def _m_date(self, text):
        L = self.lang
        for m in re.finditer(
            rf"\b(\d{{1,2}})[-–](\d{{1,2}})\s+({self._month_re})(?!\w)(?:\s+(\d{{4}}))?",
            text,
            re.I,
        ):
            mo = self._month_index(m[3])
            for day in (m[1], m[2]):
                _calendar_check(int(day), mo, int(m[4]) if m[4] else 2000)
            days = _fmt(L.RANGE, a=L.cardinal(int(m[1])), b=L.cardinal(int(m[2])))
            out = f"{days} {L.month(mo)}"
            if m[4]:
                out += f" {L.year(int(m[4]))}"
            yield m.start(), m.end(), out
        for m in re.finditer(r"(?<![\w-])(\d{4})-(\d{1,2})-(\d{1,2})(?![\d-])", text):
            yield m.start(), m.end(), self._date_words(int(m[3]), int(m[2]), int(m[1]))
        # DD/MM/YYYY or DD-MM-YYYY or DD.MM.YYYY
        for m in re.finditer(
            r"\b(\d{1,2})([/.\-])(\d{1,2})\2(\d{2,4})\b(?![/A-Za-z-])", text
        ):
            d, mo, y = int(m.group(1)), int(m.group(3)), int(m.group(4))
            if self.date_order == "MDY":
                d, mo = mo, d
            if not (1 <= d <= 31 and 1 <= mo <= 12):
                continue
            if len(m.group(4)) == 2:
                y += self.two_digit_year_base
            yield m.start(), m.end(), self._date_words(d, mo, y)
        # DD-Mon-YYYY / DD Mon YYYY / DD Mon, with native or English months
        # (?!\w) after the month: without it "25 ಮೇಲೆ" would parse as
        # "25 May" because ಮೇಲೆ starts with ಮೇ (May)
        pat = re.compile(
            rf"\b(\d{{1,2}})(?:-?({self._ord_suffix_re}))?[\-\s]({self._month_re})(?!\w)"
            rf"(?:[\-\s,]+(\d{{4}}))?",
            re.IGNORECASE,
        )
        for m in pat.finditer(text):
            d = int(m.group(1))
            mo = self._month_index(m.group(3))
            if mo is None or not (1 <= d <= 31):
                continue
            out = self._date_words(d, mo, int(m[4]) if m[4] else None, m[2])
            yield m.start(), m.end(), out
        # Month DD, YYYY (English order)
        pat2 = re.compile(
            rf"\b({self._month_re})\s+(\d{{1,2}})(?:-?({self._ord_suffix_re}))?(?:,?\s+(\d{{4}}))?(?!\d)",
            re.IGNORECASE,
        )
        for m in pat2.finditer(text):
            mo = self._month_index(m.group(1))
            d = int(m.group(2))
            if mo is None or not (1 <= d <= 31):
                continue
            out = self._date_words(
                d, mo, int(m[4]) if m[4] else None, m[3], month_first=True
            )
            yield m.start(), m.end(), out

    # --- currency ---------------------------------------------------------
    _CURRENCY_CODE_KEYS = {
        "₹": "rupee",
        "$": "dollar",
        "€": "euro",
        "£": "pound",
        "RS": "rupee",
        "INR": "rupee",
        "USD": "dollar",
        "EUR": "euro",
        "GBP": "pound",
    }

    def _cur_key(self, symtok):
        t = symtok.strip().rstrip(".").strip().upper()
        return (
            self._CURRENCY_CODE_KEYS.get(t)
            or self.lang.CURRENCY_WORDS.get(t)
            or self.lang.CURRENCY_WORDS.get(t + ".")
            or "rupee"
        )

    def _m_currency_range(self, text):
        # ₹500-700 -> "five hundred to seven hundred rupees"
        L = self.lang
        pat = re.compile(
            rf"(?<!\w)([+-]?)({self._currency_re})"
            rf"\s*([+-]?\d+(?:,\d{{2,3}})*(?:\.\d+)?)\s*[–\-]\s*([+-]?\d+(?:,\d{{2,3}})*(?:\.\d+)?)(?!\d|[.,]\d)",
            re.I,
        )
        for m in pat.finditer(text):
            if m[1] and m[3].startswith(("-", "+")):
                raise NormalizationError(
                    "conflicting_sign",
                    "Currency amount has two signs; confirm the value.",
                    span=m.span(),
                    source=text,
                )
            a, b = self._number_words(m[1] + m[3]), self._number_words(m[4])
            cw = L.SYM[self._cur_key(m[2])]
            rng = _fmt(L.RANGE, a=a, b=L.join_amount(b, cw))
            yield m.start(), m.end(), rng

    def _m_bare_rupee_suffix(self, text):
        L = self.lang
        for m in re.finditer(r"(?<![\w₹.])(\d+(?:,\d{2,3})*(?:\.\d+)?)\s?/-", text):
            num = m.group(1).replace(",", "")
            words = self._number_words(num)
            yield m.start(), m.end(), L.join_amount(words, L.SYM["rupee"])

    def _m_currency(self, text):
        L = self.lang
        S = L.SYM
        pat = re.compile(
            rf"(?<!\d)([+-]?)\s*(₹|\$|€|£|(?<!\w)(?:{self._currency_re}))"
            rf"\s*([+-]?)(\d+(?:,\d{{2,3}})*(?:\.\d+)?)"
            rf"(?:\s*({self._scale_re})(?:((?:ां|ं)?(?:चा|ची|चे|च्या|त|ना|हून|पर्यंत|मध्ये))|(?![ऀ-ःऺ-ॏঀ-ঃা-্ਁ-ਃਾ-੍ઁ-ઃા-્ଁ-ଃା-୍ஂா-்ఀ-ఄా-ౌಁ-ಃಾ-ೌഀ-ഃാ-ൌ]))(?![A-Za-z0-9]))?(?:\s?/-)?",
            re.I,
        )
        for m in pat.finditer(text):
            if m[1] and m[3]:
                raise NormalizationError(
                    "conflicting_sign",
                    "Currency amount has two signs; confirm the value.",
                    span=m.span(),
                    source=text,
                )
            key = self._cur_key(m[2])
            num = m[4].replace(",", "")
            scale = m[5]
            cur_word = S[key]
            # money with exactly 2 decimals and no scale word -> main + sub
            # unit ($84.20 -> 84 dollar 20 cent; ₹95.41 -> 95 rupees 41 paise)
            if "." in num and not scale and len(num.split(".", 1)[1]) == 2:
                ip, fp = num.split(".", 1)
                main = L.cardinal(int(ip or 0))
                if key == "rupee" and L.SYM.get("paisa"):
                    sub_word = L.SYM["paisa"]
                else:
                    sub_word = S["cent"] if key != "rupee" else None
                if fp.strip("0") == "":
                    out = L.join_amount(main, cur_word)
                elif sub_word:
                    out = (
                        f"{L.join_amount(main, cur_word)} "
                        f"{L.join_amount(L.cardinal(int(fp)), sub_word)}"
                    )
                else:
                    out = L.join_amount(L.decimal(int(ip or 0), fp), cur_word)
            else:
                words = self._number_words(num)
                if scale:
                    sl = scale.lower()
                    canon = L.SCALE_WORDS.get(scale)
                    if canon is None and sl in _EN_SCALES:
                        scale_word = scale  # 1.5 million -> ... million dollar
                    else:
                        if canon is None:
                            canon = (
                                "crore"
                                if sl in ("cr",) or "crore" in sl
                                else "lakh" if "lakh" in sl else "thousand"
                            )
                        scale_word = L.scale_noun(canon, num)
                    out = L.join_amount(L.join_amount(words, scale_word), cur_word)
                    case = m[6]
                    if case:
                        out = L.attach_postposition(out, case) or out
                else:
                    out = L.join_amount(words, cur_word)
            sign = m[1] or m[3]
            if sign:
                out = f"{S['minus'] if sign == '-' else S['plus']} {out}"
            # Keep whitespace outside the matched amount.
            space = m[0][:1].isspace() or (
                m.start() > 0 and text[m.start() - 1].isalnum()
            )
            yield m.start(), m.end(), (" " if space else "") + out

    def _m_amount_context(self, text):
        # Explicit currency suffix/amount label wins over broad identifier hints.
        units = (
            r"rupees?|rs\.?|inr|dollars?|usd|euros?|pounds?|paise|paisa|cents?|"
            r"ரூபாய்|பைசா|रुपये|रुपया|रुपए|రూ\.?|రూపాయలు|രൂപ"
        )
        units += "|" + "|".join(map(re.escape, self.lang.CURRENCY_WORDS))
        for m in re.finditer(
            rf"(?<![\w.])([+-]?\d[\d,]*(?:\.\d+)?)(?=\s*(?:{units})(?![A-Za-z]))",
            text,
            re.I,
        ):
            yield m.start(), m.end(), self._number_words(m[1])
        labels = {
            "balance",
            "amount",
            "payable",
            "outstanding",
            "payment",
            "price",
            "fee",
            "pay",
            "emi",
            "தவணை",
        }
        for m in re.finditer(r"(?<![\w.,+/-])\d+(?:,\d{2,3})*(?!\w|[.,/-]\d)", text):
            before = _sentence_before(text, m.start()).lower()
            toks = re.findall(r"\w+", before)[-3:]
            if any(t in labels for t in toks) and not any(
                t in {"number", "id", "code", "otp", "pin"} for t in toks
            ):
                yield m.start(), m.end(), self._number_words(m[0])

    # --- percent ------------------------------------------------------------
    def _m_percent_range(self, text):
        # 40-70% / 40–70% -> "40 to 70 percent"
        L = self.lang
        for m in re.finditer(r"\b(\d+(?:\.\d+)?)\s*[–\-]\s*(\d+(?:\.\d+)?)\s*%", text):
            a = self._number_words(m.group(1))
            b = self._number_words(m.group(2))
            yield m.start(), m.end(), f"{_fmt(L.RANGE, a=a, b=b)} {L.SYM['percent']}"

    def _m_percent(self, text):
        L = self.lang
        for m in re.finditer(r"(-?)(\d+(?:,\d{2,3})*(?:\.\d+)?)\s*%", text):
            words = self._number_words(m.group(2))
            if m.group(1):
                words = f"{L.SYM['minus']} {words}"
            yield m.start(), m.end(), f"{words} {L.SYM['percent']}"

    # --- designations (Sony WH-1000XM5, Kepler-186f, Soyuz T-11) -----------
    def _m_designation(self, text):
        # TitleCase brand fused: Kepler-186f, Salyut-7
        for m in re.finditer(r"\b([A-Z][a-z]{2,})-([A-Za-z0-9]+)\b", text):
            seg = m.group(2)
            if not re.search(r"\d", seg):
                continue
            if self._month_index(seg) or self._month_index(m.group(1)):
                continue
            yield m.start(), m.end(), f"{m.group(1)} {self._spell_model_segment(seg)}"
        # Brand word + short designation: Sony WH-1000XM5, Soyuz T-11
        pat = re.compile(r"\b([A-Z][a-z]{2,})\s+([A-Z]{1,4}-[A-Za-z0-9]+)\b")
        for m in pat.finditer(text):
            brand = m.group(1)
            if brand.lower() in self._id_keywords or brand.lower() in self._label_words:
                continue
            seg = m.group(2)
            if not re.search(r"\d", seg):
                continue
            yield (m.start(2), m.end(2), self._spell_model_segment(seg))

    # --- alphanumeric codes -------------------------------------------------
    def _m_code(self, text):
        # hyphenated tokens containing at least one letter and one digit
        pat = re.compile(
            r"#?\b(?=[A-Za-z0-9/-]*\d)(?=[A-Za-z0-9/-]*[A-Za-z])"
            r"[A-Za-z0-9]+(?:[-/][A-Za-z0-9]+)+(?![A-Za-z0-9])"
        )
        for m in pat.finditer(text):
            tok = m.group(0).lstrip("#")
            # date like 12-Aug-2025 -> handled by date matcher (higher prio)
            parts = re.split(r"[-/]", tok)
            if any(self._month_index(p) for p in parts) and all(
                p.isdigit() for p in parts if not self._month_index(p)
            ):
                continue  # date-like 12-Aug-2025; MAY-17Y7 is still a code
            # a lowercase word segment means this is prose, not a code
            # (900-point, 80-watt) -> handled by num-hyphen-word matcher
            if any(re.fullmatch(r"[a-z][a-z]+|[A-Z][a-z]{2,}", p) for p in parts):
                continue  # prose word segment (900-point, 7-East)
            if (
                len(parts) == 2
                and parts[0].isdigit()
                and parts[1].isalpha()
                and len(parts[1]) >= 3
            ):
                continue  # 30-DAY, 24-HOUR -> number + word
            # 15-GB -> unit matcher
            if (
                len(parts) == 2
                and parts[0].isdigit()
                and parts[1].upper() in _UNIT_INITIALISMS
            ):
                continue
            # Hall 3-A / Window 5-B -> label matcher (but CR-2A stays a code)
            if re.fullmatch(r"\d{1,3}-?[A-Za-z]", tok) and self._keyword_before(
                text, m.start(), self._label_words, window=2
            ):
                continue
            if sum(c.isalnum() for c in tok) <= 2:
                continue
            yield m.start(), m.end(), self._spell_alnum(tok, comma=True)
        # solid mixed tokens (length>=5, >=2 digits): MH12AB8821, 77K2L9
        pat2 = re.compile(
            r"\b(?=(?:[A-Za-z]*\d){2})(?=[A-Za-z0-9]*[A-Za-z])" r"[A-Za-z0-9]{5,}\b"
        )
        for m in pat2.finditer(text):
            tok = m.group(0)
            if re.fullmatch(r"\d+s", tok):
                continue  # 1980s -> decade matcher
            if re.fullmatch(r"\d+[x×]\d+", tok):
                continue  # 1920x1080 -> dimensions matcher
            m2 = re.fullmatch(r"\d+([A-Za-z]{1,4})", tok)
            if m2 and m2.group(1).upper() in _UNIT_INITIALISMS:
                continue  # 250ml -> unit matcher
            if re.fullmatch(r"(?:[A-Z][a-z]?\d*)+", tok) and self._chem_ok(tok):
                continue  # chemistry wins
            yield m.start(), m.end(), self._spell_alnum(tok, comma=True)

    def _m_vehicle_plate(self, text):
        # spaced Indian plates: AP 09 CL 4421 / TS 09 EA 1234
        L = self.lang
        pat = re.compile(r"\b([A-Z]{2})[ -](\d{1,2})[ -]([A-Z]{1,3})[ -](\d{3,4})\b")
        for m in pat.finditer(text):
            out = []
            for grp in m.groups():
                for ch in grp:
                    out.append(L.digit(int(ch)) if ch.isdigit() else L.letter(ch))
            yield m.start(), m.end(), ", ".join(out)

    # --- identifier digit runs ----------------------------------------------
    def _m_long_digit_runs(self, text):
        if os.environ.get("FR3_TN_LONG_DIGITS", "0") != "1":
            return
        if __package__:
            from .identifier_rules import long_runs
        else:
            from identifier_rules import long_runs
        reserved = [
            (s, e)
            for matcher in (self._m_percent_range, self._m_percent, self._m_negative)
            for s, e, _ in matcher(text)
        ]
        for s, e, spoken in long_runs(text, self.lang):
            if not any(s < end and e > start for start, end in reserved):
                yield s, e, spoken

    def _m_id_digits(self, text):
        L = self.lang
        # hyphen/space separated digit groups, or long runs, or # prefixed
        pat = re.compile(
            r"(\+?)(?:#?\b\d+(?:[-\s]\d+)+\b|#\s?\d{3,}\b"
            r"|(?<![\d.])(?<!\d,)\d{7,}(?![.,]?\d))"
        )
        for m in pat.finditer(text):
            has_plus = m.group(1) == "+"
            tok = m.group(0).lstrip("+# ")
            groups = re.split(r"[-\s]", tok)
            digits = "".join(groups)
            is_id = False
            if has_plus and len(digits) >= 8:
                is_id = True  # +91 ... country-code phone
            elif m.group(0).startswith("#"):
                is_id = True
            elif len(digits) >= 7 and " " not in tok:
                is_id = True
            elif any(len(g) >= 4 for g in groups) and len(groups) >= 2 and "-" in tok:
                is_id = True
            elif len(groups) >= 3 and "-" in tok:
                is_id = True
            elif self._keyword_before(text, m.start(), self._id_keywords):
                is_id = True
            else:
                a_toks = re.findall(
                    r"[\w#ऀ-෿']+", _sentence_after(text, m.end()).lower()
                )[:4]
                if any(t in self._id_keywords for t in a_toks):
                    is_id = True
            if not is_id:
                continue
            # ranges like 10-20 without keyword: ascending small pairs -> skip
            a_toks2 = re.findall(r"[\w#ऀ-෿']+", _sentence_after(text, m.end()).lower())[
                :4
            ]
            if (
                "-" in tok
                and len(groups) == 2
                and all(len(g) <= 3 for g in groups)
                and int(groups[0]) < int(groups[1])
                and not self._keyword_before(text, m.start(), self._id_keywords)
                and not any(t in self._id_keywords for t in a_toks2)
            ):
                continue
            rep = self._digits_spelled(digits, comma=True)
            if has_plus:
                rep = f"{L.SYM['plus']} {rep}"
            yield m.start(), m.end(), rep
        # keyword-driven short runs: OTP 4471, code 22377, train 207,
        # "ending in 7104"
        for m in re.finditer(r"(?<![\d.])(?<!\d,)\d{3,6}(?![.,]?\d)(?!s\b)", text):
            # "139 dial करें" — dial/डायल shortly after marks a hotline number
            after_toks = re.findall(
                r"[\w#ऀ-෿']+", _sentence_after(text, m.end()).lower()
            )[:6]
            id_after = {"dial", "डायल"} | set(self.lang.ID_AFTER_WORDS)
            if any(t in id_after for t in after_toks):
                yield m.start(), m.end(), self._digits_spelled(m.group(0), comma=True)
                continue
            # an identifier keyword within ~4 tokens AFTER also marks it
            # ("4321 हा OTP", "998877 वापरा password") — but not label/year
            if (
                any(t in self._id_keywords for t in after_toks[:4])
                and not self._keyword_before(
                    text, m.start(), self._label_words, window=2
                )
                and not self._keyword_before(
                    text,
                    m.start(),
                    {w.lower() for w in L.YEAR_WORDS} | {"year"},
                    window=2,
                )
            ):
                yield m.start(), m.end(), self._digits_spelled(m.group(0), comma=True)
                continue
            if self._keyword_before(text, m.start(), self._id_keywords):
                # label words override (Room 314 stays cardinal)
                if self._keyword_before(text, m.start(), self._label_words, window=2):
                    continue
                # year words override (साल 2026)
                if self._keyword_before(
                    text,
                    m.start(),
                    {w.lower() for w in L.YEAR_WORDS} | {"year"},
                    window=2,
                ):
                    continue
                yield m.start(), m.end(), self._digits_spelled(m.group(0), comma=True)

    # --- chemistry ------------------------------------------------------------
    _ELEMENTS = {
        "H",
        "He",
        "Li",
        "Be",
        "B",
        "C",
        "N",
        "O",
        "F",
        "Ne",
        "Na",
        "Mg",
        "Al",
        "Si",
        "P",
        "S",
        "Cl",
        "Ar",
        "K",
        "Ca",
        "Fe",
        "Cu",
        "Zn",
        "Ag",
        "Au",
        "Hg",
        "Pb",
        "I",
        "Mn",
        "Cr",
        "Ni",
        "Sn",
        "Br",
        "Mg",
    }
    _CHEM_WHITELIST = {
        "HCl",
        "NaCl",
        "KCl",
        "NaOH",
        "KOH",
        "HBr",
        "HF",
        "CO",
        "NO",
        "NH3",
        "CH4",
        "H2O",
        "CO2",
        "SO2",
        "O2",
        "N2",
        "H2",
        "H2SO4",
        "HNO3",
        "CaCO3",
        "C6H12O6",
        "NaHCO3",
        "MgSO4",
        "ZnO",
        "FeO",
    }

    def _chem_ok(self, tok):
        if tok in self._CHEM_WHITELIST:
            return True
        pieces = re.findall(r"[A-Z][a-z]?|\d+", tok)
        letters = [p for p in pieces if not p.isdigit()]
        digits = [p for p in pieces if p.isdigit()]
        if not letters or any(p not in self._ELEMENTS for p in letters):
            return False
        return (
            bool(digits)
            and len(letters) >= 2
            or (bool(digits) and tok in self._CHEM_WHITELIST)
        )

    def _m_chem(self, text):
        L = self.lang
        for m in re.finditer(r"\b(?:[A-Z][a-z]?\d*){1,8}\b", text):
            tok = m.group(0)
            if len(tok) < 2 or tok.upper() in _ACRONYM_WHITELIST:
                continue
            if not self._chem_ok(tok):
                continue
            out = []
            for piece in re.findall(r"[A-Z][a-z]?|\d+", tok):
                if piece.isdigit():
                    out.append(L.cardinal(int(piece)))
                else:
                    out.append(self._letters_spelled(piece.upper()))
            yield m.start(), m.end(), " ".join(out)

    # --- math -------------------------------------------------------------
    def _m_comparison(self, text):
        # Indic comparative phrases are postpositions: '<500' -> '500 से कम',
        # not 'से कम 500'. Treat the numeric bound as one semantic span.
        L = self.lang
        number = r"[+-]?\d+(?:,\d{2,3})*(?:\.\d+)?"
        operand = rf"(?:₹\s*)?{number}(?:\s*%)?"
        pattern = rf"(?<![\w.])(?:(?P<left>{operand})\s*)?(?P<op><=|>=|≤|≥|<|>)\s*(?P<right>{operand})"

        def words(token):
            s = token.strip()
            if s.startswith("₹"):
                return L.join_amount(self._number_words(s[1:].strip()), L.SYM["rupee"])
            if s.endswith("%"):
                return f"{self._number_words(s[:-1].strip())} {L.SYM['percent']}"
            return self._number_words(s)

        ors = {
            "hi": "या",
            "te": "లేదా",
            "ta": "அல்லது",
            "bn": "বা",
            "gu": "અથવા",
            "kn": "ಅಥವಾ",
            "ml": "അല്ലെങ്കിൽ",
            "mr": "किंवा",
            "or": "କିମ୍ବା",
            "pa": "ਜਾਂ",
        }
        for m in re.finditer(pattern, text):
            op = m["op"]
            comparison = L.SYM["lt" if op.startswith("<") or op == "≤" else "gt"]
            if len(op) > 1 or op in ("≤", "≥"):
                comparison += (
                    " or equal to"
                    if L.code == "en"
                    else f" {ors[L.code]} {L.SYM['equals']}"
                )
            left = words(m["left"]) + " " if m["left"] else ""
            right = words(m["right"])
            yield m.start(), m.end(), left + (
                f"{comparison} {right}" if L.code == "en" else f"{right} {comparison}"
            )

    def _m_math(self, text):
        L = self.lang
        S = L.SYM
        # power: 10^23  /  x^2
        for m in re.finditer(r"(\d+(?:\.\d+)?|[A-Za-z])\s*\^\s*(\d+|[A-Za-z])", text):
            base, exp = m.group(1), m.group(2)
            bw = self._number_words(base) if base[0].isdigit() else L.letter(base)
            ew = self._number_words(exp) if exp[0].isdigit() else L.letter(exp)
            yield m.start(), m.end(), _fmt(L.POWER, base=bw, exp=ew)
        # squared ²
        for m in re.finditer(r"(\d+(?:\.\d+)?|[A-Za-z])²", text):
            v = m.group(1)
            vw = self._number_words(v) if v[0].isdigit() else L.letter(v)
            yield m.start(), m.end(), f"{vw} {S['squared']}"
        # √
        for m in re.finditer(r"√\s*(\d+(?:\.\d+)?)", text):
            yield m.start(), m.end(), f"{S['sqrt']} {self._number_words(m.group(1))}"
        # operator chains: x + 5 = 12, E = mc^2, 17 × 3 = 51 — rendered only
        # when a strong math operator (= × ÷ ^ √) is present, so hyphen
        # ranges/ids and prose like "e-mail" never match
        eq = re.compile(
            r"(?<![\w.])((?:[A-Za-z]{1,2}|\d+(?:\.\d+)?)"
            r"(?:\s*[=+\-−×*÷/^]\s*(?:[A-Za-z]{1,2}|\d+(?:\.\d+)?))+)"
        )
        for m in eq.finditer(text):
            span = m.group(1)
            if not re.search(r"[=×÷^√]", span):
                continue
            yield m.start(1), m.end(1), self._render_math_expr(span)
        # single letters adjacent to verbal math words: "a squared plus b"
        for m in re.finditer(
            r"\b([A-Za-z])\b(?=\s+(?:squared|cubed|plus|minus|"
            r"times|equals|to the power)\b)",
            text,
        ):
            yield m.start(), m.end(), L.letter(m.group(1))
        # Operators are expanded in-place in _final_sweep. This also handles
        # thresholds without a left operand and preserves x < y word order.

    def _render_math_expr(self, span):
        L = self.lang
        S = L.SYM
        opmap = {
            "=": S["equals"],
            "+": S["plus"],
            "-": S["minus"],
            "−": S["minus"],
            "×": S["times"],
            "*": S["times"],
            "÷": S["divide"],
            "/": S["divide"],
        }
        toks = re.findall(r"\d+(?:\.\d+)?|[A-Za-z]+|[=+\-−×*÷/^]", span)
        out = []
        i = 0
        while i < len(toks):
            tok = toks[i]
            if tok == "^":
                # base already emitted; render exponent via the power template
                exp = toks[i + 1] if i + 1 < len(toks) else ""
                ew = (
                    self._number_words(exp)
                    if exp[:1].isdigit()
                    else self._letters_spelled(exp.upper())
                )
                base = out.pop() if out else ""
                out.append(_fmt(L.POWER, base=base, exp=ew))
                i += 2
                continue
            if tok in opmap:
                out.append(opmap[tok])
            elif tok[0].isdigit():
                out.append(self._number_words(tok))
            else:
                out.append(self._letters_spelled(tok.upper()))
            i += 1
        return " ".join(out)

    # --- ranges -----------------------------------------------------------
    def _m_num_range(self, text):
        L = self.lang
        for m in re.finditer(r"(\d+(?:\.\d+)?)\s*[–]\s*(\d+(?:\.\d+)?)", text):
            a = self._number_words(m.group(1))
            b = self._number_words(m.group(2))
            yield m.start(), m.end(), _fmt(L.RANGE, a=a, b=b)
        # hyphen range: only ascending small pairs, no id keyword
        for m in re.finditer(r"\b(\d{1,3})-(\d{1,3})\b", text):
            if self._keyword_before(text, m.start(), self._id_keywords):
                continue
            a, b = int(m.group(1)), int(m.group(2))
            if a == b:
                # 50-50 -> "fifty fifty"
                yield m.start(), m.end(), f"{L.cardinal(a)} {L.cardinal(b)}"
                continue
            if a > b:
                if re.match(
                    r"\s+(?:days?|months?|years?|hours?|minutes?)\b",
                    text[m.end() :],
                    re.I,
                ):
                    yield m.start(), m.end(), _fmt(
                        L.RANGE, a=L.cardinal(a), b=L.cardinal(b)
                    )
                    continue
                # score-style pair: "3-2 से जीते" -> "तीन दो"
                yield m.start(), m.end(), f"{L.cardinal(a)} {L.cardinal(b)}"
                continue
            yield m.start(), m.end(), _fmt(L.RANGE, a=L.cardinal(a), b=L.cardinal(b))

    # --- dimensions / "x" as times: 24x7, 1920x1080, 6 x 4 -------------------
    def _m_dimensions(self, text):
        L = self.lang
        for m in re.finditer(r"\b(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)\b", text):
            if m.group(1) == "0":
                continue  # hex literals like 0x1F
            yield m.start(), m.end(), f"{self._number_words(m.group(1))} {L.SYM['times']} " f"{self._number_words(m.group(2))}"
        # lone x between something and a number (higher-priority matcher may
        # have consumed the left operand: "₹4,583.33 x 24")
        for m in re.finditer(r"(?<=\s)[x×](?=\s\d)", text):
            yield m.start(), m.end(), L.SYM["times"]

    # --- Roman numerals after label words: Class XII, World War II ----------
    _ROMAN_VALS = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}

    def _roman_value(self, s):
        total = 0
        for i, ch in enumerate(s):
            v = self._ROMAN_VALS[ch]
            if i + 1 < len(s) and self._ROMAN_VALS[s[i + 1]] > v:
                total -= v
            else:
                total += v
        return total

    def _m_roman(self, text):
        L = self.lang
        # {2,7}: single letters stay letters (Counter 2-C reads "C", not 100)
        pat = re.compile(
            r"\b(?=[IVXLCDM]{2,7}\b)M{0,3}(?:CM|CD|D?C{0,3})"
            r"(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})\b"
        )
        ctx = self._label_words | {
            "war",
            "vol",
            "part",
            "act",
            "book",
            "youdh",
            "युद्ध",
            "अध्याय",
            "कक्षा",
            "भाग",
            "खंड",
        }
        for m in pat.finditer(text):
            tok = m.group(0)
            if not tok:
                continue
            # only with a label-ish word right before — avoids XL sizes,
            # initials, and words like "MIX"
            if not self._keyword_before(text, m.start(), ctx, window=2):
                continue
            yield m.start(), m.end(), L.cardinal(self._roman_value(tok))

    # --- K/M/B suffixes glued to numbers: 3.4K, 1.2M -------------------------
    def _m_kmb_suffix(self, text):
        L = self.lang
        for m in re.finditer(r"(?<![\d.A-Za-z])(\d+(?:\.\d+)?)([KMB])\b", text):
            yield m.start(), m.end(), f"{self._number_words(m.group(1))} {L.letter(m.group(2))}"

    # --- ordinals -----------------------------------------------------------
    def _m_ordinal(self, text):
        L = self.lang
        pat = re.compile(rf"\b(\d+)-?({self._ord_suffix_re})(?![A-Za-z0-9])", re.I)
        for m in pat.finditer(text):
            n = int(m.group(1))
            suf = m.group(2).lower()
            if suf in ("st", "nd", "rd", "th"):
                yield m.start(), m.end(), L.ordinal(n)
            else:
                yield m.start(), m.end(), L.ordinal(n, suf)

    # --- fractions -----------------------------------------------------------
    def _m_fraction(self, text):
        L = self.lang
        S = L.SYM
        pat = re.compile(r"\b(\d{1,4}(?:\.\d+)?)/(\d{1,4}(?:\.\d+)?)\b")
        for m in pat.finditer(text):
            a, b = m.group(1), m.group(2)
            if (a, b) == ("1", "2"):
                rep = S["half"]
            elif (a, b) == ("1", "4"):
                rep = S["quarter"]
            elif (a, b) == ("3", "4"):
                rep = S["three_quarters"]
            else:
                rep = _fmt(
                    L.FRACTION, num=self._number_words(a), den=self._number_words(b)
                )
            yield m.start(), m.end(), rep

    # --- ratios -----------------------------------------------------------
    def _m_ratio(self, text):
        L = self.lang
        for m in re.finditer(r"\b(\d{1,3}):(\d{1,2})\b", text):
            mm = m.group(2)
            # 2-digit right side <=59 looks like a time -> time matcher takes it
            if (
                len(mm) == 2
                and int(mm) <= 59
                and not self._keyword_before(text, m.start(), self._ratio_kws, window=4)
            ):
                continue
            after = text[m.end() : m.end() + 20].strip()
            if any(after.startswith(w) for w in self._ratio_kws if not w.isascii()):
                # source already says the ratio word ("1:3 నిష్పత్తిలో")
                yield m.start(), m.end(), f"{L.cardinal(int(m.group(1)))} {L.cardinal(int(mm))}"
                continue
            yield m.start(), m.end(), _fmt(
                L.RATIO, a=L.cardinal(int(m.group(1))), b=L.cardinal(int(mm))
            )

    # --- labels (Room 314, Hall 3-A, Seat 23A, coach B4) --------------------
    def _m_label(self, text):
        L = self.lang
        lbl = "|".join(
            re.escape(w) for w in sorted(self._label_words, key=len, reverse=True)
        )
        pat = re.compile(
            rf"\b({lbl})\s+(?:(\d+)(?![\d.]|,\d)(?!-[A-Za-z]{{2,}})(?:[-\s]?([A-Za-z])\b)?|([A-Za-z])\b(?:[-\s]?(\d+))?)",
            re.IGNORECASE,
        )
        for m in pat.finditer(text):
            word = m.group(1)
            out = [word]
            if m.group(2):
                out.append(L.cardinal(int(m.group(2))))
                if m.group(3):
                    out.append(L.letter(m.group(3)))
            elif m.group(4):
                if len(m.group(4)) == 1:
                    out.append(L.letter(m.group(4)))
                else:
                    out.append(m.group(4))
                if m.group(5):
                    out.append(L.cardinal(int(m.group(5))))
            yield m.start(), m.end(), " ".join(out)

    # --- ALLCAPS model prefix + number: "XPS 15" ----------------------------
    def _m_acronym_model(self, text):
        L = self.lang
        for m in re.finditer(r"\b([A-Z]{2,4})\s+(\d{1,3})(?![\d.])", text):
            tok = m.group(1)
            if tok in _ACRONYM_WHITELIST or tok.upper() in _UNIT_INITIALISMS:
                continue
            if tok.lower() in self._id_keywords or tok.lower() in self._label_words:
                continue
            yield m.start(), m.end(), f"{self._letters_spelled(tok)} {L.cardinal(int(m.group(2)))}"

    # --- unit initialisms after numbers: 1.5 GB, 5,000 mAh ------------------
    def _m_unit_initialism(self, text):
        L = self.lang
        units = "|".join(sorted(_UNIT_INITIALISMS, key=len, reverse=True))
        pat = re.compile(
            rf"(?<![\w.])(-?)\b(\d+(?:,\d{{2,3}})*(?:\.\d+)?)[\s\-]?({units})"
            rf"(?:\s?/\s?([A-Za-z]{{1,4}}))?\b",
            re.IGNORECASE,
        )
        for m in pat.finditer(text):
            unit = m.group(3)
            if unit.upper() not in _UNIT_INITIALISMS:
                continue
            # single-letter units (V, W) only when written in caps —
            # a lowercase "5 v" is too ambiguous
            if len(unit) == 1 and not unit.isupper():
                continue
            num = self._number_words(m.group(2))
            if m.group(1):
                num = f"{L.SYM['minus']} {num}"
            uk = re.sub(r"[^A-Za-z]", "", unit).upper()
            word = L.UNIT_WORDS.get(uk) or self._letters_spelled(uk)
            rep = L.join_amount(num, word)
            if m.group(4):
                u2 = m.group(4).upper()
                w2 = L.UNIT_WORDS.get(u2) or self._letters_spelled(u2)
                rep += f" {L.SYM.get('per', 'per')} {w2}"
            yield m.start(), m.end(), rep

    # --- number-hyphen-word: 80-watt, 900-point ------------------------------
    def _m_num_hyphen_word(self, text):
        for m in re.finditer(
            r"\b(\d+(?:,\d{2,3})*(?:\.\d+)?)-([A-Za-zऀ-ൿ]{2,})\b", text
        ):
            word = m.group(2)
            if self._month_index(word):
                continue  # 12-Aug
            if word.upper() in _UNIT_INITIALISMS:
                continue  # unit matcher handles
            num = self._number_words(m.group(1))
            yield m.start(), m.end(), f"{num} {word}"

    # --- decades ------------------------------------------------------------
    def _m_decade(self, text):
        L = self.lang
        for m in re.finditer(r"\b(19|20)(\d)0s\b", text):
            if m.group(1) == "19":
                # 1980s -> "eighty's decade" (spec)
                yield m.start(), m.end(), L.decade(int(m.group(2)) * 10, full=True)
            else:
                # 2020s -> full year words + decade template
                y = int(m.group(0)[:4])
                yield m.start(), m.end(), _fmt(L.DECADE_FULL, tens=L.year(y))
        for m in re.finditer(r"(?<![\w])'?(\d)0s\b", text):
            yield m.start(), m.end(), L.decade(int(m.group(1)) * 10, full=False)

    # --- years ---------------------------------------------------------------
    def _m_year(self, text):
        L = self.lang
        year_words = {w.lower() for w in L.YEAR_WORDS} | {
            "year",
            "in",
            "since",
            "by",
            "from",
            "साल",
            "वर्ष",
        }
        for m in re.finditer(r"\b(1[5-9]\d{2}|20\d{2})\b", text):
            y = int(m.group(0))
            ctx = self._keyword_before(text, m.start(), year_words, window=2)
            # attached native locative suffix right after: 1947में / 1947-ல்
            after = text[m.end() : m.end() + 20]
            # danda ।/॥ are inside the Devanagari block — they are sentence
            # punctuation, not a locative suffix
            suffixed = bool(re.match(r"^-?(?![।॥])[ऀ-෿]", after))
            # following token is/starts with a year word: "1947 সালে",
            # "1983 साली", "१९४७ ମସିହାରେ", or Hindi locatives
            next_tok = after.strip().split()[0] if after.strip() else ""
            follower = next_tok in (
                "में",
                "से",
                "तक",
                "માં",
                "मध्ये",
                "ਵਿੱਚ",
                "ਤੋਂ",
            ) or any(next_tok.startswith(w) for w in L.YEAR_WORDS if w)
            if ctx or suffixed or follower:
                yield m.start(), m.end(), L.year(y)

    # --- quoted single letters: reply 'Y' -----------------------------------
    def _m_quoted_letter(self, text):
        for m in re.finditer(r"['\"‘“]([A-Za-z])['\"’”]", text):
            yield m.start(), m.end(), self.lang.letter(m.group(1))

    # --- short mixed tokens: 5G, A4, B12, Y2K, S24, A380 ---------------------
    def _m_short_mixed_token(self, text):
        L = self.lang
        pat = re.compile(
            r"(?<![\d.])\b(?=[A-Za-z0-9]{2,5}\b)"
            r"(?=[A-Za-z0-9]*\d)(?=[A-Za-z0-9]*[A-Za-z])"
            r"[A-Za-z0-9]+\b"
        )
        for m in pat.finditer(text):
            tok = m.group(0)
            if re.fullmatch(r"\d+(?:st|nd|rd|th|s)", tok):
                continue  # ordinals / decades
            out = []
            for run in re.findall(r"[A-Za-z]+|\d+", tok):
                if run.isdigit():
                    out.append(L.cardinal(int(run)))
                else:
                    out.append(self._letters_spelled(run.upper()))
            yield m.start(), m.end(), " ".join(out)

    # --- decimals -------------------------------------------------------------
    def _m_decimal(self, text):
        # trailing (?!\.?\d) instead of (?![\d.]) so a sentence-final full
        # stop after the decimal ("...830.93.") doesn't block the match
        for m in re.finditer(
            r"(?<![\d.])(\d{1,3}(?:,\d{2,3})+|\d+)\.(\d+)(?!\.?\d)", text
        ):
            yield m.start(), m.end(), self.lang.decimal(
                int(m.group(1).replace(",", "")), m.group(2)
            )

    # --- negative numbers -------------------------------------------------------
    def _m_negative(self, text):
        L = self.lang
        for m in re.finditer(r"(?<![\w.,-])-(\d+(?:\.\d+)?)", text):
            yield m.start(), m.end(), f"{L.SYM['minus']} {self._number_words(m.group(1))}"

    # --- plain integers ----------------------------------------------------------
    def _m_int(self, text):
        L = self.lang
        for m in re.finditer(r"\b\d{1,3}(?:,\d{2,3})+\b|\b\d+\b", text):
            digits = m.group(0).replace(",", "")
            # street numbers read year-style (1422 गांधी मार्ग -> चौदह सौ बाईस)
            after = text[m.end() : m.end() + 30].strip().lower()
            first_words = after.split()[:2]
            if (
                len(digits) == 4
                and digits[0] != "0"
                and any(w.strip(".,") in _STREET_WORDS for w in first_words)
            ):
                yield m.start(), m.end(), L.year(int(digits))
                continue
            if digits.startswith("0") and len(digits) > 1:
                yield m.start(), m.end(), self._digits_spelled(digits, comma=True)
                continue
            yield m.start(), m.end(), L.cardinal(int(digits))

    # ------------------------------------------------------------------
    # Final sweep: leftover meaningful symbols -> words; decorative junk out
    # ------------------------------------------------------------------
    def _final_sweep(self, text):
        S = self.lang.SYM
        text = re.sub(r"<=|≤", " less than or equal to ", text)
        text = re.sub(r">=|≥", " greater than or equal to ", text)
        text = re.sub(r"°\s*C\b", f" {S['celsius']} ", text)
        text = re.sub(r"°\s*F\b", f" {S['fahrenheit']} ", text)
        text = text.replace("°", f" {S['degree']} ")
        rep = {
            "%": S["percent"],
            "@": S["at"],
            "&": S["amp"],
            "=": S["equals"],
            "+": S["plus"],
            "×": S["times"],
            "÷": S["divide"],
            "±": S["plus_minus"],
            "√": S["sqrt"],
            "_": S["underscore"],
            "#": S["hash"],
        }
        for sym, word in rep.items():
            text = text.replace(sym, f" {word} ")
        text = text.replace("<", f" {S['lt']} ").replace(">", f" {S['gt']} ")
        text = re.sub(r"[\^*]", " ", text)
        # slashes not consumed by URL/fraction/date/code matchers are junk
        text = re.sub(r"\s*/\s*", " ", text)
        # clause colons read as a pause ("అడ్రస్: Door" -> "అడ్రస్, Door")
        text = re.sub(r"\s*:\s*", ", ", text)
        text = re.sub(r"[₹$€£](?!\s?\d)", " ", text)  # bare currency signs
        text = re.sub(r"[–—]", " ", text)
        return text

    # ------------------------------------------------------------------
    # Fallback: absolutely no digits may remain
    # ------------------------------------------------------------------
    def _digit_fallback(self, text):
        def repl(m):
            d = m.group(0)
            if len(d) >= 7 or d.startswith("0"):
                return self._digits_spelled(d, comma=True)
            return self.lang.cardinal(int(d))

        return re.sub(r"\d+", repl, text)

    # ------------------------------------------------------------------
    def normalize(self, text):
        validate_text(
            text,
            date_order=self.date_order,
            two_digit_year_base=self.two_digit_year_base,
            language=self.lang.code,
        )
        # URLs/emails are opaque identifiers. Protect them before prose cleanup
        # can expand Dr/No inside a path or turn // and parentheses into pauses.
        raw = _ascii_digits(text)
        protected, originals, spans = {}, {}, []
        identifiers = (self._m_email, self._m_url)
        if os.environ.get("FR3_TN_URL_SPELL", "0") == "1":
            identifiers += (self._m_handle,)
        for matcher in identifiers:
            for s, e, spoken in matcher(raw):
                if not any(s < b and e > a for a, b, _ in spans):
                    key = chr(0xF0000 + len(protected))
                    protected[key] = spoken
                    originals[key] = raw[s:e]
                    spans.append((s, e, key))
        for s, e, key in sorted(spans, reverse=True):
            raw = raw[:s] + key + raw[e:]
        text = raw
        text = self._preclean(text)
        try:
            matches = self._gather(text)
        except NormalizationError as exc:
            if exc.span is not None:
                start, end = exc.span
                pieces = [text[:start], text[start:end], text[end:]]
                # Parser coordinates refer to precleaned text containing opaque
                # URI placeholders. Restore their source text before retrying.
                for key, original in originals.items():
                    pieces = [piece.replace(key, original) for piece in pieces]
                exc.source = "".join(pieces)
                exc.span = (len(pieces[0]), len(pieces[0]) + len(pieces[1]))
            raise
        # priority asc, then leftmost, then longest
        matches.sort(key=lambda t: (t[0], t[1], -(t[2] - t[1])))
        taken = []
        accepted = []
        for prio, s, e, rep in matches:
            if any(not (e <= ts or s >= te) for ts, te in taken):
                continue
            taken.append((s, e))
            accepted.append((s, e, rep))
        accepted.sort(key=lambda t: t[0])
        out = []
        pos = 0
        for s, e, rep in accepted:
            out.append(text[pos:s])
            pos = e
            # native postposition right after the span (spaced OR attached:
            # "గంటలు కి" / "6:30కి") -> language attaches it with sandhi
            while True:
                nxt = re.match(r"[ \t]*-?([^\s\d,।.!?;:()\-—–]{1,14})", text[pos:])
                if not nxt:
                    break
                if text[pos + nxt.end() : pos + nxt.end() + 1] == ".":
                    break  # abbreviation like కి.మీ — not a postposition
                merged = self.lang.attach_postposition(rep, nxt.group(1))
                if merged is None:
                    break
                rep = merged
                pos += nxt.end()
            out.append(rep)
        out.append(text[pos:])
        result = "".join(out)
        result = self._final_sweep(result)
        result = self._digit_fallback(result)
        for key, spoken in protected.items():
            result = result.replace(key, spoken)
        # A foreign cardinal must never be glued into an Indic suffix (fiveన).
        result = re.sub(r"(?<=[A-Za-z])(?=[\u0900-\u0DFF])", " ", result)
        # tidy whitespace / punctuation spacing
        result = re.sub(r"\s+([,।.!?;])", r"\1", result)
        result = re.sub(r",\s*([।.!?;,])", r"\1", result)
        result = re.sub(r",(?=\S)", ", ", result)  # space after every comma
        result = re.sub(r"\s{2,}", " ", result)
        result = re.sub(r"\s+’", "’", result)
        result = (
            re.sub(r"[,\s]+$", ".", result) if result.rstrip().endswith(",") else result
        )
        if self.lang.code == "hi":
            result = result.translate(_HINDI_NUKTA_FOLD)
        return result.strip()


# ---------------------------------------------------------------------------
# English table (the only language table)
# ---------------------------------------------------------------------------

_ONES_0_19 = [
    "zero",
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
_TENS = [
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


def _two_digit(n):
    """0..99 -> 'zero' .. 'ninety nine'. No hyphen: the model reads words."""
    if n < 20:
        return _ONES_0_19[n]
    t, o = divmod(n, 10)
    return _TENS[t] if not o else f"{_TENS[t]} {_ONES_0_19[o]}"


_ORDINAL_IRREGULAR = {
    1: "first",
    2: "second",
    3: "third",
    5: "fifth",
    8: "eighth",
    9: "ninth",
    12: "twelfth",
}


def _ordinal(n):
    if n in _ORDINAL_IRREGULAR:
        return _ORDINAL_IRREGULAR[n]
    w = _two_digit(n) if n < 100 else None
    if w is None:
        return None
    # twenty-first, thirty-second ... take the irregular form on the last digit
    tail = n % 10
    if n >= 20 and tail in _ORDINAL_IRREGULAR:
        return f"{_TENS[n // 10]} {_ORDINAL_IRREGULAR[tail]}"
    if w.endswith("y"):  # twenty -> twentieth
        return w[:-1] + "ieth"
    return w + "th"


class English(LangBase):
    code = "en"
    name = "English"
    native_digits = "0123456789"  # English uses ASCII digits

    ONES = [_two_digit(i) for i in range(100)]
    HUNDRED = "hundred"
    THOUSAND = "thousand"
    LAKH = "lakh"
    CRORE = "crore"

    # Letter names are already English, so each maps to itself.
    LETTERS = {chr(65 + i): chr(65 + i) for i in range(26)}

    MONTHS = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]

    SYM = {
        "rupee": "rupees",
        "dollar": "dollars",
        "euro": "euros",
        "pound": "pounds",
        "cent": "cents",
        "percent": "percent",
        "point": "point",
        "minus": "minus",
        "plus": "plus",
        "equals": "equals",
        "times": "times",
        "divide": "divided by",
        "squared": "squared",
        "sqrt": "square root of",
        "lt": "less than",
        "gt": "greater than",
        "plus_minus": "plus or minus",
        "at": "at",
        "underscore": "underscore",
        "dot": "dot",
        "slash": "slash",
        "hash": "hash",
        "amp": "and",
        "half": "half",
        "quarter": "quarter",
        "three_quarters": "three quarters",
        "degree": "degrees",
        "celsius": "celsius",
        "fahrenheit": "fahrenheit",
        "minute": "minutes",
    }

    TIME_ON_HOUR = "{h} o'clock"
    TIME_HM = "{h} {m}"  # "ten thirty" — how a clock is read aloud
    RANGE = "{a} to {b}"
    RANGE_TO = "{a} to {b}"
    RATIO = "{a} to {b}"
    POWER = "{base} to the power {exp}"
    FRACTION = "{num} over {den}"
    DECADE_SHORT = "{tens}s"
    DECADE_FULL = "{tens}s"

    PERIODS = {
        "morning": "in the morning",
        "afternoon": "in the afternoon",
        "evening": "in the evening",
        "night": "at night",
    }

    # ---- vocabulary the engine must recognise in the INPUT ----------------
    TIME_MARKERS = ["o'clock", "oclock"]
    MARKER_SUFFIXES = {}
    PERIOD_DETECT = {
        "morning": "morning",
        "afternoon": "afternoon",
        "evening": "evening",
        "night": "night",
        "am": "morning",
        "a.m.": "morning",
        "pm": "afternoon",
        "p.m.": "afternoon",
    }
    YEAR_WORDS = ["year", "years"]
    # Deliberately NOT "order"/"invoice"/"ticket" on their own: "your order is 250
    # rupees" is a price, and listing them made it read "two, five, zero". They only
    # mark an identifier in the explicit "order number ..." form, which "number" catches.
    # One normalizer serves every language, so an identifier must be recognised
    # whatever script the sentence around it is written in. Without the native
    # words, "ऑर्डर नंबर 47291" reads as a cardinal instead of digit-by-digit.
    ID_KEYWORDS = [
        "number",
        "pin",
        "otp",
        "pnr",
        "imei",
        "ifsc",
        "account",
        "नंबर",
        "नम्बर",
        "खाता",
        "खाते",
        "कोड",
        "ऑर्डर",
        "फ़ोन",
        "फोन",
        "मोबाइल",
        "నంబర్",
        "నంబరు",
        "నెంబర్",
        "నెంబరు",
        "ఖాతా",
        "ఫోన్",
        "మొబైల్",
        "కోడ్",
        "ఆర్డర్",
        "ఎక్స్ప్రెస్",
        "express",
        "అకౌంట్",
        "టికెట్",
        "రిజిస్ట్రేషన్",
        "యూజర్",
        "వాయిదా",
        "అంకె",
        "లావాదేవీ",
        "పిన్",
        "எண்",
        "நம்பர்",
        "கணக்கு",
        "தொலைபேசி",
        "போன்",
        "மொபைல்",
        "ஆர்டர்",
        "ஓடிபி",
        "பின்",
        "ஐடி",
        "கடவுச்சொல்",
        "ரயில்",
        "எக்ஸ்பிரஸ்",
        "பிஎன்ஆர்",
        "கார்டு",
        "குறியீடு",
        "டிக்கெட்",
        "கணக்கில்",
        "யூசர்",
        "தவணை",
        "இலக்கம்",
        "பரிவர்த்தனை",
        "ನಂಬರ್",
        "ಸಂಖ್ಯೆ",
        "ಖಾತೆ",
        "ಕೋಡ್",
        "ಆರ್ಡರ್",
        "ಫೋನ್",
        "ಮೊಬೈಲ್",
        "ಐಡಿ",
        "ಪಿನ್",
        "ಪಾಸ್‌ವರ್ಡ್",
        "ರೈಲು",
        "ಎಕ್ಸ್‌ಪ್ರೆಸ್",
        "ಪಿಎನ್‌ಆರ್",
        "ಕಾರ್ಡ್",
        "ರೆಫರೆನ್ಸ್",
        "ಟ್ರಾನ್ಸಾಕ್ಷನ್",
        "ಟಿಕೆಟ್",
        "ಓಟಿಪಿ",
        "ಅಕೌಂಟ್",
        "ಅಂಕಿ",
        "ಬಳಕೆದಾರ",
        "ಕಂತು",
        "ವಹಿವಾಟು",
        "നമ്പർ",
        "നമ്പര്",
        "നമ്പറിൽ",
        "അക്കൗണ്ട്",
        "കോഡ്",
        "ഓർഡർ",
        "ഫോൺ",
        "മൊബൈൽ",
        "ഒടിപി",
        "പിൻ",
        "ഐഡി",
        "പാസ്‌വേഡ്",
        "ട്രെയിൻ",
        "എക്സ്പ്രസ്",
        "പിഎൻആർ",
        "കാർഡ്",
        "റഫറൻസ്",
        "ട്രാൻസാക്ഷൻ",
        "ടിക്കറ്റ്",
        "അക്കൗണ്ടിൽ",
        "നമ്പറിലേക്ക്",
        "യൂസർ",
        "തവണ",
        "അക്കം",
        "ഇടപാട്",
        "क्रमांक",
        "खात्याचा",
        "खात्यावर",
        "मोबाईल",
        "अकाउंट",
        "अकाउंटसाठी",
        "अकाउंटमधून",
        "खात्यातून",
        "खात्यात",
        "खात्याची",
        "पिन",
        "आयडी",
        "पासवर्ड",
        "ट्रेन",
        "एक्सप्रेस",
        "पीएनआर",
        "कार्ड",
        "रेफरन्स",
        "ट्रांझॅक्शन",
        "तिकीट",
        "ओटीपी",
        "गाडी",
        "युजर",
        "यूजर",
        "हप्ता",
        "हप्त्याचा",
        "अंक",
        "व्यवहार",
        "ट्रान्झॅक्शन",
        "क्र",
        "क्रं",
        "चेक",
        "धनादेश",
        "নম্বর",
        "নাম্বার",
        "অ্যাকাউন্ট",
        "একাউন্ট",
        "কোড",
        "অর্ডার",
        "ফোন",
        "মোবাইল",
        "ওটিপি",
        "পিন",
        "আইডি",
        "পাসওয়ার্ড",
        "ট্রেন",
        "এক্সপ্রেস",
        "পিএনআর",
        "কার্ড",
        "রেফারেন্স",
        "ট্রানজেকশন",
        "টিকিট",
        "অ্যাকাউন্টের",
        "নম্বরে",
        "আইডিতে",
        "ইউজার",
        "ইনস্টলমেন্ট",
        "কিস্তি",
        "ডিজিট",
        "অঙ্ক",
        "লেনদেন",
        "નંબર",
        "નમ્બર",
        "ખાતા",
        "ખાતું",
        "ખાતાં",
        "કોડ",
        "ઓર્ડર",
        "ફોન",
        "મોબાઇલ",
        "મોબાઈલ",
        "એકાઉન્ટ",
        "એકાઉન્ટમાં",
        "એકાઉન્ટનો",
        "પિન",
        "આઈડી",
        "પાસવર્ડ",
        "ટ્રેન",
        "એક્સપ્રેસ",
        "પીએનઆર",
        "કાર્ડ",
        "રેફરન્સ",
        "ટ્રાન્ઝેક્શન",
        "ટિકિટ",
        "ઓટીપી",
        "નંબરમાં",
        "યુઝર",
        "હપ્તો",
        "અંક",
        "ડિજિટ",
        "ਨੰਬਰ",
        "ਖਾਤਾ",
        "ਖਾਤੇ",
        "ਕੋਡ",
        "ਆਰਡਰ",
        "ਫ਼ੋਨ",
        "ਫੋਨ",
        "ਮੋਬਾਈਲ",
        "ਮੋਬਾਇਲ",
        "ਅਕਾਊਂਟ",
        "ਅਕਾਊਂਟਵਿੱਚ",
        "ਪਿੰਨ",
        "ਆਈਡੀ",
        "ਪਾਸਵਰਡ",
        "ਟ੍ਰੇਨ",
        "ਐਕਸਪ੍ਰੈਸ",
        "ਪੀਐਨਆਰ",
        "ਕਾਰਡ",
        "ਰੈਫਰੈਂਸ",
        "ਟਿਕਟ",
        "ਓਟੀਪੀ",
        "ਯੂਜ਼ਰ",
        "ਕਿਸ਼ਤ",
        "ਅੰਕ",
        "ਲੈਣ-ਦੇਣ",
        "ନମ୍ବର",
        "ନମ୍ବର୍",
        "ଖାତା",
        "ଆକାଉଣ୍ଟ",
        "କୋଡ୍",
        "ଅର୍ଡର",
        "ଫୋନ୍",
        "ଫୋନ",
        "ମୋବାଇଲ୍",
        "ମୋବାଇଲ",
        "ଓଟିପି",
        "ପିନ୍",
        "ଆଇଡି",
        "ପାସୱାର୍ଡ",
        "ଏକ୍ସପ୍ରେସ",
        "ପିଏନ୍‌ଆର୍",
        "କାର୍ଡ",
        "ରେଫରେନ୍ସ",
        "ଟିକେଟ",
        "ଆକାଉଣ୍ଟରୁ",
        "ଟ୍ରେନ୍",
        "ୟୁଜର",
        "କିସ୍ତି",
        "ଅଙ୍କ",
        "ଲାଣଦେଣ",
    ]
    ID_AFTER_WORDS = [
        "number",
        "code",
        "खत्म",
        "डायल",
        "कॉल",
        "संपर्क",
        "ఎండ్",
        "ముగిసే",
        "ముగుస్తుంది",
        "చివర",
        "చివరి",
        "అంకెలతో",
        "డయల్",
        "పిన్",
        "పిన్‌కోడ్",
        "pin",
        "pincode",
        "కాల్",
        "సంప్రదించండి",
        "டயல்",
        "அழை",
        "தொடர்பு",
        "ಡಯಲ್",
        "ಕರೆ",
        "ಸಂಪರ್ಕ",
        "ডায়াল",
        "কল",
        "যোগাযোগ",
        "ડાયલ",
        "કૉલ",
        "સંપર્ક",
        "ਡਾਇਲ",
        "ਕਾਲ",
        "ਸੰਪਰਕ",
        "ଡାଏଲ",
        "କଲ",
        "ଯୋଗାଯୋଗ",
    ]
    LABEL_WORDS = [
        "room",
        "flat",
        "floor",
        "gate",
        "platform",
        "seat",
        "section",
        "chapter",
        "page",
        "block",
        "phase",
    ]
    MINUTE_WORDS = ["minute", "minutes", "min", "mins"]
    SCALE_WORDS = {
        "crore": "crore",
        "crores": "crore",
        "cr": "crore",
        "lakh": "lakh",
        "lakhs": "lakh",
        "lac": "lakh",
        "thousand": "thousand",
        "thousands": "thousand",
        "k": "thousand",
    }
    CURRENCY_WORDS = {
        "rupee": "rupee",
        "rupees": "rupee",
        "rs": "rupee",
        "inr": "rupee",
        "dollar": "dollar",
        "dollars": "dollar",
        "usd": "dollar",
        "euro": "euro",
        "euros": "euro",
        "pound": "pound",
        "pounds": "pound",
        "paisa": "cent",
        "paise": "cent",
        "cent": "cent",
        "cents": "cent",
    }

    # Unit initialisms spoken as full words instead of letters. The engine
    # falls back to letter-spelling ("K G") when a key is absent, which is
    # right for GB/MB but wrong for the metric units customers actually say.
    UNIT_WORDS = {
        "KG": "kilograms",
        "KM": "kilometers",
        "CM": "centimeters",
        "MM": "millimeters",
        "GM": "grams",
        "MG": "milligrams",
        "ML": "millilitres",
        "MCG": "micrograms",
        "KMPH": "kilometers per hour",
        "MPH": "miles per hour",
        "KWH": "kilowatt hours",
        "RPM": "R P M",
    }

    # Written-only abbreviations expanded wherever they appear. Case-insensitive,
    # guarded by word boundaries, so "Drew" and "MIN-22" are untouched.
    ABBREV_UNITS = {
        "MINS": "minutes",
        "MIN": "minutes",
        "HRS": "hours",
        "SECS": "seconds",
        "DR": "doctor",
        "MR": "mister",
        "MRS": "missus",
        "MS": "miss",
        "PROF": "professor",
        "LTD": "limited",
        "PVT": "private",
        "A/C": "account",
        "APPROX": "approximately",
        "ETA": "E T A",
    }

    # "No. 12" -> "number twelve"
    NUMBER_WORD = "number"

    # English transcribes the am/pm that was written; it never invents a period.
    # "6.45 pm" -> "six forty five p m", NOT "six forty five in the evening".
    AMPM_WORDS = {"am": "a m", "pm": "p m"}

    ORDINAL_EXCEPTIONS = {n: _ordinal(n) for n in range(1, 100) if _ordinal(n)}
    ORDINAL_SUFFIX = "th"
    ORDINAL_INPUT_SUFFIXES = ["st", "nd", "rd", "th"]

    def time_phrase(self, h, m, period_key=None):
        """English puts the period AFTER the clock time — "five in the evening", not
        the base class's Indo-Aryan "in the evening five" (शाम पाँच बजे)."""
        if h == 0:
            core = "zero " + ("hundred" if m == 0 else self.two_digit(m))
        else:
            core = super().time_phrase(h, m, None)
        return f"{core} {self.PERIODS[period_key]}" if period_key else core

    def _ordinal_compose(self, n, src_suffix=""):
        if n < 100:
            return "zeroth" if n == 0 else _ordinal(n)
        for scale, word in (
            (10**7, self.CRORE),
            (10**5, self.LAKH),
            (1000, self.THOUSAND),
            (100, self.HUNDRED),
        ):
            if n >= scale:
                count, rest = divmod(n, scale)
                head = f"{self.cardinal(count)} {word}"
                return f"{head} {self.ordinal(rest)}" if rest else head + "th"


LANG = English()
_NORMALIZER = Normalizer(LANG)


@lru_cache(maxsize=66)
def _engine(language, date_order, two_digit_year_base):
    return Normalizer(
        language_table(language),
        date_order=date_order,
        two_digit_year_base=two_digit_year_base,
    )


def normalize(text, language="en-IN", *, date_order="DMY", two_digit_year_base=None):
    """Locale-specific numbers and symbols; prose is preserved, never translated.

    Omitted language retains Indian English for compatibility. Use an explicit
    language/locale for Indic speech. language="auto" selects majority-script
    inference with a guard against introducing foreign-script expansions.
    """
    auto = isinstance(language, str) and language == "auto"
    original = text
    if auto:
        if __package__:
            from .script_guard import detect_language, guard_output
        else:
            from script_guard import detect_language, guard_output
        language = detect_language(text) if isinstance(text, str) else "en-IN"
    code = resolve_language(language)
    # Validate before accessing the cache, including unhashable option values.
    warning = None
    while True:
        try:
            validate_text(
                text,
                date_order=date_order,
                two_digit_year_base=two_digit_year_base,
                language=code,
            )
            result = _engine(code, date_order, two_digit_year_base).normalize(text)
            break
        except NormalizationError as exc:
            # Request type/options are not text tokens and cannot be rendered.
            if os.environ.get("FR3_TN_STRICT", "0") == "1" or exc.span is None:
                raise
            warning = warning or exc.code
            source = exc.source if exc.source is not None else text
            start, end = exc.span
            token = source[start:end]
            locale = language_table(code)
            symbols = {
                "/": "slash",
                ".": "dot",
                "-": "minus",
                "+": "plus",
                "@": "at",
                "_": "underscore",
                "₹": "rupee",
                "$": "dollar",
                "%": "percent",
            }
            words = []
            for char in token:
                if char.isdecimal():
                    words.append(" " + locale.ONES[unicodedata.decimal(char)] + " ")
                elif char in symbols:
                    words.append(" " + locale.SYM[symbols[char]] + " ")
                elif char in ",:;<>{}" or unicodedata.category(char) in ("Cc", "Co"):
                    words.append(" ")
                else:
                    words.append(char)
            replacement = "".join(words)
            if replacement == token:
                raise  # no progress: do not loop on a new, unhandled error
            text = source[:start] + " " + replacement + " " + source[end:]
    if warning:
        # One privacy-safe observation per normalization call, never caller text.
        try:
            import obs

            obs.event("normalization_fallback", level="warn", code=warning)
        except ImportError:
            print(
                "[obs] "
                + json.dumps(
                    {
                        "event": "normalization_fallback",
                        "level": "warn",
                        "code": warning,
                    }
                ),
                flush=True,
            )
    return guard_output(original, result, language_table(code)) if auto else result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="?", help="Plain text; otherwise read stdin.")
    parser.add_argument("--date-order", choices=("DMY", "MDY"), default="DMY")
    parser.add_argument(
        "--language",
        default="en-IN",
        help="Language code/name, including hi, te and en-IN.",
    )
    parser.add_argument("--two-digit-year-base", type=int, choices=(1900, 2000))
    args = parser.parse_args()
    try:
        print(
            normalize(
                args.text if args.text is not None else sys.stdin.read(),
                language=args.language,
                date_order=args.date_order,
                two_digit_year_base=args.two_digit_year_base,
            )
        )
    except NormalizationError as exc:
        print(f"{exc.code}: {exc}", file=sys.stderr)
        sys.exit(2)
