from __future__ import annotations

import re

ENGLISH_LANGS = {"en", "en_us", "en_gb"}

_ONES = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
}
_TEENS = {
    10: "ten",
    11: "eleven",
    12: "twelve",
    13: "thirteen",
    14: "fourteen",
    15: "fifteen",
    16: "sixteen",
    17: "seventeen",
    18: "eighteen",
    19: "nineteen",
}
_TENS = {
    2: "twenty",
    3: "thirty",
    4: "forty",
    5: "fifty",
    6: "sixty",
    7: "seventy",
    8: "eighty",
    9: "ninety",
}
_SCALES = (
    (1_000_000_000_000, "trillion"),
    (1_000_000_000, "billion"),
    (1_000_000, "million"),
    (1_000, "thousand"),
)
_ORDINAL_EXCEPTIONS = {
    "zero": "zeroth",
    "one": "first",
    "two": "second",
    "three": "third",
    "five": "fifth",
    "eight": "eighth",
    "nine": "ninth",
    "twelve": "twelfth",
}
_MONTHS = {
    "jan": "january",
    "january": "january",
    "feb": "february",
    "february": "february",
    "mar": "march",
    "march": "march",
    "apr": "april",
    "april": "april",
    "may": "may",
    "jun": "june",
    "june": "june",
    "jul": "july",
    "july": "july",
    "aug": "august",
    "august": "august",
    "sep": "september",
    "sept": "september",
    "september": "september",
    "oct": "october",
    "october": "october",
    "nov": "november",
    "november": "november",
    "dec": "december",
    "december": "december",
}
_MONTH_BY_NUMBER = {
    1: "january",
    2: "february",
    3: "march",
    4: "april",
    5: "may",
    6: "june",
    7: "july",
    8: "august",
    9: "september",
    10: "october",
    11: "november",
    12: "december",
}
_CURRENCIES = {
    "$": ("dollar", "dollars", "cent", "cents"),
    "€": ("euro", "euros", "cent", "cents"),
    "£": ("pound", "pounds", "penny", "pence"),
}
_QUANTITIES = {
    "k": "thousand",
    "m": "million",
    "b": "billion",
    "bn": "billion",
    "thousand": "thousand",
    "million": "million",
    "billion": "billion",
    "trillion": "trillion",
}
_UNITS = {
    "kg": ("kilogram", "kilograms"),
    "g": ("gram", "grams"),
    "mg": ("milligram", "milligrams"),
    "km": ("kilometer", "kilometers"),
    "cm": ("centimeter", "centimeters"),
    "mm": ("millimeter", "millimeters"),
    "m": ("meter", "meters"),
    "mi": ("mile", "miles"),
    "mph": ("mile per hour", "miles per hour"),
    "lb": ("pound", "pounds"),
    "lbs": ("pound", "pounds"),
    "ft": ("foot", "feet"),
    "in": ("inch", "inches"),
    "l": ("liter", "liters"),
    "ml": ("milliliter", "milliliters"),
    "hz": ("hertz", "hertz"),
    "khz": ("kilohertz", "kilohertz"),
    "mhz": ("megahertz", "megahertz"),
    "ghz": ("gigahertz", "gigahertz"),
    "kbps": ("kilobit per second", "kilobits per second"),
    "mbps": ("megabit per second", "megabits per second"),
    "gbps": ("gigabit per second", "gigabits per second"),
    "°c": ("degree celsius", "degrees celsius"),
    "°f": ("degree fahrenheit", "degrees fahrenheit"),
}
_UNIT_PATTERN = "|".join(
    re.escape(unit) for unit in sorted(_UNITS, key=len, reverse=True)
)
_NUMBER_PATTERN = r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"

_ISO_DATE_RE = re.compile(r"\b(?P<year>\d{4})-(?P<month>\d{1,2})-(?P<day>\d{1,2})\b")
_SLASH_DATE_RE = re.compile(
    r"\b(?P<month>\d{1,2})/(?P<day>\d{1,2})/(?P<year>\d{2,4})\b"
)
_MONTH_DATE_RE = re.compile(
    r"\b(?P<month>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
    r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sept?(?:ember)?|oct(?:ober)?|"
    r"nov(?:ember)?|dec(?:ember)?)\.?\s+"
    r"(?P<day>\d{1,2})(?:st|nd|rd|th)?(?:,\s*|\s+)?(?P<year>\d{4})?\b",
    re.IGNORECASE,
)
_TIME_RE = re.compile(
    r"(?<!\w)(?P<hour>[01]?\d|2[0-3]):(?P<minute>[0-5]\d)"
    r"(?::(?P<second>[0-5]\d))?\s*(?P<suffix>[aApP]\.?\s?[mM]\.?)?\b"
)
_MONEY_RE = re.compile(
    rf"(?<!\w)(?P<symbol>[$€£])\s*(?P<amount>{_NUMBER_PATTERN})"
    r"(?:\s*(?P<quantity>k|K|m|M|b|B|bn|BN|thousand|million|billion|trillion))?"
)
_PERCENT_RE = re.compile(rf"(?<!\w)(?P<number>{_NUMBER_PATTERN})\s*%")
_UNIT_RE = re.compile(
    rf"(?<!\w)(?P<number>{_NUMBER_PATTERN})\s*(?P<unit>{_UNIT_PATTERN})\b",
    re.IGNORECASE,
)
_FRACTION_RE = re.compile(r"(?<!\w)(?P<numerator>\d+)/(?P<denominator>\d+)(?!\w)")
_ORDINAL_RE = re.compile(
    r"(?<!\w)(?P<number>-?\d{1,3}(?:,\d{3})+|-?\d+)(?:st|nd|rd|th)\b"
)
_DECIMAL_RE = re.compile(
    r"(?<![\w.])(?P<number>-?(?:\d{1,3}(?:,\d{3})+|\d+)\.\d+)(?![\w.])"
)
_PHONE_RE = re.compile(
    r"(?<!\w)(?:\+?1[-.\s]?)?(?:(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4})(?!\w)"
)
_INTEGER_RE = re.compile(r"(?<![\w.])(?P<number>-?(?:\d{1,3}(?:,\d{3})+|\d+))(?![\w.])")
_ACRONYM_RE = re.compile(r"\b[A-Z]{2,}\b")
_SPACE_RE = re.compile(r"\s+")
_SPACE_PUNCT_RE = re.compile(r"\s+([.!?,;:])")


def default_cache_root() -> str:
    """Compatibility shim from the previous NeMo-backed implementation."""
    return ""


def _int_to_words(value: int) -> str:
    if value < 0:
        return "negative " + _int_to_words(abs(value))
    if value < 10:
        return _ONES[value]
    if value < 20:
        return _TEENS[value]
    if value < 100:
        tens, rest = divmod(value, 10)
        return _TENS[tens] if rest == 0 else f"{_TENS[tens]} {_ONES[rest]}"
    if value < 1000:
        hundreds, rest = divmod(value, 100)
        words = f"{_ONES[hundreds]} hundred"
        return words if rest == 0 else f"{words} {_int_to_words(rest)}"
    for scale_value, scale_name in _SCALES:
        if value >= scale_value:
            quotient, rest = divmod(value, scale_value)
            words = f"{_int_to_words(quotient)} {scale_name}"
            return words if rest == 0 else f"{words} {_int_to_words(rest)}"
    return str(value)


def _year_to_words(value: int) -> str:
    if 1000 <= value <= 1999:
        prefix, rest = divmod(value, 100)
        if rest == 0:
            return f"{_int_to_words(prefix)} hundred"
        return f"{_int_to_words(prefix)} {_int_to_words(rest)}"
    if 2000 <= value <= 2009:
        rest = value - 2000
        return "two thousand" if rest == 0 else f"two thousand {_int_to_words(rest)}"
    if 2010 <= value <= 2099:
        return f"twenty {_int_to_words(value - 2000)}"
    return _int_to_words(value)


def _digits_to_words(digits: str) -> str:
    return " ".join(_ONES[int(digit)] for digit in digits)


def _clean_numeric(value: str) -> str:
    return value.replace(",", "")


def _number_to_words(value: str) -> str:
    value = _clean_numeric(value)
    negative = value.startswith("-")
    if negative:
        value = value[1:]
    prefix = "negative " if negative else ""
    if "." in value:
        integer, fractional = value.split(".", 1)
        integer_words = _int_to_words(int(integer or "0"))
        return f"{prefix}{integer_words} point {_digits_to_words(fractional)}"
    if len(value) > 1 and value.startswith("0"):
        return prefix + _digits_to_words(value)
    return prefix + _int_to_words(int(value))


def _ordinal_to_words(value: int) -> str:
    words = _int_to_words(value).split()
    last = words[-1]
    if last in _ORDINAL_EXCEPTIONS:
        words[-1] = _ORDINAL_EXCEPTIONS[last]
    elif last.endswith("y"):
        words[-1] = last[:-1] + "ieth"
    else:
        words[-1] = last + "th"
    return " ".join(words)


def _numeric_is_one(value: str) -> bool:
    value = _clean_numeric(value)
    try:
        return float(value) == 1.0
    except ValueError:
        return False


def _format_date(month: int, day: int, year: str | None = None) -> str | None:
    month_name = _MONTH_BY_NUMBER.get(month)
    if month_name is None or not 1 <= day <= 31:
        return None
    parts = [month_name, _ordinal_to_words(day)]
    if year:
        year_num = int(year)
        if year_num < 100:
            year_num += 2000 if year_num < 50 else 1900
        parts.append(_year_to_words(year_num))
    return " ".join(parts)


def _replace_iso_date(match: re.Match[str]) -> str:
    replacement = _format_date(
        int(match.group("month")),
        int(match.group("day")),
        match.group("year"),
    )
    return replacement or match.group(0)


def _replace_slash_date(match: re.Match[str]) -> str:
    replacement = _format_date(
        int(match.group("month")),
        int(match.group("day")),
        match.group("year"),
    )
    return replacement or match.group(0)


def _replace_month_date(match: re.Match[str]) -> str:
    month = _MONTHS.get(match.group("month").rstrip(".").lower())
    if month is None:
        return match.group(0)
    day = int(match.group("day"))
    if not 1 <= day <= 31:
        return match.group(0)
    parts = [month, _ordinal_to_words(day)]
    year = match.group("year")
    if year:
        parts.append(_year_to_words(int(year)))
    return " ".join(parts)


def _replace_time(match: re.Match[str]) -> str:
    hour = int(match.group("hour"))
    minute = int(match.group("minute"))
    second = match.group("second")
    suffix = match.group("suffix")
    spoken_hour = hour
    if suffix:
        spoken_hour = hour % 12 or 12
    parts = [_int_to_words(spoken_hour)]
    if minute:
        minute_words = (
            f"oh {_int_to_words(minute)}" if minute < 10 else _int_to_words(minute)
        )
        parts.append(minute_words)
    if second and int(second):
        parts.extend(["and", _int_to_words(int(second)), "seconds"])
    if suffix:
        suffix_letter = "a" if suffix.lower().startswith("a") else "p"
        parts.extend([suffix_letter, "m"])
    return " ".join(parts)


def _replace_money(match: re.Match[str]) -> str:
    symbol = match.group("symbol")
    amount = _clean_numeric(match.group("amount"))
    quantity = match.group("quantity")
    major_singular, major_plural, minor_singular, minor_plural = _CURRENCIES[symbol]
    if quantity:
        quantity_word = _QUANTITIES[quantity.lower()]
        major = major_singular if _numeric_is_one(amount) else major_plural
        return f"{_number_to_words(amount)} {quantity_word} {major}"

    if "." in amount:
        integer, fractional = amount.split(".", 1)
        cents = int((fractional + "00")[:2])
    else:
        integer, cents = amount, 0

    dollars = int(integer)
    parts = []
    if dollars:
        major = major_singular if dollars == 1 else major_plural
        parts.append(f"{_int_to_words(dollars)} {major}")
    if cents:
        minor = minor_singular if cents == 1 else minor_plural
        parts.append(f"{_int_to_words(cents)} {minor}")
    if not parts:
        return f"zero {major_plural}"
    return " and ".join(parts)


def _replace_percent(match: re.Match[str]) -> str:
    return f"{_number_to_words(match.group('number'))} percent"


def _replace_unit(match: re.Match[str]) -> str:
    number = match.group("number")
    unit = match.group("unit").lower()
    singular, plural = _UNITS[unit]
    unit_words = singular if _numeric_is_one(number) else plural
    return f"{_number_to_words(number)} {unit_words}"


def _replace_fraction(match: re.Match[str]) -> str:
    numerator = int(match.group("numerator"))
    denominator = int(match.group("denominator"))
    numerator_words = _int_to_words(numerator)
    if denominator == 2:
        denominator_words = "half" if numerator == 1 else "halves"
    elif denominator == 4:
        denominator_words = "quarter" if numerator == 1 else "quarters"
    else:
        denominator_words = _ordinal_to_words(denominator)
        if numerator != 1:
            denominator_words += "s"
    return f"{numerator_words} {denominator_words}"


def _replace_ordinal(match: re.Match[str]) -> str:
    return _ordinal_to_words(int(_clean_numeric(match.group("number"))))


def _replace_phone(match: re.Match[str]) -> str:
    digits = re.sub(r"\D", "", match.group(0))
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    return _digits_to_words(digits)


def _replace_acronym(match: re.Match[str]) -> str:
    return " ".join(match.group(0).lower())


def normalize_english_text(text: str) -> str:
    text = _ISO_DATE_RE.sub(_replace_iso_date, text)
    text = _MONTH_DATE_RE.sub(_replace_month_date, text)
    text = _SLASH_DATE_RE.sub(_replace_slash_date, text)
    text = _TIME_RE.sub(_replace_time, text)
    text = _PHONE_RE.sub(_replace_phone, text)
    text = _MONEY_RE.sub(_replace_money, text)
    text = _PERCENT_RE.sub(_replace_percent, text)
    text = _UNIT_RE.sub(_replace_unit, text)
    text = _FRACTION_RE.sub(_replace_fraction, text)
    text = _ORDINAL_RE.sub(_replace_ordinal, text)
    text = _DECIMAL_RE.sub(lambda m: _number_to_words(m.group("number")), text)
    text = _INTEGER_RE.sub(lambda m: _number_to_words(m.group("number")), text)
    text = _ACRONYM_RE.sub(_replace_acronym, text)
    text = text.replace("&", " and ")
    text = text.replace("@", " at ")
    text = _SPACE_PUNCT_RE.sub(r"\1", text)
    return _SPACE_RE.sub(" ", text).strip()


class TTSTextNormalizer:
    """Dependency-free English text normalizer for ZONOS2 TTS prompts."""

    def __init__(self, cache_root: str | None = None):
        self.cache_root = cache_root

    def supported(self, language: str) -> bool:
        return language.lower() in ENGLISH_LANGS

    def normalize(self, text: str, language: str) -> str:
        if not text.strip() or not self.supported(language):
            return text
        return normalize_english_text(text)
