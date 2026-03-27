"""Text preprocessing utilities for Voxtral TTS."""

import re
from html import unescape

_INVISIBLE_UNICODE_RE = re.compile(
    "["
    "\u200b"
    "\u200e-\u200f"
    "\u2028-\u2029"
    "\u2060-\u2064"
    "\u00ad"
    "\u180e"
    "\ufeff"
    "\ufff9-\ufffb"
    "]"
)

_LINE_BREAK_RE = re.compile(r"(?:\r\n|\r|\n)+")
_REPEATED_PUNCT_RE = re.compile(r"([!?])\1+")
_ASCII_ELLIPSIS_RUN_RE = re.compile(r"\.{3,}")
_UNICODE_HYPHEN_TO_ASCII_RE = re.compile("[\u2010\u2011]")

_DASH_LIKE_CHARS = "\u002d\u2013\u2212"
_DASH_LIKE_CLASS = re.escape(_DASH_LIKE_CHARS)
_CURRENCY_CHARS = "$€£¥₹¢"
_MULTI_HYPHEN_RE = re.compile(rf"[{_DASH_LIKE_CLASS}]{{2,}}")
_STANDALONE_HYPHEN_RE = re.compile(rf"(?<=\s)[{_DASH_LIKE_CLASS}](?=\s)")
_TERMINAL_PUNCT = ".!?\u2026\u061f\u3002\uff01\uff1f"

_PARENTHETICAL_RE = re.compile(r"\s*\(([^()]+)\)\s*")
_SIMPLE_NUMERIC_PAREN_CONTENT_RE = re.compile(
    r"^\s*[$€£¥₹¢]?\s*[+\-−–]?\d+(?:[.,]\d+)?\s*%?\s*$"
)

_MD_FENCE_RE = re.compile(r"```[\s\S]*?```|~~~[\s\S]*?~~~")
_MD_IMAGE_LINK_RE = re.compile(r"!\[([^\]]*)\]\(([^)]*)\)")
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]*)\)")
_MD_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_MD_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_MD_UNORDERED_RE = re.compile(r"^\s*[-*+]\s+", re.MULTILINE)
_MD_ORDERED_RE = re.compile(r"^\s*\d{1,4}[.)]\s+", re.MULTILINE)
_MD_BLOCKQUOTE_RE = re.compile(r"^\s{0,3}>\s?", re.MULTILINE)
_MD_BR_TAG_RE = re.compile(r"(?i)<br\s*/?>")
_MD_HTML_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")
_MD_AUTOLINK_RE = re.compile(r"<https?://[^>]+>")
_MD_URL_RE = re.compile(r"https?://\S+")

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
    20: "twenty",
    30: "thirty",
    40: "forty",
    50: "fifty",
    60: "sixty",
    70: "seventy",
    80: "eighty",
    90: "ninety",
}
_SCALES = ["", "thousand", "million", "billion", "trillion", "quadrillion"]

_TOKEN_RE = re.compile(
    r"""
    (?P<currency>
        (?P<symbol>[$£€¥₹¢])
        (?P<amount>\d[\d,]*(?:\.\d+)?)
    )
    |
    (?P<number>
        (?<!\w)-?\d[\d,]*(?:\.\d+)?\b
    )
    """,
    re.VERBOSE,
)

_CURRENCY_INFO = {
    "$": ("dollar", "dollars", "cent", "cents"),
    "£": ("pound", "pounds", "penny", "pence"),
    "€": ("euro", "euros", "cent", "cents"),
    "¥": ("yen", "yen", None, None),
    "₹": ("rupee", "rupees", "paise", "paise"),
    "¢": ("cent", "cents", None, None),
}


def _flatten_markdown_for_tts(text: str) -> str:
    text = _MD_FENCE_RE.sub(" Code example omitted. ", text)
    text = unescape(text)
    text = _MD_BR_TAG_RE.sub(" ", text)
    text = _MD_IMAGE_LINK_RE.sub(lambda match: match.group(1).strip() or "image", text)
    text = _MD_LINK_RE.sub(lambda match: match.group(1), text)
    text = _MD_AUTOLINK_RE.sub("link", text)
    text = _MD_URL_RE.sub("link", text)
    text = _MD_INLINE_CODE_RE.sub(lambda match: match.group(1), text)
    text = _MD_HEADING_RE.sub("", text)
    text = _MD_UNORDERED_RE.sub("", text)
    text = _MD_ORDERED_RE.sub("", text)
    text = _MD_BLOCKQUOTE_RE.sub("", text)
    text = _MD_HTML_TAG_RE.sub(" ", text)
    return text


def _next_non_space_char(text: str, start_index: int) -> str | None:
    index = start_index
    while index < len(text) and text[index].isspace():
        index += 1
    if index >= len(text):
        return None
    return text[index]


def _replace_standalone_hyphens(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        next_char = _next_non_space_char(text, match.end())
        if next_char is None:
            return match.group(0)
        if next_char.isdigit() or next_char in _CURRENCY_CHARS:
            return match.group(0)
        return "\u2014"

    return _STANDALONE_HYPHEN_RE.sub(replace, text)


def _replace_textual_parentheticals(text: str) -> str:
    def normalize_parenthetical(inner: str) -> str:
        normalized = inner
        normalized = re.sub(r"(?i)\bN\s*/\s*A\b", "not available", normalized)
        normalized = re.sub(r"\s*(?:\+/-|±)\s*", " plus or minus ", normalized)
        normalized = re.sub(r"\s*>=\s*", " greater than or equal to ", normalized)
        normalized = re.sub(r"\s*<=\s*", " less than or equal to ", normalized)
        normalized = re.sub(r"\s*>\s*", " greater than ", normalized)
        normalized = re.sub(r"\s*<\s*", " less than ", normalized)
        normalized = re.sub(r"\s*~\s*", " about ", normalized)
        normalized = re.sub(r"#\s*(\d+)\b", r"number \1", normalized)
        normalized = re.sub(
            r"(\d+(?:\.\d+)?)\s*ms\b",
            r"\1 milliseconds",
            normalized,
            flags=re.IGNORECASE,
        )
        normalized = re.sub(
            r"(\d+(?:\.\d+)?)\s*h\b", r"\1 hours", normalized, flags=re.IGNORECASE
        )
        normalized = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"\1 percent", normalized)
        normalized = re.sub(r"\s*\+\s*", " plus ", normalized)
        normalized = re.sub(r"\s*=\s*", " equals ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    def replace(match: re.Match[str]) -> str:
        inner = match.group(1).strip()
        if not inner:
            return match.group(0)
        if _SIMPLE_NUMERIC_PAREN_CONTENT_RE.fullmatch(inner):
            return match.group(0)
        normalized = normalize_parenthetical(inner)
        return f"\u2014{normalized}\u2014"

    return _PARENTHETICAL_RE.sub(replace, text)


def _verbalize_sub_thousand(n: int, *, use_and: bool = False) -> str:
    if not 0 <= n <= 999:
        raise ValueError(f"Expected sub-thousand integer in [0, 999], got {n=}")
    parts: list[str] = []
    hundreds = n // 100
    rem = n % 100
    if hundreds:
        parts.append(f"{_ONES[hundreds]} hundred")
        if rem and use_and:
            parts.append("and")
    if rem:
        if rem < 10:
            parts.append(_ONES[rem])
        elif rem < 20:
            parts.append(_TEENS[rem])
        else:
            tens = (rem // 10) * 10
            ones = rem % 10
            if ones:
                parts.append(f"{_TENS[tens]}-{_ONES[ones]}")
            else:
                parts.append(_TENS[tens])
    return " ".join(parts) if parts else "zero"


def _verbalize_integer_en(num_str: str, *, use_and: bool = False) -> str:
    s = num_str.replace(",", "").strip()
    if not re.fullmatch(r"\d+", s):
        raise ValueError(f"Not a plain integer: {num_str}")
    n = int(s)
    if n == 0:
        return "zero"
    groups: list[int] = []
    while n > 0:
        groups.append(n % 1000)
        n //= 1000
    if len(groups) > len(_SCALES):
        raise ValueError(f"Integer magnitude too large to verbalize safely: {num_str}")
    parts: list[str] = []
    for scale_idx in range(len(groups) - 1, -1, -1):
        group_val = groups[scale_idx]
        if group_val == 0:
            continue
        group_words = _verbalize_sub_thousand(
            group_val,
            use_and=use_and and scale_idx == 0 and len(groups) > 1,
        )
        scale_word = _SCALES[scale_idx]
        parts.append(f"{group_words} {scale_word}".strip())
    return " ".join(parts)


def _verbalize_decimal_en(num_str: str, *, use_and: bool = False) -> str:
    s = num_str.replace(",", "").strip()
    match = re.fullmatch(r"(\d+)\.(\d+)", s)
    if not match:
        raise ValueError(f"Not a plain decimal: {num_str}")
    int_part, frac_part = match.groups()
    int_words = _verbalize_integer_en(int_part, use_and=use_and)
    frac_words = "-".join(_ONES[int(ch)] for ch in frac_part)
    return f"{int_words} point {frac_words}"


def _verbalize_number_en(num_str: str, *, use_and: bool = False) -> str:
    s = num_str.strip()
    if s.startswith("-"):
        inner = s[1:]
        if not inner:
            raise ValueError(f"Unsupported numeric format: {num_str}")
        return f"negative {_verbalize_number_en(inner, use_and=use_and)}"
    if re.fullmatch(r"\d{4}", s):
        year = int(s)
        if 1400 <= year < 2100:
            if year == 2000:
                return "two thousand"
            first_two = year // 100
            last_two = year % 100
            if 1400 <= year <= 1999:
                if last_two == 0:
                    return f"{_verbalize_integer_en(str(first_two), use_and=use_and)} hundred"
                return (
                    f"{_verbalize_integer_en(str(first_two), use_and=use_and)} "
                    f"{_verbalize_sub_thousand(last_two)}"
                )
            if 2001 <= year <= 2009:
                return f"two thousand {_ONES[last_two]}"
            if 2010 <= year <= 2099:
                return f"twenty {_verbalize_sub_thousand(last_two)}"
    if re.fullmatch(r"\d[\d,]*", s):
        return _verbalize_integer_en(s, use_and=use_and)
    if re.fullmatch(r"\d[\d,]*\.\d+", s):
        return _verbalize_decimal_en(s, use_and=use_and)
    raise ValueError(f"Unsupported numeric format: {num_str}")


def _parse_currency_token(token: str) -> tuple[str, int, str | None]:
    match = re.fullmatch(r"([$£€¥₹¢])(\d[\d,]*)(?:\.(\d+))?", token.strip())
    if not match:
        raise ValueError(f"Not a supported currency amount: {token}")
    symbol, whole_part, frac_part = match.groups()
    whole = int(whole_part.replace(",", ""))
    return symbol, whole, frac_part


def _verbalize_currency_decimal_en(symbol: str, whole: int, frac_part: str) -> str:
    singular_major, plural_major, _, _ = _CURRENCY_INFO[symbol]
    major_unit = singular_major if whole == 1 else plural_major
    whole_words = _verbalize_integer_en(str(whole), use_and=False)
    frac_words = "-".join(_ONES[int(ch)] for ch in frac_part)
    return f"{whole_words} point {frac_words} {major_unit}"


def _verbalize_currency_en(
    token: str, *, use_and: bool = True, short: bool = False
) -> str:
    symbol, whole, frac_part = _parse_currency_token(token)
    singular_major, plural_major, singular_minor, plural_minor = _CURRENCY_INFO[symbol]
    if short and symbol != "$":
        short = False
    if frac_part is not None and len(frac_part) > 2:
        return _verbalize_currency_decimal_en(symbol, whole, frac_part)

    minor = None
    if frac_part is not None:
        frac_2 = (frac_part + "00")[:2]
        minor = int(frac_2)

    if not short:
        whole_words = _verbalize_integer_en(str(whole), use_and=use_and)
        major_unit = singular_major if whole == 1 else plural_major
        if symbol in {"¥", "¢"}:
            if frac_part is None or set(frac_part) == {"0"}:
                return f"{whole_words} {major_unit}"
            return _verbalize_currency_decimal_en(symbol, whole, frac_part)
        if minor is None or minor == 0:
            return f"{whole_words} {major_unit}"
        if whole == 0:
            minor_words = _verbalize_integer_en(str(minor), use_and=False)
            minor_unit = singular_minor if minor == 1 else plural_minor
            return f"{minor_words} {minor_unit}"
        minor_words = _verbalize_integer_en(str(minor), use_and=False)
        minor_unit = singular_minor if minor == 1 else plural_minor
        return f"{whole_words} {major_unit} and {minor_words} {minor_unit}"

    if minor is None or minor == 0:
        whole_words = _verbalize_integer_en(str(whole), use_and=False)
        major_unit = singular_major if whole == 1 else plural_major
        return f"{whole_words} {major_unit}"
    if whole == 0:
        minor_words = _verbalize_integer_en(str(minor), use_and=False)
        minor_unit = singular_minor if minor == 1 else plural_minor
        return f"{minor_words} {minor_unit}"
    if 0 <= minor <= 9:
        return f"{_verbalize_integer_en(str(whole), use_and=False)} oh {_ONES[minor]}"
    return (
        f"{_verbalize_integer_en(str(whole), use_and=False)} "
        f"{_verbalize_integer_en(str(minor), use_and=False)}"
    )


def _should_verbalize_plain_number(token: str) -> bool:
    s = token.strip()
    if s.startswith("-"):
        s = s[1:]
    raw_int_part = s.split(".", 1)[0]
    int_no_commas = raw_int_part.replace(",", "")
    if not re.fullmatch(r"\d+", int_no_commas):
        return False
    value = int(int_no_commas)
    return 1_000_000 <= value < 1e18 and "," in raw_int_part


def _auto_verbalize_safe_numbers_in_text(text: str) -> str:
    matches = list(_TOKEN_RE.finditer(text))
    currency_symbols = [
        m.group("symbol") for m in matches if m.group("currency") is not None
    ]
    only_dollar_repetitions = (
        currency_symbols
        and set(currency_symbols) == {"$"}
        and len(currency_symbols) > 1
    )
    first_dollar_done = False

    def repl(match: re.Match[str]) -> str:
        nonlocal first_dollar_done
        if match.group("currency") is not None:
            token = match.group("currency")
            symbol = match.group("symbol")
            try:
                short = False
                if only_dollar_repetitions and symbol == "$":
                    if first_dollar_done:
                        short = True
                    else:
                        first_dollar_done = True
                return _verbalize_currency_en(token, use_and=True, short=short)
            except (ValueError, IndexError):
                return token
        token = match.group("number")
        try:
            if not _should_verbalize_plain_number(token):
                return token
            return _verbalize_number_en(token, use_and=True)
        except (ValueError, IndexError):
            return token

    return _TOKEN_RE.sub(repl, text)


def sanitize_tts_input_text_for_demo(text: str) -> str:
    """Normalize text before sending it to TTS."""
    raw_text = text
    text = _flatten_markdown_for_tts(text)
    text = _INVISIBLE_UNICODE_RE.sub("", text)
    text = _LINE_BREAK_RE.sub(" ", text)
    text = _auto_verbalize_safe_numbers_in_text(text)
    text = _replace_textual_parentheticals(text)
    text = _UNICODE_HYPHEN_TO_ASCII_RE.sub("-", text)
    text = _ASCII_ELLIPSIS_RUN_RE.sub("...", text)
    text = _REPEATED_PUNCT_RE.sub(r"\1", text)
    text = _MULTI_HYPHEN_RE.sub("\u2014", text)
    text = _replace_standalone_hyphens(text)
    text = re.sub(r"\s+", " ", text).strip()
    if text and text[-1] not in _TERMINAL_PUNCT:
        text += "."
    if not text:
        raise ValueError(f"Speech input is empty after sanitization, got {raw_text=}")
    return text
