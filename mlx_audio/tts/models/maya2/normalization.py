"""Spoken forms for Maya2's numbers, money, dates, times, and identifiers."""

import json
import re
import unicodedata
from datetime import date
from functools import lru_cache
from pathlib import Path

_NUMBER = r"\d+(?:,\d+)*(?:\.\d+)?"
_SYMBOLS = {"₹": "rupee", "$": "dollar", "€": "euro", "£": "pound"}
_CURRENCIES = {
    "rs": "rupee",
    "rs.": "rupee",
    "inr": "rupee",
    "rupees": "rupee",
    "rupee": "rupee",
    "usd": "dollar",
    "dollars": "dollar",
    "dollar": "dollar",
    "eur": "euro",
    "euro": "euro",
    "euros": "euro",
    "gbp": "pound",
    "pound": "pound",
    "pounds": "pound",
}


@lru_cache(maxsize=1)
def _data():
    return json.loads(
        Path(__file__).with_name("normalization_data.json").read_text(encoding="utf-8")
    )


def _alternatives(words):
    return "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))


def _ungroup(value):
    integer = value.lstrip("+-").split(".")[0]
    if "," in integer and not re.fullmatch(
        r"\d{1,3}(?:,\d{3})+|\d{1,2}(?:,\d{2})*,\d{3}", integer
    ):
        raise ValueError("Invalid number grouping")
    return value.replace(",", "")


class _Language:
    def __init__(self, code):
        self.code = code
        self.data = _data()["languages"][code]
        self.ones = self.data["ones"]
        self.symbols = self.data["symbols"]

    def join(self, parts):
        parts = [p for p in parts if p]
        if self.code == "ta":
            for i in range(len(parts) - 1):
                if parts[i].endswith(("த்து", "ற்று")) and parts[i + 1][0] in "கசதப":
                    parts[i] += {"க": "க்", "ச": "ச்", "த": "த்", "ப": "ப்"}[
                        parts[i + 1][0]
                    ]
        return " ".join(parts)

    def cardinal(self, value):
        if value < 100:
            return self.ones[value]
        if value < 1000:
            h, rest = divmod(value, 100)
            return (
                self.join([self.data["hundreds_comb"][h - 1], self.ones[rest]])
                if rest
                else self.data["hundreds"][h - 1]
            )
        parts = []
        for index, scale in ((2, 10**7), (1, 10**5), (0, 1000)):
            count, value = divmod(value, scale)
            if not count:
                continue
            combining = bool(value)
            word = self.data["scales"][index][combining]
            if index == 0 and self.code == "ta":
                word = word if count == 1 else self.ones[count][:-1] + "ா" + word[1:]
            elif index == 0 and self.code == "ml":
                word = self.data["thousand_exceptions"].get(str(count))
                if word is None:
                    base = self.ones[count]
                    word = (
                        base[:-1] + ("ിനായിരം" if base.endswith("ത്") else "ായിരം")
                        if base.endswith("്")
                        else base + " " + self.data["scales"][0][0]
                    )
                if combining:
                    word = word[:-1] + "ത്തി"
            elif self.code == "te" and count == 1:
                word = self.data["scale_one"][index][combining]
            elif not (self.code == "kn" and index == 0 and count == 1 and not parts):
                multiplier = (
                    self.data["count_one"]
                    if count == 1 and self.code in ("ta", "ml")
                    else self.cardinal(count)
                )
                word = multiplier + " " + word
            parts.append(word)
        if value:
            parts.append(self.cardinal(value))
        return self.join(parts)

    def year(self, value):
        prefixes = self.data["year_prefix"]
        if prefixes and 1100 <= value <= 1999:
            h, rest = divmod(value, 100)
            if self.code == "te" and not rest:
                return prefixes[h - 11] + "ు"
            return prefixes[h - 11] + (" " + self.ones[rest] if rest else "")
        return self.cardinal(value)

    def digits(self, value, comma=True):
        return (", " if comma else " ").join(self.ones[int(c)] for c in value)

    def oblique(self, words):
        return words[:-1] if self.code == "te" and words.endswith("లు") else words

    def number(self, value):
        value = _ungroup(value)
        sign = (
            self.symbols["minus" if value[0] == "-" else "plus"]
            if value.startswith(("-", "+"))
            else ""
        )
        value = value.lstrip("+-")
        whole, dot, fraction = value.partition(".")
        words = self.cardinal(int(whole))
        if dot:
            words = f"{self.oblique(words)} {self.symbols['point']} {self.digits(fraction, False)}"
        return " ".join(filter(None, (sign, words)))

    def amount(self, words, noun):
        if words == self.ones[1]:
            return self.data["count_one"] + " " + self.data["singular"].get(noun, noun)
        return self.oblique(words) + " " + noun

    def money(self, value, currency, scale=None):
        key = (
            _SYMBOLS.get(currency)
            or {**_CURRENCIES, **self.data["currency_words"]}[currency.lower()]
        )
        noun = self.symbols[key]
        value = _ungroup(value)
        sign, value = (
            (value[0], value[1:]) if value.startswith(("-", "+")) else ("", value)
        )
        whole, dot, fraction = value.partition(".")
        if dot and len(fraction) == 2 and not scale:
            words = self.amount(self.cardinal(int(whole)), noun)
            if int(fraction):
                subunit = self.symbols["paisa" if key == "rupee" else "cent"]
                words += " " + self.amount(self.cardinal(int(fraction)), subunit)
        elif scale:
            index = {"thousand": 0, "lakh": 1, "crore": 2}[
                self.data["scale_words"][scale.lower()]
            ]
            scale_word = (
                self.data["scale_one"][index][0]
                if self.code == "te" and value == "1"
                else self.data["scales"][index][0]
            )
            words = self.amount(self.amount(self.number(value), scale_word), noun)
        else:
            words = self.amount(self.number(value), noun)
        return (
            self.symbols["minus" if sign == "-" else "plus"] + " " if sign else ""
        ) + words

    def date(self, day, month, year=None, month_first=False):
        date(year or 2000, month, day)
        name = self.data["months"][month - 1]
        if self.code == "ta":
            words = f"{name} {self.data['date_days'][day]} தேதி"
        else:
            words = (
                f"{name} {self.cardinal(day)}"
                if month_first and self.code != "en"
                else f"{self.cardinal(day)} {name}"
            )
        return words + (" " + self.year(year) if year else "")

    def time(self, hour, minute, ampm=None, source_period=None):
        if hour > 23 or minute > 59 or (ampm and not 1 <= hour <= 12):
            raise ValueError("Invalid time")
        period = None
        if ampm and not self.data["ampm"] and source_period is None:
            h24 = hour % 12 + (12 if ampm == "pm" else 0)
            period = (
                "morning"
                if 4 <= h24 < 12
                else (
                    "afternoon"
                    if 12 <= h24 < 16
                    else "evening" if 16 <= h24 < 20 else "night"
                )
            )
        if self.code == "te" and 1 <= hour <= 12:
            h = "ఒక" if hour == 1 else self.ones[hour]
            noun = "గంట" if hour == 1 else "గంటలు" if minute == 0 else "గంటల"
            words = f"{h} {noun}" + (
                " ఒక నిమిషం"
                if minute == 1
                else f" {self.ones[minute]} నిమిషాలు" if minute else ""
            )
        elif self.code == "en":
            words = self.ones[hour] + (
                " " + self.ones[minute] if minute else "" if ampm else " o'clock"
            )
            if hour == 0:
                words = "zero " + (self.ones[minute] if minute else "hundred")
        else:
            template = self.data["time_hm" if minute else "time_hour"]
            words = template.format(h=self.ones[hour], m=self.ones[minute])
        if source_period:
            words = source_period + " " + words
        elif period:
            words = self.data["periods"][period] + " " + words
        if ampm and self.data["ampm"]:
            words += " " + self.data["ampm"][ampm]
        return words


@lru_cache(maxsize=12)
def _normalizer(code):
    lang = _Language(code)
    data = lang.data
    months = {
        **_data()["month_aliases"],
        **data["month_aliases"],
        **{m.lower(): i + 1 for i, m in enumerate(data["months"])},
    }
    currencies = _alternatives(
        set(data["currency_words"]) | set(_SYMBOLS) | {"Rs", "Rs.", "INR"}
    )
    scales, month_names = _alternatives(data["scale_words"]), _alternatives(months)
    markers = _alternatives(data["time_markers"])
    periods = _alternatives(data["period_words"])
    patterns = {
        "tag": r"\[[^\[\]\n]*\]",
        "uri": r"[\w.+%-]+@[\w.-]+\.[A-Za-z]{2,}|(?:https?://|www\.)[^\s\[\]]+|(?<![\w.])[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}(?:/[^\s\[\]]*)?",
        "date": r"(?<![\w-])(?:\d{4}-\d{1,2}-\d{1,2}|\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4})(?![\w/-])",
        "named_date": rf"(?<!\w)(?:\d{{1,2}}[ -](?:{month_names})(?!\w)(?:[ ,\-]+\d{{4}})?|(?:{month_names})\s+\d{{1,2}}(?:,?\s+\d{{4}})?)(?!\w)",
        "time": rf"(?:(?<!\w)(?:{periods})\s+)?(?<![\w.])\d{{1,2}}(?::\d{{2}}|\.\d{{2}}(?=\s*[ap]\.?m\.?)|(?=\s*(?:[ap]\.?m\.?|{markers})(?!\w)))(?:\s*[ap]\.?m\.?)?(?:\s*(?:{markers})(?!\w))?(?:\s+(?:{periods})(?!\w))?(?!\d)",
        "money": rf"(?<!\w)[+-]?(?:{currencies})\s*[+-]?{_NUMBER}(?:\s*(?:{scales})(?!\w))?|(?<!\w)[+-]?{_NUMBER}\s+(?:{currencies})(?!\w)",
        "percent": rf"(?<!\w)-?{_NUMBER}\s*%",
        "version": r"(?<!\w)v\d+(?:\.\d+){2,}(?!\w)",
        "identifier": r"(?<!\w)(?:\+\d{1,3}[ -]*)?(?:\d{4}(?:[ -]\d{4}){3}|\d{5}[ -]\d{5}|\d{3}[ -]\d{3}[ -]\d{4})(?!\d|[.,]\d)|(?<!\w)[A-Za-z]+[-\w]*\d[-\w]*(?!\w)|\#\s*\d+",
        "number": rf"(?<![\w.])[+-]?{_NUMBER}(?![\w.]\d)",
    }
    pattern = re.compile(
        "|".join(f"(?P<{name}>{value})" for name, value in patterns.items()),
        re.IGNORECASE,
    )
    return lang, months, pattern


def _uri(text, lang):
    tail = text[len(text.rstrip(".,;!?")) :]
    text = re.sub(r"^https?://", "", text.rstrip(".,;!?"), flags=re.IGNORECASE)
    symbols = {
        "@": "at",
        ".": "dot",
        "_": "underscore",
        "/": "slash",
        "+": "plus",
        "#": "hash",
    }
    parts = []
    for token in re.findall(r"[^\W\d_]+|\d+|[^\w\s]|_", text):
        if token.isdecimal():
            parts.append(lang.digits(token, False))
        elif token in symbols:
            parts.append(lang.symbols[symbols[token]])
        elif token == "-":
            parts.append(lang.data["dash"])
        elif (
            token.lower() == "www"
            or token.lower() in _data()["uri_acronyms"]
            or len(token) == 1
            and token.isascii()
            and token.isalpha()
        ):
            parts.append(" ".join(lang.data["letters"][c] for c in token.upper()))
        else:
            parts.append(token)
    return " ".join(parts) + tail


def normalise_text(text, lang="en"):
    """Normalize supported written forms; preserve prose, tags, and invalid dates."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if lang not in _data()["languages"]:
        raise ValueError(f"Unsupported Maya2 language: {lang}")
    language, months, pattern = _normalizer(lang)
    text = re.sub(
        r"\[[^\[\]\n]*\]|\d",
        lambda m: m[0] if m[0].startswith("[") else str(unicodedata.decimal(m[0])),
        text,
    )
    text = re.sub(
        r"\[[^\[\]\n]*\]|([\w\u0900-\u0dff]+)\s*:\s*(?=\d)",
        lambda m: m[1] + ", " if m[1] and m[1].lower() in _data()["id_words"] else m[0],
        text,
    )

    def replace(match):
        token, kind = match.group(), match.lastgroup
        try:
            if kind == "tag":
                return token
            if kind == "uri":
                return _uri(token, language)
            if kind == "date":
                if len(set(re.findall(r"[/.\-]", token))) != 1:
                    return token
                parts = re.split(r"[/.\-]", token)
                a, b, c = map(int, parts)
                year, month, day = (a, b, c) if len(parts[0]) == 4 else (c, b, a)
                if len(parts[0]) != 4 and len(parts[-1]) == 2:
                    year += 2000 if year <= 50 else 1900
                return language.date(day, month, year)
            if kind == "named_date":
                m = re.fullmatch(r"(\d{1,2})[ -](.+?)(?:[ ,\-]+(\d{4}))?", token)
                month_first = m is None
                if month_first:
                    m = re.fullmatch(r"(.+?)\s+(\d{1,2})(?:,?\s+(\d{4}))?", token)
                    name, day, year = m.groups()
                else:
                    day, name, year = m.groups()
                return language.date(
                    int(day),
                    months[name.lower()],
                    int(year) if year else None,
                    month_first,
                )
            if kind == "time":
                h = re.search(r"(\d{1,2})(?:[.:](\d{2}))?", token)
                ampm = re.search(r"([ap])\.?m\.?", token, re.IGNORECASE)
                period = re.search(
                    rf"(?<!\w)({_alternatives(language.data['period_words'])})(?!\w)",
                    token,
                    re.IGNORECASE,
                )
                words = language.time(
                    int(h[1]),
                    int(h[2] or 0),
                    ampm[1].lower() + "m" if ampm else None,
                    period[0] if period else None,
                )
                for marker, endings in language.data["time_marker_endings"].items():
                    if re.search(re.escape(marker) + r"(?!\w)", token):
                        head, _, last = words.rpartition(" ")
                        return head + " " + endings.get(last, last)
                return words
            if kind == "money":
                number = re.search(_NUMBER, token)
                before, after = (
                    token[: number.start()].strip(),
                    token[number.end() :].strip(),
                )
                currency = before.strip("+- ") or after
                scale = after if before else None
                value = (
                    "-" if "-" in before else "+" if "+" in before else ""
                ) + number.group()
                if not before.strip("+- "):
                    return language.number(value) + " " + after
                return language.money(value, currency, scale or None)
            if kind == "percent":
                return (
                    language.number(token[:-1].strip())
                    + " "
                    + language.symbols["percent"]
                )
            if kind == "version":
                parts = [language.cardinal(int(n)) for n in token[1:].split(".")]
                return (
                    language.data["letters"]["V"]
                    + " "
                    + (" " + language.symbols["point"] + " ").join(parts)
                )
            if kind == "identifier":
                words = []
                for char in token:
                    if char.isdecimal():
                        words.append(language.ones[int(char)])
                    elif char.isascii() and char.isalpha():
                        words.append(language.data["letters"][char.upper()])
                return (
                    language.symbols["plus"] + " " if token.startswith("+") else ""
                ) + (
                    " " if "-" in token and any(c.isalpha() for c in token) else ", "
                ).join(
                    words
                )
            digits = token.replace(",", "").lstrip("+-")
            if "." not in digits and "," not in token:
                before = re.findall(
                    r"[\w\u0900-\u0dff]+",
                    re.split(
                        r"[.!?;।॥\n]",
                        text[max(0, match.start() - 70) : match.start()].lower(),
                    )[-1],
                )[-3:]
                after = re.findall(
                    r"[\w\u0900-\u0dff]+",
                    re.split(
                        r"[.!?;।॥\n]", text[match.end() : match.end() + 50].lower()
                    )[0],
                )[:4]
                context = set(before + after)
                identifier = bool(
                    context.intersection(_data()["id_words"])
                ) and not context.intersection(
                    _data()["label_words"] + language.data["year_words"]
                )
                if (
                    digits.startswith("0")
                    and len(digits) > 1
                    or len(digits) >= 7
                    or identifier
                    and len(digits) >= 3
                ):
                    sign = (
                        language.symbols["minus" if token[0] == "-" else "plus"] + " "
                        if token.startswith(("-", "+"))
                        else ""
                    )
                    return sign + language.digits(digits)
            return language.number(token)
        except (ValueError, OverflowError):
            return token

    return re.sub(
        r"\[[^\[\]\n]*\]|\s+",
        lambda m: m[0] if m[0].startswith("[") else " ",
        pattern.sub(replace, text),
    ).strip()


normalize_text = normalise_text
