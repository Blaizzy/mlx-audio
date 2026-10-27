# Vendored from MayaResearch/maya2-native (https://github.com/MayaResearch/maya2-native), Copyright Maya Research,
# licensed under the Apache License 2.0 (see LICENSE in this directory). Modified for mlx-audio: reformatted with black, imports made package-relative.
"""Targeted pre/post rules around the vendored normaliser (maya3.text.tn), derived from the eval rows that still
failed with text normalisation on.

The vendored copy stays byte-identical. Each rule rewrites one input shape that the strict normaliser either rejects
(it then falls back to reading the span character by character: "15-08-25" -> "one five minus zero eight minus two
five") or reads in a way listeners do not expect (a PIN code as "five lakh sixty thousand thirty eight"):

  R1 date_2y      DD-MM-YY, DD/MM/YY, DD.MM.YY            -> four-digit year (pivot: 00-50 -> 20xx, 51-99 -> 19xx),
                                                              then the normaliser's own date reading
  R2 expiry       MM/YY or MM/YYYY next to an expiry / validity / date word or a CVV -> "<month> <year>"
  R3 fy_range     2023-24 (second part = next year)       -> "<year> <two-digit number>"
  R4 digit_groups 4111 2233 4455 6677, 98765 43210, 1800 123 4567 (>= 10 digits in space-separated groups)
                                                           -> one digit run, which the normaliser reads digit by digit
  R5 pin_keyword  PIN / पिन / ... followed by 4-6 digits    -> digit by digit
  R6 pincode      a bare 6-digit number (no currency or unit next to it, no grouping commas) -> digit by digit
  R7 version      v4.5.2                                  -> "v four point five point two" (language's decimal word)
  R8 ta_date_year Tamil "ஆகஸ்ட் 15, 2025"                  -> "15 ஆகஸ்ட் 2025" (the normaliser keeps the year)
  R10 pct_range   3.5%-4%                                 -> 3.5-4% (the normaliser's own percent range)
  R11 per_price   ₹500/kg                                 -> "<amount> <per word> K G"
  R12 per_annum   8.5% p.a.                               -> "P A" (en: "per annum")
  R9 paise        rupee amounts with two decimals read "<n> rupees <m> paise" in every language (the tables of
                  bn en gu kn ml mr pa ta had no paise word, so they read "point five zero rupees")
  post:
  P1 tag_comma    a piece that ends in "," before an inline [tag] keeps its comma (the normaliser turns a trailing
                  comma into a full stop, and the model then ends the sentence at the tag)
  P2 uri_dash     "-" inside an e-mail / URL is spoken in the request language, not as the English word "dash"
  P3 uri_acronym  lowercase acronym labels in an e-mail / URL (sbi, kyc, hdfc in hdfcbank, sbi in onlinesbi) are
                  spelled letter by letter in the request language; other labels keep their English spelling
Language-table corrections (Assamese lexicon, "per" units, sub-domain hosts) live in maya3.text.lexicon.
"""

from __future__ import annotations

import re

__all__ = ["pre", "post", "install", "RULES"]

RULES = (
    "date_2y",
    "expiry",
    "fy_range",
    "digit_groups",
    "pin_keyword",
    "pincode",
    "version",
    "ta_date_year",
    "paise",
    "pct_range",
    "per_price",
    "per_annum",
    "tag_comma",
    "uri_dash",
    "uri_acronym",
)

PAISE = {
    "en": "paise",
    "bn": "পয়সা",
    "gu": "પૈસા",
    "kn": "ಪೈಸೆ",
    "ml": "പൈസ",
    "mr": "पैसे",
    "pa": "ਪੈਸੇ",
    "ta": "பைசா",
}  # hi as or te already have one
DASH = {
    "hi": "डैश",
    "mr": "डॅश",
    "bn": "ড্যাশ",
    "as": "ডেশ্ব",
    "gu": "ડૅશ",
    "pa": "ਡੈਸ਼",
    "or": "ଡ୍ୟାସ୍",
    "ta": "டாஷ்",
    "te": "డాష్",
    "kn": "ಡ್ಯಾಶ್",
    "ml": "ഡാഷ്",
}

# expiry / validity / date words (any language; matched case-insensitively in the 40 characters before the span)
_EXPIRY_WORDS = [
    "expiry",
    "expires",
    "expire",
    "exp",
    "valid",
    "validity",
    "thru",
    "expiration",
    "एक्सपायरी",
    "समाप्ति",
    "वैधता",
    "मुदत",
    "समाप्ती",
    "মেয়াদ",
    "মেয়াদ",
    "এক্সপায়ারি",
    "এক্সপায়ারি",
    "এক্সপায়াৰী",
    "ম্যাদ",
    "એક્સપાયરી",
    "સમાપ્તિ",
    "માન્યતા",
    "ਮਿਆਦ",
    "ਐਕਸਪਾਇਰੀ",
    "காலாவதி",
    "எக்ஸ்பைரி",
    "గడువు",
    "ఎక్స్‌పైరీ",
    "ఎక్స్పైరీ",
    "ಮುಕ್ತಾಯ",
    "ಅಂತಿಮ ದಿನಾಂಕ",
    "ಎಕ್ಸ್‌ಪೈರಿ",
    "കാലാവധി",
    "എക്സ്പയറി",
    "ମିଆଦ",
    "ଏକ୍ସପାୟାରୀ",
    "ସମାପ୍ତି",
]
_DATE_WORDS = [
    "date",
    "तारीख",
    "तिथि",
    "दिनांक",
    "डेट",
    "তারিখ",
    "তাৰিখ",
    "તારીખ",
    "ડેટ",
    "ਤਾਰੀਖ਼",
    "ਤਾਰੀਖ",
    "ਡੇਟ",
    "தேதி",
    "டேட்",
    "తేదీ",
    "డేట్",
    "ದಿನಾಂಕ",
    "ಡೇಟ್",
    "തീയതി",
    "ഡേറ്റ്",
    "ତାରିଖ",
    "ଡେଟ୍",
]
_alt = lambda ws: "|".join(sorted(map(re.escape, ws), key=len, reverse=True))
_EXPIRY_RE = re.compile(_alt(_EXPIRY_WORDS), re.I)
_DATE_RE = re.compile(_alt(_DATE_WORDS), re.I)
_PIN_WORDS = [
    "pin",
    "पिन",
    "पिनकोड",
    "ಪಿನ್",
    "పిన్",
    "பின்",
    "പിൻ",
    "পিন",
    "ପିନ",
    "ਪਿੰਨ",
    "ਪਿਨ",
    "પિન",
]
_PIN_RE = re.compile(
    r"(?<![\w])(?:"
    + "|".join(map(re.escape, _PIN_WORDS))
    + r")(?:[\s-]*(?:code|कोड|কোড|કોડ|ਕੋਡ|କୋଡ|ಕೋಡ್|కోడ్|கோட்|കോഡ്))?"
    r"(?:\s*(?:no\.?|number|नंबर|नम्बर))?\s*[:\-]?\s*(\d{4,6})(?![\d,]|\.\d)",
    re.I,
)
# words right after a number that make it an amount / quantity (R6 must not read those digit by digit)
_AMOUNT_AFTER = re.compile(
    r"\s*(?:rupees?|rs\.?|inr|dollars?|usd|paise|lakh|crore|thousand|people|users|customers|km|kg|kms|"
    r"रुपये|रुपए|रुपया|रु\.?|लाख|करोड़|हज़ार|हजार|लोग|লাখ|টাকা|টকা|কোটি|રૂપિયા|લાખ|ਰੁਪਏ|ਲੱਖ|ரூபாய்|ரூ\.?|லட்சம்|"
    r"రూపాయలు|రూ\.?|లక్షల|ರೂಪಾಯಿ|ರೂ\.?|ಲಕ್ಷ|രൂപ|ലക്ഷം|ଟଙ୍କା|ଲକ୍ଷ|ਕਰੋੜ|कोटी)(?![\w])",
    re.I,
)
_CURRENCY_BEFORE = re.compile(
    r"(?:₹|\$|€|£|\brs\.?|\binr|रु\.?|ரூ\.?|రూ\.?|ರೂ\.?)\s*$", re.I
)
_NUMBER_WORD_BEFORE = re.compile(
    r"(?:\bno\.?|\bnumber|\bplot|\bflat|\bhouse|\bdoor|\bh\.\s?no\.?|नंबर|नम्बर|क्रमांक|नं\.?|নং|নম্বর|নম্বৰ|ਨੰਬਰ|નંબર|ନମ୍ବର|"
    r"எண்|నంబర్|ಸಂಖ್ಯೆ|ನಂಬರ್|നമ്പർ)\s*[:\-]?\s*$",
    re.I,
)
_TA_MONTHS = [
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
_URI_RE = re.compile(
    r"https?://\S+|www\.\S+|\b[\w.%+\-]+@[\w.\-]+\.[A-Za-z]{2,}|\b[A-Za-z][\w.\-]*@[A-Za-z][A-Za-z0-9]{1,15}\b|\b[A-Za-z][\w\-]*(?:\.[A-Za-z0-9][\w\-]*)*\.(?:com|org|net|ai|io|co|dev|app|edu|gov|in)(?:\.(?:in|uk|us))?(?:[/?#][^\s<>]*)?"
)

# P3: lowercase acronym labels inside e-mail / URL hosts and paths are spelled letter by letter (sbi.co.in ->
# "S B I dot co dot in", hdfcbank.com -> "H D F C bank dot com"); other labels keep their English spelling.
_URI_ACRONYMS = {
    "sbi",
    "hdfc",
    "icici",
    "kyc",
    "irctc",
    "pnb",
    "rbi",
    "idbi",
    "npci",
    "uidai",
    "epfo",
    "mf",
    "lic",
    "upi",
    "gst",
    "sms",
    "atm",
    "emi",
    "bob",
    "boi",
    "uco",
    "iob",
    "nsdl",
    "cdsl",
    "pnr",
    "otp",
    "ifsc",
}
_URI_ACR_PREFIX = (
    "icici",
    "irctc",
    "uidai",
    "hdfc",
    "idbi",
    "npci",
    "epfo",
    "sbi",
    "pnb",
    "kyc",
    "rbi",
)
_URI_ACR_SUFFIX = ("sbi", "mf")

_SPELL_UNITS = {
    "kg",
    "km",
    "gm",
    "g",
    "ml",
    "ltr",
    "l",
}  # R11: read letter by letter (as the normaliser reads "15 kg")

_installed = False


def install(on: bool = True):
    """R9: add the paise word to the language tables that lack it (bn en gu kn ml mr pa ta), so the normaliser's own
    main+sub-unit path reads ₹95.41 as "<n> rupees <m> paise" (with its native scale words and case suffixes).
    The tables are module-level; install(False) removes the added words again (used only by rules=False, the
    eval-parity mode; do not mix rules=False with concurrent rules=True calls)."""
    global _installed
    if on == _installed:
        return
    from .tn.normalize import language_table

    for code, word in PAISE.items():
        sym = language_table(code).SYM
        if on:
            sym.setdefault("paisa", word)
        elif sym.get("paisa") == word:
            del sym["paisa"]
    _installed = on


def _table(lang):
    from .tn.normalize import language_table

    return language_table(lang)


def _tn_id_context(L, before):
    """True when the normaliser itself will read the next number digit by digit (an ID keyword such as PIN / OTP /
    number in the three tokens before it); the rule then leaves the span alone so the normaliser's postposition
    sandhi ("...ಒಂದರಲ್ಲಿ") is kept."""
    import sys

    from .tn import normalize as _N  # the function; its module holds the keyword sets

    mod = sys.modules[__package__ + ".tn.normalize"]
    kws = (
        mod._ID_KEYWORDS_EN | mod._OTP_INPUT_WORDS | {w.lower() for w in L.ID_KEYWORDS}
    )
    toks = re.findall(r"[\w#\u0900-\u0DFF']+", before.replace("\u200c", "").lower())[
        -3:
    ]
    return any(t in kws for t in toks)


def _digits(L, s):
    return ", ".join(L.digit(int(c)) for c in s)


def _year4(yy):
    y = int(yy)
    return 2000 + y if y <= 50 else 1900 + y


def _decimal_word(L):
    w = L.decimal(1, "2")
    a, b = L.cardinal(1), L.digit(2)
    return (
        w[len(a) : len(w) - len(b)].strip()
        if w.startswith(a) and w.endswith(b)
        else "point"
    )


def pre(text: str, lang: str) -> str:
    """Apply R1-R8 outside e-mail / URL spans (those are identifiers, spelled by the normaliser)."""
    try:
        L = _table(lang)
    except Exception:
        return text
    out, pos = [], 0
    for m in _URI_RE.finditer(text):
        out.append(_pre(text[pos : m.start()], lang, L))
        out.append(m.group(0))
        pos = m.end()
    out.append(_pre(text[pos:], lang, L))
    return "".join(out)


def _pre(text, lang, L):
    t = text

    # R10 percent ranges written with the sign twice: 3.5%-4% (the normaliser read "...percentminus four percent")
    t = re.sub(
        r"(?<![\d.])(\d+(?:\.\d+)?)\s?%\s?[-–]\s?(\d+(?:\.\d+)?)\s?%", r"\1-\2%", t
    )

    # R11 price per unit: ₹500/kg (the "/" was dropped and "kg" left in Latin letters)
    def r11(m):
        unit, per = m.group(2), L.SYM.get("per") or "per"
        if lang != "en" and unit.lower() in _SPELL_UNITS:
            unit = " ".join(L.letter(c) for c in unit.upper())
        return f"{m.group(1)} {per} {unit}"

    t = re.sub(
        r"((?:[₹$€£]|\b(?:Rs\.?|INR)\s?)\s?\d[\d,]*(?:\.\d+)?)\s?/\s?(kg|km|gm|g|ml|ltr|l|day|month|night|year|person)\b",
        r11,
        t,
        flags=re.I,
    )

    # R12 "p.a." (per annum) after a rate was left as the raw token "p.a."
    t = re.sub(
        r"(?<![\w.])[pP]\.\s?[aA]\.?(?![\w])",
        lambda m: "per annum" if lang == "en" else f"{L.letter('P')} {L.letter('A')}",
        t,
    )

    # R7 version strings
    def r7(m):
        parts = m.group(1).split(".")
        dw = _decimal_word(L)
        return f"{L.letter('v')} " + f" {dw} ".join(L.cardinal(int(p)) for p in parts)

    t = re.sub(r"(?<![\w.])[vV](\d+(?:\.\d+){1,3})(?![\w]|\.\d)", r7, t)

    # R4 space-separated digit groups (card / phone numbers): join into one run (>= 10 digits in total)
    def r4(m):
        groups = m.group(0).split()
        return "".join(groups) if sum(len(g) for g in groups) >= 10 else m.group(0)

    t = re.sub(
        r"(?<![\d,.])(?:\d{4,5}(?: \d{3,5}){1,3}|\d{2,4} \d{3,4} \d{4})(?![\d,]|\.\d)",
        r4,
        t,
    )

    # R1 dates with a two-digit year
    def r1(m):
        d, sep, mo, yy = m.group(1), m.group(2), m.group(3), m.group(4)
        if not (1 <= int(d) <= 31 and 1 <= int(mo) <= 12):
            return m.group(0)
        if (
            sep == "-" and len(d) == 1 and len(mo) == 1
        ):  # 8-2-45: a door number, not a date
            return m.group(0)
        if _NUMBER_WORD_BEFORE.search(
            t[max(0, m.start() - 16) : m.start()]
        ):  # "House No. 12-5-24", "ঘৰ নং ..."
            return m.group(0)
        return f"{d}{sep}{mo}{sep}{_year4(yy)}"

    t = re.sub(
        r"(?<![\d/.\-])(\d{1,2})([/.\-])(\d{1,2})\2(\d{2})(?![\d/\-]|\.\d)", r1, t
    )

    # R2 card expiry / month-year next to an expiry, date or CVV word
    def r2(m):
        mo, y = int(m.group(1)), m.group(2)
        before = t[max(0, m.start() - 40) : m.start()]
        after = t[m.end() : m.end() + 40]
        strong = _EXPIRY_RE.search(before) or re.search(
            r"\bcvv\b", before + " " + after, re.I
        )
        weak = _DATE_RE.search(before) and (
            len(y) == 4 or int(y) > 12
        )  # second part > 12: not DD/MM, so MM/YY
        if not (strong or weak):
            return m.group(0)
        year = int(y) if len(y) == 4 else 2000 + int(y)
        return f"{L.month(mo)} {L.year(year)}"

    t = re.sub(r"(?<![\d/.\-])(0?[1-9]|1[0-2])/(\d{2}|20\d{2})(?![\d/]|\.\d)", r2, t)

    # R3 financial-year ranges 2023-24
    def r3(m):
        y, yy = int(m.group(1)), int(m.group(2))
        if (y + 1) % 100 != yy:
            return m.group(0)
        return f"{L.year(y)} {L.cardinal(yy)}"

    t = re.sub(r"(?<![\d\-])((?:19|20)\d{2})[-–](\d{2})(?![\d\-])", r3, t)

    # R5 PIN keyword + digits
    def r5(m):
        if _tn_id_context(L, t[max(0, m.start(1) - 60) : m.start(1)]):
            return m.group(0)
        return m.group(0)[: m.start(1) - m.start(0)] + _digits(L, m.group(1))

    t = _PIN_RE.sub(r5, t)

    # R6 bare six-digit numbers (Indian PIN codes): digit by digit unless an amount / quantity
    def r6(m):
        before = t[max(0, m.start() - 8) : m.start()]
        if (
            _CURRENCY_BEFORE.search(before)
            or _AMOUNT_AFTER.match(t, m.end())
            or re.search(
                r"[A-Za-z0-9]\s?-\s?$|[#÷×*/+=<>]\s*$", before
            )  # ID suffix (CUST-567890), #ref, maths
            or re.match(r"\s*[÷×*/+=<>^]", t[m.end() :])
            or _tn_id_context(
                L, re.split(r"[.!?;।॥\n]", t[max(0, m.start() - 60) : m.start()])[-1]
            )
        ):
            return m.group(0)
        return _digits(L, m.group(0))

    t = re.sub(r"(?<![\d,.₹$#\w])[1-9]\d{5}(?![\d,]|\.\d|\s*%)", r6, t)

    # R8 Tamil "Month D, YYYY" -> "D Month YYYY"
    if lang == "ta":
        months = "|".join(_TA_MONTHS)
        t = re.sub(
            rf"(?<!\w)({months})\s+(\d{{1,2}}),?\s+(\d{{4}})(?!\d)", r"\2 \1 \3", t
        )
    return t


TAG_COMMA = False  # P1 off: no measured gain (6 vs 6 backstop cuts on the 14 affected emotion rows x2 seeds)


def post(out: str, seg: str, lang: str, before_tag: bool) -> str:
    # P1 keep a trailing comma that precedes an inline [tag] (disabled, see TAG_COMMA)
    if (
        TAG_COMMA
        and before_tag
        and seg.rstrip().endswith(",")
        and out.rstrip().endswith(".")
    ):
        out = out.rstrip()[:-1] + ","
    # P2 spoken hyphen inside e-mail / URL in the request language
    uris = [m.group(0) for m in _URI_RE.finditer(seg)]
    if lang in DASH and uris:
        out = re.sub(r"(?<=\s)dash(?=\s)", DASH[lang], out)
    # P3 acronym labels inside e-mail / URL
    labels = {
        w
        for u in uris
        for w in re.findall(r"[a-z]+", re.sub(r"^https?://", "", u))
        if w in _URI_ACRONYMS
        or w.startswith(_URI_ACR_PREFIX)
        or w.endswith(_URI_ACR_SUFFIX)
    }
    if labels:
        try:
            L = _table(lang)
        except Exception:
            return out
        spell = lambda w: " ".join(L.letter(c.upper()) for c in w)

        def p3(m):
            w = m.group(0)
            if w in _URI_ACRONYMS:
                return spell(w)
            for a in _URI_ACR_PREFIX:
                if w.startswith(a) and len(w) - len(a) >= 3:
                    return spell(a) + " " + w[len(a) :]
            for a in _URI_ACR_SUFFIX:
                if w.endswith(a) and len(w) - len(a) >= 4:
                    return w[: -len(a)] + " " + spell(a)
            return w

        out = re.sub(
            r"(?<![\w.@/])(?:"
            + "|".join(sorted(map(re.escape, labels), key=len, reverse=True))
            + r")(?![\w]|[.@/\-]\w)",
            p3,
            out,
        )
    return out
