# Vendored from MayaResearch/maya2-native (https://github.com/MayaResearch/maya2-native), Copyright Maya Research,
# licensed under the Apache License 2.0 (see LICENSE in this directory). Modified for mlx-audio: reformatted with black, imports made package-relative.
"""Language-table corrections applied around the vendored normaliser (maya3.text.tn), found by a multi-language
normaliser benchmark.

The vendored copy stays byte-identical. `install(True)` (called by maya3.text.normalise_text with rules on) patches
the module-level language tables / one Normalizer method; `install(False)` restores the originals exactly, so
`normalise_text(..., rules=False)` keeps the eval-time behaviour.

  L1 as_lexicon  Assamese inherited Bengali grammar tables: Bengali र (U+09B0) instead of Assamese ৰ (U+09F0) in the
                 number words (ত্রিশ -> ত্ৰিশ), Bengali ordinals / symbol words / letter names / clock words
                 (চতুর্থ, অ্যাট দ্য রেট, আর, সকাল, দুপুর, রাত, "{h}টা", "এর বর্গ" ...). Replaced with Assamese forms;
                 Assamese clock markers (বজাত / বজাৰ / বজালৈ) and period words (পুৱা, আবেলি, বিয়লি, ৰাতি ...)
                 are recognised, and postpositions glue to the number noun (পঞ্চাশ টকা ৰ -> পঞ্চাশ টকাৰ).
  L2 per_unit    "km/h", "MB/s": the per word was the English "per" in bn as gu kn ml mr pa ta, and the
                 denominator was a spelled letter ("কে এম per এইচ"); now the language's per word + hour / second.
  L3 uri_hosts   bare domains with sub-domains (e-receipt.hdfcbank.com, netbanking.nationalbank.in) were only
                 matched from the last label, so "e-receipt.hdfcbank" stayed raw; a bare domain that ends a sentence
                 ("irctc.co.in.") was not matched at all; the local part of a UPI ID (rahul.deshmukh@okaxis) stayed
                 raw. All three are now spelled like other URLs.
"""

from __future__ import annotations

import re
import types

__all__ = ["install", "AS_ONES"]

_R, _RR = "র", "ৰ"  # Bengali ra, Assamese ra


def _as_script(s):
    return s.replace(_R, _RR) if isinstance(s, str) else s


# Assamese 0-99: AI4Bharat indic-numtowords data (tn/assamese_numbers.py) with the Bengali ra fixed to ৰ
# (ঊনত্রিশ..আঠত্রিশ -> ঊনত্ৰিশ..আঠত্ৰিশ) and two malformed entries corrected: 45 পঁঞ্চল্লিশ -> পঞ্চল্লিশ,
# 71 এসত্তৰ -> একসত্তৰ.
def _as_ones():
    from .tn.assamese_numbers import ONES

    ones = [_as_script(w) for w in ONES]
    ones[45] = "পঞ্চল্লিশ"
    ones[71] = "একসত্তৰ"
    return ones


AS_ONES = None  # filled on first install

_AS_SYM = {
    "at": "এট দ্য ৰেট",
    "underscore": "আণ্ডাৰস্ক'ৰ",
    "slash": "শ্লেচ",
    "hash": "হেচ",
    "amp": "এণ্ড",
    "squared": "বৰ্গ",
    "sqrt": "বৰ্গমূল",
    "lt": "তকৈ কম",
    "gt": "তকৈ বেছি",
    "half": "আধা",
    "quarter": "এক চতুৰ্থাংশ",
    "three_quarters": "তিনি চতুৰ্থাংশ",
    "degree": "ডিগ্ৰী",
    "celsius": "ডিগ্ৰী চেলছিয়াছ",
    "fahrenheit": "ডিগ্ৰী ফাৰেনহাইট",
    "second": "ছেকেণ্ড",
    "cent": "চেণ্ট",
    "per": "প্ৰতি",
}
_AS_PERIOD_DETECT = {
    "পুৱা": "morning",
    "পুৱাৰ": "morning",
    "পুৱাতে": "morning",
    "ৰাতিপুৱা": "morning",
    "দুপৰীয়া": "afternoon",
    "দুপৰীয়াৰ": "afternoon",
    "দুপৰীয়াত": "afternoon",
    "আবেলি": "evening",
    "আবেলিৰ": "evening",
    "আবেলিতে": "evening",
    "বিয়লি": "evening",
    "বিয়লিৰ": "evening",
    "সন্ধিয়া": "evening",
    "সন্ধিয়াৰ": "evening",
    "সন্ধিয়াতে": "evening",
    "ৰাতি": "night",
    "ৰাতিৰ": "night",
    "ৰাতিতে": "night",
}
_AS_ATTACH = {
    "টকা",
    "পইচা",
    "ডলাৰ",
    "শতাংশ",
    "মিনিট",
    "ঘণ্টা",
    "ছেকেণ্ড",
    "বজা",
    "ডিগ্ৰী",
    "কিলোমিটাৰ",
}
_AS_PP = ("ৰ", "ত", "লৈ", "তকৈ", "ৰে", "তে", "লৈকে", "ৰপৰা")


def _as_attach(self, rep, tok):
    """Assamese case endings glue to the noun without sandhi: টকা+ৰ -> টকাৰ, মিনিট+ত -> মিনিটত."""
    t = tok.lstrip("-")
    last = rep.rsplit(" ", 1)[-1]
    if last not in _AS_ATTACH or t not in _AS_PP:
        return None
    return rep + t


_PER = {
    "bn": "প্রতি",
    "gu": "પ્રતિ",
    "mr": "प्रति",
    "pa": "ਪ੍ਰਤੀ",
    "ta": "பெர்",
    "kn": "ಪರ್",
    "ml": "പെർ",
}  # hi or te already have one; as: _AS_SYM
_PER_UNITS = {  # denominators of X/h, X/s (H, S are not units on their own, so only "km/h"-style reads change)
    "en": {"H": "hour", "S": "second"},
    "hi": {"H": "घंटा", "S": "सेकंड"},
    "mr": {"H": "तास", "S": "सेकंद"},
    "bn": {"H": "ঘণ্টা", "S": "সেকেন্ড"},
    "as": {"H": "ঘণ্টা", "S": "ছেকেণ্ড"},
    "gu": {"H": "કલાક", "S": "સેકન્ડ"},
    "pa": {"H": "ਘੰਟਾ", "S": "ਸਕਿੰਟ"},
    "or": {"H": "ଘଣ୍ଟା", "S": "ସେକେଣ୍ଡ"},
    "ta": {"H": "ஹவர்", "S": "செகண்ட்"},
    "te": {"S": "సెకండ్"},  # te already has H
    "kn": {"H": "ಅವರ್", "S": "ಸೆಕೆಂಡ್"},
    "ml": {"H": "അവർ", "S": "സെക്കൻഡ്"},
}

_saved = []  # (restore callable) in apply order
_installed = False


def _set_attr(obj, name, value):
    had = name in vars(obj)
    old = vars(obj).get(name)
    setattr(obj, name, value)
    _saved.append(lambda: setattr(obj, name, old) if had else delattr(obj, name))


def _set_key(d, key, value):
    had = key in d
    old = d.get(key)
    d[key] = value
    _saved.append(lambda: d.__setitem__(key, old) if had else d.pop(key, None))


def _patch_url(N):
    orig = N._m_url

    def _m_url(self, text):
        # as the vendored _m_url, but a bare host may have sub-domain labels (e-receipt.hdfcbank.com) and may end a
        # sentence ("... irctc.co.in." was left unspelled: the vendored lookahead rejected any following ".")
        pat = (
            r"https?://[^\s<>]+|www\.[^\s<>]+"
            r"|\b[A-Za-z][\w\-]*(?:\.[A-Za-z0-9][\w\-]*)*\.(?:com|org|net|ai|io|co|dev|app|edu|gov|in)"
            r"(?:\.(?:in|uk|us))?(?:[/?#][^\s<>]*)?(?![\w]|\.\w)"
        )
        for m in re.finditer(pat, text):
            token = m.group(0).rstrip(".,!?;)")
            body = re.sub(r"^https?://", "", token)
            yield m.start(), m.start() + len(token), self._spell_uri_part(body)
        # UPI IDs with a dotted local part (rahul.deshmukh@okaxis) were left raw (the "." unspoken); spell the ID
        for m in re.finditer(
            r"(?<![\w.@])[A-Za-z][\w\-]*\.[\w.\-]*@[A-Za-z][A-Za-z0-9]{1,15}(?![\w@]|\.\w)",
            text,
        ):
            yield m.start(), m.end(), self._spell_uri_part(m.group(0))

    N._m_url = _m_url
    _saved.append(lambda: setattr(N, "_m_url", orig))


def install(on: bool = True):
    global _installed, AS_ONES
    if on == _installed:
        return
    import sys

    from .tn import normalize as _normalize_module  # noqa: F401

    mod = sys.modules[__package__ + ".tn.normalize"]
    if on:
        L = mod.language_table("as")
        AS_ONES = AS_ONES or _as_ones()
        _set_attr(L, "ONES", AS_ONES)
        _set_attr(
            L,
            "ORDINAL_EXCEPTIONS",
            {
                1: "প্ৰথম",
                2: "দ্বিতীয়",
                3: "তৃতীয়",
                4: "চতুৰ্থ",
                5: "পঞ্চম",
                6: "ষষ্ঠ",
                7: "সপ্তম",
                8: "অষ্টম",
                9: "নৱম",
                10: "দশম",
            },
        )
        _set_attr(
            L,
            "LETTERS",
            {**{k: _as_script(v) for k, v in L.LETTERS.items()}, "S": "এছ", "Y": "ৱাই"},
        )
        for k, v in list(L.SYM.items()):
            if _as_script(v) != v:
                _set_key(L.SYM, k, _as_script(v))
        for k, v in _AS_SYM.items():
            _set_key(L.SYM, k, v)
        _set_attr(
            L,
            "PERIODS",
            {
                "morning": "পুৱা",
                "afternoon": "দুপৰীয়া",
                "evening": "সন্ধিয়া",
                "night": "ৰাতি",
            },
        )
        _set_attr(L, "PERIOD_DETECT", {**L.PERIOD_DETECT, **_AS_PERIOD_DETECT})
        _set_attr(L, "TIME_ON_HOUR", "{h} বজা")
        # Bengali "টা" is a counting classifier in Assamese (2023 টা অভিযোগ), not an o'clock marker
        _set_attr(L, "TIME_MARKERS", ["বাজে", "বজাত", "বজাৰ", "বজালৈ", "বজাতকৈ", "বজা"])
        _set_attr(
            L,
            "MARKER_SUFFIXES",
            {"বজাত": "ত", "বজাৰ": "ৰ", "বজালৈ": "লৈ", "বজাতকৈ": "তকৈ"},
        )
        _set_attr(L, "POWER", "{base}ৰ ঘাত {exp}")
        _set_attr(L, "DECADE_SHORT", "{tens}ৰ দশক")
        _set_attr(L, "DECADE_FULL", "{tens}ৰ দশক")
        _set_attr(
            L,
            "ABBREV_UNITS",
            {
                "MINS": "মিনিট",
                "MIN": "মিনিট",
                "HRS": "ঘণ্টা",
                "HR": "ঘণ্টা",
                "SECS": "ছেকেণ্ড",
                "SEC": "ছেকেণ্ড",
                "KMS": "কিলোমিটাৰ",
            },
        )
        _set_attr(L, "attach_postposition", types.MethodType(_as_attach, L))
        for code, word in _PER.items():
            sym = mod.language_table(code).SYM
            if "per" not in sym:
                _set_key(sym, "per", word)
        for code, words in _PER_UNITS.items():
            T = mod.language_table(code)
            uw = dict(T.UNIT_WORDS)
            for k, w in words.items():
                uw.setdefault(k, w)
            _set_attr(T, "UNIT_WORDS", uw)
        _patch_url(mod.Normalizer)
    else:
        while _saved:
            _saved.pop()()
    mod._engine.cache_clear()  # regexes are built from the tables
    _installed = on
