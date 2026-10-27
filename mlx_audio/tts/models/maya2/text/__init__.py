# Vendored from MayaResearch/maya2-native (https://github.com/MayaResearch/maya2-native), Copyright Maya Research,
# licensed under the Apache License 2.0 (see LICENSE in this directory). Modified for mlx-audio: reformatted with black, imports made package-relative.
"""Text normalisation for Maya-2, vendored from MayaResearch/maya2-native (maya3/text, Apache-2.0).

Imports were made package-relative for mlx-audio; the logic is unchanged.

Text normalisation for Maya3 (numbers, currency, dates, OTPs, URLs, ... -> spoken words, per language).

    from mlx_audio.tts.models.maya2.text import normalise_text
    normalise_text("Your OTP is 482913 [laughs]", "en")   # 'Your OTP is four, eight, two, nine, one, three [laughs]'

Same behaviour as the normalise_text wrapper the release evals used:
  * inline [tags] (e.g. [laughs], [sighs]) are protected: the text between them is normalised piece by piece;
  * ALL-CAPS Latin words that look like names (>= 5 letters with >= 2 vowels, or 4 letters with >= 2 vowels) are
    title-cased first, so the normaliser and the model read them as words; short / vowel-poor tokens (OTP, NEFT,
    HDFC, KYC) stay acronyms;
  * each piece goes through the bundled normaliser maya3.text.tn (pure Python, standard library only);
  * if the normaliser raises on a piece, that piece is kept unchanged (the eval wrapper's second fallback, a
    separate benchmark pipeline, is not part of the release);
  * whitespace is collapsed and the result stripped;
  * `rules=True` (default) adds the targeted pre/post rules of maya3.text.rules (two-digit-year dates, card expiry,
    FY ranges, spaced card/phone digits, PIN codes, version strings, rupees+paise in every language, the comma
    before an inline [tag]). `rules=False` is the exact behaviour of the release evals before these rules were added.
`lang` is a Maya3 language code (as bn en gu hi kn ml mr or pa ta te); for an unknown code the normaliser raises,
so the text comes back with only the caps fold and whitespace collapse applied.
"""

from __future__ import annotations

import logging
import re

from . import lexicon as _lexicon
from . import rules as _rules

__all__ = ["normalise_text", "normalize_text", "SOURCE"]

SOURCE = "Maya Research text normaliser, upstream commit fe5e6ec"

_TAG = re.compile(r"(\[[a-z ]+\])")
_CAPS = re.compile(r"\b[A-Z]{4,}\b")
_VOWELS = set("AEIOU")
_TN = None

log = logging.getLogger("maya3.text")


def _caps_fold(t):
    def f(m):
        w = m.group(0)
        if (
            len(w) >= 5
            and sum(ch in _VOWELS for ch in w) >= 2
            or (len(w) == 4 and sum(ch in _VOWELS for ch in w) >= 2)
        ):
            return w.title()
        return w

    return _CAPS.sub(f, t)


def _load_tn():
    global _TN
    if _TN is None:
        from .tn.normalize import normalize

        _TN = normalize
    return _TN


def normalise_text(text: str, lang: str, rules: bool = True) -> str:
    tn = _load_tn()
    _rules.install(rules)
    _lexicon.install(rules)
    out = []
    segs = _TAG.split(text)
    for i, seg in enumerate(segs):
        if not seg or _TAG.fullmatch(seg):
            out.append(seg)
            continue
        s = _caps_fold(seg)
        try:
            s2 = tn(_rules.pre(s, lang) if rules else s, lang)
            if rules:
                s2 = _rules.post(
                    s2,
                    seg,
                    lang,
                    before_tag=any(_TAG.fullmatch(x) for x in segs[i + 1 : i + 2]),
                )
        except (
            Exception
        ) as exc:  # never fail a request on normalisation: keep this piece as written
            log.debug("normaliser fallback (%s: %s)", type(exc).__name__, exc)
            s2 = s
        lead = seg[: len(seg) - len(seg.lstrip())]
        trail = seg[len(seg.rstrip()) :]
        out.append(lead + s2.strip() + trail)
    return re.sub(r"\s+", " ", "".join(out)).strip()


normalize_text = normalise_text
