# Vendored from MayaResearch/maya2-native (https://github.com/MayaResearch/maya2-native), Copyright Maya Research,
# licensed under the Apache License 2.0 (see LICENSE in this directory). Modified for mlx-audio: reformatted with black.
"""Prosody-aware, Indic-aware, emotion-tag-safe chunker for TTS.

Python port of the TypeScript chunker (studio gateway, itself a port of
indic_chunker.py on branch local-chunker-normguard) that passes the I1-I6
invariant harness. Replaces the earlier sentence-only chunker, which had two
defects:
  * it split on any '.', '!' or '?', tearing "8.4", "example.com", "₹2,500"
    and quoted '"आओगे?"' apart;
  * it split ONLY on sentence punctuation, so a long comma-joined sentence
    became one oversized chunk that silently overran max_model_len and
    truncated mid-speech.

Splits normalized text preferring, in priority order:
  1. sentence boundaries  ( . ! ? … । ॥ — ONLY token-final, i.e. followed by
     whitespace, so decimals / URLs / currency / quoted '?' stay intact )
  2. clause boundaries    ( ; : , — –  and pan-Indic connector words )
  3. word boundaries      ( never splits a word / Devanagari cluster )

Emotion tags ([laughs], [clears throat], …) are atomic and glued to the word
they condition, so a boundary can never strand a tag into the wrong clip.

BUDGET: MAX chars is sized in SPEAKING SECONDS, but the real ceiling is the
engine's context window: a chunk costs ~50 audio tokens/sec plus ~1.0 (Hindi)
to ~1.5 (Telugu) prompt tokens/char. At max_model_len=1024 a chunk must stay
under ~13 s / ~180 chars; 25 s / 347 chars needs max_model_len=2048. Set
CHUNK_MAX_CHARS to match the deployed window.
"""

import os
import re

_DASH_ELLIPSIS = dict.fromkeys(map(ord, "―‒⸺⸻"), " ")  # bar/figure/2-3em dash -> space


# (em/en dash and … kept: clause cues)
def post_process(s):
    if not s:
        return s
    s = s.translate(_DASH_ELLIPSIS)
    s = re.sub(r"\.{2,}", "…", s)  # ASCII ".." / "..." -> ellipsis (a sentence cue)
    s = re.sub(r"\s+([,.;:!?।])", r"\1", s)  # drop space before punctuation
    s = re.sub(r"\s{2,}", " ", s)  # collapse runs of spaces
    return s.strip()


SPC = 0.060  # measured speaking rate, s/char: 40 clips,
# 10 languages x 2 voices, mean 0.058
# (ta/ml ~0.051, or ~0.062 -- 1.54x spread).
# 0.072 was a guess and over-split by ~20%.
MAX = int(
    os.environ.get("CHUNK_MAX_CHARS", str(round(20 / SPC)))
)  # 333 -> ~20.0 s (fits max_model_len=2048)
MIN = int(os.environ.get("CHUNK_MIN_CHARS", "60"))  # ~4.3 s: merge runts
HARD = MAX  # absolute ceiling (except a lone long token)
CHUNK_GAP_S = 0.15  # silence inserted between concatenated chunks
SECS_PER_CHAR = SPC  # back-compat alias
CHUNK_MAX_SEC = MAX * SPC  # back-compat alias

TAG_G = re.compile(r"\[[a-z][a-z ]*\]")  # bracketed lowercase = TTS tag
SENT_END = re.compile(r"(?:[.!?।॥…]|\.{3})+[\"'”’»)\]]*$")
CLAUSE_END = re.compile(r"[,;:—–][\"'”’»)\]]*$")
_WS = re.compile(r"\s+")
STRIP = "\"'“”‘’«»()[],.;:!?।॥…-–—"
STRIP_LEAD = "\"'“”‘’«»("

CONNECTORS = {
    "and",
    "but",
    "because",
    "while",
    "so",
    "then",
    "और",
    "एवं",
    "तथा",
    "लेकिन",
    "पर",
    "मगर",
    "क्योंकि",
    "जबकि",
    "फिर",
    "आणि",
    "पण",
    "कारण",
    "এবং",
    "আর",
    "কিন্তু",
    "কারণ",
    "তারপর",
    "ଏବଂ",
    "କିନ୍ତୁ",
    "અને",
    "પણ",
    "કારણ",
    "ਅਤੇ",
    "ਪਰ",
    "ਕਿਉਂਕਿ",
    "మరియు",
    "కానీ",
    "ఎందుకంటే",
    "ಮತ್ತು",
    "ಆದರೆ",
    "ಏಕೆಂದರೆ",
    "மற்றும்",
    "ஆனால்",
    "ஏனெனில்",
    "പക്ഷേ",
    "എന്നാൽ",
    "കാരണം",
}


def _strip(s, chars):
    return s.strip(chars)


def _groups(para):
    """Whitespace tokens -> atomic groups, with emotion tags glued to their word."""
    prot = TAG_G.sub(lambda m: m.group(0).replace(" ", "\x00"), para)
    toks = [t.replace("\x00", " ") for t in prot.split() if t]
    out, glue_right = [], False
    for t in toks:
        bare = _strip(TAG_G.sub("", t).strip(), STRIP)
        is_tag = bare == ""  # token is only tag(+punctuation)
        if glue_right:
            out[-1] = out[-1] + " " + t
            glue_right = is_tag  # consecutive tags keep gluing
            continue
        if is_tag and out and not SENT_END.search(out[-1]):
            out[-1] = out[-1] + " " + t  # mid/end tag -> preceding word
        elif is_tag:
            out.append(t)  # sentence-leading tag -> next word
            glue_right = True
        else:
            out.append(t)
    return out


def _strength(group, nxt):
    """Boundary strength AFTER this group: 3 sentence, 2 clause, 1 word."""
    if SENT_END.search(group):
        return 3
    if CLAUSE_END.search(group):
        return 2
    if nxt is not None:
        parts = nxt.split()
        first = _strip(parts[0], STRIP_LEAD) if parts else ""
        if first in CONNECTORS:
            return 2  # break BEFORE a connector word
    return 1


def _runs(gs, min_strength):
    """Join groups into runs, breaking where boundary strength >= min_strength."""
    out, buf = [], []
    for i, g in enumerate(gs):
        buf.append(g)
        nxt = gs[i + 1] if i + 1 < len(gs) else None
        if nxt is not None and _strength(g, nxt) >= min_strength:
            out.append(" ".join(buf))
            buf = []
    if buf:
        out.append(" ".join(buf))
    return out


def _split_words(clause):
    out, buf = [], ""
    for g in _groups(clause):
        cand = (buf + " " + g) if buf else g
        if len(cand) <= HARD or not buf:
            buf = cand
        else:
            out.append(buf)
            buf = g
    if buf:
        out.append(buf)
    return out


def _split_long(sent):
    out, buf = [], ""
    for clause in _runs(_groups(sent), 2):
        if len(clause) > MAX:
            if buf:
                out.append(buf)
                buf = ""
            out.extend(_split_words(clause))
            continue
        cand = (buf + " " + clause) if buf else clause
        if len(cand) <= MAX:
            buf = cand
        else:
            out.append(buf)
            buf = clause
    if buf:
        out.append(buf)
    return out


def _chunk_groups(gs):
    out, buf = [], ""
    for sent in _runs(gs, 3):
        if len(sent) > MAX:
            if buf:
                out.append(buf)
                buf = ""
            out.extend(_split_long(sent))
            continue
        cand = (buf + " " + sent) if buf else sent
        if len(cand) <= MAX:
            buf = cand
        else:
            out.append(buf)
            buf = sent
    if buf:
        out.append(buf)
    return out


def _merge_runts(pieces):
    if not pieces:
        return []
    merged = []
    for p in pieces:
        if merged and len(p) < MIN and len(merged[-1] + " " + p) <= HARD:
            merged[-1] = merged[-1] + " " + p
        else:
            merged.append(p)
    if (
        len(merged) >= 2
        and len(merged[0]) < MIN
        and len(merged[0] + " " + merged[1]) <= HARD
    ):
        merged[1] = merged[0] + " " + merged[1]
        merged.pop(0)
    return [_WS.sub(" ", m).strip() for m in merged]


def split_for_tts(text, max_sec=None, spc=SPC, hard_sec=None):
    """Split normalized text into prosody-friendly, tag-safe TTS chunks.

    max_sec/spc/hard_sec are accepted for call-site compatibility with the old
    chunker; sizing is governed by MAX/MIN/HARD (chars) above.
    """
    if not text or not text.strip():
        return []
    pieces = []
    for para in text.split("\n"):
        p = para.strip()
        if not p:
            continue
        pieces.extend(_chunk_groups(_groups(p)))
    return _merge_runts(pieces) or [text.strip()]
