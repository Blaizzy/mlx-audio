# Vendored from MayaResearch/maya2-native (https://github.com/MayaResearch/maya2-native), Copyright Maya Research,
# licensed under the Apache License 2.0 (see LICENSE in this directory). Modified for mlx-audio: reformatted with black; DurationPrior takes its table as a dict.
"""Maya-2 decode guards, vendored from MayaResearch/maya2-native (maya3/guards.py, Apache-2.0).

The only change: DurationPrior takes the prior table (shipped in the model config) instead of a file path.

Engine-internal guards (zero latency, no user configuration, on by default).

Port of the guards used by the offline reference decoder (DurationPrior, loop_detect, and the FastBatch guard logic), with the
loop detector vectorised over seats (same decision per seat as guards.loop_detect; maya3/tests/test_guards.py).

  1. duration cap   : stop at 2.2 x expected + 1.2 s (digit/pause/language-aware prior), floor 0.5 x old-prior + 8 frames
  2. loop guard     : every 2 frames from frame 32, on the cb0 tail (last 96 frames): a 32-frame stall or a periodic
                      pattern (c[t] == c[t-p] on >= 70 % of the last 64 frames, p = 2..16) ->
                      EOS boost (+4 on the AEOS logit; the model ends the phrase itself), removed when the loop clears;
                      a loop that persists 24 frames after the boost is stopped (loop_backstop).
                      Tag-aware: suspended for [-0.5 s, +3 s] around each inline [tag]'s estimated time.
"""

from __future__ import annotations

import dataclasses
import os
import re

import numpy as np

PAUSE = re.compile(r"[,;:.!?।॥…\-]")
TAG = re.compile(r"\[[a-z ]+\]")
FRAME_S = 0.08


def text_feats(t):
    letters = sum(1 for ch in t if ch.isalpha() or (0x900 <= ord(ch) <= 0xDFF))
    latin = sum(1 for ch in t if ch.isascii() and ch.isalpha())
    return np.array(
        [
            1.0,
            letters - latin,
            latin,
            len(PAUSE.findall(t)),
            sum(ch.isdigit() for ch in t),
        ]
    )


class DurationPrior:
    def __init__(self, table):
        """table: the release's duration_prior.json contents ({"langs": {lang: {"coef": [...]}}})."""
        d = table
        self.coef = {l: np.array(v["coef"]) for l, v in d["langs"].items()}
        self.default = np.mean(list(self.coef.values()), 0)

    def expected_s(self, text, lang):
        return float(max(0.35, text_feats(text) @ self.coef.get(lang, self.default)))

    def cap_frames(self, text, lang, k=2.2, margin_s=1.2):
        return int((k * self.expected_s(text, lang) + margin_s) / FRAME_S)


@dataclasses.dataclass
class GuardConfig:
    enabled: bool = True
    dur_k: float = 2.2
    dur_margin_s: float = 1.2
    loop: bool = True
    stall: int = 32
    period_win: int = 64
    period_thr: float = 0.7
    pmin: int = 2
    pmax: int = 16
    start: int = 32  # first frame the loop rule is evaluated
    every: int = 2
    mode: str = "boost"  # boost | stop
    boost: float = 4.0
    backstop: int = 24
    tag_aware: bool = True
    tag_hold_s: float = 3.0
    hist: int = 96  # cb0 tail kept per seat
    # loop guard v2b windows (suspend_windows_v2): ON by default. Measured against the v1 windows on an eval subset with text
    # normalisation: real_fail_cq 1.16 -> 0.96 %, no-stop 0 -> 0, emotion CER 7.5 -> 5.6 %. MAYA3_GUARD_V2=0 restores v1 tag windows
    aware_v2: bool = os.environ.get("MAYA3_GUARD_V2", "1") == "1"


def tag_windows(text, lang, prior, hold_s=3.0):
    out = []
    for m in TAG.finditer(text):
        t = prior.expected_s(text[: m.start()], lang) if m.start() > 0 else 0.0
        out.append((max(0, int((t - 0.5) / FRAME_S)), int((t + hold_s) / FRAME_S)))
    return tuple(out)


def loop_detect_batch(W, n, cfg: GuardConfig):
    """W: [S, hist] int array, row i = the last `hist` cb0 codes of seat i, right-aligned (newest last; -1 = empty);
    n: [S] frames generated so far. Returns list of None | 'stall' | 'period:<p>' (guards.loop_detect semantics with
    min_frames=12 on the last 96 codes)."""
    S = W.shape[0]
    out = [None] * S
    if S == 0:
        return out
    st = cfg.stall
    tail = W[:, -st:]
    stall = (n >= max(12, st)) & (tail == tail[:, :1]).all(1)
    pw = cfg.period_win
    win = W[:, -pw:]
    can_p = n >= max(12, st, pw)
    hit_p = np.full(S, 0, np.int64)
    if can_p.any():
        for p in range(cfg.pmin, cfg.pmax + 1):
            frac = (win[:, p:] == win[:, :-p]).mean(1)
            new = (hit_p == 0) & can_p & (frac >= cfg.period_thr)
            hit_p[new] = p
    for i in range(S):
        if n[i] < max(12, st):
            continue
        if stall[i]:
            out[i] = "stall"
        elif hit_p[i]:
            out[i] = f"period:{int(hit_p[i])}"
    return out


# ---- loop guard v2 (derived from the remaining eval failures; identical in the reference decoder's guards) ----------
# The v1 tag windows ([-0.5 s, +3 s] around the prior-estimated tag time) miss most backstop cuts: style tags colour all the
# speech after them (whispered / sad / excited speech trips the period rule seconds later), the real tag time drifts from the
# prior (cuts happen at 1.0-2.1 x the piece's expected frames), and fillers / dialogue speaker switches are legitimate pauses.
# v2 windows (no loop action inside; the duration cap still applies):
#   style tag   [0.8 t - 0.5 s, 1.5 t + 4.0 s]   (style_hold_s=None: to the end of the piece; rejected, see below)
#   event tag   [0.8 t - 0.5 s, 1.5 t + 4.0 s]
#   filler      [0.8 t - 0.5 s, 1.5 t + 2.5 s]       hmm / umm / uh / ಹ್ಮ್ / उम्म / ഉം ...
#   speaker     [0.8 t - 0.5 s, 1.5 t + 2.0 s]       "Name: " after a sentence end (dialogue speaker switch)
STYLE_TAGS = {
    "whispers",
    "whispering",
    "sad",
    "excited",
    "curious",
    "shouting",
    "angry",
    "frustrated",
    "sarcastic",
    "crying",
    "nervous",
    "scared",
    "happy",
    "panicked",
    "amazed",
    "worried",
    "calm",
    "serious",
    "surprised",
    "cheerful",
    "tired",
    "fearful",
    "disgusted",
    "confused",
    "sleepy",
    "annoyed",
    "softly",
    "loudly",
}
FILLER_RE = re.compile(
    r"(?<!\w)(?:hmm+|hm+|umm+|um|uh+|uhh|erm|ह्म्म?|हम्म|उम्म?|ಹ್ಮ್|ಉಮ್|ஹ்ம்|ம்ம்|உம்|উমম?|হুম|ଉମ୍|ହମ୍|ਹਮ|ਉਮ|"
    r"હમ્મ|ઉમ|ఉమ్|హ్మ్|ഉം|ഹ്മ്)(?!\w)",
    re.I,
)
SPEAKER_RE = re.compile(r"(?:(?<=[.!?।॥])\s+|\n)([^\s:,.!?।]{1,24}):\s")
TAG_RE_V2 = re.compile(r"\[([a-z _]+)\]")


def suspend_windows_v2(
    text,
    lang,
    prior,
    tag_hold_s=4.0,
    filler_hold_s=2.5,
    switch_hold_s=2.0,
    end_frames=10**6,
    style_hold_s=4.0,
):
    """Frame windows where the v2 loop guard takes no action (see the block comment above). Default (v2b): style tags
    get a bounded window like event tags; style_hold_s=None suspends to the end of the piece (the first v2 variant,
    measured worse: it turned backstop cuts into duration-cap overruns on emotion rows).
    """
    ev = []
    for m in TAG_RE_V2.finditer(text):
        ev.append(
            (
                m.start(),
                style_hold_s if m.group(1).strip() in STYLE_TAGS else tag_hold_s,
            )
        )
    for m in FILLER_RE.finditer(text):
        ev.append((m.start(), filler_hold_s))
    for m in SPEAKER_RE.finditer(text):
        ev.append((m.start(1), switch_hold_s))
    out = []
    for pos, hold in sorted(ev):
        t = prior.expected_s(text[:pos], lang) if pos > 0 else 0.0
        a = max(0, int((0.8 * t - 0.5) / FRAME_S))
        b = end_frames if hold is None else int((1.5 * t + hold) / FRAME_S)
        out.append((a, b))
    return tuple(out)
