# Portions derived from MayaResearch/maya2-native (https://github.com/MayaResearch/maya2-native), Copyright Maya
# Research, licensed under the Apache License 2.0 (see LICENSE in this directory).
"""Maya-2 prompts without PyTorch.

Ports of MayaResearch/maya2-native (Apache-2.0):
  * the token contract and tokenizer extension (tts/tokens.py),
  * the span layout of PackerV2._span and the span geometry of collate_v2 (tts/data/packing_v2.py), reduced to what a
    serving prompt needs: tokens, kinds and codes up to the target's ``<|audio|>`` sentinel, plus every span's sentinel
    column (where the speaker anchor is added),
  * the prompt builders (maya3/prompt.py: V5Builder, NamedSpeakerBuilder),
  * the per-language text -> frames prior and the long-text splitter (scripts/s2_serve_controls.py).

Layout of one span: ``[<|lang:xx|> if first] text <|audio|> frames AEOS``. A serving prompt is the reference spans of
a cloned voice (if any) followed by the target span cut right after its ``<|audio|>`` sentinel.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

N_Q = 32
N_CODES = 2048
AEOS_CLASS = N_CODES  # class 2048 of the cb0 head; never a text token
FPS = 12.5
LANGS = ["as", "bn", "en", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"]
KIND_TEXT, KIND_AUDIO, KIND_AEOS = 0, 1, 3

# appended to the backbone tokenizer in exactly this order (frozen by the checkpoint)
ADDED_TOKENS = (
    ["<|spk|>", "<|audio|>", "<|emo|>", "<|instruct|>", "<|/instruct|>", "<|cont|>"]
    + [f"<|lang:{l}|>" for l in LANGS]
    + [f"<|ctl:{i:02d}|>" for i in range(32)]
)

# conditioning mode -> (reference clips in context, ECAPA speaker memory)
MODES = {
    "plain": (False, False),
    "spk": (False, True),
    "icl": (True, False),
    "icl_spk": (True, True),
    "named": (False, False),
}
NAMED_TEMPLATE = "{{{name}}}: {text}"

PRIOR_DEFAULT = {"a": 2.0, "b": 0.72, "resid_p99": 90.0}
TERMINALS = "।.!?;।॥"


def extend_tokenizer(tokenizer):
    """Append the contract tokens; return (tokenizer, name -> id). Idempotent."""
    tokenizer.add_tokens(ADDED_TOKENS, special_tokens=True)
    ids = {t: tokenizer.convert_tokens_to_ids(t) for t in ADDED_TOKENS}
    missing = [t for t, i in ids.items() if i is None or i < 0]
    if missing:
        raise ValueError(f"tokenizer failed to register: {missing}")
    return tokenizer, ids


@dataclass
class Voice:
    """A voice: a trained name (``named``), plain, or a reference clip (``icl`` / ``icl_spk`` / ``spk``).

    clips: [{"text", "lang", "codes" (32, F) int}]; ecapa: (192,) unit vector or None.
    Model-side caches (prefix KV, depth cross-attention constants) are filled on first use.
    """

    id: str
    mode: str = "plain"
    clips: List[dict] = field(default_factory=list)
    ecapa: Optional[np.ndarray] = None
    name: Optional[str] = None
    template: str = NAMED_TEMPLATE
    meta: dict = field(default_factory=dict)
    cache: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(
                f"unknown conditioning mode {self.mode!r}; choose one of {sorted(MODES)}"
            )
        use_icl, use_mem = MODES[self.mode]
        if use_icl and not self.clips:
            raise ValueError(
                f"voice {self.id}: mode {self.mode} needs a reference clip"
            )
        if use_mem and self.ecapa is None:
            raise ValueError(
                f"voice {self.id}: mode {self.mode} needs an ECAPA embedding"
            )

    @property
    def use_icl(self):
        return MODES[self.mode][0]

    @property
    def use_mem(self):
        return MODES[self.mode][1]

    def render(self, text):
        return (
            self.template.format(name=self.name or self.id, text=text.strip())
            if self.mode == "named"
            else text
        )


def load_named_voices(voices: dict) -> Dict[str, Voice]:
    """voices.json of the release ({"template"?, "voices": [...]}) -> {lowercase name: Voice}."""
    items = voices.get("voices", voices) if isinstance(voices, dict) else voices
    template = (
        voices.get("template") if isinstance(voices, dict) else None
    ) or NAMED_TEMPLATE
    out = {}
    for it in items:
        name = it.get("name") or it.get("display_name") or it.get("id")
        if (it.get("mode") or "named") != "named":
            continue  # the release only ships named voices; reference-clip voices are registered at runtime
        meta = {
            k: v for k, v in it.items() if k not in ("ecapa", "centroid", "ref_clips")
        }
        out[name.lower()] = Voice(
            name, "named", name=name, template=template, meta=meta
        )
    return out


@dataclass
class Prompt:
    tokens: np.ndarray  # (T,) int, token ids (0 at audio positions)
    kind: np.ndarray  # (T,) int8, KIND_*
    codes: np.ndarray  # (T, 32) int, -1 where unused
    sentinels: List[
        int
    ]  # column of every span's <|audio|> sentinel (speaker-anchor positions)
    prefix_len: (
        int  # tokens that belong to the voice's reference spans (cacheable per voice)
    )


class Prompter:
    def __init__(self, tokenizer, ids):
        self.tok, self.ids = tokenizer, ids

    def span(self, text, lang, codes, first):
        """PackerV2._span (cont=False): returns (tokens, kind, codes, n_text)."""
        head = [self.ids[f"<|lang:{lang}|>"]] if first else []
        if not isinstance(text, str) or not text.strip():
            raise ValueError("empty text")
        head += self.tok(text, add_special_tokens=False)["input_ids"]
        head.append(self.ids["<|audio|>"])
        n_text, frames = len(head), codes.shape[1]
        length = n_text + frames + 1
        t = np.zeros(length, np.int64)
        t[:n_text] = head
        k = np.zeros(length, np.int8)
        k[n_text : n_text + frames] = KIND_AUDIO
        k[-1] = KIND_AEOS
        c = np.full((length, N_Q), -1, np.int64)
        c[n_text : n_text + frames] = codes.T
        c[-1, 0] = AEOS_CLASS
        return t, k, c, n_text

    def build(self, voice: Voice, text: str, lang: str) -> Prompt:
        """The serving prompt of V5Builder.example + prompt_from_example (q = 32), cut after the target sentinel."""
        spans = []
        if voice.use_icl:
            for i, clip in enumerate(voice.clips):
                spans.append(
                    self.span(
                        clip["text"],
                        clip["lang"],
                        np.asarray(clip["codes"], np.int64),
                        first=(i == 0),
                    )
                )
        prefix_len = sum(len(s[0]) for s in spans)
        # the target span gets its own language tag when it opens the row, or when the references are in another language
        xling = bool(spans) and any(c["lang"] != lang for c in voice.clips)
        target = self.span(
            voice.render(text),
            lang,
            np.zeros((N_Q, 1), np.int64),
            first=(not spans) or xling,
        )
        spans.append(target)
        tokens = np.concatenate([s[0] for s in spans])
        kind = np.concatenate([s[1] for s in spans])
        codes = np.concatenate([s[2] for s in spans])
        sentinels, at = [], 0
        for t, _, _, n_text in spans:
            sentinels.append(at + n_text - 1)
            at += len(t)
        cut = (
            prefix_len + target[3]
        )  # the target's first audio column; the prompt ends at its sentinel
        return Prompt(tokens[:cut], kind[:cut], codes[:cut], sentinels, prefix_len)


# ----------------------------------------------------------------------------- text -> frames prior, splitting
def load_frames_prior(table: dict) -> dict:
    return {
        lang: {
            "a": float(d["affine_a"]),
            "b": float(d["affine_b"]),
            "resid_p99": float(d["resid_p99"]),
        }
        for lang, d in (table or {}).items()
    }


def expected_frames(prior, lang, nchar):
    p = prior.get(lang, PRIOR_DEFAULT)
    return p["a"] + p["b"] * nchar, p["resid_p99"]


def split_text(text, lang, prior, chunk_s):
    """Split at sentence terminals into pieces of about chunk_s seconds (prior-estimated); falls back to commas, then
    whitespace, for run-on text. Byte-for-byte the reference splitter."""
    if chunk_s <= 0:
        return [text]
    target = chunk_s * FPS
    pieces = [
        p.strip()
        for p in re.split(f"(?<=[{re.escape(TERMINALS)}])\\s+", text)
        if p.strip()
    ]
    out, cur = [], ""
    for p in pieces:
        cand = (cur + " " + p).strip() if cur else p
        if cur and expected_frames(prior, lang, len(cand))[0] > target:
            out.append(cur)
            cur = p
        else:
            cur = cand
    if cur:
        out.append(cur)
    final = []
    for piece in out:
        if expected_frames(prior, lang, len(piece))[0] <= target * 1.3:
            final.append(piece)
            continue
        words = re.split(r"(?<=[,،])\s+|\s+", piece)
        cur = ""
        for w in words:
            cand = (cur + " " + w).strip() if cur else w
            if cur and expected_frames(prior, lang, len(cand))[0] > target:
                final.append(cur)
                cur = w
            else:
                cur = cand
        if cur:
            final.append(cur)
    return final
