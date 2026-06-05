"""Detokenizer for Nemotron 3.5 ASR (SentencePiece BPE pieces).

The .nemo's tokenizer is SentencePiece BPE (13087 pieces). The converter dumps
the piece list into config `vocabulary`, so detok is dependency-free:
join pieces and turn the SP space marker into spaces. The model emits a trailing
language tag piece (e.g. `<en-US>`); `strip_lang_tag` removes it.
"""

import re

_SP_SPACE = "▁"  # ▁
_LANG_TAG = re.compile(r"\s*<[a-z]{2}(?:-[A-Za-z]{2,3})?>\s*$")


class VocabTokenizer:
    def __init__(self, vocabulary: list[str]):
        self.vocabulary = vocabulary

    def decode(self, ids: list[int]) -> str:
        pieces = [self.vocabulary[i] for i in ids if 0 <= i < len(self.vocabulary)]
        return "".join(pieces).replace(_SP_SPACE, " ").strip()


def strip_lang_tag(text: str) -> tuple[str, str | None]:
    """Return (clean_text, lang_tag) where lang_tag is e.g. 'en-US' or None."""
    m = _LANG_TAG.search(text)
    if not m:
        return text, None
    return _LANG_TAG.sub("", text), m.group(0).strip().strip("<>")
