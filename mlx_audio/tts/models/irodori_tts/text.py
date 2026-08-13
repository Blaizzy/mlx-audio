from __future__ import annotations

import re
import unicodedata
from typing import Optional

import mlx.core as mx
import numpy as np

# ---------------------------------------------------------------------------
# Japanese text normalisation
# Ported from Irodori-TTS/irodori_tts/text_normalization.py (pure Python).
# ---------------------------------------------------------------------------

_SIMPLE_REPLACE_MAP: dict[str, str] = {
    "\t": "",
    "[n]": "",
    r"\[n\]": "",
    "\u3000": "",  # ideographic space
    "？": "?",
    "！": "!",
    "♥": "♡",
    "●": "○",
    "◯": "○",
    "〇": "○",
}

_REGEX_REPLACE_MAP: dict[re.Pattern[str], str] = {
    re.compile(r"[;▼♀♂《》≪≫①②③④⑤⑥]"): "",
    re.compile(
        r"[\u02d7\u2010-\u2015\u2043\u2212\u23af\u23e4\u2500\u2501\u2e3a\u2e3b]"
    ): "",
    re.compile(r"[\uff5e\u301C]"): "ー",
    re.compile(r"…{3,}"): "……",
}

_BRACKET_PAIRS = {"「": "」", "『": "』", "（": "）", "【": "】", "(": ")"}


def strip_outer_brackets(text: str) -> str:
    """
    Remove bracket pairs that enclose the whole string, repeatedly.

    Depth tracking matters: in 「前半」と「後半」 the leading 「 closes before the
    end, so nothing is stripped.
    """
    while True:
        if len(text) < 2:
            break

        start_char = text[0]
        end_char = text[-1]

        if start_char in _BRACKET_PAIRS and _BRACKET_PAIRS[start_char] == end_char:
            depth = 0
            is_enclosing_all = True

            for i, char in enumerate(text):
                if char == start_char:
                    depth += 1
                elif char == end_char:
                    depth -= 1

                if depth == 0 and i < len(text) - 1:
                    is_enclosing_all = False
                    break

            if is_enclosing_all and depth == 0:
                text = text[1:-1]
                continue

        break

    return text


def normalize_text(text: str) -> str:
    """
    Normalise Japanese text for TTS input.

    Mirrors Irodori-TTS/irodori_tts/text_normalization.py step for step,
    including the NFKC pass that folds fullwidth alphanumerics, halfwidth
    kana and characters such as ㈱ or Ⅲ.
    """
    for old, new in _SIMPLE_REPLACE_MAP.items():
        text = text.replace(old, new)

    for pattern, replacement in _REGEX_REPLACE_MAP.items():
        text = pattern.sub(replacement, text)

    text = strip_outer_brackets(text)

    text = unicodedata.normalize("NFKC", text)

    text = text.replace("...", "…")
    text = text.replace("..", "…")

    return text


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------


def encode_text(
    text: str,
    tokenizer,
    max_length: int,
    add_bos: bool = True,
) -> tuple[mx.array, mx.array]:
    """
    Tokenise a single text string using a HuggingFace tokenizer.

    Matches Irodori's PretrainedTextTokenizer behaviour:
      - special tokens are NOT added by the HF tokenizer
      - BOS is prepended manually when add_bos=True
      - right-padding to max_length with pad_token_id

    Returns
    -------
    input_ids : mx.array  shape (1, max_length)  int32
    mask      : mx.array  shape (1, max_length)  bool
    """
    # Ensure right-padding (tokenizer default may differ)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            raise ValueError(
                "Tokenizer has no pad_token_id. Set a pad token before inference."
            )

    token_ids: list[int] = tokenizer.encode(text, add_special_tokens=False)

    if add_bos:
        if tokenizer.bos_token_id is None:
            raise ValueError("Tokenizer has no bos_token_id but add_bos=True.")
        token_ids.insert(0, int(tokenizer.bos_token_id))

    # Truncate
    token_ids = token_ids[:max_length]
    n = len(token_ids)

    # Pad
    pad_id = int(tokenizer.pad_token_id)
    padded = token_ids + [pad_id] * (max_length - n)

    ids_np = np.array([padded], dtype=np.int32)
    mask_np = np.zeros((1, max_length), dtype=bool)
    mask_np[0, :n] = True

    return mx.array(ids_np), mx.array(mask_np)
