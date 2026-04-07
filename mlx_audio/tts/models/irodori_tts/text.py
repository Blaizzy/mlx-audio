from __future__ import annotations

import re
from typing import Optional

import mlx.core as mx
import numpy as np

# ---------------------------------------------------------------------------
# Japanese text normalisation
# Ported from Irodori-TTS/irodori_tts/text_normalization.py (pure Python).
# ---------------------------------------------------------------------------

_REPLACE_MAP: dict[str, str] = {
    r"\t": "",
    r"\[n\]": "",
    r" ": "",  # narrow no-break space (U+202F) / ideographic space handled below
    r"　": "",  # ideographic space
    r"[;▼♀♂《》≪≫①②③④⑤⑥]": "",
    r"[\u02d7\u2010-\u2015\u2043\u2212\u23af\u23e4\u2500\u2501\u2e3a\u2e3b]": "",
    r"[\uff5e\u301C]": "ー",
    r"？": "?",
    r"！": "!",
    r"[●◯〇]": "○",
    r"♥": "♡",
}

# Fullwidth A-Z a-z → halfwidth
_FULLWIDTH_ALPHA_TO_HALFWIDTH = str.maketrans(
    {
        chr(full): chr(half)
        for full, half in zip(
            list(range(0xFF21, 0xFF3B)) + list(range(0xFF41, 0xFF5B)),
            list(range(0x41, 0x5B)) + list(range(0x61, 0x7B)),
        )
    }
)

# Fullwidth 0-9 → halfwidth
_FULLWIDTH_DIGITS_TO_HALFWIDTH = str.maketrans(
    {
        chr(full): chr(half)
        for full, half in zip(range(0xFF10, 0xFF1A), range(0x30, 0x3A))
    }
)

# Halfwidth katakana → fullwidth katakana
_HW_KANA = "ｦｧｨｩｪｫｬｭｮｯｰｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄﾅﾆﾇﾈﾉﾊﾋﾌﾍﾎﾏﾐﾑﾒﾓﾔﾕﾖﾗﾘﾙﾚﾛﾜﾝ"
_FW_KANA = "ヲァィゥェォャュョッーアイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワン"
_HALFWIDTH_KANA_TO_FULLWIDTH = str.maketrans(_HW_KANA, _FW_KANA)


def normalize_text(text: str) -> str:
    """
    Normalise Japanese text for TTS input.
    - Removes noise characters (tabs, special symbols, etc.)
    - Converts fullwidth alphanumerics and digits to halfwidth
    - Converts halfwidth katakana to fullwidth
    - Strips surrounding brackets and trailing punctuation
    """
    for pattern, replacement in _REPLACE_MAP.items():
        text = re.sub(pattern, replacement, text)

    text = text.translate(_FULLWIDTH_ALPHA_TO_HALFWIDTH)
    text = text.translate(_FULLWIDTH_DIGITS_TO_HALFWIDTH)
    text = text.translate(_HALFWIDTH_KANA_TO_FULLWIDTH)

    # Collapse runs of 3+ ellipses to double
    text = re.sub(r"…{3,}", "……", text)

    # Strip surrounding bracket pairs
    for open_br, close_br in [
        ("「", "」"),
        ("『", "』"),
        ("（", "）"),
        ("【", "】"),
        ("(", ")"),
    ]:
        if text.startswith(open_br) and text.endswith(close_br):
            text = text[1:-1]

    # Strip trailing Japanese sentence-ending punctuation
    if text.endswith(("。", "、")):
        text = text.rstrip("。、")

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
