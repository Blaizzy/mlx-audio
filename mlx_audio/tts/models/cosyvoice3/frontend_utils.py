"""Text-normalization helpers for CosyVoice3."""

import re
from functools import partial
from typing import Callable, List

_chinese_char_pattern = re.compile(r"[一-鿿]+")


def contains_chinese(text: str) -> bool:
    """Return ``True`` if *text* contains any CJK Unified Ideograph."""
    return bool(_chinese_char_pattern.search(text))


def replace_corner_mark(text: str) -> str:
    """Replace superscript corner marks with their spoken forms."""
    text = text.replace("²", "平方")  # ² → 平方
    text = text.replace("³", "立方")  # ³ → 立方
    return text


def remove_bracket(text: str) -> str:
    """Strip full-width brackets / backticks / em-dashes (meaningless for TTS)."""
    text = text.replace("（", "").replace("）", "")  # （  ）
    text = text.replace("【", "").replace("】", "")  # 【  】
    text = text.replace("`", "").replace("`", "")
    text = text.replace("——", " ")  # ——
    return text


def spell_out_number(text: str, inflect_parser=None) -> str:
    """Spell Arabic-numeral runs as English words (e.g. 42 → forty-two).

    Requires ``inflect``. When it is unavailable, numbers are left unchanged.
    """
    if inflect_parser is None:
        try:
            import inflect

            inflect_parser = inflect.engine()
        except ImportError:
            return text

    new_text: List[str] = []
    st = None
    for i, c in enumerate(text):
        if not c.isdigit():
            if st is not None:
                num_str = inflect_parser.number_to_words(text[st:i])
                new_text.append(num_str)
                st = None
            new_text.append(c)
        else:
            if st is None:
                st = i
    if st is not None and st < len(text):
        num_str = inflect_parser.number_to_words(text[st:])
        new_text.append(num_str)
    return "".join(new_text)


def replace_blank(text: str) -> str:
    """Remove spaces that sit between two Chinese characters.

    Spaces between ASCII tokens (or ASCII↔Chinese boundaries) are kept so
    English word boundaries aren't collapsed.
    """
    out_str: List[str] = []
    for i, c in enumerate(text):
        if c == " ":
            if (
                i + 1 < len(text)
                and i - 1 >= 0
                and text[i + 1].isascii()
                and text[i + 1] != " "
                and text[i - 1].isascii()
                and text[i - 1] != " "
            ):
                out_str.append(c)
        else:
            out_str.append(c)
    return "".join(out_str)


def is_only_punctuation(text: str) -> bool:
    """Return ``True`` if *text* consists solely of punctuation / symbols."""
    import regex  # heavier Unicode property support than stdlib ``re``

    return bool(regex.fullmatch(r"^[\p{P}\p{S}]*$", text))


def split_paragraph(
    text: str,
    tokenize: Callable[[str], List[int]],
    lang: str = "zh",
    token_max_n: int = 80,
    token_min_n: int = 60,
    merge_len: int = 20,
    comma_split: bool = False,
):
    """Split *text* into sub-sentences bounded by *token_max_n* / *token_min_n*.

    Parameters match ``cosyvoice.utils.frontend_utils.split_paragraph`` exactly.
    """

    def _calc_utt_length(t: str) -> int:
        if lang == "zh":
            return len(t)
        return len(tokenize(t))

    def _should_merge(t: str) -> bool:
        if lang == "zh":
            return len(t) < merge_len
        return len(tokenize(t)) < merge_len

    if lang == "zh":
        pounc = ["。", "？", "！", "；", "：", "、", ".", "?", "!", ";"]
    else:
        pounc = [".", "?", "!", ";", ":"]
    if comma_split:
        pounc.extend(["，", ","])

    if text[-1] not in pounc:
        text += "。" if lang == "zh" else "."

    st = 0
    utts = []
    for i, c in enumerate(text):
        if c in pounc:
            if len(text[st:i]) > 0:
                utts.append(text[st : i] + c)
            if i + 1 < len(text) and text[i + 1] in ['"', "”"]:
                tmp = utts.pop(-1)
                utts.append(tmp + text[i + 1])
                st = i + 2
            else:
                st = i + 1

    final_utts = []
    cur_utt = ""
    for utt in utts:
        if _calc_utt_length(cur_utt + utt) > token_max_n and _calc_utt_length(cur_utt) > token_min_n:
            final_utts.append(cur_utt)
            cur_utt = ""
        cur_utt = cur_utt + utt
    if len(cur_utt) > 0:
        if _should_merge(cur_utt) and len(final_utts) != 0:
            final_utts[-1] = final_utts[-1] + cur_utt
        else:
            final_utts.append(cur_utt)

    return final_utts
