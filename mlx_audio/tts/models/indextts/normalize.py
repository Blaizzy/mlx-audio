import re
from typing import Dict, List, Tuple

CHAR_MAP = {
    "：": ",",
    "；": ",",
    ";": ",",
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "\n": " ",
    "·": "-",
    "、": ",",
    "...": "…",
    ",,,": "…",
    "，，，": "…",
    "……": "…",
    """: "'", """: "'",
    '"': "'",
    "'": "'",
    "（": "'",
    "）": "'",
    "(": "'",
    ")": "'",
    "《": "'",
    "》": "'",
    "【": "'",
    "】": "'",
    "[": "'",
    "]": "'",
    "—": "-",
    "～": "-",
    "~": "-",
    "「": "'",
    "」": "'",
    ":": ",",
}

ZH_CHAR_MAP = {"$": ".", **CHAR_MAP}

PINYIN_PATTERN = r"(?<![a-z])((?:[bpmfdtnlgkhjqxzcsryw]|[zcs]h)?(?:[aeiouüv]|[ae]i|u[aio]|ao|ou|i[aue]|[uüv]e|[uvü]ang?|uai|[aeiuv]n|[aeio]ng|ia[no]|i[ao]ng)|ng|er)([1-5])"
NAME_PATTERN = r"[\u4e00-\u9fff]+(?:[-·—][\u4e00-\u9fff]+){1,2}"
CONTRACTION_PATTERN = r"(what|where|who|which|how|t?here|it|s?he|that|this)'s"
EMAIL_PATTERN = r"^[a-zA-Z0-9]+@[a-zA-Z0-9]+\.[a-zA-Z]+$"


def is_email(text: str) -> bool:
    return bool(re.match(EMAIL_PATTERN, text))


def has_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def has_alpha(text: str) -> bool:
    return bool(re.search(r"[a-zA-Z]", text))


def has_pinyin(text: str) -> bool:
    return bool(re.search(PINYIN_PATTERN, text, re.IGNORECASE))


def use_chinese(text: str) -> bool:
    return (
        has_chinese(text) or not has_alpha(text) or is_email(text) or has_pinyin(text)
    )


def replace_chars(text: str, char_map: Dict[str, str]) -> str:
    pattern = re.compile("|".join(re.escape(p) for p in char_map.keys()))
    return pattern.sub(lambda x: char_map[x.group()], text)


def extract_all_digits(text):
    return "".join(filter(str.isdigit, text))


def expand_contractions(text: str) -> str:
    return re.sub(CONTRACTION_PATTERN, r"\1 is", text, flags=re.IGNORECASE)


def correct_pinyin(pinyin: str) -> str:
    if pinyin[0] not in "jqxJQX":
        return pinyin
    return re.sub(
        r"([jqx])[uü](n|e|an)*(\d)", r"\g<1>v\g<2>\g<3>", pinyin, flags=re.IGNORECASE
    ).upper()


def extract_patterns(text: str, pattern: str) -> List[str]:
    matches = re.findall(re.compile(pattern, re.IGNORECASE), text)
    return list(set("".join(m) for m in matches))


def create_placeholders(items: List[str], prefix: str) -> Dict[str, str]:
    return {item: f"<{prefix}_{chr(ord('a') + i)}>" for i, item in enumerate(items)}


def apply_placeholders(text: str, placeholders: Dict[str, str]) -> str:
    result = text
    for original, placeholder in placeholders.items():
        result = result.replace(original, placeholder)
    return result


def restore_placeholders(
    text: str, placeholders: Dict[str, str], transform_fn=None
) -> str:
    result = text
    for original, placeholder in placeholders.items():
        replacement = transform_fn(original) if transform_fn else original
        result = result.replace(placeholder, replacement)
    return result


def save_and_replace(
    text: str, pattern: str, prefix: str
) -> Tuple[str, Dict[str, str]]:
    items = extract_patterns(text, pattern)
    if not items:
        return text, {}
    placeholders = create_placeholders(items, prefix)
    return apply_placeholders(text, placeholders), placeholders


# number normalizers
def number_to_words(n: int):
    ones = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    teens = [
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
    ]
    tens = [
        "",
        "",
        "twenty",
        "thirty",
        "forty",
        "fifty",
        "sixty",
        "seventy",
        "eighty",
        "ninety",
    ]
    thousands = ["", "thousand", "million", "billion", "trillion"]

    def convert_hundreds(num):
        if num == 0:
            return ""
        elif num < 10:
            return ones[num]
        elif num < 20:
            return teens[num - 10]
        elif num < 100:
            return tens[num // 10] + (" " + ones[num % 10] if num % 10 else "")
        else:
            return (
                ones[num // 100]
                + " hundred"
                + (" " + convert_hundreds(num % 100) if num % 100 else "")
            )

    def convert_number(num: int):
        if num == 0:
            return "zero"

        groups = []
        group_idx = 0

        while num > 0:
            group = num % 1000
            if group != 0:
                group_words = convert_hundreds(group)
                if thousands[group_idx]:
                    group_words += " " + thousands[group_idx]
                groups.append(group_words)
            num //= 1000
            group_idx += 1

        return " ".join(reversed(groups))

    return convert_number(n)


# @lru_cache(maxsize=1)
# def get_normalizers():
#     """Lazy load normalizers"""
#     from wetext import Normalizer  # type: ignore

#     return (
#         Normalizer(remove_erhua=False, lang="zh", operator="tn"),
#         Normalizer(lang="en", operator="tn"),
#     )


def normalize_chinese(text: str) -> str:
    # zh_normalizer, _ = get_normalizers()

    text = expand_contractions(text.rstrip())
    text, pinyin_map = save_and_replace(text, PINYIN_PATTERN, "pinyin")
    text, name_map = save_and_replace(text, NAME_PATTERN, "n")

    try:
        result = text  # TODO: improve Chinese normalizers
        # result = zh_normalizer.normalize(text)
    except Exception:
        return ""

    result = restore_placeholders(result, name_map)
    result = restore_placeholders(result, pinyin_map, correct_pinyin)
    result = replace_chars(result, ZH_CHAR_MAP)

    return result


def normalize_english(text: str) -> str:
    # _, en_normalizer = get_normalizers()

    text = expand_contractions(text)

    try:
        # currently dollar only
        def process_currency(match):
            digits = extract_all_digits(match.group(0))
            if not digits:
                return match.group(0)

            num = int(digits)
            word_form = number_to_words(num)

            return f"{word_form} dollar{'s' if num != 1 else ''} "

        text = re.sub(r"\$\s*[0-9,.\s]+", process_currency, text).rstrip()

        def process_digits(match):
            parts = match.group(0).split()
            if all(len(part) == 1 and part.isdigit() for part in parts):
                return " ".join(number_to_words(int(digit)) for digit in parts)
            return number_to_words(int(extract_all_digits(match.group(0))))

        text = re.sub(r"\b\d(\s+\d)+\b", process_digits, text)

        def process_number(match):
            digits = extract_all_digits(match.group(0))
            if digits:
                return number_to_words(int(digits))
            return match.group(0)

        text = re.sub(r"\b\d+(?:,\d+)*\b", process_number, text)

        result = re.sub(r"\s+", " ", text).strip()
    except Exception:
        result = text

    return replace_chars(result, CHAR_MAP)


def normalize(text: str) -> str:
    normalize_fn = normalize_chinese if use_chinese(text) else normalize_english
    return normalize_fn(text)


def tokenize_by_CJK_char(line: str, do_upper_case=True) -> str:
    """
    Tokenize a line of text with CJK char.

    Note: All return charaters will be upper case.

    Example:
      input = "你好世界是 hello world 的中文"
      output = "你 好 世 界 是 HELLO WORLD 的 中 文"

    Args:
      line:
        The input text.

    Return:
      A new string tokenize by CJK char.
    """
    # The CJK ranges is from https://github.com/alvations/nltk/blob/79eed6ddea0d0a2c212c1060b477fc268fec4d4b/nltk/tokenize/util.py
    CJK_RANGE_PATTERN = r"([\u1100-\u11ff\u2e80-\ua4cf\ua840-\uD7AF\uF900-\uFAFF\uFE30-\uFE4F\uFF65-\uFFDC\U00020000-\U0002FFFF])"
    chars = re.split(CJK_RANGE_PATTERN, line.strip())
    return " ".join(
        [w.strip().upper() if do_upper_case else w.strip() for w in chars if w.strip()]
    )
