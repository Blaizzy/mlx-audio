"""
Text processing pipeline for MeloTTS English.

Pipeline: text -> normalize -> G2P (phones, tones, word2ph) -> symbol IDs + BERT features.
"""

import re
from typing import List, Tuple

import mlx.core as mx

# ---------------------------------------------------------------------------
# Symbols — loaded from model config.json at runtime via load_symbols_from_config().
# The defaults below are minimal English-only fallbacks.
# ---------------------------------------------------------------------------

punctuation = ["!", "?", "...", ",", ".", "'", "-"]
pu_symbols = punctuation + ["SP", "UNK"]
pad = "_"

en_symbols = [
    "aa", "ae", "ah", "ao", "aw", "ay",
    "b", "ch", "d", "dh",
    "eh", "er", "ey",
    "f", "g", "hh",
    "ih", "iy",
    "jh", "k", "l", "m", "n", "ng",
    "ow", "oy",
    "p", "r", "s", "sh",
    "t", "th",
    "uh", "uw",
    "V",  # uppercase V is intentional in MeloTTS
    "w", "y", "z", "zh",
]

# Default symbol table (English-only fallback; overwritten by load_symbols_from_config)
symbols = [pad] + sorted(set(en_symbols)) + pu_symbols
_symbol_to_id = {s: i for i, s in enumerate(symbols)}


def load_symbols_from_config(config_symbols):
    """Update symbol table from model config.json symbols list."""
    global symbols, _symbol_to_id
    symbols = config_symbols
    _symbol_to_id = {s: i for i, s in enumerate(symbols)}


language_id_map = {
    "ZH": 0, "JP": 1, "EN": 2, "ZH_MIX_EN": 3,
    "KR": 4, "ES": 5, "SP": 5, "FR": 6,
}

language_tone_start_map = {
    "ZH": 0, "ZH_MIX_EN": 0,
    "JP": 6, "EN": 7,
    "KR": 11, "ES": 12, "SP": 12,
    "FR": 13,
}

arpa = {
    "AH0", "S", "AH1", "EY2", "AE2", "EH0", "OW2", "UH0", "NG",
    "B", "G", "AY0", "M", "AA0", "F", "AO0", "ER2", "UH1", "IY1",
    "AH2", "DH", "IY0", "EY1", "IH0", "K", "N", "W", "IY2", "T",
    "AA1", "ER1", "EH2", "OY0", "UH2", "UW1", "Z", "AW2", "AW1",
    "V", "UW2", "AA2", "ER", "AW0", "UW0", "R", "OW1", "EH1",
    "ZH", "AE0", "IH2", "IH1", "OY2", "JH", "EY0", "AE1", "OW0",
    "AY1", "TH", "HH", "P", "SH", "CH", "AO1", "OY1", "AO2",
    "IH", "UW", "AY2", "AY", "EH", "L", "ER0", "D", "AE",
}


_ones = [
    "", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen",
]
_tens = [
    "", "", "twenty", "thirty", "forty", "fifty",
    "sixty", "seventy", "eighty", "ninety",
]


def _number_to_words(n: int) -> str:
    """Convert integer to English words (simplified, up to 999,999)."""
    if n == 0:
        return "zero"
    if n < 0:
        return "minus " + _number_to_words(-n)
    parts = []
    if n >= 1000:
        parts.append(_number_to_words(n // 1000) + " thousand")
        n %= 1000
    if n >= 100:
        parts.append(_ones[n // 100] + " hundred")
        n %= 100
    if n >= 20:
        word = _tens[n // 10]
        if n % 10:
            word += " " + _ones[n % 10]
        parts.append(word)
    elif n > 0:
        parts.append(_ones[n])
    return " ".join(parts)


_comma_number_re = re.compile(r"(\d{1,3}(,\d{3})+)")
_decimal_number_re = re.compile(r"(\d+\.\d+)")
_number_re = re.compile(r"\d+")


def _remove_commas(m):
    return m.group(0).replace(",", "")


def _expand_decimal(m):
    num = m.group(0)
    parts = num.split(".")
    integer = _number_to_words(int(parts[0]))
    decimal = " ".join(_ones[int(d)] if int(d) < 20 else str(d) for d in parts[1])
    return integer + " point " + decimal


def _expand_number(m):
    return _number_to_words(int(m.group(0)))


_abbreviations = [
    (re.compile(r"\b%s\." % x[0], re.IGNORECASE), x[1])
    for x in [
        ("mrs", "missis"),
        ("mr", "mister"),
        ("dr", "doctor"),
        ("st", "saint"),
        ("co", "company"),
        ("jr", "junior"),
        ("maj", "major"),
        ("gen", "general"),
        ("drs", "doctors"),
        ("rev", "reverend"),
        ("lt", "lieutenant"),
        ("hon", "honorable"),
        ("sgt", "sergeant"),
        ("capt", "captain"),
        ("esq", "esquire"),
        ("ltd", "limited"),
        ("col", "colonel"),
        ("ft", "fort"),
    ]
]


def text_normalize(text: str) -> str:
    """Normalize English text for G2P."""
    text = text.lower()
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    text = re.sub(_comma_number_re, _remove_commas, text)
    text = re.sub(_decimal_number_re, _expand_decimal, text)
    text = re.sub(_number_re, _expand_number, text)
    return text


_g2p_instance = None
_eng_dict = None
_tokenizer = None


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    return _tokenizer


def _get_g2p():
    global _g2p_instance
    if _g2p_instance is None:
        from g2p_en import G2p
        _g2p_instance = G2p()
    return _g2p_instance


def _get_eng_dict():
    """Load CMU pronouncing dictionary."""
    global _eng_dict
    if _eng_dict is not None:
        return _eng_dict

    try:
        import cmudict
        _eng_dict = cmudict.dict()
    except ImportError:
        # Fallback: use g2p_en's built-in CMU dict
        g2p = _get_g2p()
        _eng_dict = g2p.cmu if hasattr(g2p, "cmu") else {}

    return _eng_dict


def _refine_ph(phn: str) -> Tuple[str, int]:
    """Convert ARPAbet phone to (lowercase_phone, tone)."""
    if phn[-1].isdigit():
        return phn[:-1].lower(), int(phn[-1]) + 1
    return phn.lower(), 0


def _post_replace_ph(ph: str) -> str:
    """Map phone to symbol table, handling special cases."""
    if ph == "v":
        return "V"
    if ph in _symbol_to_id:
        return ph
    if ph in punctuation:
        return ph
    return "UNK"


def _distribute_phone(n_phone: int, n_word: int) -> List[int]:
    """Distribute phones evenly across wordpiece sub-tokens."""
    phones_per_word = [0] * n_word
    for _ in range(n_phone):
        min_val = min(phones_per_word)
        min_idx = phones_per_word.index(min_val)
        phones_per_word[min_idx] += 1
    return phones_per_word


def g2p(text: str, pad_start_end: bool = True) -> Tuple[List[str], List[int], List[int]]:
    """Convert normalized English text to phones, tones, and word2ph alignment."""
    tokenizer = _get_tokenizer()
    g2p_fn = _get_g2p()
    eng_dict = _get_eng_dict()

    tokenized = tokenizer.tokenize(text)

    ph_groups = []
    for t in tokenized:
        if not t.startswith("##"):
            ph_groups.append([t])
        else:
            ph_groups[-1].append(t.replace("##", ""))

    phones = []
    tones = []
    word2ph = []

    for group in ph_groups:
        word = "".join(group)
        word_len = len(group)
        phone_len = 0

        upper = word.upper()
        if upper in eng_dict:
            pron = eng_dict[upper]
            if isinstance(pron, list):
                pron = pron[0]
            for ph in pron:
                p, t = _refine_ph(ph)
                phones.append(p)
                tones.append(t)
                phone_len += 1
        else:
            phone_list = [p for p in g2p_fn(word) if p != " "]
            for ph in phone_list:
                if ph in arpa:
                    p, t = _refine_ph(ph)
                    phones.append(p)
                    tones.append(t)
                else:
                    phones.append(ph)
                    tones.append(0)
                phone_len += 1

        word2ph += _distribute_phone(phone_len, word_len)

    phones = [_post_replace_ph(p) for p in phones]

    if pad_start_end:
        phones = ["_"] + phones + ["_"]
        tones = [0] + tones + [0]
        word2ph = [1] + word2ph + [1]

    return phones, tones, word2ph


def cleaned_text_to_sequence(
    phones: List[str],
    tones: List[int],
    language: str = "EN",
) -> Tuple[List[int], List[int], List[int]]:
    """Convert phone symbols to integer IDs with tone and language offsets."""
    phone_ids = [_symbol_to_id.get(p, _symbol_to_id["UNK"]) for p in phones]
    tone_start = language_tone_start_map[language]
    tone_ids = [t + tone_start for t in tones]
    lang_id = language_id_map[language]
    lang_ids = [lang_id] * len(phone_ids)
    return phone_ids, tone_ids, lang_ids


def get_bert_features(
    text: str,
    word2ph: List[int],
    bert_model,
    add_blank: bool = True,
) -> mx.array:
    """Extract phone-level BERT features. Returns (768, num_phones)."""
    tokenizer = _get_tokenizer()

    inputs = tokenizer(text, return_tensors="np")
    input_ids = mx.array(inputs["input_ids"])
    token_type_ids = mx.array(inputs.get("token_type_ids", [[0] * input_ids.shape[1]]))
    attention_mask = mx.array(inputs["attention_mask"])

    features = bert_model.extract_features(
        input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
    )
    features = features[0]

    w2ph = list(word2ph)
    if add_blank:
        w2ph = [p * 2 for p in w2ph]
        w2ph[0] += 1

    assert features.shape[0] == len(w2ph), (
        f"BERT tokens ({features.shape[0]}) != word2ph length ({len(w2ph)}). "
        f"Text: {text!r}"
    )

    phone_features = []
    for i, count in enumerate(w2ph):
        if count > 0:
            repeated = mx.repeat(features[i : i + 1], count, axis=0)
            phone_features.append(repeated)

    phone_features = mx.concatenate(phone_features, axis=0)
    return phone_features.T


def process_text(
    text: str,
    bert_model=None,
    language: str = "EN",
    add_blank: bool = True,
) -> dict:
    """Full text processing pipeline: text -> phone IDs + BERT features."""
    norm_text = text_normalize(text)
    phones, tones, word2ph = g2p(norm_text)

    if add_blank:
        phones_with_blank = [pad]
        tones_with_blank = [0]
        for p, t in zip(phones, tones):
            phones_with_blank.extend([p, pad])
            tones_with_blank.extend([t, 0])
        phones = phones_with_blank
        tones = tones_with_blank

    phone_ids, tone_ids, lang_ids = cleaned_text_to_sequence(phones, tones, language)

    if bert_model is not None:
        bert_features = get_bert_features(
            norm_text, word2ph, bert_model, add_blank=add_blank
        )
        n_phones = len(phone_ids)
        if bert_features.shape[1] < n_phones:
            pad_size = n_phones - bert_features.shape[1]
            bert_features = mx.pad(bert_features, [(0, 0), (0, pad_size)])
        elif bert_features.shape[1] > n_phones:
            bert_features = bert_features[:, :n_phones]
    else:
        bert_features = mx.zeros((768, len(phone_ids)))

    return {
        "phone_ids": phone_ids,
        "tone_ids": tone_ids,
        "lang_ids": lang_ids,
        "bert_features": bert_features,
        "phones": phones,
        "norm_text": norm_text,
    }
