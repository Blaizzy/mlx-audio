"""Text frontend helpers shared by IndexTTS2 and IndexTTS-2.5.

The multilingual tokenizer and pronunciation syntax match index-tts's
``indextts/utils/tokenizer.py`` and ``indextts/infer_v2_5.py``.
"""

from __future__ import annotations

import base64
import re
from functools import lru_cache
from pathlib import Path
from typing import Callable, List

# Order is load-bearing: language ids come from enumeration order.
LANGUAGES = {
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    "tr": "turkish",
    "pl": "polish",
    "ca": "catalan",
    "nl": "dutch",
    "ar": "arabic",
    "sv": "swedish",
    "it": "italian",
    "id": "indonesian",
    "hi": "hindi",
    "fi": "finnish",
    "vi": "vietnamese",
    "he": "hebrew",
    "uk": "ukrainian",
    "el": "greek",
    "ms": "malay",
    "cs": "czech",
    "ro": "romanian",
    "da": "danish",
    "hu": "hungarian",
    "ta": "tamil",
    "no": "norwegian",
    "th": "thai",
    "ur": "urdu",
    "hr": "croatian",
    "bg": "bulgarian",
    "lt": "lithuanian",
    "la": "latin",
    "mi": "maori",
    "ml": "malayalam",
    "cy": "welsh",
    "sk": "slovak",
    "te": "telugu",
    "fa": "persian",
    "lv": "latvian",
    "bn": "bengali",
    "sr": "serbian",
    "az": "azerbaijani",
    "sl": "slovenian",
    "kn": "kannada",
    "et": "estonian",
    "mk": "macedonian",
    "br": "breton",
    "eu": "basque",
    "is": "icelandic",
    "hy": "armenian",
    "ne": "nepali",
    "mn": "mongolian",
    "bs": "bosnian",
    "kk": "kazakh",
    "sq": "albanian",
    "sw": "swahili",
    "gl": "galician",
    "mr": "marathi",
    "pa": "punjabi",
    "si": "sinhala",
    "km": "khmer",
    "sn": "shona",
    "yo": "yoruba",
    "so": "somali",
    "af": "afrikaans",
    "oc": "occitan",
    "ka": "georgian",
    "be": "belarusian",
    "tg": "tajik",
    "sd": "sindhi",
    "gu": "gujarati",
    "am": "amharic",
    "yi": "yiddish",
    "lo": "lao",
    "uz": "uzbek",
    "fo": "faroese",
    "ht": "haitian creole",
    "ps": "pashto",
    "tk": "turkmen",
    "nn": "nynorsk",
    "mt": "maltese",
    "sa": "sanskrit",
    "lb": "luxembourgish",
    "my": "myanmar",
    "bo": "tibetan",
    "tl": "tagalog",
    "mg": "malagasy",
    "as": "assamese",
    "tt": "tatar",
    "haw": "hawaiian",
    "ln": "lingala",
    "ha": "hausa",
    "ba": "bashkir",
    "jw": "javanese",
    "su": "sundanese",
    "yue": "cantonese",
    "minnan": "minnan",
    "wuyu": "wuyu",
    "dialect": "dialect",
    "zh/en": "zh/en",
    "en/zh": "en/zh",
    "common": "common",
}

LANGUAGE_DICT = {lang: index for index, lang in enumerate(LANGUAGES)}

AUDIO_EVENT = (
    "ASR",
    "AED",
    "SER",
    "Speech",
    "/Speech",
    "BGM",
    "/BGM",
    "Laughter",
    "/Laughter",
    "Applause",
    "/Applause",
)
EMOTION = ("HAPPY", "SAD", "ANGRY", "NEUTRAL")
TTS_VOCAL_TOKEN = (
    "TTS/B",
    "TTS/O",
    "TTS/Q",
    "TTS/A",
    "TTS/CO",
    "TTS/CL",
    "TTS/H",
    *(f"TTS/SP{i:02d}" for i in range(1, 14)),
)

PRONUNCIATION_ANNOTATION_PATTERN = re.compile(r"<([^|>\n]+)\|([^>\n]+)>")
SPLIT_PROTECTED_PATTERN = re.compile(
    r"<\|SPECIAL_TOKEN_\d+\|>.*?<\|SPECIAL_TOKEN_\d+\|>"
)


def lang_to_token(lang: str) -> int:
    return LANGUAGE_DICT.get(lang.lower(), LANGUAGE_DICT["common"])


def is_kana(text: str) -> bool:
    hira = re.compile(r"^[぀-ゟ]+$")
    kata = re.compile(r"^[゠-ヿ]+$")
    return bool(hira.fullmatch(text) or kata.fullmatch(text))


def apply_pronunciation_annotations(text: str) -> str:
    """Convert `<text|pronunciation>` into IndexTTS-2.5 special tokens."""

    def replace(match: re.Match[str]) -> str:
        word = match.group(1)
        pronunciation = match.group(2).upper()
        if is_kana(pronunciation):
            return f" {pronunciation} "
        token = "SPECIAL_TOKEN_2" if re.search(r"[一-鿿]", word) else "SPECIAL_TOKEN_1"
        return f"<|{token}|>{pronunciation}<|{token}|>"

    return PRONUNCIATION_ANNOTATION_PATTERN.sub(replace, text)


def normalize_case(text: str, lang: str) -> str:
    lang = lang.lower()
    if lang in {"ja", "zh", "zhen", "en"}:
        return text.lower()
    if lang == "es":
        return text.upper()
    return text


def _split_atomic_pieces(text: str) -> List[tuple[str, bool]]:
    pieces: List[tuple[str, bool]] = []
    pos = 0
    for match in SPLIT_PROTECTED_PATTERN.finditer(text):
        if match.start() > pos:
            pieces.append((text[pos : match.start()], False))
        pieces.append((match.group(0), True))
        pos = match.end()
    if pos < len(text):
        pieces.append((text[pos:], False))
    return pieces


def split_text_by_tokens(
    text: str,
    max_tokens: int,
    token_len: Callable[[str], int],
    *,
    capacity: int,
    lang_prefix: str = "",
) -> List[str]:
    """Split text without breaking protected pronunciation spans."""
    budget = min(max_tokens, capacity - 2) - token_len(lang_prefix)
    budget = max(1, budget)
    if token_len(text) <= budget:
        return [text]

    chunks: List[str] = []
    for piece, atomic in _split_atomic_pieces(text):
        if atomic:
            chunks.append(piece)
            continue
        for part in re.split(r"(?<=[，。！？、；：,\.!\?;:\n])", piece):
            if not part:
                continue
            if token_len(part) <= budget:
                chunks.append(part)
                continue
            current = ""
            for char in part:
                if current and token_len(current + char) > budget:
                    chunks.append(current)
                    current = char
                else:
                    current += char
            if current:
                chunks.append(current)

    segments: List[str] = []
    current = ""
    for chunk in chunks:
        if current and token_len(current + chunk) > budget:
            segments.append(current)
            current = chunk
        else:
            current += chunk
    if current:
        segments.append(current)
    return segments or [text]


def split_text_by_punctuation(text: str, max_chars: int = 40) -> List[str]:
    parts = re.split(r"(?<=[，。！？、；：,\.!\?;:\n])", text)
    segments: List[str] = []
    current = ""
    for part in parts:
        if not part:
            continue
        if current and len(current) + len(part) > max_chars:
            segments.append(current)
            current = part
        else:
            current += part
    if current:
        segments.append(current)
    return segments


class MultilingualTokenizer:
    """Tiktoken-based tokenizer used by IndexTTS-2.5."""

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.encoding = _load_encoding(model_dir)

    def encode(self, text: str, *, allowed_special: str = "all") -> List[int]:
        return self.encoding.encode(text, allowed_special=allowed_special)

    def token_len(self, text: str) -> int:
        return len(self.encode(text))


@lru_cache(maxsize=None)
def _load_encoding(model_dir: str):
    try:
        import tiktoken
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "IndexTTS-2.5 text tokenization requires tiktoken. "
            "Install the project dependencies before loading this model."
        ) from exc

    directory = Path(model_dir)
    candidates = (
        directory / "multilingual_zh_ja_yue_char_del.tiktoken",
        directory / "gpt2.tiktoken",
    )
    vocab_path = next((path for path in candidates if path.exists()), candidates[0])
    ranks = {
        base64.b64decode(token): int(rank)
        for token, rank in (line.split() for line in open(vocab_path, encoding="utf-8"))
    }
    n_vocab = len(ranks)
    special_tokens = {}
    specials = [
        "<|endoftext|>",
        "<|startoftranscript|>",
        *[f"<|{lang}|>" for lang in list(LANGUAGES.keys())[:99]],
        *[f"<|{event}|>" for event in AUDIO_EVENT],
        *[f"<|{emotion}|>" for emotion in EMOTION],
        "<|translate|>",
        "<|transcribe|>",
        "<|startoflm|>",
        "<|startofprev|>",
        "<|nospeech|>",
        "<|notimestamps|>",
        *[f"<|SPECIAL_TOKEN_{i}|>" for i in range(1, 31)],
        *[f"<|{token}|>" for token in TTS_VOCAL_TOKEN],
        *[f"<|{i * 0.02:.2f}|>" for i in range(1501)],
    ]
    for token in specials:
        special_tokens[token] = n_vocab
        n_vocab += 1

    return tiktoken.Encoding(
        name=vocab_path.name,
        explicit_n_vocab=n_vocab,
        pat_str=(
            r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| "
            r"?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
        ),
        mergeable_ranks=ranks,
        special_tokens=special_tokens,
    )
