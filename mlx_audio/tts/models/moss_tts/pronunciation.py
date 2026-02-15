"""Pronunciation/input-type validation and optional conversion helpers for MOSS-TTS."""

from __future__ import annotations

from importlib import import_module
import re
from typing import Optional

VALID_INPUT_TYPES = {"text", "pinyin", "ipa"}

_HAS_CJK_RE = re.compile(r"[\u3400-\u9fff]")
_HAS_TONE_NUMBER_RE = re.compile(r"[1-5]")
_PINYIN_TONE_SYLLABLE_RE = re.compile(r"^[A-Za-zvV\u00fc\u00dc]+[1-5]$")
_PUNCT_STRIP_CHARS = " \t\r\n.,!?;:，。！？；：、()[]{}\"'“”‘’`"
_CN_PUNCTUATION = "，。！？；：、（）“”‘’"


class PronunciationHelperUnavailableError(RuntimeError):
    """Raised when an optional pronunciation helper dependency is unavailable."""


def validate_input_type(input_type: str) -> str:
    """Validate and normalize a requested input type."""
    normalized = str(input_type).strip().lower()
    if normalized not in VALID_INPUT_TYPES:
        raise ValueError(
            f"Unsupported input_type '{input_type}'. "
            f"Expected one of {sorted(VALID_INPUT_TYPES)}"
        )
    return normalized


def validate_pronunciation_input_contract(text: Optional[str], input_type: str) -> str:
    """Fail-fast validation for pronunciation-specific input modes."""
    normalized = validate_input_type(input_type)
    if normalized == "text":
        return normalized

    if text is None or not str(text).strip():
        raise ValueError(
            f"input_type='{normalized}' requires non-empty text content. "
            "Use input_type='text' for free-form prompts."
        )

    if normalized == "pinyin":
        _validate_tone_numbered_pinyin_text(str(text))
    else:
        _validate_ipa_wrapped_text(str(text))
    return normalized


def convert_text_to_tone_numbered_pinyin(text: str, *, strict: bool = True) -> str:
    """Convert Chinese text to tone-numbered pinyin using optional `pypinyin`."""
    normalized_text = _require_non_empty_text(
        text,
        helper_name="convert_text_to_tone_numbered_pinyin",
    )
    try:
        pypinyin = import_module("pypinyin")
    except ImportError as exc:  # pragma: no cover - environment specific
        raise PronunciationHelperUnavailableError(
            "pypinyin is not installed. Install it to enable zh->pinyin conversion."
        ) from exc

    try:
        pinyin_fn = getattr(pypinyin, "pinyin")
        style = getattr(pypinyin, "Style").TONE3
    except AttributeError as exc:  # pragma: no cover - defensive
        raise PronunciationHelperUnavailableError(
            "Installed pypinyin package is missing `pinyin`/`Style.TONE3`."
        ) from exc

    kwargs = {
        "style": style,
        "heteronym": False,
        "strict": bool(strict),
        "errors": "default",
    }
    try:
        converted_rows = pinyin_fn(
            normalized_text,
            neutral_tone_with_five=True,
            **kwargs,
        )
    except TypeError:
        converted_rows = pinyin_fn(normalized_text, **kwargs)

    converted = " ".join(str(item[0]) for item in converted_rows if item)
    converted = _fix_cjk_punctuation_spacing(converted)
    validate_pronunciation_input_contract(converted, "pinyin")
    return converted


def convert_text_to_ipa(
    text: str,
    *,
    language: str = "en-us",
    deep_phonemizer_checkpoint: Optional[str] = None,
    batch_size: int = 8,
) -> str:
    """Convert text to a `/.../`-wrapped IPA span using optional helpers."""
    normalized_text = _require_non_empty_text(text, helper_name="convert_text_to_ipa")
    if deep_phonemizer_checkpoint is not None:
        ipa = _ipa_via_deep_phonemizer(
            normalized_text,
            language=language,
            checkpoint=deep_phonemizer_checkpoint,
            batch_size=batch_size,
        )
    else:
        ipa = _ipa_via_phonemizer(normalized_text, language=language)

    wrapped = f"/{ipa}/"
    validate_pronunciation_input_contract(wrapped, "ipa")
    return wrapped


def _strip_token_edges(token: str) -> str:
    return token.strip(_PUNCT_STRIP_CHARS)


def _validate_tone_numbered_pinyin_text(text: str) -> None:
    if _HAS_CJK_RE.search(text):
        raise ValueError(
            "input_type='pinyin' expects tone-numbered pinyin text, not Han characters. "
            "Example: 'ni3 hao3'."
        )

    raw_tokens = text.split()
    tokens = [_strip_token_edges(token) for token in raw_tokens]
    tokens = [token for token in tokens if token]
    if not tokens:
        raise ValueError(
            "input_type='pinyin' expects whitespace-separated syllables with tone numbers. "
            "Example: 'ni3 hao3'."
        )

    latin_tokens = [
        token
        for token in tokens
        if re.search(r"[A-Za-zvV\u00fc\u00dc]", token) is not None
    ]
    if not latin_tokens:
        raise ValueError(
            "input_type='pinyin' expects latin syllables with tone numbers (1..5)."
        )

    tone_syllables = [
        token for token in latin_tokens if _PINYIN_TONE_SYLLABLE_RE.fullmatch(token)
    ]
    if not tone_syllables:
        raise ValueError(
            "input_type='pinyin' requires tone-numbered syllables like 'ni3 hao3'."
        )

    tone_coverage = len(tone_syllables) / len(latin_tokens)
    if tone_coverage < 0.6 or not _HAS_TONE_NUMBER_RE.search(text):
        raise ValueError(
            "input_type='pinyin' expects mostly whitespace-separated tone-numbered "
            "syllables (digits 1..5). Example: 'wo3 xiang3 qu4 shang4 hai3'."
        )


def _validate_ipa_wrapped_text(text: str) -> None:
    slash_count = text.count("/")
    if slash_count < 2:
        raise ValueError(
            "input_type='ipa' expects IPA spans wrapped in '/.../'. "
            "Example: '/h\u0259\u02c8lo\u028a/'."
        )
    if slash_count % 2 != 0:
        raise ValueError(
            "input_type='ipa' found unbalanced '/' delimiters. "
            "Wrap each IPA segment with a matching '/.../'."
        )

    spans = []
    open_index: Optional[int] = None
    for idx, char in enumerate(text):
        if char != "/":
            continue
        if open_index is None:
            open_index = idx
            continue
        span = text[open_index + 1 : idx].strip()
        if not span:
            raise ValueError(
                "input_type='ipa' found an empty IPA span ('//'). "
                "Use non-empty '/.../' spans."
            )
        spans.append(span)
        open_index = None

    if (
        open_index is not None
    ):  # pragma: no cover - defensive, guarded by slash_count parity
        raise ValueError("input_type='ipa' has an unclosed '/' delimiter.")
    if not spans:
        raise ValueError(
            "input_type='ipa' expects at least one '/.../' IPA span in the text."
        )


def _require_non_empty_text(text: str, *, helper_name: str) -> str:
    normalized = str(text).strip()
    if not normalized:
        raise ValueError(f"{helper_name} requires a non-empty text input")
    return normalized


def _fix_cjk_punctuation_spacing(text: str) -> str:
    text = re.sub(rf"\s+([{_CN_PUNCTUATION}])", r"\1", text)
    text = re.sub(rf"([{_CN_PUNCTUATION}])\s+", r"\1", text)
    return text.strip()


def _ipa_via_deep_phonemizer(
    text: str,
    *,
    language: str,
    checkpoint: str,
    batch_size: int,
) -> str:
    try:
        deep_phonemizer = import_module("dp.phonemizer")
    except ImportError as exc:  # pragma: no cover - environment specific
        raise PronunciationHelperUnavailableError(
            "DeepPhonemizer is not installed. Install `deep-phonemizer` or use "
            "the phonemizer fallback by omitting deep_phonemizer_checkpoint."
        ) from exc

    try:
        phonemizer_cls = getattr(deep_phonemizer, "Phonemizer")
    except AttributeError as exc:  # pragma: no cover - defensive
        raise PronunciationHelperUnavailableError(
            "DeepPhonemizer import succeeded but `Phonemizer` is unavailable."
        ) from exc

    try:
        phonemizer = phonemizer_cls.from_checkpoint(str(checkpoint))
        output = phonemizer(
            text,
            lang=str(language).replace("-", "_"),
            batch_size=int(batch_size),
        )
    except Exception as exc:  # pragma: no cover - environment specific
        raise PronunciationHelperUnavailableError(
            "DeepPhonemizer conversion failed. Verify checkpoint path and runtime deps."
        ) from exc

    ipa = _normalize_phonemizer_output(output)
    if not ipa:
        raise ValueError("DeepPhonemizer produced empty IPA output")
    return ipa


def _ipa_via_phonemizer(text: str, *, language: str) -> str:
    try:
        phonemizer = import_module("phonemizer")
    except ImportError as exc:  # pragma: no cover - environment specific
        raise PronunciationHelperUnavailableError(
            "phonemizer is not installed. Install `phonemizer-fork` (or equivalent) "
            "to enable ipa conversion helpers."
        ) from exc

    phonemize_fn = getattr(phonemizer, "phonemize", None)
    if phonemize_fn is None:  # pragma: no cover - defensive
        raise PronunciationHelperUnavailableError(
            "phonemizer package does not expose `phonemize`."
        )

    try:
        output = phonemize_fn(
            text,
            language=str(language).replace("_", "-"),
            backend="espeak",
            strip=True,
            preserve_punctuation=True,
            with_stress=True,
        )
    except Exception as exc:  # pragma: no cover - environment specific
        raise PronunciationHelperUnavailableError(
            "phonemizer conversion failed. Ensure backend dependencies (e.g. espeak-ng) "
            "are available in this environment."
        ) from exc

    ipa = _normalize_phonemizer_output(output)
    if not ipa:
        raise ValueError("phonemizer produced empty IPA output")
    return ipa


def _normalize_phonemizer_output(output) -> str:
    if isinstance(output, str):
        return output.strip()
    if isinstance(output, (list, tuple)):
        parts = [str(item).strip() for item in output if str(item).strip()]
        return " ".join(parts)
    return str(output).strip()


__all__ = [
    "PronunciationHelperUnavailableError",
    "VALID_INPUT_TYPES",
    "convert_text_to_ipa",
    "convert_text_to_tone_numbered_pinyin",
    "validate_input_type",
    "validate_pronunciation_input_contract",
]
