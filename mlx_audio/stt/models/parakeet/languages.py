"""Language support for Parakeet v3 multilingual models.

Parakeet v3 supports 25 European languages with automatic language detection.
"""

from typing import Dict, List, Optional, Tuple

# The 25 European languages supported by Parakeet v3
# Maps ISO 639-1 code to (full name, native name)
PARAKEET_V3_LANGUAGES: Dict[str, Tuple[str, str]] = {
    "bg": ("Bulgarian", "Български"),
    "hr": ("Croatian", "Hrvatski"),
    "cs": ("Czech", "Čeština"),
    "da": ("Danish", "Dansk"),
    "nl": ("Dutch", "Nederlands"),
    "en": ("English", "English"),
    "et": ("Estonian", "Eesti"),
    "fi": ("Finnish", "Suomi"),
    "fr": ("French", "Français"),
    "de": ("German", "Deutsch"),
    "el": ("Greek", "Ελληνικά"),
    "hu": ("Hungarian", "Magyar"),
    "it": ("Italian", "Italiano"),
    "lv": ("Latvian", "Latviešu"),
    "lt": ("Lithuanian", "Lietuvių"),
    "mt": ("Maltese", "Malti"),
    "pl": ("Polish", "Polski"),
    "pt": ("Portuguese", "Português"),
    "ro": ("Romanian", "Română"),
    "sk": ("Slovak", "Slovenčina"),
    "sl": ("Slovenian", "Slovenščina"),
    "es": ("Spanish", "Español"),
    "sv": ("Swedish", "Svenska"),
    "ru": ("Russian", "Русский"),
    "uk": ("Ukrainian", "Українська"),
}

# Special tokens related to language in Parakeet v3
SPECIAL_TOKENS = {
    "unklang": "<|unklang|>",  # Unknown language
    "predict_lang": "<|predict_lang|>",  # Predict language marker
    "nopredict_lang": "<|nopredict_lang|>",  # No language prediction marker
}


def get_language_token(lang_code: str) -> str:
    """Get the token string for a language code.

    Args:
        lang_code: ISO 639-1 language code (e.g., "en", "de")

    Returns:
        Token string (e.g., "<|en|>")
    """
    return f"<|{lang_code}|>"


def parse_language_token(token: str) -> Optional[str]:
    """Parse a language code from a token string.

    Args:
        token: Token string (e.g., "<|en|>")

    Returns:
        ISO 639-1 language code if valid, None otherwise
    """
    if token.startswith("<|") and token.endswith("|>") and len(token) == 6:
        lang_code = token[2:-2]
        # Check if it's a valid 2-letter code (not a special token)
        if lang_code.isalpha() and len(lang_code) == 2:
            return lang_code
    return None


def is_language_token(token: str) -> bool:
    """Check if a token string is a language token.

    Args:
        token: Token string to check

    Returns:
        True if the token is a language token
    """
    return parse_language_token(token) is not None


def is_supported_language(lang_code: str) -> bool:
    """Check if a language code is supported by Parakeet v3.

    Args:
        lang_code: ISO 639-1 language code

    Returns:
        True if the language is in the 25 supported European languages
    """
    return lang_code in PARAKEET_V3_LANGUAGES


def get_language_name(lang_code: str, native: bool = False) -> Optional[str]:
    """Get the full name of a language.

    Args:
        lang_code: ISO 639-1 language code
        native: If True, return the native name

    Returns:
        Language name, or None if not found
    """
    if lang_code in PARAKEET_V3_LANGUAGES:
        return PARAKEET_V3_LANGUAGES[lang_code][1 if native else 0]
    return None


def get_supported_languages() -> List[str]:
    """Get list of supported language codes.

    Returns:
        List of ISO 639-1 language codes supported by Parakeet v3
    """
    return list(PARAKEET_V3_LANGUAGES.keys())


def build_language_token_map(vocabulary: List[str]) -> Dict[int, str]:
    """Build a mapping from token IDs to language codes.

    Args:
        vocabulary: The model's vocabulary list

    Returns:
        Dict mapping token indices to language codes
    """
    lang_map = {}
    for idx, token in enumerate(vocabulary):
        lang_code = parse_language_token(token)
        if lang_code is not None:
            lang_map[idx] = lang_code
    return lang_map


def detect_language_from_tokens(
    token_ids: List[int],
    vocabulary: List[str],
    lang_token_map: Optional[Dict[int, str]] = None,
) -> Optional[str]:
    """Detect language from a sequence of token IDs.

    The model typically outputs a language token early in the sequence.
    This function finds the first language token in the output.

    Args:
        token_ids: List of decoded token IDs
        vocabulary: The model's vocabulary list
        lang_token_map: Pre-built language token map (optional, for efficiency)

    Returns:
        Detected language code, or None if not found
    """
    if lang_token_map is None:
        lang_token_map = build_language_token_map(vocabulary)

    for token_id in token_ids:
        if token_id in lang_token_map:
            return lang_token_map[token_id]
    return None
