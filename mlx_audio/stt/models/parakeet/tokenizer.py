from typing import Dict, List, Optional, Tuple

from .languages import build_language_token_map, parse_language_token


def decode(tokens: list[int], vocabulary: list[str]):
    return "".join([vocabulary[token].replace("▁", " ") for token in tokens])


def decode_with_language(
    tokens: List[int],
    vocabulary: List[str],
    lang_token_map: Optional[Dict[int, str]] = None,
) -> Tuple[str, Optional[str]]:
    """Decode tokens to text and extract detected language.

    For multilingual models (like Parakeet v3), this function detects
    the language token in the output and returns both the text and
    the detected language code.

    Args:
        tokens: List of token IDs to decode
        vocabulary: The model's vocabulary list
        lang_token_map: Pre-built language token map (optional, for efficiency)

    Returns:
        Tuple of (decoded_text, language_code)
        - decoded_text: The transcribed text with language tokens removed
        - language_code: ISO 639-1 language code if detected, None otherwise
    """
    if lang_token_map is None:
        lang_token_map = build_language_token_map(vocabulary)

    detected_language = None
    text_tokens = []

    for token_id in tokens:
        # Check if this is a language token
        if token_id in lang_token_map:
            if detected_language is None:
                detected_language = lang_token_map[token_id]
            # Skip language tokens in output
            continue

        # Check if this is any special token (starts with <| and ends with |>)
        token_str = vocabulary[token_id] if token_id < len(vocabulary) else ""
        if token_str.startswith("<|") and token_str.endswith("|>"):
            # Skip special tokens
            continue

        text_tokens.append(token_id)

    text = decode(text_tokens, vocabulary)
    return text, detected_language


def is_special_token(token_id: int, vocabulary: List[str]) -> bool:
    """Check if a token ID corresponds to a special token.

    Args:
        token_id: The token ID to check
        vocabulary: The model's vocabulary list

    Returns:
        True if the token is a special token (like <|en|>, <|pnc|>, etc.)
    """
    if token_id >= len(vocabulary):
        return False
    token_str = vocabulary[token_id]
    return token_str.startswith("<|") and token_str.endswith("|>")


def filter_special_tokens(tokens: List[int], vocabulary: List[str]) -> List[int]:
    """Filter out special tokens from a token list.

    Args:
        tokens: List of token IDs
        vocabulary: The model's vocabulary list

    Returns:
        List of token IDs with special tokens removed
    """
    return [t for t in tokens if not is_special_token(t, vocabulary)]
