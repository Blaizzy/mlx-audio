from typing import List


def decode(tokens: list[int], vocabulary: list[str]):
    return "".join([vocabulary[token].replace("▁", " ") for token in tokens])


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
