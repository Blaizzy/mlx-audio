def _is_special_piece(piece: str) -> bool:
    return (piece.startswith("<|") and piece.endswith("|>")) or piece in (
        "<unk>",
        "<pad>",
    )


def is_special_token(token_id: int, vocabulary: list[str]) -> bool:
    if token_id < 0 or token_id >= len(vocabulary):
        return False
    return _is_special_piece(vocabulary[token_id])


def decode(tokens: list[int], vocabulary: list[str]) -> str:
    parts: list[str] = []
    for token in tokens:
        if token < 0 or token >= len(vocabulary):
            continue
        piece = vocabulary[token]
        if _is_special_piece(piece):
            continue
        parts.append(piece.replace("▁", " "))
    return "".join(parts)
