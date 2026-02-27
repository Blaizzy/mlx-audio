from __future__ import annotations

from typing import List, Tuple

import mlx.core as mx
import numpy as np


def normalize_text_prompt(text: str) -> str:
    text = text.replace("…", "...")
    text = text.replace("’", "'")
    text = text.replace("”", '"')
    text = text.replace("\n", " ")
    text = text.replace(":", ",")
    text = text.replace(";", ",")
    text = text.replace("—", ", ")

    if (
        not text.startswith("[")
        and not text.startswith("(")
        and "S1" not in text
        and "S2" not in text
    ):
        text = "[S1] " + text

    return text


def tokenizer_encode(
    text: str, append_bos: bool = True, normalize: bool = True
) -> mx.array:
    if normalize:
        text = normalize_text_prompt(text)

    tokens = list(text.encode("utf-8"))
    if append_bos:
        tokens.insert(0, 0)

    return mx.array(tokens, dtype=mx.int32)


def get_text_input_ids_and_mask(
    text_arr: List[str],
    max_length: int | None,
    normalize: bool = True,
    return_normalized_text: bool = False,
    pad_to_max: bool = True,
) -> Tuple[mx.array, mx.array] | Tuple[mx.array, mx.array, List[str]]:
    normalized_texts: List[str] = []
    encoded_texts: List[mx.array] = []
    for text in text_arr:
        normalized = normalize_text_prompt(text) if normalize else text
        normalized_texts.append(normalized)
        encoded_texts.append(
            tokenizer_encode(normalized, append_bos=True, normalize=False)
        )

    if max_length is None:
        max_length = max(int(enc.shape[0]) for enc in encoded_texts)

    tokens_np = np.zeros((len(text_arr), max_length), dtype=np.int32)
    mask_np = np.zeros((len(text_arr), max_length), dtype=bool)

    for i, encoded in enumerate(encoded_texts):
        length = min(int(encoded.shape[0]), max_length)
        tokens_np[i, :length] = np.array(encoded[:length], dtype=np.int32)
        mask_np[i, :length] = True

    if not pad_to_max and max_length is not None:
        actual_max_length = max(
            min(int(enc.shape[0]), max_length) for enc in encoded_texts
        )
        tokens_np = tokens_np[:, :actual_max_length]
        mask_np = mask_np[:, :actual_max_length]

    tokens = mx.array(tokens_np, dtype=mx.int32)
    mask = mx.array(mask_np, dtype=mx.bool_)

    if return_normalized_text:
        return tokens, mask, normalized_texts

    return tokens, mask
