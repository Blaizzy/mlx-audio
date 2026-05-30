from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol, Sequence

from .config import ModelConfig

# Prompt template ported from OpenMOSS/MOSS-TTS-Nano (Apache-2.0).
USER_ROLE_PREFIX = "user\n"
USER_TEMPLATE_REFERENCE_PREFIX = "<user_inst>\n- Reference(s):\n"
USER_TEMPLATE_AFTER_REFERENCE = (
    "\n- Instruction:\nNone\n"
    "- Tokens:\nNone\n"
    "- Quality:\nNone\n"
    "- Sound Event:\nNone\n"
    "- Ambient Sound:\nNone\n"
    "- Language:\nNone\n"
    "- Text:\n"
)
USER_TEMPLATE_SUFFIX = "\n</user_inst>"
ASSISTANT_TURN_PREFIX = "\n"
ASSISTANT_ROLE_PREFIX = "assistant\n"

SENTENCE_END_PUNCTUATION = frozenset(".!?。！？；;")
CLAUSE_SPLIT_PUNCTUATION = frozenset(",，、；;：:")
CLOSING_PUNCTUATION = frozenset("\"'”’)]}）】》」』")


class TextTokenizer(Protocol):
    def encode(self, text: str, *args, **kwargs) -> list[int]: ...

    def decode(self, token_ids: Sequence[int], *args, **kwargs) -> str: ...


class SentencePieceTextTokenizer:
    def __init__(self, model_path: str | Path):
        try:
            import sentencepiece as spm
        except ImportError as exc:
            raise ImportError(
                "MOSS-TTS-Nano text tokenization requires sentencepiece. "
                "Install it with `pip install sentencepiece` or the `tts` extra."
            ) from exc
        self.processor = spm.SentencePieceProcessor(model_file=str(model_path))

    def encode(self, text: str, *args, **kwargs) -> list[int]:
        del args, kwargs
        return [
            int(token_id) for token_id in self.processor.encode(str(text), out_type=int)
        ]

    def decode(self, token_ids: Sequence[int], *args, **kwargs) -> str:
        del args, kwargs
        return str(self.processor.decode(list(token_ids)))


def load_tokenizer(model_path: str | Path) -> SentencePieceTextTokenizer:
    root = Path(model_path)
    tokenizer_model = root / "tokenizer.model"
    if not tokenizer_model.exists():
        raise FileNotFoundError(
            f"MOSS-TTS-Nano tokenizer.model not found: {tokenizer_model}"
        )
    return SentencePieceTextTokenizer(tokenizer_model)


def encode_text(tokenizer: TextTokenizer, text: str) -> list[int]:
    try:
        return [
            int(token_id)
            for token_id in tokenizer.encode(text, add_special_tokens=False)
        ]
    except TypeError:
        return [int(token_id) for token_id in tokenizer.encode(text)]


def build_user_prompt_prefix(
    tokenizer: TextTokenizer, config: ModelConfig
) -> list[int]:
    return (
        [config.im_start_token_id]
        + encode_text(tokenizer, USER_ROLE_PREFIX)
        + encode_text(tokenizer, USER_TEMPLATE_REFERENCE_PREFIX)
    )


def build_user_prompt_after_reference(tokenizer: TextTokenizer) -> list[int]:
    return encode_text(tokenizer, USER_TEMPLATE_AFTER_REFERENCE)


def build_assistant_prompt_prefix(
    tokenizer: TextTokenizer, config: ModelConfig
) -> list[int]:
    return (
        encode_text(tokenizer, USER_TEMPLATE_SUFFIX)
        + [config.im_end_token_id]
        + encode_text(tokenizer, ASSISTANT_TURN_PREFIX)
        + [config.im_start_token_id]
        + encode_text(tokenizer, ASSISTANT_ROLE_PREFIX)
    )


def build_prompt_token_ids(
    tokenizer: TextTokenizer,
    config: ModelConfig,
    text_token_ids: Sequence[int],
) -> list[int]:
    return (
        build_user_prompt_prefix(tokenizer, config)
        + encode_text(tokenizer, "None")
        + build_user_prompt_after_reference(tokenizer)
        + [int(token_id) for token_id in text_token_ids]
        + build_assistant_prompt_prefix(tokenizer, config)
    )


def contains_cjk(text: str) -> bool:
    return any(
        "\u4e00" <= ch <= "\u9fff"
        or "\u3400" <= ch <= "\u4dbf"
        or "\u3040" <= ch <= "\u30ff"
        or "\uac00" <= ch <= "\ud7af"
        for ch in str(text)
    )


def prepare_text_for_sentence_chunking(text: str) -> str:
    normalized_text = str(text or "").strip()
    if not normalized_text:
        raise ValueError("Text prompt cannot be empty.")

    normalized_text = normalized_text.replace("\r", " ").replace("\n", " ")
    while "  " in normalized_text:
        normalized_text = normalized_text.replace("  ", " ")

    if contains_cjk(normalized_text):
        if normalized_text[-1] not in SENTENCE_END_PUNCTUATION:
            normalized_text += "。"
        return normalized_text

    if normalized_text[:1].islower():
        normalized_text = normalized_text[:1].upper() + normalized_text[1:]
    if normalized_text[-1].isalnum():
        normalized_text += "."
    if len([item for item in normalized_text.split() if item]) < 5:
        normalized_text = f"        {normalized_text}"
    return normalized_text


def split_text_by_punctuation(
    text: str, punctuation: set[str] | frozenset[str]
) -> list[str]:
    sentences: list[str] = []
    current_chars: list[str] = []
    index = 0
    normalized_text = str(text or "")
    while index < len(normalized_text):
        character = normalized_text[index]
        current_chars.append(character)
        if character in punctuation:
            lookahead = index + 1
            while (
                lookahead < len(normalized_text)
                and normalized_text[lookahead] in CLOSING_PUNCTUATION
            ):
                current_chars.append(normalized_text[lookahead])
                lookahead += 1
            sentence = "".join(current_chars).strip()
            if sentence:
                sentences.append(sentence)
            current_chars.clear()
            while (
                lookahead < len(normalized_text)
                and normalized_text[lookahead].isspace()
            ):
                lookahead += 1
            index = lookahead
            continue
        index += 1

    tail = "".join(current_chars).strip()
    if tail:
        sentences.append(tail)
    return sentences


def join_sentence_parts(left: str, right: str) -> str:
    if not left:
        return right
    if not right:
        return left
    if contains_cjk(left) or contains_cjk(right):
        return left + right
    return f"{left} {right}"


def split_text_by_token_budget(
    tokenizer: TextTokenizer,
    text: str,
    max_tokens: int,
) -> list[str]:
    remaining_text = str(text or "").strip()
    if not remaining_text:
        return []

    pieces: list[str] = []
    preferred_boundary_chars = (
        set(CLAUSE_SPLIT_PUNCTUATION) | set(SENTENCE_END_PUNCTUATION) | {" "}
    )
    while remaining_text:
        if len(encode_text(tokenizer, remaining_text)) <= max_tokens:
            pieces.append(remaining_text)
            break

        low = 1
        high = len(remaining_text)
        best_prefix_length = 1
        while low <= high:
            middle = (low + high) // 2
            candidate = remaining_text[:middle].strip()
            if not candidate:
                low = middle + 1
                continue
            if len(encode_text(tokenizer, candidate)) <= max_tokens:
                best_prefix_length = middle
                low = middle + 1
            else:
                high = middle - 1

        cut_index = best_prefix_length
        prefix = remaining_text[:best_prefix_length]
        preferred_index = -1
        scan_min = max(-1, len(prefix) - 25)
        for scan_index in range(len(prefix) - 1, scan_min, -1):
            if prefix[scan_index] in preferred_boundary_chars:
                preferred_index = scan_index + 1
                break
        if preferred_index > 0:
            cut_index = preferred_index

        piece = remaining_text[:cut_index].strip()
        if not piece:
            piece = remaining_text[:best_prefix_length].strip()
            cut_index = best_prefix_length
        pieces.append(piece)
        remaining_text = remaining_text[cut_index:].strip()
    return pieces


def split_text_into_best_sentences(
    tokenizer: TextTokenizer,
    text: str,
    max_tokens: int = 75,
) -> list[str]:
    normalized_text = str(text or "").strip()
    if not normalized_text:
        return []
    safe_max_tokens = max(1, int(max_tokens))
    prepared_text = prepare_text_for_sentence_chunking(normalized_text)
    sentence_candidates = split_text_by_punctuation(
        prepared_text, SENTENCE_END_PUNCTUATION
    ) or [prepared_text.strip()]

    sentence_slices: list[tuple[int, str]] = []
    for sentence_text in sentence_candidates:
        normalized_sentence = sentence_text.strip()
        if not normalized_sentence:
            continue
        sentence_token_count = len(encode_text(tokenizer, normalized_sentence))
        if sentence_token_count <= safe_max_tokens:
            sentence_slices.append((sentence_token_count, normalized_sentence))
            continue
        clause_candidates = split_text_by_punctuation(
            normalized_sentence, CLAUSE_SPLIT_PUNCTUATION
        )
        if len(clause_candidates) <= 1:
            clause_candidates = [normalized_sentence]
        for clause_text in clause_candidates:
            normalized_clause = clause_text.strip()
            if not normalized_clause:
                continue
            clause_token_count = len(encode_text(tokenizer, normalized_clause))
            if clause_token_count <= safe_max_tokens:
                sentence_slices.append((clause_token_count, normalized_clause))
                continue
            for piece in split_text_by_token_budget(
                tokenizer, normalized_clause, safe_max_tokens
            ):
                normalized_piece = piece.strip()
                if normalized_piece:
                    sentence_slices.append(
                        (
                            len(encode_text(tokenizer, normalized_piece)),
                            normalized_piece,
                        )
                    )

    chunks: list[str] = []
    current_chunk = ""
    current_chunk_token_count = 0
    for sentence_token_count, sentence_text in sentence_slices:
        if not current_chunk:
            current_chunk = sentence_text
            current_chunk_token_count = sentence_token_count
            continue
        if current_chunk_token_count + sentence_token_count > safe_max_tokens:
            chunks.append(current_chunk.strip())
            current_chunk = sentence_text
            current_chunk_token_count = sentence_token_count
        else:
            current_chunk = join_sentence_parts(current_chunk, sentence_text)
            current_chunk_token_count = len(encode_text(tokenizer, current_chunk))
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks if len(chunks) > 1 else [normalized_text]


def lightweight_normalize_text(text: str) -> str:
    normalized = str(text or "").replace("\r", " ").replace("\n", " ").strip()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized
