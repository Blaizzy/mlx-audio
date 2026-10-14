"""Drama 3 prompt rows and long-text chunking.

Ports build_drama3_prompt (fish_omni/generate.py) and fish_omni/chunking.py.
"""

from __future__ import annotations

import re
from collections import deque
from typing import Iterator, List, Optional, Sequence, Tuple

import mlx.core as mx
import numpy as np

DRAMA3_TTS_INSTRUCTION = "convert the provided text to speech"
DRAMA3_REFERENCE_INSTRUCTION = (
    "convert the provided text to speech reference to the following:"
)
DRAMA3_MESSAGE_END = "<|im_end|>\n"
DRAMA3_GENERATION_START = "<|im_start|>assistant\n<|voice|>"
# Only the prompt builder writes these; other special tokens in caller text are content.
DRAMA3_STRUCTURAL_TOKENS = (
    "<|im_start|>",
    "<|im_end|>",
    "<|voice|>",
    "<|text|>",
    "<|interleave|>",
    "<|semantic:",
)
SPEAKER_TAG_PATTERN = re.compile(r"<\|speaker:([0-9]+)\|>")
MAX_SPEAKER_ID = 15
# The Unicode White_Space property; str.strip() without arguments also strips
# U+001C..U+001F, which would change the empty check and the speaker-tag rule.
UNICODE_WHITE_SPACE = "".join(
    map(
        chr,
        (*range(0x9, 0xE), 0x20, 0x85, 0xA0, 0x1680, *range(0x2000, 0x200B))
        + (0x2028, 0x2029, 0x202F, 0x205F, 0x3000),
    )
)
DRAMA3_MAX_PROMPT_ROWS = 8192

# A threshold, not a hard cap: a protected span longer than this stays whole.
DRAMA3_CHUNK_BYTES = 300
DRAMA3_CHUNK_MAX_SPEAKERS = 3
# Frames per chunk, about 27.9 s at 2048 samples per frame and 44.1 kHz.
DRAMA3_CHUNK_MAX_NEW_TOKENS = 600
DRAMA3_HISTORY_RECENT_TURNS = 1

SENTENCE_ENDS = frozenset(".!?;。！？；…\n")
CLAUSE_MARKS = frozenset(",:，：、")
SPACES = frozenset(" ")
# Never split inside: special tokens, paired tags such as <whisper>…</whisper>,
# and [bracketed] cues.
PROTECTED_SPAN = re.compile(
    r"<\|[^|>]*\|>|<([A-Za-z][\w-]*)\b[^>]*>.*?</\1>|\[[^\[\]]*\]", re.S
)

HistoryTurn = Tuple[str, mx.array]  # (chunk text, codes [num_codebooks, frames])


class PromptTooLongError(ValueError):
    """The prompt has more than DRAMA3_MAX_PROMPT_ROWS rows."""


def validate_text(text: str, location: str) -> None:
    """Non-empty, no structural tokens, and only canonical speaker tags 0..15."""
    if not text.strip(UNICODE_WHITE_SPACE):
        raise ValueError(f"{location}: text must not be empty")
    for token in DRAMA3_STRUCTURAL_TOKENS:
        if token in text:
            raise ValueError(f"{location}: {token} is written by the prompt builder")
    for marker in re.finditer(r"<\|speaker", text):
        tag = SPEAKER_TAG_PATTERN.match(text, marker.start())
        digits = tag.group(1) if tag else ""
        if not digits or str(int(digits)) != digits or int(digits) > MAX_SPEAKER_ID:
            raise ValueError(
                f"{location}: speaker tags must be <|speaker:0|> to "
                f"<|speaker:{MAX_SPEAKER_ID}|>; got {text!r}"
            )


def with_speaker_tag(text: str) -> str:
    """Prefix <|speaker:0|> unless a tag already leads the text after white space."""
    if SPEAKER_TAG_PATTERN.match(text.lstrip(UNICODE_WHITE_SPACE)) is None:
        return f"<|speaker:0|> {text}"
    return text


def build_prompt(
    text: str,
    tokenizer,
    *,
    num_codebooks: int,
    codebook_size: int,
    semantic_start: int,
    prompt_texts: Sequence[str] = (),
    prompt_tokens: Sequence[mx.array] = (),
    history: Sequence[HistoryTurn] = (),
) -> np.ndarray:
    """int32 rows [rows, 1 + num_codebooks] ending in the open assistant voice turn.

    Text rows are [token, 0, ..., 0]; audio rows are [c0 + semantic_start, c0..cN-1].
    Every literal is tokenized on its own: tokenizing the joined string changes
    ids at the seams. `tokenizer` is a `tokenizers.Tokenizer`.
    """
    if len(prompt_texts) != len(prompt_tokens):
        raise ValueError("every reference needs both a transcript and codes")
    validate_text(text, "text")

    def check_codes(codes: np.ndarray, location: str) -> np.ndarray:
        if codes.ndim != 2 or codes.shape[0] != num_codebooks or codes.shape[1] == 0:
            raise ValueError(
                f"{location}: codes must be [{num_codebooks}, frames >= 1]; "
                f"got {codes.shape}"
            )
        if codes.min() < 0 or codes.max() >= codebook_size:
            raise ValueError(f"{location}: codes must be in [0, {codebook_size})")
        return codes

    segments: list = [
        "<|im_start|>system\n",
        DRAMA3_REFERENCE_INSTRUCTION if prompt_tokens else DRAMA3_TTS_INSTRUCTION,
    ]
    for index, (transcript, codes) in enumerate(zip(prompt_texts, prompt_tokens), 1):
        location = f"reference {index}"
        validate_text(transcript, location)
        codes = check_codes(np.array(codes, dtype=np.int64), location)
        # The tag must follow "Text: " directly; leading white space moves behind it.
        tagged = with_speaker_tag(transcript)
        body = tagged.lstrip(UNICODE_WHITE_SPACE)
        tag = SPEAKER_TAG_PATTERN.match(body)
        leading = tagged[: len(tagged) - len(body)]
        transcript = tag.group(0) + leading + body[tag.end() :]
        segments += [f"\n\n[Reference {index}]\nText: {transcript}\nSpeech: ", codes]
    segments.append(DRAMA3_MESSAGE_END)
    for index, (turn_text, codes) in enumerate(history, 1):
        location = f"history {index}"
        validate_text(turn_text, location)
        codes = check_codes(np.array(codes, dtype=np.int64), location)
        segments += [
            "<|im_start|>user\n",
            with_speaker_tag(turn_text),
            DRAMA3_MESSAGE_END,
            DRAMA3_GENERATION_START,
            codes,
            DRAMA3_MESSAGE_END,
        ]
    segments += [
        "<|im_start|>user\n",
        with_speaker_tag(text),
        DRAMA3_MESSAGE_END,
        DRAMA3_GENERATION_START,
    ]

    rows = []
    for segment in segments:
        if isinstance(segment, str):
            ids = np.array(tokenizer.encode(segment).ids, dtype=np.int64)
            text_rows = np.zeros((len(ids), 1 + num_codebooks), dtype=np.int64)
            text_rows[:, 0] = ids
            rows.append(text_rows)
        else:
            main = segment[:1] + semantic_start
            rows.append(np.concatenate([main, segment]).T)
    rows = np.concatenate(rows).astype(np.int32)
    if len(rows) > DRAMA3_MAX_PROMPT_ROWS:
        raise PromptTooLongError(
            f"prompt has {len(rows)} rows; at most {DRAMA3_MAX_PROMPT_ROWS}"
        )
    return rows


def utf8_len(text: str) -> int:
    return len(text.encode("utf-8"))


def _protected_pieces(text: str) -> list:
    pieces, last = [], 0
    for match in PROTECTED_SPAN.finditer(text):
        if match.start() > last:
            pieces.append((text[last : match.start()], False))
        pieces.append((match.group(0), True))
        last = match.end()
    if last < len(text):
        pieces.append((text[last:], False))
    return pieces


def _breaks_at(text: str, index: int, marks: frozenset) -> bool:
    if text[index] not in marks:
        return False
    # Keep decimals such as 3.14 in one piece.
    return not (
        text[index] == "."
        and 0 < index < len(text) - 1
        and text[index - 1].isdigit()
        and text[index + 1].isdigit()
    )


def _break_at_marks(pieces: list, length: int, marks: frozenset) -> list:
    """Split free pieces longer than length bytes after each mark, keeping it."""
    out = []
    for text, protected in pieces:
        if protected or utf8_len(text) <= length:
            out.append((text, protected))
            continue
        start = 0
        for index in range(len(text)):
            if _breaks_at(text, index, marks):
                out.append((text[start : index + 1], False))
                start = index + 1
        if start < len(text):
            out.append((text[start:], False))
    return out


def _break_by_length(pieces: list, length: int) -> list:
    """Split the free pieces still longer than length on character boundaries."""
    out = []
    for text, protected in pieces:
        if protected or utf8_len(text) <= length:
            out.append((text, protected))
            continue
        current, size = "", 0
        for char in text:
            if current and size + utf8_len(char) > length:
                out.append((current, False))
                current, size = "", 0
            current += char
            size += utf8_len(char)
        out.append((current, False))
    return out


def split_by_length(text: str, length: int = DRAMA3_CHUNK_BYTES) -> List[str]:
    """Chunks of about length bytes, packed sentence by sentence."""
    if not text.strip(UNICODE_WHITE_SPACE):
        return []
    if utf8_len(text) <= length:
        return [text]
    pieces = _break_at_marks(_protected_pieces(text), 0, SENTENCE_ENDS)
    for marks in (CLAUSE_MARKS, SPACES):
        pieces = _break_at_marks(pieces, length, marks)
    pieces = _break_by_length(pieces, length)
    chunks, current = [], ""
    for piece, _ in pieces:
        if current and utf8_len(current + piece) > length:
            chunks.append(current)
            current = ""
        current += piece
    chunks.append(current)
    return [
        chunk.strip(UNICODE_WHITE_SPACE)
        for chunk in chunks
        if chunk.strip(UNICODE_WHITE_SPACE)
    ]


def parse_speaker_segments(text: str) -> List[Tuple[str, str]]:
    """(speaker id, text) per tagged segment; text before the first tag is speaker 0."""
    parts = SPEAKER_TAG_PATTERN.split(text)
    segments = [("0", parts[0])] if parts[0].strip(UNICODE_WHITE_SPACE) else []
    for speaker, body in zip(parts[1::2], parts[2::2]):
        if body.strip(UNICODE_WHITE_SPACE):
            segments.append((speaker, body))
    return segments


def split_text(
    text: str,
    max_bytes: int = DRAMA3_CHUNK_BYTES,
    max_speakers: int = DRAMA3_CHUNK_MAX_SPEAKERS,
) -> List[str]:
    """Generation chunks for text, in order (split_drama3_text)."""
    if SPEAKER_TAG_PATTERN.search(text) is None:
        return split_by_length(text, max_bytes)
    segments = []
    for speaker, body in parse_speaker_segments(text):
        for piece in split_by_length(body, max_bytes):
            segments.append((speaker, f"<|speaker:{speaker}|> {piece}"))
    chunks, batch, speakers, size = [], [], [], 0
    for speaker, segment in segments:
        new_speaker = speaker not in speakers
        too_many_speakers = new_speaker and len(speakers) >= max_speakers
        too_long = bool(batch) and size + utf8_len(segment) > max_bytes
        if too_many_speakers or too_long:
            chunks.append("\n".join(batch))
            batch, speakers, size = [], [], 0
            new_speaker = True
        batch.append(segment)
        if new_speaker:
            speakers.append(speaker)
        size += utf8_len(segment)
    if batch:
        chunks.append("\n".join(batch))
    return chunks


class History:
    """Context for the next chunk: the first chunk stays pinned, plus the latest
    DRAMA3_HISTORY_RECENT_TURNS chunks."""

    def __init__(self, recent_turns: int = DRAMA3_HISTORY_RECENT_TURNS) -> None:
        self.pinned: Optional[HistoryTurn] = None
        self.recent: deque = deque(maxlen=recent_turns)

    def push(self, text: str, codes: mx.array) -> None:
        if self.pinned is None:
            self.pinned = (text, codes)
        else:
            self.recent.append((text, codes))

    def candidates(self) -> Iterator[List[HistoryTurn]]:
        """Histories to try, longest first: drop the oldest recent turns, then
        the pinned one, ending with no history."""
        pinned = [self.pinned] if self.pinned is not None else []
        recent = list(self.recent)
        for start in range(len(recent) + 1):
            yield pinned + recent[start:]
        if pinned:
            yield []
