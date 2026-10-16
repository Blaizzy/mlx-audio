"""Text rules of the Drama 3 prompt, and splitting long text into chunks.

Drama 3 is generated chunk by chunk: about CHUNK_BYTES of UTF-8 text with at most
CHUNK_SPEAKERS speakers each, so the per-chunk frame budget is never the limit.
Cuts prefer sentence ends, then clause marks, then spaces. Special tokens,
paired tags such as <whisper>...</whisper>, and [bracketed] cues are never cut,
and every piece of a speaker's turn keeps that speaker's tag.
"""

import re

CHUNK_BYTES = 300
CHUNK_SPEAKERS = 3
MAX_SPEAKER_ID = 15  # the tokenizer has <|speaker:0|> to <|speaker:15|>

# Written only by the prompt builder; caller text containing them would inject
# conversation structure or audio rows.
STRUCTURAL_TOKENS = (
    "<|im_start|>",
    "<|im_end|>",
    "<|voice|>",
    "<|text|>",
    "<|interleave|>",
    "<|semantic:",
)
SPEAKER_TAG = re.compile(r"<\|speaker:([0-9]+)\|>")
# Unicode White_Space. str.strip() would also strip U+001C..U+001F.
WHITE_SPACE = "".join(
    map(
        chr,
        [*range(0x9, 0xE), 0x20, 0x85, 0xA0, 0x1680, *range(0x2000, 0x200B)]
        + [0x2028, 0x2029, 0x202F, 0x205F, 0x3000],
    )
)
UNCUTTABLE = re.compile(
    r"<\|[^|>]*\|>|<([A-Za-z][\w-]*)\b[^>]*>.*?</\1>|\[[^\[\]]*\]", re.S
)
SENTENCE_ENDS = frozenset(".!?;。！？；…\n")
CLAUSE_MARKS = frozenset(",:，：、")
SPACES = frozenset(" ")


def validate_text(text: str, location: str) -> None:
    """Rejects empty text, structural tokens, and speaker tags outside 0..15."""
    if not text.strip(WHITE_SPACE):
        raise ValueError(f"{location}: text is empty")
    for token in STRUCTURAL_TOKENS:
        if token in text:
            raise ValueError(f"{location}: text must not contain {token}")
    for start in (m.start() for m in re.finditer(r"<\|speaker", text)):
        tag = SPEAKER_TAG.match(text, start)
        number = tag.group(1) if tag else ""
        if not number or number != str(int(number)) or int(number) > MAX_SPEAKER_ID:
            raise ValueError(
                f"{location}: speaker tags run from <|speaker:0|> to "
                f"<|speaker:{MAX_SPEAKER_ID}|>"
            )


def with_speaker_tag(text: str) -> str:
    """Text that starts (after white space) with a speaker tag; speaker 0 by default."""
    if SPEAKER_TAG.match(text.lstrip(WHITE_SPACE)):
        return text
    return f"<|speaker:0|> {text}"


def reference_transcript(text: str) -> str:
    """A reference transcript whose tag directly follows "Text: " in the prompt."""
    tagged = with_speaker_tag(text)
    body = tagged.lstrip(WHITE_SPACE)
    tag = SPEAKER_TAG.match(body)
    leading = tagged[: len(tagged) - len(body)]
    return tag.group(0) + leading + body[tag.end() :]


def _bytes(text: str) -> int:
    return len(text.encode("utf-8"))


def _cut_after(text: str, marks: frozenset[str]) -> list[str]:
    """Cut after every mark, except a period between two digits (3.14)."""
    parts, start = [], 0
    for i, char in enumerate(text):
        decimal = (
            char == "."
            and 0 < i < len(text) - 1
            and text[i - 1].isdigit()
            and text[i + 1].isdigit()
        )
        if char in marks and not decimal:
            parts.append(text[start : i + 1])
            start = i + 1
    if start < len(text):
        parts.append(text[start:])
    return parts


def _fit(text: str, limit: int, marks: tuple[frozenset[str], ...]) -> list[str]:
    """Pieces of at most limit bytes, cut at the first kind of mark that helps."""
    if _bytes(text) <= limit:
        return [text]
    if marks:
        return [
            piece
            for part in _cut_after(text, marks[0])
            for piece in _fit(part, limit, marks[1:])
        ]
    pieces, current = [], ""
    for char in text:
        if current and _bytes(current + char) > limit:
            pieces.append(current)
            current = ""
        current += char
    return pieces + [current]


def _pieces(text: str, limit: int) -> list[str]:
    pieces, last = [], 0
    for match in UNCUTTABLE.finditer(text):
        pieces += _free_pieces(text[last : match.start()], limit)
        pieces.append(match.group(0))
        last = match.end()
    return pieces + _free_pieces(text[last:], limit)


def _free_pieces(text: str, limit: int) -> list[str]:
    return [
        piece
        for sentence in _cut_after(text, SENTENCE_ENDS)
        for piece in _fit(sentence, limit, (CLAUSE_MARKS, SPACES))
    ]


def split_by_length(text: str, limit: int = CHUNK_BYTES) -> list[str]:
    """Pieces packed into chunks of about limit bytes. An uncuttable span longer
    than limit becomes a chunk of its own."""
    if not text.strip(WHITE_SPACE):
        return []
    if _bytes(text) <= limit:
        return [text]
    chunks, current = [], ""
    for piece in _pieces(text, limit):
        if current and _bytes(current + piece) > limit:
            chunks.append(current)
            current = ""
        current += piece
    chunks.append(current)
    return [c.strip(WHITE_SPACE) for c in chunks if c.strip(WHITE_SPACE)]


def split_text(
    text: str, limit: int = CHUNK_BYTES, max_speakers: int = CHUNK_SPEAKERS
) -> list[str]:
    """Generation chunks in order. Tagged turns are split per speaker, each piece
    re-tagged, then regrouped so a chunk stays within limit and max_speakers."""
    if not SPEAKER_TAG.search(text):
        return split_by_length(text, limit)

    parts = SPEAKER_TAG.split(text)
    turns = [("0", parts[0])] + list(zip(parts[1::2], parts[2::2]))
    tagged = [
        (speaker, f"<|speaker:{speaker}|> {piece}")
        for speaker, body in turns
        for piece in split_by_length(body, limit)
    ]

    chunks: list[list[str]] = []
    speakers: set[str] = set()
    size = 0
    for speaker, piece in tagged:
        new_speaker = speaker not in speakers
        if chunks and (
            (new_speaker and len(speakers) >= max_speakers)
            or size + _bytes(piece) > limit
        ):
            chunks.append([])
            speakers, size = set(), 0
        if not chunks:
            chunks.append([])
        chunks[-1].append(piece)
        speakers.add(speaker)
        size += _bytes(piece)
    return ["\n".join(chunk) for chunk in chunks]
