from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal, Optional

import mlx.core as mx

from .tokenizer import IM_END_TOKEN, IM_START_TOKEN, MODALITY_TOKENS, FishTokenizer


@dataclass
class TextPart:
    text: str


@dataclass
class VQPart:
    codes: mx.array

    def __init__(self, codes):
        self.codes = mx.array(codes, dtype=mx.int32)


Part = TextPart | VQPart


@dataclass
class Message:
    role: Literal["system", "user", "assistant"]
    parts: list[Part] = field(default_factory=list)
    add_im_start: bool = True
    add_im_end: bool = True
    modality: Optional[Literal["text", "voice", "interleave"]] = None


@dataclass
class Conversation:
    messages: list[Message] = field(default_factory=list)

    def append(self, message: Message) -> None:
        self.messages.append(message)

    def encode_for_inference(
        self, tokenizer: FishTokenizer, num_codebooks: int
    ) -> mx.array:
        segments: list[tuple[mx.array, Optional[mx.array]]] = []

        for message in self.messages:
            if message.add_im_start:
                modality_token = (
                    MODALITY_TOKENS[message.modality] if message.modality else ""
                )
                text = f"{IM_START_TOKEN}{message.role}\n{modality_token}"
                segments.append(
                    (mx.array(tokenizer.encode(text), dtype=mx.int32), None)
                )

            for part in message.parts:
                if isinstance(part, TextPart):
                    segments.append(
                        (mx.array(tokenizer.encode(part.text), dtype=mx.int32), None)
                    )
                elif isinstance(part, VQPart):
                    semantic_tokens = part.codes[0] + tokenizer.semantic_begin_id
                    segments.append(
                        (semantic_tokens.astype(mx.int32), part.codes.astype(mx.int32))
                    )
                else:
                    raise TypeError(f"Unsupported prompt part: {type(part).__name__}")

            if message.add_im_end:
                segments.append(
                    (
                        mx.array(tokenizer.encode(IM_END_TOKEN + "\n"), dtype=mx.int32),
                        None,
                    )
                )

        if not segments:
            raise ValueError("Conversation produced an empty prompt.")

        tokens = mx.concatenate([segment for segment, _ in segments], axis=0)
        values = mx.zeros((num_codebooks + 1, tokens.shape[0]), dtype=mx.int32)
        values[0] = tokens

        vq_segments = [vq for _, vq in segments if vq is not None]
        if not vq_segments:
            return values

        positions = []
        cursor = 0
        for segment, vq in segments:
            length = int(segment.shape[0])
            if vq is not None:
                positions.extend(range(cursor, cursor + length))
            cursor += length

        all_vq_codes = mx.concatenate(vq_segments, axis=1)
        values[1:, mx.array(positions, dtype=mx.int32)] = all_vq_codes
        return values


def split_text_by_speaker(text: str) -> list[str]:
    pattern = r"(<\|speaker:\d+\|>)"
    parts = re.split(pattern, text)

    turns = []
    i = 0
    while i < len(parts):
        part = parts[i].strip()
        if re.match(pattern, part):
            if i + 1 < len(parts):
                turns.append((part + parts[i + 1]).strip())
                i += 2
            else:
                turns.append(part)
                i += 1
        else:
            i += 1

    return turns


def group_turns_into_batches(
    turns: list[str], max_speakers: int = 5, max_bytes: int = 200
) -> list[str]:
    if not turns:
        return []

    batches = []
    current_batch = []
    current_bytes = 0
    for turn in turns:
        turn_bytes = len(turn.encode("utf-8"))
        would_exceed_speakers = len(current_batch) >= max_speakers
        would_exceed_bytes = current_batch and current_bytes + turn_bytes > max_bytes
        if would_exceed_speakers or would_exceed_bytes:
            batches.append("\n".join(current_batch))
            current_batch = [turn]
            current_bytes = turn_bytes
        else:
            current_batch.append(turn)
            current_bytes += turn_bytes

    if current_batch:
        batches.append("\n".join(current_batch))

    return batches
