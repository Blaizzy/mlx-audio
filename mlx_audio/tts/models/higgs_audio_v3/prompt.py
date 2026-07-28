from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

import mlx.core as mx

AUDIO_PLACEHOLDER_ID = -100

_REQUIRED_SPECIALS = (
    "<|tts|>",
    "<|ref_audio|>",
    "<|text|>",
    "<|audio|>",
)


@dataclass
class ReferenceCodes:
    codes: mx.array
    text: Optional[str] = None


@dataclass
class PromptParts:
    token_ids: list[int]
    audio_segments: list[tuple[int, mx.array]]


class HiggsAudioV3PromptBuilder:
    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer
        vocab = dict(tokenizer.get_added_vocab())
        missing = [token for token in _REQUIRED_SPECIALS if token not in vocab]
        if missing:
            raise ValueError(f"Tokenizer is missing Higgs v3 specials: {missing}")
        self.tts_id = int(vocab["<|tts|>"])
        self.ref_audio_id = int(vocab["<|ref_audio|>"])
        self.text_id = int(vocab["<|text|>"])
        self.audio_id = int(vocab["<|audio|>"])
        self.ref_text_id = (
            int(vocab["<|ref_text|>"]) if "<|ref_text|>" in vocab else None
        )

    def encode_text(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def build_prompt(
        self,
        text: str,
        *,
        references: Iterable[ReferenceCodes] = (),
    ) -> PromptParts:
        ids: list[int] = [self.tts_id]
        segments: list[tuple[int, mx.array]] = []

        for reference in references:
            if reference.text and self.ref_text_id is not None:
                ids.append(self.ref_text_id)
                ids.extend(self.encode_text(reference.text))
            ids.append(self.ref_audio_id)
            start = len(ids)
            num_rows = int(reference.codes.shape[0])
            ids.extend([AUDIO_PLACEHOLDER_ID] * num_rows)
            segments.append((start, reference.codes))

        ids.append(self.text_id)
        ids.extend(self.encode_text(text))
        ids.append(self.audio_id)
        return PromptParts(token_ids=ids, audio_segments=segments)
