from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Iterable

from transformers import AutoTokenizer

EOS_TOKEN = "<|endoftext|>"
PAD_TOKEN = "<|pad|>"
IM_START_TOKEN = "<|im_start|>"
IM_END_TOKEN = "<|im_end|>"
MODALITY_TEXT_TOKEN = "<|text|>"
MODALITY_VOICE_TOKEN = "<|voice|>"
MODALITY_INTERLEAVE_TOKEN = "<|interleave|>"
SEMANTIC_TOKEN_TEMPLATE = "<|semantic:{i}|>"

MODALITY_TOKENS = {
    "text": MODALITY_TEXT_TOKEN,
    "voice": MODALITY_VOICE_TOKEN,
    "interleave": MODALITY_INTERLEAVE_TOKEN,
}


@dataclass
class FishTokenizer:
    tokenizer: any
    semantic_begin_id: int
    semantic_end_id: int
    _vocab_size: int

    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        vocab = self.tokenizer.get_vocab()
        self._vocab_size = max(vocab.values()) + 1
        semantic_ids = []
        for code_idx in range(4096):
            token = SEMANTIC_TOKEN_TEMPLATE.format(i=code_idx)
            token_id = vocab.get(token)
            if token_id is not None:
                semantic_ids.append(token_id)

        if len(semantic_ids) != 4096:
            raise ValueError(
                "Fish tokenizer is missing semantic tokens; expected 4096 semantic IDs."
            )

        self.semantic_begin_id = min(semantic_ids)
        self.semantic_end_id = max(semantic_ids)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        kwargs = {}
        sig = inspect.signature(self.tokenizer.encode)
        if "allowed_special" in sig.parameters:
            kwargs["allowed_special"] = "all"
        return self.tokenizer.encode(
            text, add_special_tokens=add_special_tokens, **kwargs
        )

    def decode(self, tokens: Iterable[int] | int, **kwargs) -> str:
        return self.tokenizer.decode(tokens, **kwargs)

    def get_token_id(self, token: str) -> int:
        return self.tokenizer.convert_tokens_to_ids(token)

    @classmethod
    def from_pretrained(cls, model_path: str) -> "FishTokenizer":
        return cls(model_path)
