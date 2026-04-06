"""Minimal Tekken tokenizer for Voxtral TTS."""

from __future__ import annotations

import base64
import json
from functools import cached_property
from itertools import groupby
from pathlib import Path
from typing import TypedDict

try:
    import tiktoken
except ImportError:
    tiktoken = None


class TokenInfo(TypedDict):
    rank: int
    token_bytes: str
    token_str: str | None


class SpecialTokenInfo(TypedDict):
    rank: int
    token_str: str
    is_control: bool


class TekkenConfig(TypedDict):
    pattern: str
    num_vocab_tokens: int
    default_vocab_size: int
    default_num_special_tokens: int
    version: str


def is_tekken(path: str | Path) -> bool:
    """Return True when `path` points to a tekken tokenizer JSON file."""
    if isinstance(path, str):
        path = Path(path)
    return path.is_file() and path.suffix == ".json" and "tekken" in path.name


def _reload_mergeable_ranks(
    vocab: list[TokenInfo],
    max_vocab: int | None = None,
) -> dict[bytes, int]:
    """Convert tekken vocab entries to the byte->rank format expected by tiktoken."""
    if max_vocab is not None and len(vocab) > max_vocab:
        vocab = vocab[:max_vocab]

    ranks: dict[bytes, int] = {}
    for idx, token in enumerate(vocab):
        if token["rank"] != idx:
            raise ValueError(f"Invalid tekken vocab rank at index {idx}: {token}")
        merge = base64.b64decode(token["token_bytes"])
        if idx < 256 and merge != bytes([idx]):
            raise ValueError(f"Expected byte token {idx} to decode to {bytes([idx])!r}")
        ranks[merge] = token["rank"]

    if len(ranks) != len(vocab):
        raise ValueError("Duplicate mergeable ranks found in tekken vocab.")
    return ranks


class TekkenTokenizer:
    """Small Tekken tokenizer wrapper for Voxtral's text + special-token needs."""

    SPECIAL_TOKEN_TEMPLATE = "<SPECIAL_{id}>"

    def __init__(
        self,
        vocab: list[TokenInfo],
        special_tokens: list[SpecialTokenInfo],
        pattern: str,
        vocab_size: int,
        num_special_tokens: int,
        *,
        path: str | Path | None = None,
        audio_config: dict | None = None,
        name: str = "tekkenizer",
    ):
        if vocab_size > len(vocab) + num_special_tokens:
            raise ValueError(
                f"Invalid Tekken vocab size: {vocab_size=} exceeds {len(vocab)} + {num_special_tokens}"
            )
        if tiktoken is None:
            raise ImportError(
                "TekkenTokenizer requires `tiktoken`. Install mlx-audio with the "
                "`tts` extra or add `tiktoken` to your environment."
            )

        special_tokens = sorted(special_tokens, key=lambda token: token["rank"])
        defined_specials = {token["token_str"] for token in special_tokens}
        if len(defined_specials) != len(special_tokens):
            raise ValueError("Tekken special tokens must be unique.")
        if [token["rank"] for token in special_tokens] != list(
            range(len(special_tokens))
        ):
            raise ValueError("Tekken special token ranks must be contiguous from zero.")

        filler = [
            SpecialTokenInfo(
                rank=i,
                token_str=self.SPECIAL_TOKEN_TEMPLATE.format(id=i),
                is_control=True,
            )
            for i in range(len(special_tokens), num_special_tokens)
        ]
        special_tokens = list(special_tokens) + filler

        inner_vocab_size = vocab_size - num_special_tokens
        mergeable_ranks = _reload_mergeable_ranks(vocab, max_vocab=inner_vocab_size)
        self._model = tiktoken.Encoding(
            name=name,
            pat_str=pattern,
            mergeable_ranks=mergeable_ranks,
            special_tokens={},
        )

        self._file_path = Path(path) if path is not None else None
        self._audio_config = audio_config or {}
        self._vocab_size = vocab_size
        self._num_special_tokens = num_special_tokens
        self._special_tokens = special_tokens
        self._special_tokens_reverse_vocab = {
            token["token_str"]: token["rank"] for token in special_tokens
        }
        self._special_token_ids = {token["rank"] for token in special_tokens}
        self._vocab = [self.id_to_piece(i) for i in range(vocab_size)]

    @classmethod
    def from_file(cls, path: str | Path) -> "TekkenTokenizer":
        """Load a Tekken tokenizer from `tekken.json`."""
        if isinstance(path, str):
            path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        config: TekkenConfig = data["config"]
        special_tokens: list[SpecialTokenInfo] = data.get("special_tokens", [])
        return cls(
            vocab=data["vocab"],
            special_tokens=special_tokens,
            pattern=config["pattern"],
            vocab_size=config["default_vocab_size"],
            num_special_tokens=config["default_num_special_tokens"],
            path=path,
            audio_config=data.get("audio"),
            name=path.stem,
        )

    @property
    def file_path(self) -> Path:
        if self._file_path is None:
            raise ValueError("Tokenizer was not loaded from a file.")
        return self._file_path

    @property
    def audio(self) -> dict:
        return self._audio_config

    @property
    def num_special_tokens(self) -> int:
        return self._num_special_tokens

    @property
    def n_words(self) -> int:
        return self._vocab_size

    @cached_property
    def bos_id(self) -> int:
        return self.get_special_token("<s>")

    @cached_property
    def eos_id(self) -> int:
        return self.get_special_token("</s>")

    @cached_property
    def pad_id(self) -> int:
        return self.get_special_token("<pad>")

    @cached_property
    def unk_id(self) -> int:
        return self.get_special_token("<unk>")

    def vocab(self) -> list[str]:
        return self._vocab

    def token_to_id(self, token: str) -> int | None:
        """Return the id for a special token string, or None if it is unknown."""
        return self._special_tokens_reverse_vocab.get(token)

    def get_special_token(self, token: str) -> int:
        token_id = self.token_to_id(token)
        if token_id is None:
            raise ValueError(f"Unknown special token {token!r}")
        return token_id

    def is_special(self, token: int | str) -> bool:
        if isinstance(token, int):
            return token in self._special_token_ids
        return token in self._special_tokens_reverse_vocab

    def encode(
        self,
        text: str,
        add_special_tokens: bool | None = None,
        *,
        bos: bool = False,
        eos: bool = False,
    ) -> list[int]:
        """Encode text to token ids.

        `add_special_tokens=False` matches the HF-style callsite used in MLX.
        """
        if add_special_tokens:
            bos = True
            eos = True
        token_ids = [t + self.num_special_tokens for t in self._model.encode(text)]
        if bos:
            token_ids = [self.bos_id, *token_ids]
        if eos:
            token_ids = [*token_ids, self.eos_id]
        return token_ids

    def _decode_all(self, token_ids: list[int], keep_special: bool) -> list[str]:
        decoded: list[str] = []
        for is_special, group in groupby(
            token_ids, lambda token_id: token_id < self.num_special_tokens
        ):
            group_list = list(group)
            if is_special:
                if keep_special:
                    decoded.extend(
                        self._special_tokens[token_id]["token_str"]
                        for token_id in group_list
                    )
                continue
            decoded.append(
                self._model.decode(
                    [token_id - self.num_special_tokens for token_id in group_list]
                )
            )
        return decoded

    def decode(self, token_ids: list[int], *, keep_special: bool = False) -> str:
        return "".join(self._decode_all(token_ids, keep_special=keep_special))

    def id_to_piece(self, token_id: int) -> str:
        return self.decode([token_id], keep_special=True)

    def id_to_byte_piece(self, token_id: int) -> bytes:
        if token_id < self.num_special_tokens:
            return self._special_tokens[token_id]["token_str"].encode("utf-8")
        return self._model.decode_single_token_bytes(token_id - self.num_special_tokens)
