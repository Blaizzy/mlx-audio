import logging
import math
from typing import NamedTuple

import mlx.core as mx
import mlx.nn as nn
import sentencepiece

from .utils import download_if_necessary

logger = logging.getLogger(__name__)


class TokenizedText(NamedTuple):
    tokens: mx.array


class SpeakerCountConditioner(nn.Module):
    def __init__(self, n_bins: int, dim: int, output_dim: int):
        super().__init__()
        self.embed = nn.Embedding(n_bins + 1, dim)
        self.output_proj = nn.Linear(dim, output_dim, bias=False)

    def __call__(self, tokens: mx.array) -> mx.array:
        return self.output_proj(self.embed(tokens))


class ContinuousConditioner(nn.Module):
    def __init__(
        self,
        dim: int,
        output_dim: int,
        scale_factor: float,
        max_period: float = 10000.0,
    ):
        super().__init__()
        if dim < 4 or dim % 2:
            raise ValueError("Continuous conditioning requires an even dim >= 4.")
        self.scale_factor = scale_factor
        self._freqs = mx.exp(
            -math.log(max_period)
            * mx.arange(dim // 2, dtype=mx.float32)
            / (dim // 2 - 1)
        )
        self.output_proj = (
            nn.Linear(dim, output_dim, bias=False) if dim != output_dim else None
        )

    def __call__(self, values: mx.array) -> mx.array:
        phase = values[..., None] * self.scale_factor * self._freqs
        embedding = mx.concatenate([mx.cos(phase), mx.sin(phase)], axis=-1)
        if self.output_proj is not None:
            embedding = embedding.astype(self.output_proj.weight.dtype)
            embedding = self.output_proj(embedding)
        return embedding


class SentencePieceTokenizer:
    def __init__(self, n_bins: int, tokenizer_path: str) -> None:
        logger.info("Loading sentencepiece tokenizer from %s", tokenizer_path)
        tokenizer_path = download_if_necessary(tokenizer_path)
        self.sp = sentencepiece.SentencePieceProcessor(str(tokenizer_path))
        if n_bins != self.sp.vocab_size():
            raise ValueError(
                f"sentencepiece tokenizer has vocab size={self.sp.vocab_size()} "
                f"but n_bins={n_bins} was specified"
            )

    def __call__(self, text: str) -> TokenizedText:
        tokens = self.sp.encode(text, out_type=int)
        return TokenizedText(mx.array(tokens, dtype=mx.int32)[None, :])

    def decode(self, tokens: list[int]) -> str:
        return self.sp.decode(tokens)


class HuggingFaceTokenizer:
    def __init__(self, n_bins: int, tokenizer_path: str):
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(str(download_if_necessary(tokenizer_path)))
        if self.tokenizer.get_vocab_size() != n_bins:
            raise ValueError("Tokenizer vocabulary does not match n_bins.")

    def __call__(self, text: str) -> TokenizedText:
        tokens = self.tokenizer.encode(text, add_special_tokens=False).ids
        return TokenizedText(mx.array(tokens, dtype=mx.int32)[None, :])

    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=False)


class LUTConditioner(nn.Module):
    def __init__(
        self,
        n_bins: int,
        tokenizer_path: str,
        dim: int,
        output_dim: int,
        tokenizer: str = "sentencepiece",
    ):
        super().__init__()
        tokenizer_cls = (
            HuggingFaceTokenizer
            if tokenizer == "huggingface"
            else SentencePieceTokenizer
        )
        self.tokenizer = tokenizer_cls(n_bins, tokenizer_path)
        self.dim = dim
        self.output_dim = output_dim
        self.embed = nn.Embedding(n_bins + 1, dim)
        self.output_proj = (
            None if dim == output_dim else nn.Linear(dim, output_dim, bias=False)
        )

    def prepare(self, text: str) -> TokenizedText:
        return self.tokenizer(text)

    def __call__(self, inputs: TokenizedText) -> mx.array:
        embeds = self.embed(inputs.tokens)
        if self.output_proj is not None:
            embeds = self.output_proj(embeds)
        return embeds
