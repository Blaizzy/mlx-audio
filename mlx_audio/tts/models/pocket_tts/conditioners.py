import logging
from typing import NamedTuple

import mlx.core as mx
import mlx.nn as nn
import sentencepiece

from .utils import download_if_necessary

logger = logging.getLogger(__name__)


class TokenizedText(NamedTuple):
    tokens: mx.array


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


class LUTConditioner(nn.Module):
    def __init__(self, n_bins: int, tokenizer_path: str, dim: int, output_dim: int):
        super().__init__()
        self.tokenizer = SentencePieceTokenizer(n_bins, tokenizer_path)
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
