from pathlib import Path
from typing import List, Union

import mlx.core as mx

try:
    from tokenizers import Tokenizer
except ImportError:
    Tokenizer = None

# Special tokens
SOT = "[START]"
EOT = "[STOP]"
UNK = "[UNK]"
SPACE = "[SPACE]"
SPECIAL_TOKENS = [SOT, EOT, UNK, SPACE, "[PAD]", "[SEP]", "[CLS]", "[MASK]"]


class EnTokenizer:
    """
    English text tokenizer for Chatterbox TTS.

    Uses Hugging Face tokenizers library to load vocab from tokenizer.json.
    """

    def __init__(self, vocab_file_path: Union[str, Path]):
        if Tokenizer is None:
            raise ImportError(
                "tokenizers library required for Chatterbox text tokenization. "
                "Install with: pip install tokenizers"
            )
        self.tokenizer = Tokenizer.from_file(str(vocab_file_path))
        self._check_vocab()

    def _check_vocab(self):
        """Verify required special tokens exist in vocabulary."""
        vocab = self.tokenizer.get_vocab()
        if SOT not in vocab:
            raise ValueError(f"Tokenizer missing required token: {SOT}")
        if EOT not in vocab:
            raise ValueError(f"Tokenizer missing required token: {EOT}")

    def text_to_tokens(self, text: str) -> mx.array:

        token_ids = self.encode(text)
        return mx.array([token_ids], dtype=mx.int32)

    def encode(self, text: str) -> List[int]:

        text = text.replace(" ", SPACE)
        encoding = self.tokenizer.encode(text)
        return encoding.ids

    def decode(self, token_ids: Union[mx.array, List[int]]) -> str:
        if isinstance(token_ids, mx.array):
            token_ids = token_ids.tolist()

        if isinstance(token_ids[0], list):
            token_ids = token_ids[0]

        text = self.tokenizer.decode(token_ids, skip_special_tokens=False)

        text = text.replace(" ", "")
        text = text.replace(SPACE, " ")
        text = text.replace(EOT, "")
        text = text.replace(UNK, "")
        return text

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def get_sot_token_id(self) -> int:
        return self.tokenizer.token_to_id(SOT)

    def get_eot_token_id(self) -> int:
        return self.tokenizer.token_to_id(EOT)
