# Ported from https://github.com/resemble-ai/chatterbox

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
        """
        Convert text to token IDs.

        Args:
            text: Input text string

        Returns:
            Token IDs as MLX array with shape (1, seq_len)
        """
        token_ids = self.encode(text)
        return mx.array([token_ids], dtype=mx.int32)

    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs.

        Replaces spaces with SPACE token before encoding.

        Args:
            text: Input text string

        Returns:
            List of token IDs
        """
        text = text.replace(" ", SPACE)
        encoding = self.tokenizer.encode(text)
        return encoding.ids

    def decode(self, token_ids: Union[mx.array, List[int]]) -> str:
        """
        Decode token IDs back to text.

        Args:
            token_ids: Token IDs as MLX array or list

        Returns:
            Decoded text string
        """
        if isinstance(token_ids, mx.array):
            token_ids = token_ids.tolist()

        # Flatten if 2D
        if isinstance(token_ids[0], list):
            token_ids = token_ids[0]

        text = self.tokenizer.decode(token_ids, skip_special_tokens=False)
        # Clean up special tokens
        text = text.replace(" ", "")
        text = text.replace(SPACE, " ")
        text = text.replace(EOT, "")
        text = text.replace(UNK, "")
        return text

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.tokenizer.get_vocab_size()

    def get_sot_token_id(self) -> int:
        """Get start-of-text token ID."""
        return self.tokenizer.token_to_id(SOT)

    def get_eot_token_id(self) -> int:
        """Get end-of-text token ID."""
        return self.tokenizer.token_to_id(EOT)
