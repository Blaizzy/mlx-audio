from pathlib import Path
from typing import Dict, List, Optional


class CanaryTokenizer:
    """SentencePiece-based tokenizer for canary models.

    Handles both the SentencePiece model for text encoding/decoding
    and the special tokens used by the Canary prompt format.
    """

    def __init__(self, model_path: str, tokens_path: Optional[str] = None):
        """Initialize tokenizer.

        Args:
            model_path: Path to sentencepiece .model file
            tokens_path: Optional path to tokens.txt for ID mapping
        """
        import sentencepiece as spm

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)

        self.vocab_size = self.sp.get_piece_size()

        self.token2id: Dict[str, int] = {}
        self.id2token: Dict[int, str] = {}

        if tokens_path and Path(tokens_path).exists():
            self._load_tokens_file(tokens_path)
        else:
            self._build_token_maps()

    def _load_tokens_file(self, path: str):
        """Load token mapping from tokens.txt (sherpa-onnx format)."""
        with open(path, encoding="utf-8") as f:
            for line in f:
                fields = line.strip().split()
                if len(fields) == 2:
                    token, idx = fields[0], int(fields[1])
                    if line[0] == " ":
                        token = " " + token
                elif len(fields) == 1:
                    token = " "
                    idx = int(fields[0])
                else:
                    continue
                self.token2id[token] = idx
                self.id2token[idx] = token

    def _build_token_maps(self):
        """Build token maps from sentencepiece model."""
        for i in range(self.vocab_size):
            token = self.sp.id_to_piece(i)
            self.token2id[token] = i
            self.id2token[i] = token

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        return self.sp.encode(text)

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text."""
        return self.sp.decode(ids)

    def get_special_token_id(self, token: str) -> Optional[int]:
        """Get ID for a special token like <|en|>."""
        return self.token2id.get(token, None)

    def build_prompt_tokens(
        self,
        source_lang: str = "en",
        target_lang: str = "en",
        use_pnc: bool = True,
    ) -> List[int]:
        """Build the Canary prompt token sequence.

        Format:
        <|startofcontext|> <|startoftranscript|> <|emo:undefined|>
        <|{src_lang}|> <|{tgt_lang}|> <|pnc|> <|noitn|> <|notimestamp|> <|nodiarize|>
        """
        tokens = []
        tokens.append(self.token2id["<|startofcontext|>"])
        tokens.append(self.token2id["<|startoftranscript|>"])
        tokens.append(self.token2id["<|emo:undefined|>"])
        tokens.append(self.token2id[f"<|{source_lang}|>"])
        tokens.append(self.token2id[f"<|{target_lang}|>"])
        tokens.append(
            self.token2id["<|pnc|>"] if use_pnc else self.token2id["<|nopnc|>"]
        )
        tokens.append(self.token2id["<|noitn|>"])
        tokens.append(self.token2id["<|notimestamp|>"])
        tokens.append(self.token2id["<|nodiarize|>"])
        return tokens

    @property
    def eos_id(self) -> int:
        """End-of-text token ID."""
        return self.token2id.get("<|endoftext|>", 0)
