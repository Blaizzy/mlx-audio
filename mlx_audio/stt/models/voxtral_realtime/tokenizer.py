"""Tekken tokenizer for Voxtral Realtime (decode-only).

Token layout:
- IDs 0..999: Special tokens (BOS=1, EOS=2, STREAMING_PAD=32)
- IDs 1000+: Regular vocabulary (index = token_id - n_special)
  Each entry has base64-encoded UTF-8 bytes in tekken.json.
"""

import base64
import json
from pathlib import Path


class TekkenTokenizer:
    def __init__(self, tekken_path: str):
        with open(tekken_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.vocab = data["vocab"]
        config = data.get("config", {})
        self.n_special = int(config.get("default_num_special_tokens", 1000))
        self.special_ids = {
            int(st["rank"]) for st in data.get("special_tokens", []) if "rank" in st
        }

        # Cache decoded bytes per token ID
        self._bytes_cache = {}

    def token_bytes(self, token_id: int) -> bytes:
        cached = self._bytes_cache.get(token_id)
        if cached is not None:
            return cached

        if token_id < 0 or token_id < self.n_special or token_id in self.special_ids:
            self._bytes_cache[token_id] = b""
            return b""

        vocab_id = token_id - self.n_special
        if vocab_id < 0 or vocab_id >= len(self.vocab):
            self._bytes_cache[token_id] = b""
            return b""

        b = base64.b64decode(self.vocab[vocab_id]["token_bytes"])
        self._bytes_cache[token_id] = b
        return b

    def decode(self, token_ids) -> str:
        """Decode a sequence of token IDs to text."""
        out = bytearray()
        for token_id in token_ids:
            tid = int(token_id)
            if tid < self.n_special or tid in self.special_ids:
                continue
            out += self.token_bytes(tid)
        return out.decode("utf-8", errors="replace")

    @classmethod
    def from_model_path(cls, model_path) -> "TekkenTokenizer":
        """Load tokenizer from a model directory."""
        model_path = Path(model_path)
        tekken_path = model_path / "tekken.json"
        if not tekken_path.exists():
            raise FileNotFoundError(f"tekken.json not found at {model_path}")
        return cls(str(tekken_path))
