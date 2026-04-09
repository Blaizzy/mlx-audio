import json
from pathlib import Path
from typing import Iterable, List, Optional


class CohereAsrTokenizer:
    def __init__(
        self,
        model_path: str,
        tokenizer_config_path: Optional[str] = None,
        special_tokens_map_path: Optional[str] = None,
    ):
        import sentencepiece as spm

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)

        tokenizer_config = self._load_json(tokenizer_config_path)
        special_tokens_map = self._load_json(special_tokens_map_path)

        self.bos_token = tokenizer_config.get(
            "bos_token",
            special_tokens_map.get("bos_token", "<|startoftranscript|>"),
        )
        self.eos_token = tokenizer_config.get(
            "eos_token", special_tokens_map.get("eos_token", "<|endoftext|>")
        )
        self.pad_token = tokenizer_config.get(
            "pad_token", special_tokens_map.get("pad_token", "<pad>")
        )
        self.unk_token = tokenizer_config.get(
            "unk_token", special_tokens_map.get("unk_token", "<unk>")
        )

        additional = tokenizer_config.get("additional_special_tokens", [])
        if not additional:
            additional = special_tokens_map.get("additional_special_tokens", [])
        self.additional_special_tokens = list(additional)

        self.special_tokens = {
            self.bos_token,
            self.eos_token,
            self.pad_token,
            self.unk_token,
            *self.additional_special_tokens,
        }
        self.special_token_ids = {
            self.sp.piece_to_id(token)
            for token in self.special_tokens
            if self.sp.piece_to_id(token) >= 0
        }

        self.vocab_size = self.sp.get_piece_size()

    @staticmethod
    def _load_json(path: Optional[str]) -> dict:
        if path is None:
            return {}
        p = Path(path)
        if not p.exists():
            return {}
        with open(p, encoding="utf-8") as f:
            return json.load(f)

    @property
    def bos_token_id(self) -> int:
        return self.sp.piece_to_id(self.bos_token)

    @property
    def eos_token_id(self) -> int:
        return self.sp.piece_to_id(self.eos_token)

    @property
    def pad_token_id(self) -> int:
        return self.sp.piece_to_id(self.pad_token)

    @property
    def unk_token_id(self) -> int:
        return self.sp.piece_to_id(self.unk_token)

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        token_ids = list(self.sp.encode(text))
        if add_special_tokens:
            token_ids = [self.bos_token_id, *token_ids, self.eos_token_id]
        return token_ids

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        token_ids = [int(token_id) for token_id in ids if int(token_id) >= 0]
        if skip_special_tokens:
            filtered = [
                token_id
                for token_id in token_ids
                if token_id not in self.special_token_ids
            ]
            return self.sp.decode(filtered)

        output = []
        buffer = []
        for token_id in token_ids:
            piece = self.sp.id_to_piece(token_id)
            if piece in self.special_tokens:
                if buffer:
                    output.append(self.sp.decode(buffer))
                    buffer = []
                output.append(piece)
            else:
                buffer.append(token_id)
        if buffer:
            output.append(self.sp.decode(buffer))
        return "".join(output)

    def batch_decode(
        self, batch_ids: List[Iterable[int]], skip_special_tokens: bool = True
    ) -> List[str]:
        return [
            self.decode(token_ids, skip_special_tokens=skip_special_tokens)
            for token_ids in batch_ids
        ]

    def build_prompt_tokens(self, language: str, punctuation: bool = True) -> List[int]:
        tokens = [
            "<|startofcontext|>",
            "<|startoftranscript|>",
            "<|emo:undefined|>",
            f"<|{language}|>",
            f"<|{language}|>",
            "<|pnc|>" if punctuation else "<|nopnc|>",
            "<|noitn|>",
            "<|notimestamp|>",
            "<|nodiarize|>",
        ]
        return [self.sp.piece_to_id(token) for token in tokens]
