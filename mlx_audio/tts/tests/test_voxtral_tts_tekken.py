import base64
import json
import tempfile
import unittest
from pathlib import Path

from mlx_audio.tts.models.voxtral_tts.tekken import TekkenTokenizer

TEKKEN_PATTERN = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|"
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|"
    r"\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


def _byte_vocab() -> list[dict]:
    vocab = []
    for i in range(256):
        vocab.append(
            {
                "rank": i,
                "token_bytes": base64.b64encode(bytes([i])).decode("ascii"),
                "token_str": None,
            }
        )
    return vocab


class TestVoxtralTTSTekkenTokenizer(unittest.TestCase):
    def _write_tekken(self) -> Path:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        path = Path(tmpdir.name) / "tekken.json"
        path.write_text(
            json.dumps(
                {
                    "config": {
                        "pattern": TEKKEN_PATTERN,
                        "num_vocab_tokens": 256,
                        "default_vocab_size": 264,
                        "default_num_special_tokens": 8,
                        "version": "v3",
                    },
                    "vocab": _byte_vocab(),
                    "special_tokens": [
                        {"rank": 0, "token_str": "<unk>", "is_control": True},
                        {"rank": 1, "token_str": "<s>", "is_control": True},
                        {"rank": 2, "token_str": "</s>", "is_control": True},
                        {"rank": 3, "token_str": "<pad>", "is_control": True},
                        {
                            "rank": 4,
                            "token_str": "[REPEAT_AUDIO_TEXT]",
                            "is_control": True,
                        },
                        {
                            "rank": 5,
                            "token_str": "[NEXT_AUDIO_TEXT]",
                            "is_control": True,
                        },
                    ],
                    "audio": {
                        "sampling_rate": 24000,
                        "frame_rate": 12.5,
                        "chunk_length_s": None,
                        "audio_encoding_config": {
                            "num_mel_bins": 128,
                            "hop_length": 256,
                            "window_size": 1024,
                        },
                        "voice_num_audio_tokens": {"casual_male": 147},
                    },
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_from_file_exposes_special_tokens_and_audio_metadata(self):
        tokenizer = TekkenTokenizer.from_file(self._write_tekken())

        self.assertEqual(tokenizer.bos_id, 1)
        self.assertEqual(tokenizer.eos_id, 2)
        self.assertEqual(tokenizer.pad_id, 3)
        self.assertEqual(tokenizer.token_to_id("[NEXT_AUDIO_TEXT]"), 5)
        self.assertEqual(tokenizer.audio["voice_num_audio_tokens"]["casual_male"], 147)

    def test_encode_decode_roundtrip_handles_multilingual_text(self):
        tokenizer = TekkenTokenizer.from_file(self._write_tekken())
        text = "Hello مرحبا 123"

        token_ids = tokenizer.encode(text)

        self.assertTrue(token_ids)
        self.assertTrue(
            all(token_id >= tokenizer.num_special_tokens for token_id in token_ids)
        )
        self.assertEqual(tokenizer.decode(token_ids), text)

    def test_encode_with_bos_and_eos(self):
        tokenizer = TekkenTokenizer.from_file(self._write_tekken())

        token_ids = tokenizer.encode("hello", bos=True, eos=True)

        self.assertEqual(token_ids[0], tokenizer.bos_id)
        self.assertEqual(token_ids[-1], tokenizer.eos_id)

    def test_optionally_matches_mistral_common_tekkenizer(self):
        try:
            from mistral_common.tokens.tokenizers.tekken import Tekkenizer
        except ImportError:
            return

        path = self._write_tekken()
        tokenizer = TekkenTokenizer.from_file(path)
        reference = Tekkenizer.from_file(path)

        self.assertEqual(tokenizer.bos_id, reference.bos_id)
        self.assertEqual(tokenizer.eos_id, reference.eos_id)
        self.assertEqual(tokenizer.pad_id, reference.pad_id)
        self.assertEqual(
            tokenizer.audio["voice_num_audio_tokens"]["casual_male"],
            reference.audio.voice_num_audio_tokens["casual_male"],
        )

        for special_token in (
            "<unk>",
            "<s>",
            "</s>",
            "<pad>",
            "[REPEAT_AUDIO_TEXT]",
            "[NEXT_AUDIO_TEXT]",
        ):
            token_id = tokenizer.get_special_token(special_token)
            self.assertEqual(token_id, reference.get_special_token(special_token))
            self.assertEqual(
                tokenizer.id_to_piece(token_id), reference.id_to_piece(token_id)
            )

        for text in (
            "Hello world.",
            "Hello مرحبا 123",
            "Line 1\nLine 2",
            "Symbols: %$#",
        ):
            token_ids = tokenizer.encode(text)
            self.assertEqual(token_ids, reference.encode(text, bos=False, eos=False))
            self.assertEqual(tokenizer.decode(token_ids), reference.decode(token_ids))
            self.assertEqual(
                tokenizer.encode(text, bos=True, eos=True),
                reference.encode(text, bos=True, eos=True),
            )


if __name__ == "__main__":
    unittest.main()
