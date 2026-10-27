import unittest

from mlx_audio.stt.models.whisper.decoding import get_suppress_tokens
from mlx_audio.stt.models.whisper.whisper import HFTokenizerWrapper

EOT = 50257


class VocabTokenizer:
    """Mimics HF Whisper tokenizers: missing tokens map to unk, which is eot."""

    eos_token_id = EOT

    def __init__(self, extra_tokens):
        self.vocab = {
            "<|endoftext|>": EOT,
            "<|startoftranscript|>": 50258,
            "<|translate|>": 50358,
            "<|transcribe|>": 50359,
            "<|startoflm|>": 50360,
            "<|startofprev|>": 50361,
            **extra_tokens,
        }

    def get_vocab(self):
        return dict(self.vocab)

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token, EOT)


class TestHFTokenizerWrapper(unittest.TestCase):
    def test_non_speech_tokens_skips_empty_seed_encodings(self):
        class EmptySeedTokenizer:
            def encode(self, text, add_special_tokens=False):
                if text in {" -", " '"}:
                    return []
                return [101]

        tokenizer = HFTokenizerWrapper(EmptySeedTokenizer(), multilingual=False)

        self.assertEqual(tokenizer.non_speech_tokens, (101,))

    def test_non_speech_tokens_skips_empty_miscellaneous_encodings(self):
        miscellaneous = set("♩♪♫♬♭♮♯")

        class EmptyMiscellaneousTokenizer:
            def encode(self, text, add_special_tokens=False):
                if text.strip() in miscellaneous:
                    return []
                return [101]

        tokenizer = HFTokenizerWrapper(
            EmptyMiscellaneousTokenizer(), multilingual=False
        )

        self.assertEqual(tokenizer.non_speech_tokens, (101,))

    def test_no_speech_uses_nocaptions_in_pre_v3_vocabulary(self):
        # tiny...large-v2 vocabularies name this token <|nocaptions|>
        tokenizer = HFTokenizerWrapper(VocabTokenizer({"<|nocaptions|>": 50362}))

        self.assertEqual(tokenizer.no_speech, 50362)
        self.assertNotIn(EOT, get_suppress_tokens(tokenizer, []))

    def test_no_speech_uses_nospeech_in_v3_vocabulary(self):
        tokenizer = HFTokenizerWrapper(VocabTokenizer({"<|nospeech|>": 50363}))

        self.assertEqual(tokenizer.no_speech, 50363)

    def test_missing_no_speech_token_does_not_suppress_eot(self):
        tokenizer = HFTokenizerWrapper(VocabTokenizer({}))

        self.assertIsNone(tokenizer.no_speech)
        self.assertNotIn(EOT, get_suppress_tokens(tokenizer, []))


if __name__ == "__main__":
    unittest.main()
