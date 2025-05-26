import re
from typing import Union

from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from .tokens import SpecialTokens


class PromptProcessor:
    def __init__(
        self, tokenizer: Union[str, PreTrainedTokenizer, PreTrainedTokenizerFast]
    ):
        self.special_tokens = SpecialTokens()

        if tokenizer:
            if isinstance(tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)):
                self.tokenizer = tokenizer
            elif isinstance(tokenizer, str):
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
            else:
                raise ValueError(f"Invalid tokenizer: {type(tokenizer)}")

            self.c1 = {}
            self.c2 = {}
            self.get_audio_token_map()

        self.input_prompt = "{bos}\n{text_start}{text}{text_end}\n{audio_start}\n"
        self.global_features = "{fs}{codes}{fe}\n"

    def get_audio_token_map(self):
        self.c1 = {
            self.tokenizer.encode(
                self.special_tokens.c1.format(i), add_special_tokens=False
            )[0]: i
            for i in range(1025)
        }
        self.c2 = {
            self.tokenizer.encode(
                self.special_tokens.c2.format(i), add_special_tokens=False
            )[0]: i
            for i in range(1025)
        }

    def get_features(self, f: dict):
        features = {
            "energy": f.get("energy", 0),
            "spectral_centroid": f.get("spectral_centroid", 0),
            "pitch": f.get("pitch", 0),
        }
        return [f"<|{k}_{v}|>" for k, v in features.items()]

    def get_global_features(self, f: dict):
        return self.global_features.format(
            fs=self.special_tokens.global_features_start,
            codes="".join(self.get_features(f)),
            fe=self.special_tokens.global_features_end,
        )

    def create_codes(self, words: dict):
        codes = []
        for i in words:
            word = (
                i["word"]
                + self.special_tokens.features
                + self.special_tokens.time.format(i["duration"])
            )
            word += "".join(self.get_features(i["features"]))
            pairs = []

            for idx in range(len(i["c1"])):
                c1 = self.special_tokens.c1.format(i["c1"][idx])
                c2 = self.special_tokens.c2.format(i["c2"][idx])
                pairs.append(f"{c1}{c2}")

            word += self.special_tokens.code + "".join(pairs)
            codes.append(
                self.special_tokens.word_start + word + self.special_tokens.word_end
            )

        return "\n".join(codes)

    def _init_prompt(self, text):
        return self.input_prompt.format(
            bos=self.special_tokens.bos,
            text_start=self.special_tokens.text_start,
            text=text,
            text_end=self.special_tokens.text_end,
            audio_start=self.special_tokens.audio_start,
        )

    def _get_separator(self, text: str) -> str:
        has_hiragana = any("\u3040" <= c <= "\u309f" for c in text)
        has_katakana = any("\u30a0" <= c <= "\u30ff" for c in text)
        has_han = any("\u4e00" <= c <= "\u9fff" for c in text)
        has_hangul = any("\uac00" <= c <= "\ud7af" for c in text)

        if has_hiragana or has_katakana or has_han:
            return "。"
        elif has_hangul:
            return ". "
        else:
            return ". "

    def merge_speaker_text(self, input_text: str, speaker_text: str) -> str:
        speaker_text = speaker_text.strip()
        separator = self._get_separator(speaker_text)

        # Determine allowed endings based on the separator
        if separator == "。":
            allowed_ends = ["。", "？", "！", "?", "!"]
        else:
            allowed_ends = [".", "?", "!"]

        rs = ""
        if speaker_text:
            last_char = speaker_text[-1]
            if last_char not in allowed_ends:
                rs = separator
            else:
                if separator != "。":
                    rs = " "

        output = speaker_text.strip() + rs + input_text.strip()

        return output, rs.strip()

    @staticmethod
    def text_normalizations(text: str) -> str:
        # Normalize whitespace characters (newlines, tabs, etc.) to single spaces
        text = re.sub(r"\s+", " ", text)
        text = text.replace("…", "...")  # Replace ellipsis character with three dots

        # Strip leading/trailing whitespace
        text = text.strip()

        # Normalize common Unicode characters to ASCII equivalents
        text = re.sub(r"[“”]", '"', text)  # Curly quotes to straight quotes
        text = re.sub(r"[‘’]", "'", text)  # Curly single quotes
        text = re.sub(r"[–—]", "-", text)  # Various dashes to hyphen

        # Remove control characters
        text = re.sub(r"[\x00-\x1F\x7F-\x9F]", "", text)

        return text

    def get_completion_prompt(self, text: str, speaker: dict = None):
        text = self.text_normalizations(text)

        if speaker is not None:
            text, separator = self.merge_speaker_text(text, speaker["text"])
            speaker["words"][-1]["word"] += separator
            codes = self.create_codes(speaker["words"])

        prompt = self._init_prompt(text)

        if speaker is not None:
            prompt += codes + "\n" + self.special_tokens.word_start

        return prompt

    def get_training_prompt(self, speaker: dict) -> str:
        text = self.text_normalizations(speaker["text"])
        words = speaker["words"]
        global_features = speaker["global_features"]

        prompt = self._init_prompt(text)
        prompt += self.get_global_features(global_features)
        prompt += self.create_codes(words)
        prompt += (
            "\n" + self.special_tokens.audio_end + "\n" + self.special_tokens.eos + "\n"
        )

        return prompt

    def extract_audio_from_tokens(self, tokens: list[int]):
        codebook1 = [self.c1[i] for i in tokens if i in self.c1]
        codebook2 = [self.c2[i] for i in tokens if i in self.c2]
        t = min(len(codebook1), len(codebook2))
        codebook1 = codebook1[:t]
        codebook2 = codebook2[:t]
        return [codebook1, codebook2]
