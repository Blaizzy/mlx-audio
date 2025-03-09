from functools import cache
from typing import Any, Literal, Iterable

import torch
import torch.nn as nn

from tts.models.zonos.config import PrefixConditionerConfig
from tts.models.zonos.utils import DEFAULT_DEVICE


class Conditioner(nn.Module):
    def __init__(
        self,
        output_dim: int,
        name: str,
        cond_dim: int | None = None,
        projection: Literal["none", "linear", "mlp"] = "none",
        uncond_type: Literal["learned", "none"] = "none",
        **kwargs,
    ):
        super().__init__()
        self.name = name
        self.output_dim = output_dim
        self.cond_dim = cond_dim = cond_dim or output_dim

        if projection == "linear":
            self.project = nn.Linear(cond_dim, output_dim)
        elif projection == "mlp":
            self.project = nn.Sequential(
                nn.Linear(cond_dim, output_dim),
                nn.SiLU(),
                nn.Linear(output_dim, output_dim),
            )
        else:
            self.project = nn.Identity()

        self.uncond_vector = None
        if uncond_type == "learned":
            self.uncond_vector = nn.Parameter(torch.zeros(output_dim))

    def apply_cond(self, *inputs: Any) -> torch.Tensor:
        raise NotImplementedError()

    def forward(self, inputs: tuple[Any, ...] | None) -> torch.Tensor:
        if inputs is None:
            assert self.uncond_vector is not None
            return self.uncond_vector.data.view(1, 1, -1)

        cond = self.apply_cond(*inputs)
        cond = self.project(cond)
        return cond


# ------- ESPEAK CONTAINMENT ZONE ------------------------------------------------------------------------------------------------------------------------------------------------
import os
import sys
import re
import unicodedata

import inflect
import torch
import torch.nn as nn
from kanjize import number2kanji
from phonemizer.backend import EspeakBackend
from sudachipy import Dictionary, SplitMode

if sys.platform == "darwin":
    os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = "/opt/homebrew/lib/libespeak-ng.dylib"

# --- Number normalization code from https://github.com/daniilrobnikov/vits2/blob/main/text/normalize_numbers.py ---

_inflect = inflect.engine()
_comma_number_re = re.compile(r"([0-9][0-9\,]+[0-9])")
_decimal_number_re = re.compile(r"([0-9]+\.[0-9]+)")
_pounds_re = re.compile(r"£([0-9\,]*[0-9]+)")
_dollars_re = re.compile(r"\$([0-9\.\,]*[0-9]+)")
_ordinal_re = re.compile(r"[0-9]+(st|nd|rd|th)")
_number_re = re.compile(r"[0-9]+")


def _remove_commas(m: re.Match) -> str:
    return m.group(1).replace(",", "")


def _expand_decimal_point(m: re.Match) -> str:
    return m.group(1).replace(".", " point ")


def _expand_dollars(m: re.Match) -> str:
    match = m.group(1)
    parts = match.split(".")
    if len(parts) > 2:
        return match + " dollars"  # Unexpected format
    dollars = int(parts[0]) if parts[0] else 0
    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    if dollars and cents:
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        cent_unit = "cent" if cents == 1 else "cents"
        return "%s %s, %s %s" % (dollars, dollar_unit, cents, cent_unit)
    elif dollars:
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        return "%s %s" % (dollars, dollar_unit)
    elif cents:
        cent_unit = "cent" if cents == 1 else "cents"
        return "%s %s" % (cents, cent_unit)
    else:
        return "zero dollars"


def _expand_ordinal(m: re.Match) -> str:
    return _inflect.number_to_words(m.group(0))


def _expand_number(m: re.Match) -> str:
    num = int(m.group(0))
    if num > 1000 and num < 3000:
        if num == 2000:
            return "two thousand"
        elif num > 2000 and num < 2010:
            return "two thousand " + _inflect.number_to_words(num % 100)
        elif num % 100 == 0:
            return _inflect.number_to_words(num // 100) + " hundred"
        else:
            return _inflect.number_to_words(num, andword="", zero="oh", group=2).replace(", ", " ")
    else:
        return _inflect.number_to_words(num, andword="")


def normalize_numbers(text: str) -> str:
    text = re.sub(_comma_number_re, _remove_commas, text)
    text = re.sub(_pounds_re, r"\1 pounds", text)
    text = re.sub(_dollars_re, _expand_dollars, text)
    text = re.sub(_decimal_number_re, _expand_decimal_point, text)
    text = re.sub(_ordinal_re, _expand_ordinal, text)
    text = re.sub(_number_re, _expand_number, text)
    return text


# --- Number normalization code end ---


PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3
SPECIAL_TOKEN_IDS = [PAD_ID, UNK_ID, BOS_ID, EOS_ID]

_punctuation = ';:,.!?¡¿—…"«»“”() *~-/\\&'
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_letters_ipa = (
    "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
)

symbols = [*_punctuation, *_letters, *_letters_ipa]
_symbol_to_id = {s: i for i, s in enumerate(symbols, start=len(SPECIAL_TOKEN_IDS))}


def _get_symbol_id(s: str) -> int:
    return _symbol_to_id.get(s, 1)


def get_symbol_ids(text: str) -> list[int]:
    return list(map(_get_symbol_id, text))


def tokenize_phonemes(phonemes: list[str]) -> tuple[torch.Tensor, list[int]]:
    phoneme_ids = [[BOS_ID, *get_symbol_ids(phonemes), EOS_ID] for phonemes in phonemes]
    lengths = list(map(len, phoneme_ids))
    longest = max(lengths)
    phoneme_ids = [[PAD_ID] * (longest - len(ids)) + ids for ids in phoneme_ids]
    return torch.tensor(phoneme_ids), lengths


def normalize_jp_text(text: str, tokenizer=Dictionary(dict="full").create()) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\d+", lambda m: number2kanji(int(m[0])), text)
    final_text = " ".join([x.reading_form() for x in tokenizer.tokenize(text, SplitMode.A)])
    return final_text


def clean(texts: list[str], languages: list[str]) -> list[str]:
    texts_out = []
    for text, language in zip(texts, languages):
        if "ja" in language:
            text = normalize_jp_text(text)
        else:
            text = normalize_numbers(text)
        texts_out.append(text)
    return texts_out


@cache
def get_backend(language: str) -> "EspeakBackend":
    import logging

    from phonemizer.backend import EspeakBackend

    logger = logging.getLogger("phonemizer")
    backend = EspeakBackend(
        language,
        preserve_punctuation=True,
        with_stress=True,
        punctuation_marks=_punctuation,
        logger=logger,
    )
    logger.setLevel(logging.ERROR)
    return backend


def phonemize(texts: list[str], languages: list[str]) -> list[str]:
    texts = clean(texts, languages)

    batch_phonemes = []
    for text, language in zip(texts, languages):
        backend = get_backend(language)
        phonemes = backend.phonemize([text], strip=True)
        batch_phonemes.append(phonemes[0])

    return batch_phonemes


class EspeakPhonemeConditioner(Conditioner):
    def __init__(self, output_dim: int, **kwargs):
        super().__init__(output_dim, **kwargs)
        self.phoneme_embedder = nn.Embedding(len(SPECIAL_TOKEN_IDS) + len(symbols), output_dim)

    def apply_cond(self, texts: list[str], languages: list[str]) -> torch.Tensor:
        """
        Args:
            texts: list of texts to convert to phonemes
            languages: ISO 639-1 -or otherwise eSpeak compatible- language code
        """
        device = self.phoneme_embedder.weight.device

        phonemes = phonemize(texts, languages)
        phoneme_ids, _ = tokenize_phonemes(phonemes)
        phoneme_embeds = self.phoneme_embedder(phoneme_ids.to(device))

        return phoneme_embeds


# ------- ESPEAK CONTAINMENT ZONE ------------------------------------------------------------------------------------------------------------------------------------------------


class FourierConditioner(Conditioner):
    def __init__(
        self,
        output_dim: int,
        input_dim: int = 1,
        std: float = 1.0,
        min_val: float = 0.0,
        max_val: float = 1.0,
        **kwargs,
    ):
        assert output_dim % 2 == 0
        super().__init__(output_dim, **kwargs)
        self.register_buffer("weight", torch.randn([output_dim // 2, input_dim]) * std)
        self.input_dim, self.min_val, self.max_val = input_dim, min_val, max_val

    def apply_cond(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.input_dim
        x = (x - self.min_val) / (self.max_val - self.min_val)  # [batch_size, seq_len, input_dim]
        f = 2 * torch.pi * x.to(self.weight.dtype) @ self.weight.T  # [batch_size, seq_len, output_dim // 2]
        return torch.cat([f.cos(), f.sin()], dim=-1)  # [batch_size, seq_len, output_dim]


class IntegerConditioner(Conditioner):
    def __init__(self, output_dim: int, min_val: int = 0, max_val: int = 512, **kwargs):
        super().__init__(output_dim, **kwargs)
        self.min_val = min_val
        self.max_val = max_val
        self.int_embedder = nn.Embedding(max_val - min_val + 1, output_dim)

    def apply_cond(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == 1
        return self.int_embedder(x.squeeze(-1) - self.min_val)  # [batch_size, seq_len, output_dim]


class PassthroughConditioner(Conditioner):
    def __init__(self, output_dim: int, **kwargs):
        super().__init__(output_dim, **kwargs)

    def apply_cond(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.cond_dim
        return x


_cond_cls_map = {
    "PassthroughConditioner": PassthroughConditioner,
    "EspeakPhonemeConditioner": EspeakPhonemeConditioner,
    "FourierConditioner": FourierConditioner,
    "IntegerConditioner": IntegerConditioner,
}


def build_conditioners(conditioners: list[dict], output_dim: int) -> list[Conditioner]:
    return [_cond_cls_map[config["type"]](output_dim, **config) for config in conditioners]


class PrefixConditioner(Conditioner):
    def __init__(self, config: PrefixConditionerConfig, output_dim: int):
        super().__init__(output_dim, "prefix", projection=config.projection)
        self.conditioners = nn.ModuleList(build_conditioners(config.conditioners, output_dim))
        self.norm = nn.LayerNorm(output_dim)
        self.required_keys = {c.name for c in self.conditioners if c.uncond_vector is None}

    def forward(self, cond_dict: dict) -> torch.Tensor:
        if not set(cond_dict).issuperset(self.required_keys):
            raise ValueError(f"Missing required keys: {self.required_keys - set(cond_dict)}")
        conds = []
        for conditioner in self.conditioners:
            conds.append(conditioner(cond_dict.get(conditioner.name)))
        max_bsz = max(map(len, conds))
        assert all(c.shape[0] in (max_bsz, 1) for c in conds)
        conds = [c.expand(max_bsz, -1, -1) for c in conds]
        return self.norm(self.project(torch.cat(conds, dim=-2)))


supported_language_codes = [
    'af', 'am', 'an', 'ar', 'as', 'az', 'ba', 'bg', 'bn', 'bpy', 'bs', 'ca', 'cmn',
    'cs', 'cy', 'da', 'de', 'el', 'en-029', 'en-gb', 'en-gb-scotland', 'en-gb-x-gbclan',
    'en-gb-x-gbcwmd', 'en-gb-x-rp', 'en-us', 'eo', 'es', 'es-419', 'et', 'eu', 'fa',
    'fa-latn', 'fi', 'fr-be', 'fr-ch', 'fr-fr', 'ga', 'gd', 'gn', 'grc', 'gu', 'hak',
    'hi', 'hr', 'ht', 'hu', 'hy', 'hyw', 'ia', 'id', 'is', 'it', 'ja', 'jbo', 'ka',
    'kk', 'kl', 'kn', 'ko', 'kok', 'ku', 'ky', 'la', 'lfn', 'lt', 'lv', 'mi', 'mk',
    'ml', 'mr', 'ms', 'mt', 'my', 'nb', 'nci', 'ne', 'nl', 'om', 'or', 'pa', 'pap',
    'pl', 'pt', 'pt-br', 'py', 'quc', 'ro', 'ru', 'ru-lv', 'sd', 'shn', 'si', 'sk',
    'sl', 'sq', 'sr', 'sv', 'sw', 'ta', 'te', 'tn', 'tr', 'tt', 'ur', 'uz', 'vi',
    'vi-vn-x-central', 'vi-vn-x-south', 'yue'
]  # fmt: off


def make_cond_dict(
    text: str = "It would be nice to have time for testing, indeed.",
    language: str = "en-us",
    speaker: torch.Tensor | None = None,
    emotion: list[float] = [0.3077, 0.0256, 0.0256, 0.0256, 0.0256, 0.0256, 0.2564, 0.3077],
    fmax: float = 22050.0,
    pitch_std: float = 20.0,
    speaking_rate: float = 15.0,
    vqscore_8: list[float] = [0.78] * 8,
    ctc_loss: float = 0.0,
    dnsmos_ovrl: float = 4.0,
    speaker_noised: bool = False,
    unconditional_keys: Iterable[str] = {"vqscore_8", "dnsmos_ovrl"},
    device: torch.device | str = DEFAULT_DEVICE,
) -> dict:
    """
    A helper to build the 'cond_dict' that the model expects.
    By default, it will generate a random speaker embedding
    """
    assert language.lower() in supported_language_codes, "Please pick a supported language"

    language_code_to_id = {lang: i for i, lang in enumerate(supported_language_codes)}

    cond_dict = {
        "espeak": ([text], [language]),
        "speaker": speaker,
        "emotion": emotion,
        "fmax": fmax,
        "pitch_std": pitch_std,
        "speaking_rate": speaking_rate,
        "language_id": language_code_to_id[language],
        "vqscore_8": vqscore_8,
        "ctc_loss": ctc_loss,
        "dnsmos_ovrl": dnsmos_ovrl,
        "speaker_noised": int(speaker_noised),
    }

    for k in unconditional_keys:
        cond_dict.pop(k, None)

    for k, v in cond_dict.items():
        if isinstance(v, (float, int, list)):
            v = torch.tensor(v)
        if isinstance(v, torch.Tensor):
            cond_dict[k] = v.view(1, 1, -1).to(device)

        if k == "emotion":
            cond_dict[k] /= cond_dict[k].sum(dim=-1)

    return cond_dict