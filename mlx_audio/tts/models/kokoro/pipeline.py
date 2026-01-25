import logging
import re
from dataclasses import dataclass
from numbers import Number
from pathlib import Path
from typing import Any, Generator, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download
from misaki import en, espeak

from .voice import load_voice_tensor

ALIASES = {
    "en": "a",
    "en-us": "a",
    "en-gb": "b",
    "es": "e",
    "fr-fr": "f",
    "fr": "f",
    "hi": "h",
    "it": "i",
    "pt-br": "p",
    "pt": "p",
    "ja": "j",
    "zh": "z",
}

LANG_CODES = dict(
    # pip install misaki[en]
    a="American English",
    b="British English",
    # espeak-ng
    e="es",
    f="fr-fr",
    h="hi",
    i="it",
    p="pt-br",
    # pip install misaki[ja]
    j="Japanese",
    # pip install misaki[zh]
    z="Mandarin Chinese",
)


class KokoroPipeline:
    """
    KokoroPipeline is a language-aware support class with 2 main responsibilities:
    1. Perform language-specific G2P, mapping (and chunking) text -> phonemes
    2. Manage and store voices, lazily downloaded from HF if needed

    You are expected to have one KokoroPipeline per language. If you have multiple
    KokoroPipeline instances, you should reuse one KokoroModel instance across all of them.

    KokoroPipeline is designed to work with a KokoroModel, but this is not required.
    There are 2 ways to pass an existing model into a pipeline:
    1. On init: us_pipeline = KokoroPipeline(lang_code='a', model=model)
    2. On call: us_pipeline(text, voice, model=model)

    By default, KokoroPipeline will automatically initialize its own KokoroModel. To
    suppress this, construct a "quiet" KokoroPipeline with model=False.

    A "quiet" KokoroPipeline yields (graphemes, phonemes, None) without generating
    any audio. You can use this to phonemize and chunk your text in advance.

    A "loud" KokoroPipeline _with_ a model yields (graphemes, phonemes, audio).
    """

    def __init__(
        self,
        lang_code: str,
        model: nn.Module,
        repo_id: str,
        trf: bool = False,
    ):
        """Initialize a KokoroPipeline.

        Args:
            lang_code: Language code for G2P processing
            model: KokoroModel instance, True to create new model, False for no model
            trf: Whether to use transformer-based G2P
            device: Override default device selection ('cuda' or 'cpu', or None for auto)
                   If None, will auto-select cuda if available
                   If 'cuda' and not available, will explicitly raise an error
        """
        lang_code = lang_code.lower()
        lang_code = ALIASES.get(lang_code, lang_code)
        assert lang_code in LANG_CODES, (lang_code, LANG_CODES)
        self.lang_code = lang_code
        self.repo_id = repo_id
        if repo_id is None:
            raise ValueError("repo_id is required to load voices")
        self.model = model
        self.voices = {}
        if lang_code in "ab":
            try:
                fallback = espeak.EspeakFallback(british=lang_code == "b")
            except Exception as e:
                logging.warning("EspeakFallback not Enabled: OOD words will be skipped")
                logging.warning({str(e)})
                fallback = None
            self.g2p = en.G2P(
                trf=trf, british=lang_code == "b", fallback=fallback, unk=""
            )
        elif lang_code == "j":
            try:
                from misaki import ja

                self.g2p = ja.JAG2P()
            except ImportError:
                logging.error(
                    "You need to `pip install misaki[ja]` to use lang_code='j'"
                )
                raise
        elif lang_code == "z":
            try:
                from pypinyin import Style, pinyin

                self.pinyin = pinyin
                self.pinyin_style = Style
                # Also initialize English G2P for mixed Chinese/English text
                try:
                    self.en_g2p = en.G2P(trf=False, fallback=None, unk="")
                except Exception as e:
                    logging.warning(f"English G2P not available for mixed text: {e}")
                    self.en_g2p = None
                # Use a simple wrapper as g2p for compatibility
                self.g2p = lambda text: (self._chinese_to_bopomofo(text), None)
            except ImportError:
                logging.error("You need to `pip install pypinyin` to use lang_code='z'")
                raise
        else:
            language = LANG_CODES[lang_code]
            logging.warning(
                f"Using EspeakG2P(language='{language}'). Chunking logic not yet implemented, so long texts may be truncated unless you split them with '\\n'."
            )
            self.g2p = espeak.EspeakG2P(language=language)

    def load_single_voice(self, voice: str) -> mx.array:
        if voice in self.voices:
            return self.voices[voice]

        if voice.endswith(".safetensors"):
            # Direct path to safetensors file
            f = voice
        else:
            # Check if voice exists in local snapshot first
            try:
                local_dir = Path(
                    snapshot_download(
                        repo_id=self.repo_id,
                        allow_patterns=[f"voices/{voice}.safetensors"],
                        local_files_only=True,
                    )
                )
                local_voice = local_dir / "voices" / f"{voice}.safetensors"
                if local_voice.exists():
                    f = str(local_voice)
                else:
                    raise FileNotFoundError
            except (FileNotFoundError, Exception):
                # Download the specific voice file
                local_dir = Path(
                    snapshot_download(
                        repo_id=self.repo_id,
                        allow_patterns=[f"voices/{voice}.safetensors"],
                    )
                )
                f = str(local_dir / "voices" / f"{voice}.safetensors")

            if not voice.startswith(self.lang_code):
                v = LANG_CODES.get(voice, voice)
                p = LANG_CODES.get(self.lang_code, self.lang_code)
                logging.warning(
                    f"Language mismatch, loading {v} voice into {p} pipeline."
                )

        pack = load_voice_tensor(f)
        self.voices[voice] = pack
        return pack

    """
    load_voice is a helper function that lazily downloads and loads a voice:
    Single voice can be requested (e.g. 'af_bella') or multiple voices (e.g. 'af_bella,af_jessica').
    If multiple voices are requested, they are averaged.
    Delimiter is optional and defaults to ','.
    """

    def load_voice(self, voice: str, delimiter: str = ",") -> mx.array:
        if voice in self.voices:
            return self.voices[voice]
        logging.debug(f"Loading voice: {voice}")
        packs = [self.load_single_voice(v) for v in voice.split(delimiter)]
        if len(packs) == 1:
            return packs[0]
        self.voices[voice] = mx.mean(mx.stack(packs), axis=0)
        return self.voices[voice]

    def _number_to_chinese(self, num_str: str) -> str:
        """Convert Arabic numerals to Chinese characters.

        Examples:
            "23" -> "二十三"
            "100" -> "一百"
            "1000" -> "一千"
        """
        digits = "零一二三四五六七八九"
        units = ["", "十", "百", "千"]
        big_units = ["", "万", "亿"]

        if not num_str:
            return ""

        # Handle decimal numbers
        if "." in num_str:
            integer_part, decimal_part = num_str.split(".", 1)
            integer_chinese = (
                self._number_to_chinese(integer_part) if integer_part else ""
            )
            decimal_chinese = "".join(digits[int(d)] for d in decimal_part)
            return f"{integer_chinese}点{decimal_chinese}"

        num = int(num_str)
        if num == 0:
            return "零"

        if num < 0:
            return "负" + self._number_to_chinese(str(-num))

        result = ""
        unit_index = 0

        while num > 0:
            section = num % 10000
            if section > 0:
                section_str = ""
                for i, unit in enumerate(units):
                    digit = section % 10
                    section = section // 10
                    if digit > 0:
                        section_str = digits[digit] + unit + section_str
                    elif section_str and not section_str.startswith("零"):
                        section_str = "零" + section_str
                    if section == 0:
                        break
                result = section_str + big_units[unit_index] + result
            num = num // 10000
            unit_index += 1

        # Special case: 10-19 don't need leading "一"
        if result.startswith("一十"):
            result = result[1:]

        return result

    def _chinese_to_bopomofo(self, text: str) -> str:
        """Convert Chinese text to Bopomofo with numeric tones.

        The Kokoro ZH model expects Bopomofo symbols with numeric tones (1-5).
        """
        # Tone mark to number mapping
        tone_map = {
            "\u02ca": "2",  # ˊ tone 2
            "\u02c7": "3",  # ˇ tone 3
            "\u02cb": "4",  # ˋ tone 4
            "\u02d9": "5",  # ˙ neutral tone
        }

        # First, convert numbers to Chinese characters
        # Match sequences of digits (including decimals)
        text = re.sub(
            r"(\d+\.?\d*)",
            lambda m: self._number_to_chinese(m.group(1)),
            text,
        )

        result = []
        for char in text:
            # Chinese character range
            if "\u4e00" <= char <= "\u9fff":
                bpmf = self.pinyin(char, style=self.pinyin_style.BOPOMOFO)[0][0]

                # Extract tone mark and convert to number
                tone = "1"  # default tone 1
                clean_bpmf = ""
                for c in bpmf:
                    if c in tone_map:
                        tone = tone_map[c]
                    else:
                        clean_bpmf += c

                result.append(clean_bpmf + tone)
            elif char.isascii() and char.isalpha():
                # English letters - will be processed separately
                result.append(char)
            else:
                # Punctuation and other characters
                result.append(char)

        return " ".join(result)

    def _process_mixed_zh_en(self, text: str) -> str:
        """Process mixed Chinese/English text by using appropriate G2P for each part.

        Args:
            text: Input text containing Chinese and/or English

        Returns:
            Combined phoneme string with proper phonemes for both languages
        """
        # Pattern to match English sequences (letters, spaces, and common punctuation)
        pattern = r"([a-zA-Z][a-zA-Z\s,.'\"!\?\-]*)"

        parts = re.split(pattern, text)
        phonemes = []

        for part in parts:
            if not part.strip():
                continue

            # Check if this part starts with English letter
            if re.match(r"^[a-zA-Z]", part):
                # Process as English
                if self.en_g2p:
                    try:
                        _, tokens = self.en_g2p(part)
                        ps = "".join(
                            t.phonemes + (" " if t.whitespace else "")
                            for t in tokens
                            if t.phonemes
                        )
                        if ps.strip():
                            phonemes.append(ps.strip())
                    except Exception as e:
                        logging.warning(f"English G2P failed for '{part}': {e}")
                        # Keep English as-is if G2P fails
                        phonemes.append(part.strip())
                else:
                    # No English G2P available, keep as-is
                    phonemes.append(part.strip())
            else:
                # Process as Chinese using Bopomofo
                ps = self._chinese_to_bopomofo(part)
                if ps.strip():
                    phonemes.append(ps.strip())

        return " ".join(phonemes)

    @classmethod
    def tokens_to_ps(cls, tokens: List[en.MToken]) -> str:
        return "".join(
            t.phonemes + (" " if t.whitespace else "") for t in tokens
        ).strip()

    @classmethod
    def waterfall_last(
        cls,
        tokens: List[en.MToken],
        next_count: int,
        waterfall: List[str] = ["!.?…", ":;", ",—"],
        bumps: List[str] = [")", "”"],
    ) -> int:
        for w in waterfall:
            z = next(
                (
                    i
                    for i, t in reversed(list(enumerate(tokens)))
                    if t.phonemes in set(w)
                ),
                None,
            )
            if z is None:
                continue
            z += 1
            if z < len(tokens) and tokens[z].phonemes in bumps:
                z += 1
            if next_count - len(cls.tokens_to_ps(tokens[:z])) <= 510:
                return z
        return len(tokens)

    @classmethod
    def tokens_to_text(cls, tokens: List[en.MToken]) -> str:
        return "".join(t.text + t.whitespace for t in tokens).strip()

    def en_tokenize(
        self, tokens: List[en.MToken]
    ) -> Generator[Tuple[str, str, List[en.MToken]], None, None]:
        tks = []
        pcount = 0
        for t in tokens:
            # American English: ɾ => T
            t.phonemes = "" if t.phonemes is None else t.phonemes.replace("ɾ", "T")
            next_ps = t.phonemes + (" " if t.whitespace else "")
            next_pcount = pcount + len(next_ps.rstrip())
            if next_pcount > 510:
                z = KokoroPipeline.waterfall_last(tks, next_pcount)
                text = KokoroPipeline.tokens_to_text(tks[:z])
                logging.debug(
                    f"Chunking text at {z}: '{text[:30]}{'...' if len(text) > 30 else ''}'"
                )
                ps = KokoroPipeline.tokens_to_ps(tks[:z])
                yield text, ps, tks[:z]
                tks = tks[z:]
                pcount = len(KokoroPipeline.tokens_to_ps(tks))
                if not tks:
                    next_ps = next_ps.lstrip()
            tks.append(t)
            pcount += len(next_ps)
        if tks:
            text = KokoroPipeline.tokens_to_text(tks)
            ps = KokoroPipeline.tokens_to_ps(tks)
            yield "".join(text).strip(), "".join(ps).strip(), tks

    @classmethod
    def infer(
        cls,
        model: nn.Module,
        ps: str,
        pack: mx.array,
        speed: Number = 1,
    ):
        return model(ps, pack[len(ps) - 1], speed, return_output=True)

    def generate_from_tokens(
        self,
        tokens: Union[str, List[en.MToken]],
        voice: str,
        speed: Number = 1,
        model: Optional[nn.Module] = None,
    ) -> Generator["KokoroPipeline.Result", None, None]:
        """Generate audio from either raw phonemes or pre-processed tokens.

        Args:
            tokens: Either a phoneme string or list of pre-processed MTokens
            voice: The voice to use for synthesis
            speed: Speech speed modifier (default: 1)
            model: Optional Model instance (uses pipeline's model if not provided)

        Yields:
            KokoroPipeline.Result containing the input tokens and generated audio

        Raises:
            ValueError: If no voice is provided or token sequence exceeds model limits
        """
        model = model or self.model
        if model and voice is None:
            raise ValueError(
                'Specify a voice: pipeline.generate_from_tokens(..., voice="af_heart")'
            )

        pack = self.load_voice(voice) if model else None

        # Handle raw phoneme string
        if isinstance(tokens, str):
            logging.debug("Processing phonemes from raw string")
            if len(tokens) > 510:
                raise ValueError(f"Phoneme string too long: {len(tokens)} > 510")
            output = KokoroPipeline.infer(model, tokens, pack, speed) if model else None
            yield self.Result(graphemes="", phonemes=tokens, output=output)
            return

        logging.debug("Processing MTokens")
        # Handle pre-processed tokens
        for gs, ps, tks in self.en_tokenize(tokens):
            if not ps:
                continue
            elif len(ps) > 510:
                logging.warning(
                    f"Unexpected len(ps) == {len(ps)} > 510 and ps == '{ps}'"
                )
                logging.warning("Truncating to 510 characters")
                ps = ps[:510]
            output = KokoroPipeline.infer(model, ps, pack, speed) if model else None
            if output is not None and output.pred_dur is not None:
                KokoroPipeline.join_timestamps(tks, output.pred_dur)
            yield self.Result(graphemes=gs, phonemes=ps, tokens=tks, output=output)

    @classmethod
    def join_timestamps(cls, tokens: List[en.MToken], pred_dur: mx.array):
        # Multiply by 600 to go from pred_dur frames to sample_rate 24000
        # Equivalent to dividing pred_dur frames by 40 to get timestamp in seconds
        # We will count nice round half-frames, so the divisor is 80
        MAGIC_DIVISOR = 80
        if not tokens or len(pred_dur) < 3:
            # We expect at least 3: <bos>, token, <eos>
            return
        # We track 2 counts, measured in half-frames: (left, right)
        # This way we can cut space characters in half
        # TODO: Is -3 an appropriate offset?
        left = right = 2 * max(0, pred_dur[0].item() - 3)
        # Updates:
        # left = right + (2 * token_dur) + space_dur
        # right = left + space_dur
        i = 1
        for t in tokens:
            if i >= len(pred_dur) - 1:
                break
            if not t.phonemes:
                if t.whitespace:
                    i += 1
                    left = right + pred_dur[i].item()
                    right = left + pred_dur[i].item()
                    i += 1
                continue
            j = i + len(t.phonemes)
            if j >= len(pred_dur):
                break
            t.start_ts = left / MAGIC_DIVISOR
            token_dur = pred_dur[i:j].sum().item()
            space_dur = pred_dur[j].item() if t.whitespace else 0
            left = right + (2 * token_dur) + space_dur
            t.end_ts = left / MAGIC_DIVISOR
            right = left + space_dur
            i = j + (1 if t.whitespace else 0)

    @dataclass
    class Result:
        graphemes: str
        phonemes: str
        tokens: Optional[List[en.MToken]] = None
        output: Optional[Any] = None
        text_index: Optional[int] = None

        @property
        def audio(self) -> Optional[mx.array]:
            return None if self.output is None else self.output.audio

        @property
        def pred_dur(self) -> Optional[mx.array]:
            return None if self.output is None else self.output.pred_dur

        ### MARK: BEGIN BACKWARD COMPAT ###
        def __iter__(self):
            yield self.graphemes
            yield self.phonemes
            yield self.audio

        def __getitem__(self, index):
            return [self.graphemes, self.phonemes, self.audio][index]

        def __len__(self):
            return 3

    def __call__(
        self,
        text: Union[str, List[str]],
        voice: Optional[str] = None,
        speed: Number = 1,
        split_pattern: Optional[str] = r"\n+",
    ) -> Generator["KokoroPipeline.Result", None, None]:
        if voice is None:
            raise ValueError(
                'Specify a voice: en_us_pipeline(text="Hello world!", voice="af_heart")'
            )
        pack = self.load_voice(voice) if self.model else None
        if isinstance(text, str):
            text = re.split(split_pattern, text.strip()) if split_pattern else [text]
        # Process each segment
        for graphemes_index, graphemes in enumerate(text):
            if not graphemes.strip():  # Skip empty segments
                continue

            # English processing (unchanged)
            if self.lang_code in "ab":
                # print(f"Processing English text: {graphemes[:50]}{'...' if len(graphemes) > 50 else ''}")
                _, tokens = self.g2p(graphemes)
                for gs, ps, tks in self.en_tokenize(tokens):
                    if not ps:
                        continue
                    elif len(ps) > 510:
                        logging.warning(
                            f"Unexpected len(ps) == {len(ps)} > 510 and ps == '{ps}'"
                        )
                        ps = ps[:510]
                    output = (
                        KokoroPipeline.infer(self.model, ps, pack, speed)
                        if self.model
                        else None
                    )
                    if output is not None and output.pred_dur is not None:
                        KokoroPipeline.join_timestamps(tks, output.pred_dur)
                    yield self.Result(
                        graphemes=gs,
                        phonemes=ps,
                        tokens=tks,
                        output=output,
                        text_index=graphemes_index,
                    )

            # Non-English processing with chunking
            else:
                # Split long text into smaller chunks (roughly 400 characters each)
                # Using sentence boundaries when possible
                chunk_size = 400
                chunks = []

                # Try to split on sentence boundaries first
                sentences = re.split(r"([.!?]+)", graphemes)
                current_chunk = ""

                for i in range(0, len(sentences), 2):
                    sentence = sentences[i]
                    # Add the punctuation back if it exists
                    if i + 1 < len(sentences):
                        sentence += sentences[i + 1]

                    if len(current_chunk) + len(sentence) <= chunk_size:
                        current_chunk += sentence
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence

                if current_chunk:
                    chunks.append(current_chunk.strip())

                # If no chunks were created (no sentence boundaries), fall back to character-based chunking
                if not chunks:
                    chunks = [
                        graphemes[i : i + chunk_size]
                        for i in range(0, len(graphemes), chunk_size)
                    ]

                # Process each chunk
                for chunk in chunks:
                    if not chunk.strip():
                        continue

                    # For Chinese, use mixed language processing if English G2P is available
                    if (
                        self.lang_code == "z"
                        and hasattr(self, "en_g2p")
                        and self.en_g2p
                    ):
                        ps = self._process_mixed_zh_en(chunk)
                    else:
                        ps, _ = self.g2p(chunk)

                    if not ps:
                        continue
                    elif len(ps) > 510:
                        logging.warning(f"Truncating len(ps) == {len(ps)} > 510")
                        ps = ps[:510]

                    output = (
                        KokoroPipeline.infer(self.model, ps, pack, speed)
                        if self.model
                        else None
                    )
                    yield self.Result(
                        graphemes=chunk,
                        phonemes=ps,
                        output=output,
                        text_index=graphemes_index,
                    )
