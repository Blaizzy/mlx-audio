"""CosyVoice3 frontend for text tokenization and prompt-feature extraction."""

import logging
import re
from functools import partial
from pathlib import Path
from typing import Dict, Generator, List, Optional, Union

import mlx.core as mx

from mlx_audio.utils import load_audio

from .frontend_utils import (
    contains_chinese,
    is_only_punctuation,
    remove_bracket,
    replace_blank,
    replace_corner_mark,
    spell_out_number,
    split_paragraph,
)

S3_SAMPLE_RATE = 16000


class CosyVoice3FrontEnd:
    def __init__(
        self,
        tokenizer=None,
        speech_tokenizer=None,
        speaker_encoder=None,
        sample_rate: int = 24000,
        endofprompt_token: int = 151646,
        speech_tokenizer_onnx_path: Optional[Path] = None,
    ):
        self.tokenizer = tokenizer
        self.speech_tokenizer = speech_tokenizer
        self.speaker_encoder = speaker_encoder
        self.sample_rate = sample_rate
        self.endofprompt_token = endofprompt_token
        # Optional onnxruntime speech-tokenizer model path.
        self.speech_tokenizer_onnx_path = speech_tokenizer_onnx_path

        # Cached zero-shot speaker presets.
        self.spk2info: Dict[str, Dict[str, mx.array]] = {}

        # Use wetext when available, otherwise leave the text unchanged.
        try:
            from wetext import Normalizer as ZhNormalizer  # noqa: F811
            from wetext import Normalizer as EnNormalizer

            self.zh_tn_model = ZhNormalizer(remove_erhua=False)
            self.en_tn_model = EnNormalizer()
            self._text_frontend = "wetext"
            logging.info("cosyvoice3 frontend: using wetext normalizer")
        except Exception:
            self._text_frontend = ""
            logging.info(
                "cosyvoice3 frontend: wetext not available — text will not "
                "be normalised (numbers, brackets, etc. left as-is)"
            )

    # ------------------------------------------------------------------ #
    @classmethod
    def from_model_path(
        cls,
        model_path: Union[str, Path],
        sample_rate: int = 24000,
        endofprompt_token: int = 151646,
        load_speaker_encoder: bool = True,
        use_onnx_speech_tokenizer: bool = True,
    ) -> "CosyVoice3FrontEnd":
        """Load the Qwen text tokenizer (and, if present, CAMPPlus) from
        the model directory. Speech tokenizer is loaded lazily on first use
        (see ensure_speech_tokenizer) since it comes from a separate repo.
        """
        from transformers import AutoTokenizer

        model_path = Path(model_path)
        tok = AutoTokenizer.from_pretrained(str(model_path))
        # Register model control tokens so they encode as atomic tokens.
        _vocal_events = [
            "[breath]", "<strong>", "</strong>", "[noise]",
            "[laughter]", "[cough]", "[clucking]", "[accent]",
            "[quick_breath]", "<laughter>", "</laughter>",
            "[hissing]", "[sigh]", "[vocalized-noise]",
            "[lipsmack]", "[mn]",
        ]
        # ARPABET phonemes (English pronunciation hotfix)
        _arpabet_consonants = [
            "B", "CH", "D", "DH", "F", "G", "HH", "JH", "K", "L", "M", "N",
            "NG", "P", "R", "S", "SH", "T", "TH", "V", "W", "Y", "Z", "ZH",
        ]
        _arpabet_vowels = [
            "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY",
            "IH", "IY", "OW", "OY", "UH", "UW",
        ]
        _arpabet = []
        for _ph in _arpabet_consonants + _arpabet_vowels:
            _arpabet.append(f"[{_ph}]")
        for _ph in _arpabet_vowels:
            for _stress in ("0", "1", "2"):
                _arpabet.append(f"[{_ph}{_stress}]")
        # Pinyin initials + finals (Chinese pronunciation hotfix)
        _pinyin_initials = [
            "b", "c", "ch", "d", "f", "g", "h", "j", "k", "l", "m", "n",
            "p", "q", "r", "s", "sh", "t", "w", "x", "y", "z", "zh",
        ]
        _pinyin_finals_tone = [
            # tone 4 (à)
            "a", "ai", "an", "ang", "ao",
            "e", "ei", "en", "eng", "er",
            "i", "in", "ing", "iu",
            "o", "ong", "ou",
            "u", "uang", "ue", "un", "uo",
        ]
        _pinyin_finals_tone2 = [  # tone 2 (á)
            "a", "ai", "an", "ang", "ao",
            "e", "ei", "en", "eng", "er",
            "i", "in", "ing",
            "o", "ong", "ou",
            "u", "uai", "uan", "uang",
            "v",
        ]
        _pinyin_finals_tone3 = [  # tone 3 (ǎ)
            "a", "ai", "an", "ang", "ao",
            "e", "ei", "en", "eng", "er",
            "i", "in", "ing",
            "o", "ong", "ou",
            "u", "uai", "uan", "uang",
            "v",
        ]
        _pinyin_finals_tone1 = [  # tone 1 (ā)
            "a", "ai", "an", "ang", "ao",
            "e", "ei", "en", "eng",
            "i", "in", "ing",
            "o", "ong", "ou",
            "u", "uai", "uan", "uang", "ue", "un", "uo",
        ]
        _pinyin = []
        for _py in _pinyin_initials:
            _pinyin.append(f"[{_py}]")
        for _py in (
            "a ai an ang ao e ei en eng i ian in ing iu o ong ou u uang ue un uo"
            .split()
        ):
            _pinyin.append(f"[{_py}]")
        # Toned pinyin finals.
        _toned = (
            # tone 4 (à)
            "ià iàn iàng iào iè iòng iù uà uài uàn uàng uè uì uò vè "
            "à ài àn àng ào è èi èn èng èr ì ìn ìng ò òng òu ù ùn "
            # tone 2 (á)
            "iá ián iáng iáo ié ióng iú uá uái uán uáng ué uí uó "
            "á ái án áng áo é éi én éng ér í ín íng ó óng óu ú ún "
            # tone 3 (ǎ)
            "iǎ iǎn iǎng iǎo iě iǒng iǔ uǎ uǎi uǎn uǎng uǐ uǒ ǚ "
            "ǎ ǎi ǎn ǎng ǎo ě ěi ěn ěng ěr ǐ ǐn ǐng ǒ ǒng ǒu ǔ ǔn "
            # tone 1 (ā)
            "iā iān iāng iāo iē iōng iū uā uāi uān uāng uē uī uō "
            "ā āi ān āng āo ē ēi ēn ēng ī īn īng ō ōng ōu ū ūn ǘ ǜ"
        )
        for _py in _toned.split():
            _pinyin.append(f"[{_py}]")

        tok.add_special_tokens(
            {
                "eos_token": "<|endoftext|>",
                "pad_token": "<|endoftext|>",
                "additional_special_tokens": [
                    "<|im_start|>",
                    "<|im_end|>",
                    "<|endofprompt|>",
                    "<|endofsystem|>",
                    *_vocal_events,
                    *_arpabet,
                    *_pinyin,
                ],
            }
        )

        onnx_path = None
        if use_onnx_speech_tokenizer:
            onnx_path = _first_existing_optional(
                model_path, "speech_tokenizer_v3.onnx", "speech_tokenizer_v2.onnx"
            )

        fe = cls(
            tokenizer=tok,
            sample_rate=sample_rate,
            endofprompt_token=endofprompt_token,
            speech_tokenizer_onnx_path=onnx_path,
        )

        if load_speaker_encoder:
            campplus_path = _first_existing_optional(
                model_path, "campplus.safetensors", "campplus.onnx"
            )
            if campplus_path is not None:
                fe.load_speaker_encoder(campplus_path)
        return fe

    def load_speaker_encoder(self, path: Union[str, Path]) -> None:
        """Load CAMPPlus speaker-encoder weights from .safetensors or .onnx."""
        from mlx_audio.codec.models.stepaudio2.convert import load_campplus_weights
        from mlx_audio.codec.models.stepaudio2.speaker import StepAudio2CAMPPlus

        self.speaker_encoder = StepAudio2CAMPPlus()
        load_campplus_weights(self.speaker_encoder, path, strict=True)
        # base_load_model's model.eval() runs before post_load_hook builds the
        # frontend, so this module (outside the main param tree) never gets
        # switched out of training mode on its own — its BatchNorm(affine=False)
        # layers would otherwise normalize using this single sample's own
        # batch/time statistics (batch=1, time=1 at the final dense layer),
        # whose variance is exactly 0, zeroing the entire speaker embedding.
        self.speaker_encoder.eval()
        mx.eval(self.speaker_encoder.parameters())

    def encode_text(self, text: str) -> mx.array:
        """Tokenize text to Qwen ids as (1, L) int32."""
        if self.tokenizer is None:
            raise RuntimeError("text tokenizer not loaded")
        ids = self.tokenizer.encode(text)
        return mx.array([ids], dtype=mx.int32)

    def text_normalize(
        self,
        text: str,
        split: bool = True,
        text_frontend: bool = True,
    ) -> Union[List[str], str]:
        """Normalize and optionally split text before TTS synthesis.

        When ``split=True``, returns token-balanced sub-sentences; otherwise,
        returns the cleaned text as a single string.
        """
        # Preserve explicit control-token sequences.
        if "<|" in text and "|>" in text:
            text_frontend = False
        if text_frontend is False or text == "":
            return [text] if split else text

        text = text.strip()

        if contains_chinese(text):
            if self._text_frontend == "wetext":
                text = self.zh_tn_model.normalize(text)
            text = text.replace("\n", "")
            text = replace_blank(text)
            text = replace_corner_mark(text)
            text = text.replace(".", "。")
            text = text.replace(" - ", "，")
            text = remove_bracket(text)
            text = re.sub(r"[，,、]+$", "。", text)
            if split:
                texts = list(
                    split_paragraph(
                        text,
                        partial(self.tokenizer.encode),
                        "zh",
                        token_max_n=80,
                        token_min_n=60,
                        merge_len=20,
                        comma_split=False,
                    )
                )
            else:
                texts = [text]
        else:
            if self._text_frontend == "wetext":
                text = self.en_tn_model.normalize(text)
            text = spell_out_number(text)
            if split:
                texts = list(
                    split_paragraph(
                        text,
                        partial(self.tokenizer.encode),
                        "en",
                        token_max_n=80,
                        token_min_n=60,
                        merge_len=20,
                        comma_split=False,
                    )
                )
            else:
                texts = [text]

        texts = [t for t in texts if not is_only_punctuation(t)]
        return texts if split else texts[0]

    def ensure_speech_tokenizer(self):
        """Build the configured speech tokenizer on first use."""
        if self.speech_tokenizer is None:
            if self.speech_tokenizer_onnx_path is not None:
                import onnxruntime as ort

                self.speech_tokenizer = ort.InferenceSession(
                    str(self.speech_tokenizer_onnx_path),
                    providers=["CPUExecutionProvider"],
                )
            else:
                from mlx_audio.codec.models.s3 import S3TokenizerV2

                self.speech_tokenizer = S3TokenizerV2.from_pretrained(
                    "speech_tokenizer_v2_25hz"
                )
                mx.eval(self.speech_tokenizer.parameters())
        return self.speech_tokenizer

    def extract_speech_token(self, prompt_wav) -> mx.array:
        """Prompt wav -> discrete speech tokens (1, N) int32."""
        import numpy as np

        from mlx_audio.codec.models.s3 import log_mel_spectrogram

        tok = self.ensure_speech_tokenizer()
        audio_16k = load_audio(prompt_wav, sample_rate=S3_SAMPLE_RATE)
        mels = log_mel_spectrogram(audio_16k)
        # Drop the trailing STFT frame to preserve tokenizer frame alignment.
        mels = mels[..., :-1]
        mels = mx.expand_dims(mels, 0)
        mel_lens = mx.array([mels.shape[-1]], dtype=mx.int32)

        import onnxruntime as ort

        if isinstance(tok, ort.InferenceSession):
            feats = np.array(mels, dtype=np.float32)
            feats_length = np.array(mel_lens, dtype=np.int32)
            indices = tok.run(
                None,
                {
                    tok.get_inputs()[0].name: feats,
                    tok.get_inputs()[1].name: feats_length,
                },
            )[0]
            return mx.array(indices, dtype=mx.int32)

        tokens, _ = tok.quantize(mels, mel_lens)
        return tokens.astype(mx.int32)

    def extract_speaker_embedding(self, prompt_wav) -> mx.array:
        """Prompt wav -> 192-d CAMPPlus x-vector (1, 192)."""
        if self.speaker_encoder is None:
            raise RuntimeError(
                "speaker_encoder not loaded; call load_speaker_encoder(path) "
                "or construct via from_model_path(), or pass a precomputed "
                "speaker_embedding explicitly"
            )
        audio_16k = load_audio(prompt_wav, sample_rate=S3_SAMPLE_RATE)
        return self.speaker_encoder.inference(audio_16k)

    def extract_prompt_feat(self, prompt_wav) -> mx.array:
        """Prompt wav (24k) -> mel [1, T, n_mels] for flow conditioning.

        ``fmax=None`` preserves the full Nyquist band of the 24 kHz input.
        A periodic Hann window aligns STFT framing with the model's training
        features.
        """
        from mlx_audio.dsp import hanning
        from mlx_audio.tts.models.chatterbox.s3gen.mel import mel_spectrogram

        audio_24k = load_audio(prompt_wav, sample_rate=self.sample_rate)
        window = hanning(1920, periodic=True)
        mels = mel_spectrogram(
            mx.expand_dims(audio_24k, 0), fmax=None, window=window
        )  # (1, n_mels, T)
        return mx.transpose(mels, (0, 2, 1))  # (1, T, n_mels)

    def frontend_zero_shot(
        self, tts_text: str, prompt_text: str, prompt_wav, spk_id: str = ""
    ) -> Dict[str, mx.array]:
        """Build the model_input dict for zero-shot voice cloning.

        When ``spk_id`` is a key in ``self.spk2info``, the cached prompt
        features (speech tokens, mel, and speaker embedding) are reused,
        saving a full pass through the speech tokenizer, mel extractor,
        and speaker encoder — useful for repeated synthesis with the same
        reference audio.
        """
        text = self.encode_text(tts_text)

        if spk_id and spk_id in self.spk2info:
            cached = self.spk2info[spk_id]
            return {
                "text": text,
                "prompt_text": cached["prompt_text"],
                "prompt_speech_token": cached["prompt_speech_token"],
                "prompt_feat": cached["prompt_feat"],
                "embedding": cached["embedding"],
            }

        prompt_text_ids = self.encode_text(prompt_text)
        prompt_speech_token = self.extract_speech_token(prompt_wav)
        prompt_feat = self.extract_prompt_feat(prompt_wav)
        embedding = self.extract_speaker_embedding(prompt_wav)

        # Keep prompt features and prompt tokens aligned at the model's 2:1
        # mel-to-token ratio before they are passed to the flow decoder.
        n_tok = prompt_speech_token.shape[1]
        token_len = min(prompt_feat.shape[1] // 2, n_tok)
        prompt_feat = prompt_feat[:, : 2 * token_len]
        prompt_speech_token = prompt_speech_token[:, :token_len]

        return {
            "text": text,
            "prompt_text": prompt_text_ids,
            "prompt_speech_token": prompt_speech_token,
            "prompt_feat": prompt_feat,
            "embedding": embedding,
        }

    def frontend_instruct2(
        self, tts_text: str, instruct: str, prompt_wav
    ) -> Dict[str, mx.array]:
        """Build frontend inputs for instruction-based generation.

        The instruction replaces the LLM prompt while the reference audio
        continues to provide speaker and acoustic conditioning.
        """
        # instruct replaces the usual prompt_text for the LLM, while
        # prompt_wav still drives flow/HiFT conditioning (speaker + mel prompt).
        model_input = self.frontend_zero_shot(tts_text, instruct, prompt_wav)
        return model_input

    # ---- spk2info (zero-shot speaker presets) ----

    def list_spks(self) -> list:
        """Return the list of cached zero-shot speaker ids."""
        return list(self.spk2info.keys())

    def add_zero_shot_spk(
        self, prompt_text: str, prompt_wav, spk_id: str
    ) -> bool:
        """Extract and cache prompt features under a speaker identifier."""
        if not spk_id:
            raise ValueError("spk_id must not be empty")
        model_input = self.frontend_zero_shot("", prompt_text, prompt_wav)
        del model_input["text"]
        self.spk2info[spk_id] = model_input
        return True

    def save_spkinfo(self, path: Union[str, Path]) -> None:
        """Persist ``spk2info`` to *path* as a safetensors file.

        Each speaker's tensors are stored under keys ``{spk_id}.{field}``
        so the flat safetensors format can round-trip the nested dict.
        """
        flat: Dict[str, mx.array] = {}
        for spk_id, info in self.spk2info.items():
            for key, tensor in info.items():
                flat[f"{spk_id}.{key}"] = tensor
        mx.save_safetensors(str(path), flat)

    def load_spkinfo(self, path: Union[str, Path]) -> int:
        """Load ``spk2info`` from a safetensors file saved by ``save_spkinfo``.

        Returns the number of speakers loaded. Existing entries with the same
        ``spk_id`` are overwritten.
        """
        flat = mx.load(str(path))
        # Group flat keys back into per-speaker dicts
        loaded: Dict[str, Dict[str, mx.array]] = {}
        for flat_key, tensor in flat.items():
            spk_id, field = flat_key.split(".", 1)
            loaded.setdefault(spk_id, {})[field] = tensor
        self.spk2info.update(loaded)
        return len(loaded)


def _first_existing_optional(root: Path, *names: str) -> Optional[Path]:
    for name in names:
        path = root / name
        if path.exists():
            return path
    return None
