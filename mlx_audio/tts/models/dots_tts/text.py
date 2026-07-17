"""Tokenizer + generation-schedule builder for the dots.tts MLX runtime.

Imports ONLY the ``tokenizers`` library (the HF Rust BPE backend) + stdlib —
NEVER the expected format. ``transformers.AutoTokenizer`` transitively imports the expected format, so we load
the saved fast tokenizer directly via ``tokenizers.Tokenizer.from_file`` (identical
BPE merges + special-token table, zero the expected format).

Ports the dots.tts ``tts`` template + ``build_generation_schedule`` (upstream
``data/pipelines/{tts_pipeline,tokenizing}.py`` + ``runtime._prepare_inputs``):

  template "tts" = ``[文本]{text}[文本对应语音]{audio}``

  schedule_ids = BPE("[文本]")
               + BPE(prompt_text + text)            # {text}
               + BPE("[文本对应语音]")
               + [audio_gen_start_id]               # {audio} ->
               + [audio_gen_span_id] * max_audio_tokens

All literals are PLAIN BPE (``encode(s, add_special_tokens=False)``; NO BOS/EOS).
``prompt_text`` is concatenated directly before the target ``text`` (with a trailing
space appended to ``prompt_text`` for non-ZH/JA/YUE langs, mirroring upstream
``_process_prompt_text``). When a ``language`` is given, the ``[CODE]`` uppercase-ISO
tag is prepended to the ``prompt_text`` (if present), and to the target ``text`` only
in the no-prompt-text case — matching upstream ``runtime._process_prompt_text`` /
``_prepare_inputs``. ``YUE`` maps to ``[口音:粤语]``; tagging is idempotent.
"""

from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer

# Special tokens (verbatim from dots_tts.utils.tokenizer).
AUDIO_GEN_START_TOKEN = "<|audio_gen_start|>"
AUDIO_GEN_SPAN_TOKEN = "<|audio_gen_span|>"
AUDIO_GEN_END_TOKEN = "<|audio_gen_end|>"
TEXT_COND_END_TOKEN = "<|text_cond_end|>"
AUDIO_COMP_SPAN_TOKEN = "<|audio_comp_span|>"

# tts template literals (dots_tts.data.pipelines.tts_pipeline).
TTS_TEXT_PREFIX = "[文本]"
TTS_AUDIO_PREFIX = "[文本对应语音]"

# Languages that do NOT get a trailing space appended to the prompt transcript
# (upstream _process_prompt_text); a CJK/Cantonese set.
_NO_TRAILING_SPACE_LANGS = {"ZH", "YUE", "JA", "口音:粤语"}


class DotsTokenizer:
    """Thin wrapper over a ``tokenizers.Tokenizer`` with the dots.tts schedule logic.

    Loaded from the saved tokenizer dir (``tokenizer.json``). Exposes the special
    token ids the AR loop needs and ``build_generation_schedule``.
    """

    def __init__(self, tokenizer: Tokenizer):
        self._tk = tokenizer
        self.audio_gen_start_id = self._require(AUDIO_GEN_START_TOKEN)
        self.audio_gen_span_id = self._require(AUDIO_GEN_SPAN_TOKEN)
        self.audio_gen_end_id = self._require(AUDIO_GEN_END_TOKEN)
        self.text_cond_end_id = self._require(TEXT_COND_END_TOKEN)
        self.audio_comp_span_id = self._require(AUDIO_COMP_SPAN_TOKEN)
        # Both span ids are treated as audio placeholders during decode.
        self.audio_span_token_ids = (self.audio_gen_span_id, self.audio_comp_span_id)

    @classmethod
    def from_pretrained(cls, tokenizer_dir: str | Path) -> "DotsTokenizer":
        path = Path(tokenizer_dir)
        tk_json = path / "tokenizer.json"
        if not tk_json.exists():
            raise FileNotFoundError(f"missing tokenizer.json: {tk_json}")
        return cls(Tokenizer.from_file(str(tk_json)))

    def _require(self, token: str) -> int:
        tid = self._tk.token_to_id(token)
        if tid is None:
            raise ValueError(f"tokenizer missing required special token: {token!r}")
        return int(tid)

    @staticmethod
    def _attach_language_tag(text: str, language: str | None) -> str:
        """Prepend the ``[CODE]`` language tag (mirrors upstream ``attach_language_tag``).

        Uppercase ISO code; ``YUE`` maps to ``口音:粤语``; idempotent (won't double-
        prepend). Empty ``text`` / ``language`` is a no-op. We skip langdetect-based
        normalization (callers pass explicit codes); the code is just uppercased.
        """
        if not text or not language:
            return text
        code = language.strip().upper()
        if not code:
            return text
        if code == "YUE":
            code = "口音:粤语"
        tag = f"[{code}]"
        if text.startswith(tag):
            return text
        return f"{tag}{text}"

    def encode(self, text: str) -> list[int]:
        """Plain BPE (no BOS/EOS), matching ``encode(s, add_special_tokens=False)``."""
        return list(self._tk.encode(text, add_special_tokens=False).ids)

    def decode(self, ids: list[int]) -> str:
        return self._tk.decode(ids, skip_special_tokens=True)

    def _process_prompt_text(
        self, prompt_text: str | None, *, language: str | None
    ) -> str:
        """Mirror ``runtime._process_prompt_text`` (sans the auto language detector).

        Appends a trailing space for non-CJK languages, then attaches the ``[CODE]``
        language tag to the prompt transcript when a ``language`` is given (matching
        upstream order: trailing-space THEN tag). We do NOT auto-detect the language
        (no ``langdetect`` dependency); callers pass an explicit code.
        """
        if not prompt_text:
            return ""
        prompt_text = prompt_text.strip()
        if not prompt_text:
            return ""
        lang = (language or "").upper() if language else None
        if lang not in _NO_TRAILING_SPACE_LANGS:
            prompt_text += " "
        # Upstream attaches the language tag to the prompt transcript whenever a
        # language is given (runtime._process_prompt_text:260-261).
        if language:
            prompt_text = self._attach_language_tag(prompt_text, language)
        return prompt_text

    def build_generation_schedule(
        self,
        text: str,
        *,
        prompt_text: str | None = None,
        language: str | None = None,
        max_audio_tokens: int = 500,
    ) -> list[int]:
        """Build the flat ``schedule_ids`` for the ``tts`` template.

        Args:
            text: the target text to synthesize.
            prompt_text: the reference-audio transcript (prefixed before ``text``);
                empty/None for plain text-to-speech (no reference).
            language: optional uppercase ISO code; only prepended as a ``[CODE]`` tag
                to ``text`` when there is NO prompt_text (matches upstream).
            max_audio_tokens: number of ``audio_gen_span`` placeholders to emit
                (= ``max_generate_length``, default 500).

        Returns:
            ``schedule_ids``: a flat list of token ids.
        """
        if max_audio_tokens <= 0:
            raise ValueError("max_audio_tokens must be positive for generation.")

        normalized_prompt_text = self._process_prompt_text(
            prompt_text, language=language
        )
        normalized_text = text.strip()
        # Language tag attaches to the target text only in the no-prompt-text case
        # (runtime._prepare_inputs:325-326); the prompt-text path is tagged inside
        # _process_prompt_text above.
        if language and not normalized_prompt_text:
            normalized_text = self._attach_language_tag(normalized_text, language)

        combined_text = f"{normalized_prompt_text}{normalized_text}"

        schedule_ids: list[int] = []
        schedule_ids.extend(self.encode(TTS_TEXT_PREFIX))
        schedule_ids.extend(self.encode(combined_text))
        schedule_ids.extend(self.encode(TTS_AUDIO_PREFIX))
        schedule_ids.append(self.audio_gen_start_id)
        schedule_ids.extend([self.audio_gen_span_id] * max_audio_tokens)
        return schedule_ids
