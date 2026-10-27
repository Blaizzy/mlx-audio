# Vendored from MayaResearch/maya2-native (https://github.com/MayaResearch/maya2-native), Copyright Maya Research,
# licensed under the Apache License 2.0 (see LICENSE in this directory). Modified for mlx-audio: reformatted with black.
"""CPU-only request preparation shared by HTTP and WebSocket serving.

No model imports, logging of caller text, or silent raw-text fallback.
"""

if __package__:
    from .normalize import (
        LANGUAGE_NAMES,
        NormalizationError,
        resolve_language,
        validate_text,
    )
    from .script_guard import detect_language
else:
    from normalize import (
        LANGUAGE_NAMES,
        NormalizationError,
        resolve_language,
        validate_text,
    )
    from script_guard import detect_language

NORMALIZATION_LANGUAGES = tuple(LANGUAGE_NAMES)


class PipelineError(Exception):
    def __init__(self, code, message, status=422):
        self.code, self.status = code, status
        super().__init__(message)

    def payload(self):
        return {"error": str(self), "code": self.code}


def prepare_text(
    text,
    *,
    normalizer,
    splitter,
    post_process,
    normalize_enabled=True,
    chunk_enabled=True,
    date_order="DMY",
    two_digit_year_base=None,
    max_chunk_chars=None,
    normalization_language="en-IN",
):
    try:
        if not normalize_enabled:
            validate_text(
                text,
                date_order=date_order,
                two_digit_year_base=two_digit_year_base,
                language=(
                    detect_language(text)
                    if normalization_language is None and isinstance(text, str)
                    else normalization_language
                ),
            )
        if not isinstance(text, str):
            raise NormalizationError("invalid_text", "text must be a string.")
        t = text.strip()
        if not t:
            return []
        if normalize_enabled:
            if normalizer is None:
                raise PipelineError(
                    "normalizer_unavailable", "Text normalizer unavailable.", 503
                )
            t = normalizer(
                t,
                language=(
                    "auto" if normalization_language is None else normalization_language
                ),
                date_order=date_order,
                two_digit_year_base=two_digit_year_base,
            )
            if not isinstance(t, str) or not any(c.isalnum() for c in t):
                raise PipelineError(
                    "empty_normalized_text", "No speakable text remains."
                )
        if chunk_enabled:
            if splitter is None or post_process is None:
                raise PipelineError(
                    "chunker_unavailable", "Text chunker unavailable.", 503
                )
            t = post_process(t)
            parts = splitter(t)
            if not parts or any(not isinstance(p, str) or not p.strip() for p in parts):
                raise PipelineError(
                    "invalid_chunks", "Text chunker returned invalid chunks.", 503
                )
            # Current chunker changes whitespace only. Detect dropped/reordered text
            # before audio starts; do not send a too-long whole-text fallback.
            if " ".join(" ".join(parts).split()) != " ".join(t.split()):
                raise PipelineError(
                    "chunk_content_mismatch", "Text chunker changed content.", 503
                )
            if max_chunk_chars and any(len(p) > max_chunk_chars for p in parts):
                raise PipelineError(
                    "unsplittable_text",
                    "A text token exceeds the safe chunk size; shorten it.",
                )
            return parts
        return [t]
    except NormalizationError as exc:
        raise PipelineError(exc.code, str(exc)) from None
    except PipelineError:
        raise
    except Exception:
        raise PipelineError(
            "text_pipeline_failed",
            "Text preparation failed; no audio was generated.",
            503,
        ) from None


def resolve_conditioning(settings, supported):
    """Never invent an untrained language tag. Legacy omission remains Hindi.

    `language`/`lang` requests a model tag unless `conditioning_language` is
    explicitly supplied. The latter is for a checkpoint trained with a tag that
    differs from the text language, and requires deployment-owner confirmation.
    """
    value = (
        settings.get("conditioning_language")
        or settings.get("language")
        or settings.get("lang")
        or "Hindi"
    )
    aliases = {
        "en": "English",
        "en-in": "English",
        "en-us": "English",
        "hi": "Hindi",
        "hi-in": "Hindi",
        "indian english": "English",
        "indian-english": "English",
    }
    if not isinstance(value, str):
        raise PipelineError("invalid_language", "language must be a string.")
    by_name = {x.lower(): x for x in supported}
    value = aliases.get(value.lower(), by_name.get(value.lower(), value))
    if value not in supported:
        raise PipelineError(
            "unsupported_conditioning_language",
            "No supported model tag for this language. Confirm the checkpoint conditioning_language.",
        )
    return value


def resolve_normalization(settings):
    """Normalization locale and model conditioning are deliberately separate.

    Explicit normalization_language wins, then language/lang; omission preserves
    Indian English. Never guess Hindi versus Marathi from Devanagari text.
    """
    value = next(
        (
            settings[key]
            for key in ("normalization_language", "language", "lang")
            if key in settings
        ),
        "en-IN",
    )
    try:
        return resolve_language(value)
    except NormalizationError as exc:
        raise PipelineError(exc.code, str(exc)) from None


def merge_settings(current, incoming, sticky):
    """Snapshot per queued utterance; newer aliases replace older aliases."""
    result = dict(current)
    # A language change must not inherit a previous explicit override.
    if "language" in incoming or "lang" in incoming:
        for key in ("conditioning_language", "normalization_language"):
            if key not in incoming:
                result.pop(key, None)
    for a, b in (("voice_id", "voice"), ("language", "lang")):
        if a in incoming or b in incoming:
            result.pop(a, None)
            result.pop(b, None)
    result.update({k: incoming[k] for k in sticky if k in incoming})
    return result


_REQUEST_LANGUAGE_KEYS = frozenset(
    ("language", "lang", "normalization_language", "conditioning_language")
)


def retain_session_settings(current, incoming, sticky):
    """Text-frame locales belong to their context, not the next customer turn.

    start/config frames still use merge_settings directly and establish explicit
    session defaults. Other sticky options retain the existing protocol behavior.
    """
    return merge_settings(
        current,
        {k: v for k, v in incoming.items() if k not in _REQUEST_LANGUAGE_KEYS},
        sticky,
    )
