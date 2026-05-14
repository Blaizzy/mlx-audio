from __future__ import annotations

import re

_LAUGH_VERBS = {
    r"\blaugh(?:s|ed|ing)?\b": 1.5,
    r"\bcackl(?:e|es|ed|ing)\b": 1.5,
    r"\bchuckl(?:e|es|ed|ing)\b": 1.0,
    r"\bgiggl(?:e|es|ed|ing)\b": 1.0,
    r"\bsnicker(?:s|ed|ing)?\b": 0.8,
    r"\bcru?el laugh\b": 1.5,
}


def _contextual_laugh_duration(text: str) -> float:
    short_mod = re.compile(
        r"^\s*(?:[a-z]+ly )?(?:briefly|shortly|once|quickly)",
        re.IGNORECASE,
    )
    long_mod = re.compile(
        r"^\s*(?:[a-z]+ly )?(?:maniacally|heartily|uproariously|"
        r"uncontrollably|hysterically|darkly|wickedly|evilly|loudly|long)"
        r"|^\s*between phrases",
        re.IGNORECASE,
    )

    total = 0.0
    for pattern, base_duration in _LAUGH_VERBS.items():
        for match in re.finditer(pattern, text, re.IGNORECASE):
            context = text[match.end() : match.end() + 40]
            if short_mod.match(context):
                total += base_duration * 0.4
            elif long_mod.match(context):
                total += base_duration * 1.2
            else:
                total += base_duration

    quoted = re.findall(r'"([^"]+)"', text)
    quoted += re.findall(r"'((?:[^']|'(?![\s.,!?)\]]))+)'", text)
    for quote in quoted:
        for run in re.findall(r"(?:h[ae]){3,}|(?:h[ae][ \-]?){3,}", quote, re.I):
            syllables = len(re.findall(r"h[ae]", run, re.I))
            total += 0.2 * max(syllables - 2, 0)
    return total


def _estimate_nonverbal_duration(text: str) -> float:
    patterns = {
        r"\bsighs?\b": 0.8,
        r"\bshaky breath\b": 1.0,
        r"\bbreathing deeply\b": 1.0,
        r"\bgasps?\b": 0.5,
        r"\bburps?\b": 0.5,
        r"\byawns?\b": 1.0,
        r"\bpants?\b": 0.8,
        r"\bwheezes?\b": 0.8,
        r"\bcoughs?\b": 0.8,
        r"\bsniffles?\b": 0.5,
        r"\bsnorts?\b": 0.3,
        r"\bgroans?\b": 0.8,
        r"\blong pause\b": 1.0,
        r"\bpauses? briefly\b": 0.3,
        r"\bpauses?\b": 0.5,
        r"\bsilence\b": 1.0,
        r"\blets? the .{1,20} hang\b": 1.0,
        r"\blets? .{1,20} sink in\b": 1.0,
        r"\bslams?\b": 0.5,
        r"\bclaps?\b": 0.3,
        r"\bdraws? (?:his|her|a) sword\b": 0.5,
        r"\btakes? a (?:drag|swig|sip|drink)\b": 0.5,
        r"\bwhistles?\b": 1.0,
        r"\bhums?\b": 0.8,
        r"\bmutters?\b": 1.5,
        r"\bmumbles?\b": 1.0,
        r"\bwhispers?\b": 0.0,
        r"\bclears? (?:his|her) throat\b": 0.5,
        r"\bgulps?\b": 0.5,
        r"\bswallows?\b": 0.5,
        r"\bvoice (?:breaks?|cracks?|trembles?|drops?|rises?)\b": 0.5,
        r"\bsteadies? (?:him|her)self\b": 1.0,
        r"\bcatches? (?:his|her) breath\b": 1.0,
        r"\bcomposes? (?:him|her)self\b": 0.8,
        r"\bdemeanor shifts?\b": 0.5,
        r"\bsettles? in\b": 0.5,
        r"\bleans? in\b": 0.3,
        r"\bwipes? (?:his|her) eyes\b": 0.5,
    }
    extra = 0.0
    for pattern, duration in patterns.items():
        extra += duration * len(re.findall(pattern, text, re.IGNORECASE))
    return extra + _contextual_laugh_duration(text)


def estimate_speech_duration(text: str, speed: float = 1.0) -> float:
    quotes = re.findall(r'"([^"]+)"', text)
    if not quotes:
        quotes = re.findall(r"'((?:[^']|'(?![\s.,!?)\]]))+)'", text)
        quotes = [quote for quote in quotes if len(quote.split()) > 3]
    if quotes:
        spoken = " ".join(quotes)
    elif ":" in text:
        spoken = text.split(":", 1)[1].strip()
    else:
        spoken = text

    chars_per_second = 14.0
    text_length = len(spoken)
    if text_length < 40:
        chars_per_second *= 0.6
    elif text_length < 80:
        chars_per_second *= 0.8

    duration = text_length / (chars_per_second * speed)
    duration += (spoken.count(".") + spoken.count("!") + spoken.count("?")) * 0.3
    duration += _estimate_nonverbal_duration(text)
    return max(3.0, round(duration + 2.0, 1))
