#!/usr/bin/env python
"""Pronunciation-control workflows for MOSS-TTS (`text` / `pinyin` / `ipa`).

This example demonstrates:
- fail-fast `input_type` validation semantics (no silent conversion)
- optional helper-based conversion (`pypinyin`, DeepPhonemizer/phonemizer)
- graceful fallback to curated examples when helper deps are unavailable

Usage:
    uv run python examples/moss_tts_pronunciation_control.py --input-type pinyin
    uv run python examples/moss_tts_pronunciation_control.py --input-type ipa
    uv run python examples/moss_tts_pronunciation_control.py --input-type pinyin --auto-convert --text "您好，请问您来自哪座城市？"
    uv run python examples/moss_tts_pronunciation_control.py --input-type ipa --auto-convert --text "Hello, welcome to MOSS."
"""

from __future__ import annotations

import argparse

from mlx_audio.tts.generate import generate_audio
from mlx_audio.tts.models.moss_tts import (
    PronunciationHelperUnavailableError,
    convert_text_to_ipa,
    convert_text_to_tone_numbered_pinyin,
    validate_pronunciation_input_contract,
)

DEFAULT_PLAIN_TEXT = "Hello from the MOSS-TTS pronunciation control example."
DEFAULT_ZH_SOURCE_TEXT = "您好，请问您来自哪座城市？"
DEFAULT_PINYIN_TEXT = "nin2 hao3 qing3 wen4 nin2 lai2 zi4 na3 zuo4 cheng2 shi4"
DEFAULT_EN_SOURCE_TEXT = "Hello, may I ask which city you are from?"
DEFAULT_IPA_TEXT = "/həloʊ, meɪ aɪ æsk wɪtʃ sɪti juː ɑːr frʌm?/"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MOSS-TTS pronunciation control example",
    )
    parser.add_argument(
        "--model",
        default="OpenMOSS-Team/MOSS-TTS-Local-Transformer",
        help="MOSS model id or local path",
    )
    parser.add_argument(
        "--preset",
        default="moss_tts_local",
        help="Sampling preset (typically moss_tts_local or moss_tts)",
    )
    parser.add_argument(
        "--input-type",
        choices=["text", "pinyin", "ipa"],
        default="pinyin",
        help="Input representation contract to validate",
    )
    parser.add_argument(
        "--text",
        default=None,
        help="Source text; interpreted based on --input-type and --auto-convert",
    )
    parser.add_argument(
        "--auto-convert",
        action="store_true",
        help=(
            "Opt in to helper-based conversion. "
            "When disabled, your text is used as-is and only validated."
        ),
    )
    parser.add_argument(
        "--ipa-language",
        default="en-us",
        help="IPA conversion language code for helper mode",
    )
    parser.add_argument(
        "--deep-phonemizer-checkpoint",
        default=None,
        help=(
            "Optional DeepPhonemizer checkpoint path. "
            "If omitted, helper mode uses phonemizer fallback."
        ),
    )
    parser.add_argument(
        "--tokens",
        type=int,
        default=180,
        help="Target token budget",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=280,
        help="Safety generation cap",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/moss_tts_pronunciation_control",
        help="Directory for generated outputs",
    )
    parser.add_argument(
        "--file-prefix",
        default="pronunciation_control",
        help="Output filename prefix",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print runtime stats",
    )
    return parser.parse_args()


def _resolve_input_text(args: argparse.Namespace) -> str:
    if args.input_type == "text":
        return args.text or DEFAULT_PLAIN_TEXT

    if args.input_type == "pinyin":
        if args.auto_convert:
            source = args.text or DEFAULT_ZH_SOURCE_TEXT
            try:
                return convert_text_to_tone_numbered_pinyin(source)
            except PronunciationHelperUnavailableError as exc:
                print(
                    f"[moss_tts_pronunciation_control] pinyin helper unavailable: {exc}\n"
                    f"Falling back to curated tone-numbered text: {DEFAULT_PINYIN_TEXT}"
                )
                return DEFAULT_PINYIN_TEXT
        return args.text or DEFAULT_PINYIN_TEXT

    if args.auto_convert:
        source = args.text or DEFAULT_EN_SOURCE_TEXT
        try:
            return convert_text_to_ipa(
                source,
                language=args.ipa_language,
                deep_phonemizer_checkpoint=args.deep_phonemizer_checkpoint,
            )
        except PronunciationHelperUnavailableError as exc:
            print(
                f"[moss_tts_pronunciation_control] ipa helper unavailable: {exc}\n"
                f"Falling back to curated IPA span: {DEFAULT_IPA_TEXT}"
            )
            return DEFAULT_IPA_TEXT
    return args.text or DEFAULT_IPA_TEXT


def main() -> None:
    args = parse_args()
    text_payload = _resolve_input_text(args)
    validate_pronunciation_input_contract(text_payload, args.input_type)

    print(f"[moss_tts_pronunciation_control] input_type={args.input_type}")
    print(f"[moss_tts_pronunciation_control] text={text_payload}")

    generate_audio(
        text=text_payload,
        model=args.model,
        preset=args.preset,
        input_type=args.input_type,
        tokens=args.tokens,
        max_tokens=args.max_tokens,
        output_path=args.output_dir,
        file_prefix=args.file_prefix,
        verbose=args.verbose,
        play=False,
    )


if __name__ == "__main__":
    main()
