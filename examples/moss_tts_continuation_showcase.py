#!/usr/bin/env python
"""Continuation-style prompting showcase for MOSS-TTS Local/Delay.

This example demonstrates upstream-style continuation packing using explicit
conversation messages:
- final `assistant` message carries prefix audio (`<|audio|>` placeholder)
- generation continues from assistant audio context
- no `ref_audio` argument is passed to `model.generate(...)`

Usage:
    uv run python examples/moss_tts_continuation_showcase.py
    uv run python examples/moss_tts_continuation_showcase.py --model OpenMOSS-Team/MOSS-TTS --preset moss_tts
    uv run python examples/moss_tts_continuation_showcase.py --assistant-prefix-audio /path/to/prefix.wav --stream
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from mlx_audio.audio_io import write as audio_write
from mlx_audio.tts.utils import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MOSS-TTS continuation prompting showcase"
    )
    parser.add_argument(
        "--model",
        default="OpenMOSS-Team/MOSS-TTS-Local-Transformer",
        help="MOSS Local/Delay model id or local path",
    )
    parser.add_argument(
        "--preset",
        default="moss_tts_local",
        help="Sampling preset (typically moss_tts_local or moss_tts)",
    )
    parser.add_argument(
        "--user-text",
        default=(
            "Continue speaking in the same voice and cadence, then conclude with a "
            "short call to action."
        ),
        help="User instruction that the assistant should continue from the prefix audio",
    )
    parser.add_argument(
        "--assistant-prefix-audio",
        default="REFERENCE/MOSS-Audio-Tokenizer/demo/demo_gt.wav",
        help="Assistant prefix audio path used in continuation message",
    )
    parser.add_argument(
        "--instruct",
        default="Warm narrator voice, steady tempo.",
        help="Optional style instruction encoded in the user message",
    )
    parser.add_argument(
        "--input-type",
        choices=["text", "pinyin", "ipa"],
        default="text",
        help="Input representation contract",
    )
    parser.add_argument(
        "--tokens",
        type=int,
        default=180,
        help="Prompt-side token hint in the user message",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=320,
        help="Safety cap for generation loop",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Emit/persist streaming chunks",
    )
    parser.add_argument(
        "--streaming-interval",
        type=float,
        default=0.5,
        help="Chunk interval in seconds for stream mode",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/moss_tts_continuation_showcase",
        help="Directory for generated outputs",
    )
    parser.add_argument(
        "--file-prefix",
        default="continuation",
        help="Output filename prefix",
    )
    parser.add_argument(
        "--write-conversation-json",
        action="store_true",
        help="Write serialized conversation payload for inspection",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Enable strict checkpoint loading",
    )
    return parser.parse_args()


def _serialize_message_value(value: Any) -> Any:
    if isinstance(value, list):
        return [_serialize_message_value(item) for item in value]
    if isinstance(value, mx.array):
        return {
            "type": "mx.array",
            "shape": [int(dim) for dim in value.shape],
            "dtype": str(value.dtype),
        }
    if isinstance(value, Path):
        return str(value)
    return value


def _serialize_conversation(conversation: list[dict[str, Any]]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for message in conversation:
        payload.append(
            {
                key: _serialize_message_value(raw_value)
                for key, raw_value in message.items()
            }
        )
    return payload


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    assistant_prefix_audio = Path(args.assistant_prefix_audio)
    if not assistant_prefix_audio.exists():
        raise FileNotFoundError(
            f"Assistant prefix audio not found: {assistant_prefix_audio}"
        )

    model = load_model(args.model, strict=args.strict)
    model_type = getattr(model, "model_type", None)
    if model_type == "moss_tts_realtime":
        raise ValueError(
            "This showcase targets Local/Delay-style continuation. "
            "Use examples/moss_tts_realtime_multiturn_agent.py for realtime."
        )

    processor = getattr(model, "processor", None)
    if processor is None:
        raise RuntimeError("Loaded model does not expose a MOSS processor")

    user_message = processor.build_user_message(
        text=args.user_text,
        instruction=args.instruct,
        tokens=args.tokens,
        input_type=args.input_type,
    )
    assistant_message = processor.build_assistant_message(
        audio_codes_list=[str(assistant_prefix_audio)]
    )
    conversation = [user_message, assistant_message]

    if args.write_conversation_json:
        conversation_path = output_dir / f"{args.file_prefix}_conversation.json"
        conversation_path.write_text(
            json.dumps(_serialize_conversation(conversation), indent=2),
            encoding="utf-8",
        )
        print(f"Conversation payload saved: {conversation_path}")

    print("Running continuation showcase with explicit conversation messages...")
    print("`ref_audio` is intentionally not passed to model.generate().")

    emitted_chunks: list[mx.array] = []
    results = model.generate(
        conversation=conversation,
        preset=args.preset,
        input_type=args.input_type,
        max_tokens=args.max_tokens,
        stream=args.stream,
        streaming_interval=args.streaming_interval,
    )

    for idx, result in enumerate(results):
        audio = result.audio
        file_path = output_dir / f"{args.file_prefix}_{idx:03d}.wav"
        audio_write(
            str(file_path),
            np.array(audio),
            int(result.sample_rate),
            format="wav",
        )
        print(
            f"chunk[{idx}] -> {file_path} | samples={int(audio.shape[0])} "
            f"duration={result.audio_duration}"
        )

        if args.stream and int(audio.shape[0]) > 0:
            emitted_chunks.append(audio)

    if args.stream:
        if not emitted_chunks:
            raise RuntimeError("No non-empty stream chunks were emitted")
        merged = (
            emitted_chunks[0]
            if len(emitted_chunks) == 1
            else mx.concatenate(emitted_chunks, axis=0)
        )
        merged_path = output_dir / f"{args.file_prefix}_merged.wav"
        audio_write(
            str(merged_path),
            np.array(merged),
            int(model.sample_rate),
            format="wav",
        )
        print(f"Merged stream output -> {merged_path} | samples={int(merged.shape[0])}")


if __name__ == "__main__":
    main()
