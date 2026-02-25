#!/usr/bin/env python
"""Multi-turn realtime agent showcase for MOSS-TTS-Realtime.

This demo mirrors upstream multiturn flow with explicit lifecycle controls:
- persistent voice prompt timbre (`set_voice_prompt_audio`)
- per-turn user audio conditioning (`reset_turn(..., user_audio_tokens=...)`)
- assistant response streamed from text deltas (`bridge_text_stream`)

Default assets are local/offline under:
`REFERENCE/MOSS-TTS-GitHub/moss_tts_realtime/audio/*`

Usage:
    uv run python examples/moss_tts_realtime_multiturn_agent.py
    uv run python examples/moss_tts_realtime_multiturn_agent.py --save-chunks
    uv run python examples/moss_tts_realtime_multiturn_agent.py --dialogue-json ./my_turns.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from mlx_audio.audio_io import write as audio_write
from mlx_audio.tts.models.moss_tts_realtime import (
    MossTTSRealtimeInference,
    RealtimeSession,
)
from mlx_audio.tts.utils import load_model

DEFAULT_VOICE_PROMPT_AUDIO = (
    "REFERENCE/MOSS-TTS-GitHub/moss_tts_realtime/audio/user1.wav"
)
DEFAULT_TURNS = [
    {
        "user_text": (
            "I just landed in Paris and only have six hours before my next flight. "
            "What should I prioritize?"
        ),
        "user_audio": "REFERENCE/MOSS-TTS-GitHub/moss_tts_realtime/audio/user1.wav",
        "assistant_text": (
            "Welcome to Paris. With six hours, I recommend a focused walking route "
            "near the Seine, a short coffee stop, and one museum highlight."
        ),
    },
    {
        "user_text": "I prefer a relaxed pace and mostly outdoor sights.",
        "user_audio": "REFERENCE/MOSS-TTS-GitHub/moss_tts_realtime/audio/user2.wav",
        "assistant_text": (
            "Great, start from Notre-Dame, walk toward the Louvre gardens, and end "
            "at the river with sunset views. Keep travel light and avoid long queues."
        ),
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MOSS-TTS-Realtime multiturn agent showcase"
    )
    parser.add_argument(
        "--model",
        default="OpenMOSS-Team/MOSS-TTS-Realtime",
        help="Realtime model id or local path",
    )
    parser.add_argument(
        "--voice-prompt-audio",
        default=DEFAULT_VOICE_PROMPT_AUDIO,
        help="Persistent conversation-level voice prompt audio",
    )
    parser.add_argument(
        "--dialogue-json",
        default=None,
        help=(
            "Optional JSON list of turns. Each turn supports: user_text, "
            "assistant_text, optional user_audio, optional assistant_deltas"
        ),
    )
    parser.add_argument(
        "--delta-chars",
        type=int,
        default=10,
        help="Assistant text split size when assistant_deltas is not provided",
    )
    parser.add_argument(
        "--hold-back",
        type=int,
        default=0,
        help="Text-delta tokenizer hold-back",
    )
    parser.add_argument(
        "--drain-step",
        type=int,
        default=1,
        help="Drain step budget after end-of-text",
    )
    parser.add_argument(
        "--chunk-frames",
        type=int,
        default=40,
        help="Decoder chunk size in token frames",
    )
    parser.add_argument(
        "--overlap-frames",
        type=int,
        default=4,
        help="Crossfade overlap in token frames",
    )
    parser.add_argument(
        "--decode-chunk-duration",
        type=float,
        default=0.32,
        help="Codec decode chunk duration override",
    )
    parser.add_argument(
        "--max-pending-frames",
        type=int,
        default=4096,
        help="Backpressure cap for queued frames",
    )
    parser.add_argument(
        "--prefill-text-len",
        type=int,
        default=12,
        help="Text token count before first prefill",
    )
    parser.add_argument(
        "--text-buffer-size",
        type=int,
        default=32,
        help="Buffer size before whitespace fallback split",
    )
    parser.add_argument(
        "--min-text-chunk-chars",
        type=int,
        default=8,
        help="Minimum chars before punctuation split",
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.6)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument(
        "--repetition-window",
        type=int,
        default=50,
        help="Repetition-penalty lookback window",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Disable sampling randomness (do_sample=False)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1200,
        help="Inferencer max generation steps",
    )
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=None,
        help="Optional context-window cap",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/moss_tts_realtime_multiturn_agent",
        help="Directory for generated outputs",
    )
    parser.add_argument(
        "--file-prefix",
        default="turn",
        help="Per-turn output file prefix",
    )
    parser.add_argument(
        "--save-chunks",
        action="store_true",
        help="Persist each emitted chunk in addition to merged turn wav",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Enable strict checkpoint loading",
    )
    return parser.parse_args()


def _split_text_deltas(text: str, chunk_chars: int) -> list[str]:
    source = str(text)
    step = int(chunk_chars)
    if step <= 0:
        return [source]
    return [source[idx : idx + step] for idx in range(0, len(source), step)]


def _load_turns(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.dialogue_json is None:
        return [dict(item) for item in DEFAULT_TURNS]

    payload = json.loads(Path(args.dialogue_json).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("--dialogue-json must contain a JSON list")

    turns: list[dict[str, Any]] = []
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Turn {idx} is not an object")
        if "assistant_text" not in item and "assistant_deltas" not in item:
            raise ValueError(f"Turn {idx} requires assistant_text or assistant_deltas")
        turns.append(dict(item))
    return turns


def _resolve_assistant_deltas(
    turn: dict[str, Any], default_delta_chars: int
) -> list[str]:
    if "assistant_deltas" in turn and turn["assistant_deltas"] is not None:
        deltas = turn["assistant_deltas"]
        if not isinstance(deltas, list):
            raise ValueError("assistant_deltas must be a JSON list")
        return [str(item) for item in deltas]
    return _split_text_deltas(str(turn.get("assistant_text", "")), default_delta_chars)


def _write_markdown_manifest(manifest: dict[str, Any], path: Path) -> None:
    lines = [
        "# MOSS-TTS-Realtime Multiturn Showcase Manifest",
        "",
        f"- Model: `{manifest['model']}`",
        f"- Voice prompt: `{manifest['voice_prompt_audio']}`",
        f"- Turns requested: `{manifest['summary']['requested_turns']}`",
        f"- Turns completed: `{manifest['summary']['completed_turns']}`",
        "",
        "## Turn Results",
        "",
        "| Turn | Output | Chunks | Duration (s) | User Audio |",
        "|---|---|---:|---:|---|",
    ]

    for turn in manifest["turns"]:
        if turn.get("status") != "ok":
            lines.append(
                f"| {turn['turn_index']} | error | 0 | 0.00 | `{turn.get('user_audio', '-')}` |"
            )
            continue
        lines.append(
            "| "
            f"{turn['turn_index']} | `{turn['merged_output']}` | "
            f"{turn['chunk_count']} | {turn['duration_seconds']:.2f} | "
            f"`{turn.get('user_audio', '-')}` |"
        )

    if manifest.get("errors"):
        lines.extend(["", "## Errors", ""])
        for error in manifest["errors"]:
            lines.append(f"- Turn {error['turn_index']}: {error['error']}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    turns = _load_turns(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.model, strict=args.strict)
    if getattr(model, "model_type", None) != "moss_tts_realtime":
        raise ValueError(
            f"Expected realtime checkpoint, got model_type={getattr(model, 'model_type', None)!r}"
        )

    inferencer = MossTTSRealtimeInference(
        model=model.model,
        tokenizer=model.tokenizer,
        config=model.config,
        max_length=args.max_tokens,
        max_context_tokens=args.max_context_tokens,
    )

    session = RealtimeSession(
        inferencer=inferencer,
        processor=model.processor,
        chunk_frames=args.chunk_frames,
        overlap_frames=args.overlap_frames,
        decode_kwargs={"chunk_duration": float(args.decode_chunk_duration)},
        max_pending_frames=args.max_pending_frames,
        prefill_text_len=args.prefill_text_len,
        text_buffer_size=args.text_buffer_size,
        min_text_chunk_chars=args.min_text_chunk_chars,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        do_sample=not args.deterministic,
        repetition_penalty=args.repetition_penalty,
        repetition_window=args.repetition_window,
    )

    voice_prompt_audio = Path(args.voice_prompt_audio)
    if not voice_prompt_audio.exists():
        raise FileNotFoundError(f"Voice prompt audio not found: {voice_prompt_audio}")

    manifest: dict[str, Any] = {
        "model": args.model,
        "voice_prompt_audio": str(voice_prompt_audio),
        "turns": [],
        "errors": [],
        "summary": {
            "requested_turns": len(turns),
            "completed_turns": 0,
        },
    }

    try:
        session.set_voice_prompt_audio(str(voice_prompt_audio))

        for turn_idx, turn in enumerate(turns):
            user_text = str(turn.get("user_text", ""))
            user_audio = turn.get("user_audio")
            if user_audio is not None and not Path(str(user_audio)).exists():
                raise FileNotFoundError(
                    f"Turn {turn_idx} user_audio not found: {user_audio}"
                )

            include_system_prompt = turn_idx == 0
            reset_cache = turn_idx == 0
            print(
                f"Turn {turn_idx}: include_system_prompt={include_system_prompt} "
                f"reset_cache={reset_cache}"
            )
            session.reset_turn(
                user_text=user_text,
                user_audio_tokens=user_audio,
                include_system_prompt=include_system_prompt,
                reset_cache=reset_cache,
            )

            assistant_deltas = _resolve_assistant_deltas(turn, args.delta_chars)
            print(
                f"Turn {turn_idx}: streaming {len(assistant_deltas)} assistant deltas"
            )

            emitted_chunks: list[mx.array] = []
            for chunk_idx, chunk in enumerate(
                session.bridge_text_stream(
                    assistant_deltas,
                    hold_back=args.hold_back,
                    drain_step=args.drain_step,
                )
            ):
                if int(chunk.shape[0]) == 0:
                    continue
                emitted_chunks.append(chunk)
                if args.save_chunks:
                    chunk_path = (
                        output_dir
                        / f"{args.file_prefix}_{turn_idx:02d}_chunk_{chunk_idx:03d}.wav"
                    )
                    audio_write(
                        str(chunk_path),
                        np.array(chunk),
                        int(model.sample_rate),
                        format="wav",
                    )

            if not emitted_chunks:
                error_message = "No non-empty chunks emitted"
                manifest["errors"].append(
                    {"turn_index": turn_idx, "error": error_message}
                )
                manifest["turns"].append(
                    {
                        "turn_index": turn_idx,
                        "status": "error",
                        "user_audio": (
                            str(user_audio) if user_audio is not None else None
                        ),
                        "error": error_message,
                    }
                )
                continue

            merged = (
                emitted_chunks[0]
                if len(emitted_chunks) == 1
                else mx.concatenate(emitted_chunks, axis=0)
            )
            merged_path = output_dir / f"{args.file_prefix}_{turn_idx:02d}.wav"
            audio_write(
                str(merged_path),
                np.array(merged),
                int(model.sample_rate),
                format="wav",
            )
            duration_seconds = float(merged.shape[0]) / float(model.sample_rate)
            print(
                f"Turn {turn_idx}: wrote {merged_path} "
                f"samples={int(merged.shape[0])} duration={duration_seconds:.2f}s"
            )

            manifest["summary"]["completed_turns"] += 1
            manifest["turns"].append(
                {
                    "turn_index": turn_idx,
                    "status": "ok",
                    "user_text": user_text,
                    "assistant_text": str(turn.get("assistant_text", "")),
                    "user_audio": str(user_audio) if user_audio is not None else None,
                    "chunk_count": len(emitted_chunks),
                    "samples": int(merged.shape[0]),
                    "sample_rate": int(model.sample_rate),
                    "duration_seconds": duration_seconds,
                    "merged_output": str(merged_path),
                }
            )

    finally:
        session.clear_voice_prompt_tokens()
        session.close()

    manifest_json_path = output_dir / "multiturn_manifest.json"
    manifest_md_path = output_dir / "multiturn_manifest.md"
    manifest_json_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_markdown_manifest(manifest, manifest_md_path)
    print(f"Wrote manifest: {manifest_json_path}")
    print(f"Wrote manifest: {manifest_md_path}")


if __name__ == "__main__":
    main()
