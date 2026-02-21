#!/usr/bin/env python
"""Realtime text-delta streaming with MOSS-TTS-Realtime.

This example exercises the explicit session lifecycle:
- `reset_turn()`
- text delta ingestion via `bridge_text_stream(...)`
- chunk decoding with backpressure controls
- final merged output write

Usage:
    uv run python examples/moss_tts_realtime_text_deltas.py
    uv run python examples/moss_tts_realtime_text_deltas.py --text "Hello realtime world" --delta-chars 6
    uv run python examples/moss_tts_realtime_text_deltas.py --deltas-json ./my_deltas.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import mlx.core as mx
import numpy as np

from mlx_audio.audio_io import write as audio_write
from mlx_audio.tts.models.moss_tts_realtime import (
    MossTTSRealtimeInference,
    RealtimeSession,
)
from mlx_audio.tts.utils import load_model
from mlx_audio.utils import load_audio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MOSS-TTS-Realtime text-delta streaming example"
    )
    parser.add_argument(
        "--model",
        default="OpenMOSS-Team/MOSS-TTS-Realtime",
        help="Realtime model id or local path",
    )
    parser.add_argument(
        "--text",
        default=(
            "Realtime synthesis with incremental text deltas. "
            "This demonstrates explicit lifecycle control and chunked decoding."
        ),
        help="Source text used when --deltas-json is not provided",
    )
    parser.add_argument(
        "--deltas-json",
        default=None,
        help="Optional JSON file containing a list of text delta strings",
    )
    parser.add_argument(
        "--delta-chars",
        type=int,
        default=8,
        help="Chunk size for splitting --text into synthetic deltas",
    )
    parser.add_argument(
        "--hold-back",
        type=int,
        default=0,
        help="Tokenizer hold-back (stability vs latency)",
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
        help="Chunk duration override passed to codec decode",
    )
    parser.add_argument(
        "--max-pending-frames",
        type=int,
        default=4096,
        help="Backpressure cap for queued audio token frames",
    )
    parser.add_argument(
        "--prefill-text-len",
        type=int,
        default=12,
        help="Text-token count before first prefill",
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
        help="Minimum chars before punctuation-based split",
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.6)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
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
        help="Optional cache context cap override",
    )
    parser.add_argument(
        "--ref-audio",
        default=None,
        help="Optional reference audio path to prime turn voice",
    )
    parser.add_argument(
        "--no-system-prompt",
        action="store_true",
        help="Do not include system prompt during reset_turn",
    )
    parser.add_argument(
        "--reset-cache",
        action="store_true",
        help="Force cache reset at reset_turn",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/moss_tts_realtime_text_deltas",
        help="Directory for generated chunks and merged wav",
    )
    parser.add_argument(
        "--file-prefix",
        default="realtime",
        help="Output file prefix",
    )
    parser.add_argument(
        "--save-chunks",
        action="store_true",
        help="Persist every emitted chunk (in addition to merged audio)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Enable strict checkpoint loading",
    )
    return parser.parse_args()


def _load_text_deltas(args: argparse.Namespace) -> list[str]:
    if args.deltas_json:
        payload = json.loads(Path(args.deltas_json).read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("--deltas-json must contain a JSON list")
        return [str(item) for item in payload]

    text = str(args.text)
    step = int(args.delta_chars)
    if step <= 0:
        return [text]
    return [text[idx : idx + step] for idx in range(0, len(text), step)]


def _iter_chunks(
    session: RealtimeSession, deltas: Iterable[str], hold_back: int, drain_step: int
):
    for chunk in session.bridge_text_stream(
        deltas, hold_back=hold_back, drain_step=drain_step
    ):
        if int(chunk.shape[0]) > 0:
            yield chunk


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.model, strict=args.strict)
    if getattr(model, "model_type", None) != "moss_tts_realtime":
        raise ValueError(
            f"Expected a realtime checkpoint, got model_type={getattr(model, 'model_type', None)!r}"
        )

    inferencer = MossTTSRealtimeInference(
        model=model.model,
        tokenizer=model.tokenizer,
        config=model.config,
        max_length=args.max_tokens,
        max_context_tokens=args.max_context_tokens,
    )

    decode_kwargs = {"chunk_duration": float(args.decode_chunk_duration)}
    session = RealtimeSession(
        inferencer=inferencer,
        processor=model.processor,
        chunk_frames=args.chunk_frames,
        overlap_frames=args.overlap_frames,
        decode_kwargs=decode_kwargs,
        max_pending_frames=args.max_pending_frames,
        prefill_text_len=args.prefill_text_len,
        text_buffer_size=args.text_buffer_size,
        min_text_chunk_chars=args.min_text_chunk_chars,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        do_sample=not args.deterministic,
        repetition_penalty=args.repetition_penalty,
    )

    prompt_audio = None
    if args.ref_audio is not None:
        prompt_audio = load_audio(args.ref_audio, sample_rate=model.sample_rate)

    deltas = _load_text_deltas(args)
    print(f"Loaded {len(deltas)} text deltas")

    emitted_chunks: list[mx.array] = []
    try:
        session.reset_turn(
            user_text="",
            user_audio_tokens=prompt_audio,
            include_system_prompt=not args.no_system_prompt,
            reset_cache=args.reset_cache,
        )

        for idx, chunk in enumerate(
            _iter_chunks(
                session,
                deltas,
                hold_back=args.hold_back,
                drain_step=args.drain_step,
            )
        ):
            emitted_chunks.append(chunk)
            if args.save_chunks:
                chunk_path = output_dir / f"{args.file_prefix}_chunk_{idx:03d}.wav"
                audio_write(
                    str(chunk_path),
                    np.array(chunk),
                    model.sample_rate,
                    format="wav",
                )
                print(f"chunk[{idx}] -> {chunk_path}")

    finally:
        session.close()

    if not emitted_chunks:
        raise RuntimeError("No realtime audio chunks were emitted")

    merged = (
        emitted_chunks[0]
        if len(emitted_chunks) == 1
        else mx.concatenate(emitted_chunks, axis=0)
    )
    merged_path = output_dir / f"{args.file_prefix}_merged.wav"
    audio_write(
        str(merged_path),
        np.array(merged),
        model.sample_rate,
        format="wav",
    )

    print(
        f"Merged output: {merged_path} | chunks={len(emitted_chunks)} "
        f"samples={int(merged.shape[0])}"
    )


if __name__ == "__main__":
    main()
