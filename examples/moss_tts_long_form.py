#!/usr/bin/env python
"""Segmented long-form generation with MOSS-TTS.

This example demonstrates the long-form planner/continuity path and writes
both audio outputs and segment metrics.

Usage:
    uv run python examples/moss_tts_long_form.py
    uv run python examples/moss_tts_long_form.py --stream
    uv run python examples/moss_tts_long_form.py --long-form-target-chars 280 --long-form-prefix-text-chars 64
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_audio.audio_io import write as audio_write
from mlx_audio.tts.utils import load_model

DEFAULT_TEXT = (
    "Long-form synthesis in MOSS-TTS works by planning bounded text segments, "
    "generating each segment with explicit cache boundaries, and carrying a short "
    "audio tail forward to preserve continuity. This example intentionally uses a "
    "multi-sentence paragraph so the planner can split at natural boundaries and "
    "emit deterministic segment metrics. You can tune minimum, target, and maximum "
    "character budgets to trade off latency, seam quality, and memory behavior."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MOSS-TTS long-form generation example"
    )
    parser.add_argument(
        "--model",
        default="OpenMOSS-Team/MOSS-TTS-Local-Transformer",
        help="MOSS model id or local path",
    )
    parser.add_argument(
        "--text",
        default=DEFAULT_TEXT,
        help="Long-form source text",
    )
    parser.add_argument("--preset", default="moss_tts_local", help="Sampling preset")
    parser.add_argument(
        "--input-type",
        choices=["text", "pinyin", "ipa"],
        default="text",
        help="Input representation",
    )
    parser.add_argument("--language", default=None, help="Optional language hint")
    parser.add_argument("--instruct", default=None, help="Optional style instruction")
    parser.add_argument("--quality", default="balanced", help="Quality hint")
    parser.add_argument(
        "--tokens",
        type=int,
        default=None,
        help="Target token budget (takes precedence over --duration-s)",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=None,
        help="Optional duration hint when --tokens is omitted",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=260,
        help="Per-segment safety cap",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Emit per-segment streaming chunks",
    )
    parser.add_argument(
        "--streaming-interval",
        type=float,
        default=0.5,
        help="Streaming chunk interval",
    )

    parser.add_argument("--long-form-min-chars", type=int, default=160)
    parser.add_argument("--long-form-target-chars", type=int, default=320)
    parser.add_argument("--long-form-max-chars", type=int, default=520)
    parser.add_argument("--long-form-prefix-audio-seconds", type=float, default=2.0)
    parser.add_argument("--long-form-prefix-audio-max-tokens", type=int, default=25)
    parser.add_argument("--long-form-prefix-text-chars", type=int, default=0)
    parser.add_argument("--long-form-retry-attempts", type=int, default=0)

    parser.add_argument(
        "--output-dir",
        default="outputs/moss_tts_long_form",
        help="Directory for generated audio and metrics",
    )
    parser.add_argument(
        "--file-prefix",
        default="long_form",
        help="Output filename prefix",
    )
    parser.add_argument(
        "--metrics-json",
        default=None,
        help="Optional explicit metrics output path",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Enable strict checkpoint loading",
    )
    return parser.parse_args()


def _prune_none_values(payload: dict) -> dict:
    return {key: value for key, value in payload.items() if value is not None}


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.model, strict=args.strict)
    generation_kwargs = _prune_none_values(
        {
            "text": args.text,
            "preset": args.preset,
            "input_type": args.input_type,
            "language": args.language,
            "instruct": args.instruct,
            "quality": args.quality,
            "tokens": args.tokens,
            "duration_s": args.duration_s,
            "max_tokens": args.max_tokens,
            "stream": args.stream,
            "streaming_interval": args.streaming_interval,
            "long_form": True,
            "long_form_min_chars": args.long_form_min_chars,
            "long_form_target_chars": args.long_form_target_chars,
            "long_form_max_chars": args.long_form_max_chars,
            "long_form_prefix_audio_seconds": args.long_form_prefix_audio_seconds,
            "long_form_prefix_audio_max_tokens": args.long_form_prefix_audio_max_tokens,
            "long_form_prefix_text_chars": args.long_form_prefix_text_chars,
            "long_form_retry_attempts": args.long_form_retry_attempts,
        }
    )

    results = list(model.generate(**generation_kwargs))
    if not results:
        raise RuntimeError("Long-form generation returned no results")

    for idx, result in enumerate(results):
        suffix = f"_{idx:03d}" if (args.stream or len(results) > 1) else ""
        chunk_path = output_dir / f"{args.file_prefix}{suffix}.wav"
        audio_write(
            str(chunk_path),
            np.array(result.audio),
            result.sample_rate,
            format="wav",
        )
        print(
            f"[{idx}] {chunk_path} | duration={result.audio_duration} "
            f"tokens={result.token_count} rtf={result.real_time_factor:.3f}"
        )

    if len(results) > 1:
        merged_audio = mx.concatenate([result.audio for result in results], axis=0)
        merged_path = output_dir / f"{args.file_prefix}_merged.wav"
        audio_write(
            str(merged_path),
            np.array(merged_audio),
            results[0].sample_rate,
            format="wav",
        )
        print(f"Merged audio: {merged_path}")

    segment_metrics = [
        asdict(metric)
        for metric in getattr(model, "_last_long_form_segment_metrics", [])
    ]
    boundary_metrics = [
        asdict(metric)
        for metric in getattr(model, "_last_long_form_boundary_metrics", [])
    ]

    metrics_payload = {
        "model": args.model,
        "generation_kwargs": generation_kwargs,
        "segment_metrics": segment_metrics,
        "boundary_metrics": boundary_metrics,
    }
    metrics_path = (
        Path(args.metrics_json)
        if args.metrics_json
        else output_dir / f"{args.file_prefix}_metrics.json"
    )
    metrics_path.write_text(
        json.dumps(metrics_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
