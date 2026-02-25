#!/usr/bin/env python
"""One-command MOSS-TTS family showcase album generator.

This script runs a curated prompt suite across MOSS family variants and writes:
- per-track audio files (`wav`)
- `manifest.json` with run metadata and per-track stats
- `manifest.md` for sharing in PRs/issues/docs

By default the suite targets all family variants:
Local, Delay, TTSD, VoiceGenerator, SoundEffect, Realtime.

Usage:
    uv run python examples/moss_tts_showcase_album.py
    uv run python examples/moss_tts_showcase_album.py --variants local,voice_generator,realtime
    uv run python examples/moss_tts_showcase_album.py --fail-fast
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from mlx_audio.audio_io import write as audio_write
from mlx_audio.tts.utils import load_model


@dataclass(frozen=True)
class ShowcaseTrack:
    track_id: str
    variant: str
    title: str
    model: str
    generate_kwargs: dict[str, Any]


SUPPORTED_VARIANTS = [
    "local",
    "delay",
    "ttsd",
    "voice_generator",
    "soundeffect",
    "realtime",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MOSS-TTS family showcase album generator"
    )
    parser.add_argument(
        "--variants",
        default="all",
        help=(
            "Comma-separated subset of variants to run "
            "(local,delay,ttsd,voice_generator,soundeffect,realtime) or 'all'"
        ),
    )
    parser.add_argument(
        "--reference-audio",
        default="REFERENCE/MOSS-Audio-Tokenizer/demo/demo_gt.wav",
        help="Reference audio used for TTSD speaker schema",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/moss_tts_showcase_album",
        help="Output directory for tracks and manifest files",
    )
    parser.add_argument(
        "--manifest-prefix",
        default="showcase_album",
        help="Filename prefix for manifest outputs",
    )
    parser.add_argument(
        "--max-tokens-override",
        type=int,
        default=None,
        help="Optional max_tokens override applied to every track",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at first track failure instead of continuing",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Enable strict checkpoint loading",
    )
    return parser.parse_args()


def _parse_variants(raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        return list(SUPPORTED_VARIANTS)

    requested = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = sorted(set(requested) - set(SUPPORTED_VARIANTS))
    if unknown:
        raise ValueError(f"Unsupported variants: {', '.join(unknown)}")

    seen: set[str] = set()
    ordered: list[str] = []
    for item in requested:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _build_showcase_suite(reference_audio: str) -> list[ShowcaseTrack]:
    ttsd_speakers = [
        {
            "speaker_id": 1,
            "ref_audio": reference_audio,
            "ref_text": "Speaker one prompt: professional and concise.",
        },
        {
            "speaker_id": 2,
            "ref_audio": reference_audio,
            "ref_text": "Speaker two prompt: warm and conversational.",
        },
    ]

    return [
        ShowcaseTrack(
            track_id="01_local_intro",
            variant="local",
            title="Local Transformer Intro",
            model="OpenMOSS-Team/MOSS-TTS-Local-Transformer",
            generate_kwargs={
                "text": (
                    "Welcome to the MOSS-TTS showcase album. This local track "
                    "demonstrates clear speech at lower memory cost."
                ),
                "instruct": "Friendly studio narrator, medium pace.",
                "preset": "moss_tts_local",
                "input_type": "text",
                "tokens": 180,
                "max_tokens": 300,
            },
        ),
        ShowcaseTrack(
            track_id="02_delay_story",
            variant="delay",
            title="Delay Model Story Clip",
            model="OpenMOSS-Team/MOSS-TTS",
            generate_kwargs={
                "text": (
                    "In this delay-model sample, the narration remains expressive "
                    "while preserving long-range coherence."
                ),
                "instruct": "Confident storyteller with smooth prosody.",
                "preset": "moss_tts",
                "input_type": "text",
                "tokens": 200,
                "max_tokens": 340,
            },
        ),
        ShowcaseTrack(
            track_id="03_ttsd_dialogue",
            variant="ttsd",
            title="TTSD Dialogue",
            model="OpenMOSS-Team/MOSS-TTSD-v1.0",
            generate_kwargs={
                "text": (
                    "[S1] Thanks for joining this product review. "
                    "[S2] Happy to be here, let's cover the key tradeoffs quickly."
                ),
                "dialogue_speakers": ttsd_speakers,
                "preset": "ttsd",
                "language": "en",
                "tokens": 220,
                "max_tokens": 360,
            },
        ),
        ShowcaseTrack(
            track_id="04_voice_generator",
            variant="voice_generator",
            title="VoiceGenerator Design",
            model="OpenMOSS-Team/MOSS-Voice-Generator",
            generate_kwargs={
                "text": "This clip is synthesized from a style-first voice design prompt.",
                "instruct": "Clear premium-product presenter with calm authority.",
                "preset": "voice_generator",
                "language": "en",
                "quality": "high",
                "tokens": 180,
                "max_tokens": 320,
            },
        ),
        ShowcaseTrack(
            track_id="05_soundeffect",
            variant="soundeffect",
            title="SoundEffect Atmosphere",
            model="OpenMOSS-Team/MOSS-SoundEffect",
            generate_kwargs={
                "text": "Thunder and rain over a city street at night.",
                "ambient_sound": "Thunder and rain over a city street.",
                "sound_event": "storm",
                "quality": "high",
                "preset": "soundeffect",
                "tokens": 140,
                "max_tokens": 260,
            },
        ),
        ShowcaseTrack(
            track_id="06_realtime",
            variant="realtime",
            title="Realtime Delta Synthesis",
            model="OpenMOSS-Team/MOSS-TTS-Realtime",
            generate_kwargs={
                "text": (
                    "Realtime synthesis combines incremental text ingestion with "
                    "stream-capable audio decoding for interactive agents."
                ),
                "preset": "realtime",
                "chunk_frames": 40,
                "overlap_frames": 4,
                "decode_chunk_duration": 0.32,
                "repetition_window": 50,
                "max_tokens": 1200,
                "stream": False,
            },
        ),
    ]


def _merge_results(results: list[Any]) -> tuple[mx.array, int, int, float]:
    if not results:
        raise RuntimeError("Model returned zero generation results")

    sample_rate = int(results[0].sample_rate)
    token_count = sum(int(getattr(item, "token_count", 0)) for item in results)
    processing_time_seconds = sum(
        float(getattr(item, "processing_time_seconds", 0.0)) for item in results
    )

    audio_segments = [
        result.audio for result in results if int(result.audio.shape[0]) > 0
    ]
    if not audio_segments:
        raise RuntimeError("All generated segments were empty")

    merged = (
        audio_segments[0]
        if len(audio_segments) == 1
        else mx.concatenate(audio_segments, axis=0)
    )
    return merged, sample_rate, token_count, processing_time_seconds


def _write_markdown_manifest(manifest: dict[str, Any], path: Path) -> None:
    lines = [
        "# MOSS-TTS Showcase Album Manifest",
        "",
        f"- Requested variants: `{', '.join(manifest['requested_variants'])}`",
        f"- Tracks requested: `{manifest['summary']['requested_tracks']}`",
        f"- Tracks succeeded: `{manifest['summary']['succeeded_tracks']}`",
        "",
        "## Track Results",
        "",
        "| Track | Variant | Model | Status | Duration (s) | Output |",
        "|---|---|---|---|---:|---|",
    ]

    for track in manifest["tracks"]:
        if track["status"] == "ok":
            lines.append(
                "| "
                f"{track['track_id']} | {track['variant']} | `{track['model']}` | ok | "
                f"{track['duration_seconds']:.2f} | `{track['output_file']}` |"
            )
        else:
            lines.append(
                "| "
                f"{track['track_id']} | {track['variant']} | `{track['model']}` | error | "
                "0.00 | - |"
            )

    if manifest.get("errors"):
        lines.extend(["", "## Errors", ""])
        for error in manifest["errors"]:
            lines.append(
                f"- `{error['track_id']}` ({error['variant']}): {error['error']}"
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_audio = Path(args.reference_audio)
    if not reference_audio.exists():
        raise FileNotFoundError(f"Reference audio not found: {reference_audio}")

    requested_variants = _parse_variants(args.variants)
    suite = _build_showcase_suite(str(reference_audio))
    selected_tracks = [item for item in suite if item.variant in requested_variants]

    manifest: dict[str, Any] = {
        "requested_variants": requested_variants,
        "tracks": [],
        "errors": [],
        "summary": {
            "requested_tracks": len(selected_tracks),
            "succeeded_tracks": 0,
        },
    }

    for track in selected_tracks:
        track_output_dir = output_dir / track.variant
        track_output_dir.mkdir(parents=True, exist_ok=True)
        output_file = track_output_dir / f"{track.track_id}.wav"

        print(f"[{track.track_id}] Loading model: {track.model}")

        try:
            model = load_model(track.model, strict=args.strict)
            kwargs = dict(track.generate_kwargs)
            if args.max_tokens_override is not None:
                kwargs["max_tokens"] = int(args.max_tokens_override)

            results = list(model.generate(**kwargs))
            merged_audio, sample_rate, token_count, processing_time = _merge_results(
                results
            )

            audio_write(
                str(output_file),
                np.array(merged_audio),
                sample_rate,
                format="wav",
            )

            duration_seconds = float(merged_audio.shape[0]) / float(sample_rate)
            manifest["summary"]["succeeded_tracks"] += 1
            manifest["tracks"].append(
                {
                    "track_id": track.track_id,
                    "title": track.title,
                    "variant": track.variant,
                    "model": track.model,
                    "status": "ok",
                    "token_count": token_count,
                    "processing_time_seconds": processing_time,
                    "sample_rate": sample_rate,
                    "samples": int(merged_audio.shape[0]),
                    "duration_seconds": duration_seconds,
                    "output_file": str(output_file),
                    "generate_kwargs": kwargs,
                }
            )
            print(
                f"[{track.track_id}] wrote {output_file} "
                f"duration={duration_seconds:.2f}s"
            )

            del model
            mx.clear_cache()

        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            manifest["errors"].append(
                {
                    "track_id": track.track_id,
                    "variant": track.variant,
                    "error": error_text,
                }
            )
            manifest["tracks"].append(
                {
                    "track_id": track.track_id,
                    "title": track.title,
                    "variant": track.variant,
                    "model": track.model,
                    "status": "error",
                    "error": error_text,
                }
            )
            print(f"[{track.track_id}] ERROR: {error_text}")
            mx.clear_cache()
            if args.fail_fast:
                raise

    manifest_json_path = output_dir / f"{args.manifest_prefix}.json"
    manifest_md_path = output_dir / f"{args.manifest_prefix}.md"
    manifest_json_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_markdown_manifest(manifest, manifest_md_path)

    print(
        "Album complete: "
        f"{manifest['summary']['succeeded_tracks']}/"
        f"{manifest['summary']['requested_tracks']} tracks succeeded"
    )
    print(f"Manifest JSON: {manifest_json_path}")
    print(f"Manifest MD:   {manifest_md_path}")


if __name__ == "__main__":
    main()
