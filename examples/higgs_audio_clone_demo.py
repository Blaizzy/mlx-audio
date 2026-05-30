#!/usr/bin/env python3
"""Higgs Audio v2 voice cloning demo.

Uses the Higgs-specific HiggsAudioServer API for full parameter surface.
For a drop-in example against the standard mlx_audio.tts.generate CLI,
see docs/models/tts/higgs_audio.md.

Quick start with the bundled `en_woman` sample voice:
    python examples/higgs_audio_clone_demo.py \\
        --text "Text to synthesize in the cloned voice."

Supply your own reference:
    python examples/higgs_audio_clone_demo.py \\
        --ref_audio reference.wav \\
        --ref_text "Reference transcript text." \\
        --text "Text to synthesize in the cloned voice."

Reference audio is encoded through the in-tree HiggsAudioTokenizer and
stitched into the assistant turn of a ChatML prompt. ref_text is the
transcript of the reference clip — required for stable alignment
between the cloned voice and the target text.

Best results come from 5-15 seconds of clean reference speech.
Three sample voices live in examples/voice_prompts/ (en_woman,
en_man, en_man_deep) for drop-in use.
"""

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_audio.audio_io import write as audio_write
from mlx_audio.tts.models.higgs_audio import HiggsAudioServer


def _quantize_predicate(name: str, module: nn.Module) -> bool:
    """Keep audio head + audio codebook embeddings at bf16 — they're most
    sensitive to quantization noise. Everything else (Llama backbone + text
    head) gets compressed."""
    if not isinstance(module, (nn.Linear, nn.Embedding)):
        return False
    protected = ("audio_codebook_embeddings", "audio_decoder_proj.audio_lm_head")
    return not any(b in name for b in protected)


def _default_voice_prompt() -> tuple[str, str]:
    """Return the bundled `en_woman` voice prompt (wav path + transcript)."""
    here = Path(__file__).resolve().parent / "voice_prompts"
    return str(here / "en_woman.wav"), (here / "en_woman.txt").read_text().strip()


def main() -> int:
    p = argparse.ArgumentParser(description="Higgs Audio v2 voice cloning demo")
    default_ref_audio, default_ref_text = _default_voice_prompt()
    p.add_argument(
        "--ref_audio",
        default=default_ref_audio,
        help="Reference audio WAV (defaults to bundled en_woman sample)",
    )
    p.add_argument(
        "--ref_text", default=default_ref_text, help="Transcript of the reference audio"
    )
    p.add_argument("--text", required=True, help="Target text to synthesize")
    p.add_argument("--output", default="higgs_clone_output.wav", help="Output WAV path")
    p.add_argument(
        "--model",
        default="mlx-community/higgs-audio-v2-3B-mlx-bf16",
        help="Higgs Audio v2 MLX model repo or path",
    )
    p.add_argument(
        "--codec",
        default="mlx-community/higgs-audio-v2-tokenizer",
        help="Higgs Audio v2 tokenizer repo or path",
    )
    p.add_argument(
        "--quantize_bits",
        type=int,
        default=None,
        choices=[4, 6, 8],
        help="Optionally quantize the loaded model in-place (4/6/8-bit)",
    )
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_new_frames", type=int, default=1200)
    p.add_argument("--ras_win_len", type=int, default=7)
    p.add_argument("--ras_max_repeat", type=int, default=2)
    p.add_argument(
        "--fade_in_ms",
        type=float,
        default=30.0,
        help="Leading fade (ms) — 30ms suppresses the first-frame transient cleanly",
    )
    p.add_argument("--fade_out_ms", type=float, default=15.0)
    args = p.parse_args()

    print(f"[load] HiggsAudioServer from {args.model}")
    t0 = time.monotonic()
    server = HiggsAudioServer.from_pretrained(
        model_path=args.model,
        codec_path=args.codec,
    )
    if args.quantize_bits is not None:
        print(
            f"[quantize] group_size=64 bits={args.quantize_bits} (audio head protected)"
        )
        nn.quantize(
            server.model,
            group_size=64,
            bits=args.quantize_bits,
            class_predicate=_quantize_predicate,
        )
        mx.eval(server.model.parameters())
    print(f"  loaded in {time.monotonic() - t0:.2f}s")

    print(f"[generate] target: {args.text!r}")
    t_gen = time.monotonic()
    result = server.generate(
        target_text=args.text,
        reference_audio_path=args.ref_audio,
        reference_text=args.ref_text,
        max_new_frames=args.max_new_frames,
        temperature=args.temperature,
        top_p=args.top_p,
        ras_win_len=args.ras_win_len,
        ras_max_repeat=args.ras_max_repeat,
        fade_in_ms=args.fade_in_ms,
        fade_out_ms=args.fade_out_ms,
    )
    wall = time.monotonic() - t_gen
    audio_sec = len(result.pcm) / result.sampling_rate
    rtf = wall / audio_sec if audio_sec > 0 else float("inf")

    audio_write(args.output, result.pcm, result.sampling_rate)
    print(
        f"[done] {audio_sec:.2f}s audio in {wall:.2f}s wall "
        f"(RTF {rtf:.2f}×, {result.num_frames_raw} frames, "
        f"stop={result.stop_reason}) → {args.output}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
