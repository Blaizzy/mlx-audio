"""Benchmark script for Voxtral Realtime.

Usage:
    # With a local converted model:
    python -m mlx_audio.stt.models.voxtral_realtime.benchmark --model /path/to/converted

    # Generate test audio with TTS (requires pocket-tts):
    python -m mlx_audio.stt.models.voxtral_realtime.benchmark --model /path/to/converted --generate-audio

    # With a specific audio file:
    python -m mlx_audio.stt.models.voxtral_realtime.benchmark --model /path/to/converted --audio test.wav
"""

import argparse
import subprocess
import tempfile
import time
from pathlib import Path

import mlx.core as mx


def generate_test_audio(text: str, output_path: str) -> bool:
    """Generate test audio using pocket-tts via uvx."""
    try:
        subprocess.run(
            ["uvx", "pocket-tts", "generate", "--text", text,
             "--output-path", output_path, "-q"],
            capture_output=True, check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("pocket-tts not available. Install with: uvx pocket-tts")
        return False


def benchmark(model_path: str, audio_path: str, n_runs: int = 3, max_tokens: int = 200):
    """Run benchmark: transcription quality + TTFT + throughput."""
    import mlx_audio.stt.utils as stt_utils

    print(f"Loading model from {model_path}...")
    model = stt_utils.load(model_path, strict=True)
    print("Model loaded.\n")

    # Warmup
    print("=== Warmup ===")
    result = model.generate(audio_path, max_tokens=50, verbose=True)
    print(f"Text: {result.text}\n")

    # Full transcription
    print("=== Full Transcription ===")
    result = model.generate(audio_path, max_tokens=max_tokens, verbose=True)
    print(f"Text: {result.text}\n")

    # TTFT measurements
    print(f"=== TTFT ({n_runs} runs) ===")
    ttfts = []
    for i in range(n_runs):
        t0 = time.time()
        result = model.generate(audio_path, max_tokens=1)
        ttft = time.time() - t0
        ttfts.append(ttft)
        print(f"  Run {i+1}: {ttft*1000:.0f}ms")

    avg_ttft = sum(ttfts) / len(ttfts)
    min_ttft = min(ttfts)
    print(f"  Average: {avg_ttft*1000:.0f}ms, Min: {min_ttft*1000:.0f}ms\n")

    # Streaming test
    print("=== Streaming Output ===")
    stream_gen = model.generate(audio_path, max_tokens=max_tokens, stream=True)
    t0 = time.time()
    full_text = ""
    for delta in stream_gen:
        full_text += delta
        print(delta, end="", flush=True)
    stream_time = time.time() - t0
    print(f"\n  ({stream_time:.1f}s total)\n")

    print("=== Summary ===")
    print(f"  TTFT (avg):    {avg_ttft*1000:.0f}ms")
    print(f"  TTFT (min):    {min_ttft*1000:.0f}ms")
    print(f"  Decode TPS:    {result.generation_tps:.1f}")
    print(f"  Transcription: {result.text[:80]}...")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Voxtral Realtime")
    parser.add_argument("--model", type=str, required=True, help="Path to converted model")
    parser.add_argument("--audio", type=str, help="Path to audio file")
    parser.add_argument("--generate-audio", action="store_true",
                        help="Generate test audio with pocket-tts")
    parser.add_argument("--text", type=str,
                        default="The quick brown fox jumps over the lazy dog. "
                                "Today is a beautiful day for testing speech recognition systems.",
                        help="Text for TTS generation")
    parser.add_argument("--runs", type=int, default=3, help="Number of TTFT measurement runs")
    parser.add_argument("--max-tokens", type=int, default=200, help="Max tokens to generate")
    args = parser.parse_args()

    audio_path = args.audio
    if audio_path is None:
        if args.generate_audio:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                audio_path = f.name
            print(f"Generating test audio: {args.text!r}")
            if not generate_test_audio(args.text, audio_path):
                return
            print(f"Saved to {audio_path}\n")
        else:
            parser.error("Provide --audio or --generate-audio")

    benchmark(args.model, audio_path, n_runs=args.runs, max_tokens=args.max_tokens)


if __name__ == "__main__":
    main()
