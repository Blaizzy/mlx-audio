#!/usr/bin/env python3
"""
Benchmark for Qwen3-TTS: measures TTFB, inter-chunk latency, and throughput.

Usage:
    python benchmarks/qwen3_tts_benchmark.py
    python benchmarks/qwen3_tts_benchmark.py --model mlx-community/Qwen3-TTS-0.6B-bf16
    python benchmarks/qwen3_tts_benchmark.py --num-trials 3 --streaming-interval 1.0
    python benchmarks/qwen3_tts_benchmark.py --prompts short medium long
"""

import argparse
import gc
import statistics
import time
from dataclasses import dataclass, field
from typing import List, Optional

import mlx.core as mx

PROMPTS = {
    "short": "Hello, how are you today?",
    "medium": (
        "The quick brown fox jumps over the lazy dog. "
        "This is a test of the text-to-speech system."
    ),
    "long": (
        "Artificial intelligence has transformed the way we interact with technology. "
        "From voice assistants to autonomous vehicles, machine learning models are "
        "becoming increasingly sophisticated. Text-to-speech synthesis, in particular, "
        "has seen remarkable improvements in naturalness and expressiveness, enabling "
        "more human-like interactions between people and machines."
    ),
}


@dataclass
class ChunkMetrics:
    """Metrics for a single audio chunk."""

    chunk_idx: int
    token_count: int
    audio_samples: int
    latency_ms: float  # Time since previous chunk (or start for first)
    cumulative_ms: float  # Time since generation start


@dataclass
class TrialResult:
    """Result from a single benchmark trial."""

    prompt_key: str
    prompt_text: str
    ttfb_ms: float  # Time to first audio byte
    total_time_ms: float
    total_tokens: int
    total_audio_samples: int
    audio_duration_s: float
    real_time_factor: float  # audio_duration / generation_time
    chunk_metrics: List[ChunkMetrics] = field(default_factory=list)
    peak_memory_gb: float = 0.0

    @property
    def inter_chunk_latencies_ms(self) -> List[float]:
        """Latencies between chunks (excluding TTFB)."""
        return [c.latency_ms for c in self.chunk_metrics[1:]]

    @property
    def avg_inter_chunk_ms(self) -> float:
        lats = self.inter_chunk_latencies_ms
        return statistics.mean(lats) if lats else 0.0

    @property
    def p50_inter_chunk_ms(self) -> float:
        lats = self.inter_chunk_latencies_ms
        return statistics.median(lats) if lats else 0.0

    @property
    def p95_inter_chunk_ms(self) -> float:
        lats = self.inter_chunk_latencies_ms
        if len(lats) < 2:
            return lats[0] if lats else 0.0
        sorted_lats = sorted(lats)
        idx = int(len(sorted_lats) * 0.95)
        return sorted_lats[min(idx, len(sorted_lats) - 1)]

    @property
    def tokens_per_second(self) -> float:
        return (
            self.total_tokens / (self.total_time_ms / 1000)
            if self.total_time_ms > 0
            else 0.0
        )


@dataclass
class BenchmarkSummary:
    """Aggregated results across trials."""

    prompt_key: str
    num_trials: int
    ttfb_avg_ms: float
    ttfb_min_ms: float
    ttfb_max_ms: float
    ttfb_std_ms: float
    inter_chunk_avg_ms: float
    inter_chunk_p50_ms: float
    inter_chunk_p95_ms: float
    total_time_avg_ms: float
    tokens_per_sec_avg: float
    rtf_avg: float  # Real-time factor
    peak_memory_gb: float


def run_trial(
    model,
    prompt_key: str,
    prompt_text: str,
    voice: str = "Chelsie",
    streaming_interval: float = 2.0,
    max_tokens: int = 4096,
    temperature: float = 0.9,
    sample_rate: int = 24000,
) -> TrialResult:
    """Run a single generation trial and collect metrics."""
    mx.clear_cache()
    gc.collect()

    chunk_metrics = []
    total_tokens = 0
    total_audio_samples = 0
    chunk_idx = 0

    mx.reset_peak_memory()
    gen_start = time.perf_counter()
    last_chunk_time = gen_start
    ttfb = None

    for result in model.generate(
        text=prompt_text,
        voice=voice,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
        streaming_interval=streaming_interval,
    ):
        now = time.perf_counter()

        if ttfb is None:
            ttfb = (now - gen_start) * 1000

        latency = (now - last_chunk_time) * 1000
        cumulative = (now - gen_start) * 1000

        samples = result.audio.shape[0] if result.audio is not None else 0
        tokens = result.token_count

        chunk_metrics.append(
            ChunkMetrics(
                chunk_idx=chunk_idx,
                token_count=tokens,
                audio_samples=samples,
                latency_ms=latency,
                cumulative_ms=cumulative,
            )
        )

        total_tokens += tokens
        total_audio_samples += samples
        chunk_idx += 1
        last_chunk_time = now

    gen_end = time.perf_counter()
    total_time_ms = (gen_end - gen_start) * 1000
    audio_duration_s = total_audio_samples / sample_rate if sample_rate > 0 else 0
    rtf = audio_duration_s / (total_time_ms / 1000) if total_time_ms > 0 else 0

    return TrialResult(
        prompt_key=prompt_key,
        prompt_text=prompt_text,
        ttfb_ms=ttfb or 0.0,
        total_time_ms=total_time_ms,
        total_tokens=total_tokens,
        total_audio_samples=total_audio_samples,
        audio_duration_s=audio_duration_s,
        real_time_factor=rtf,
        chunk_metrics=chunk_metrics,
        peak_memory_gb=mx.get_peak_memory() / 1e9,
    )


def summarize_trials(prompt_key: str, trials: List[TrialResult]) -> BenchmarkSummary:
    """Aggregate metrics across trials."""
    ttfbs = [t.ttfb_ms for t in trials]
    total_times = [t.total_time_ms for t in trials]
    tps_vals = [t.tokens_per_second for t in trials]
    rtfs = [t.real_time_factor for t in trials]
    peak_mems = [t.peak_memory_gb for t in trials]

    # Collect all inter-chunk latencies across trials
    all_inter_chunks = []
    for t in trials:
        all_inter_chunks.extend(t.inter_chunk_latencies_ms)

    return BenchmarkSummary(
        prompt_key=prompt_key,
        num_trials=len(trials),
        ttfb_avg_ms=statistics.mean(ttfbs),
        ttfb_min_ms=min(ttfbs),
        ttfb_max_ms=max(ttfbs),
        ttfb_std_ms=statistics.stdev(ttfbs) if len(ttfbs) > 1 else 0.0,
        inter_chunk_avg_ms=(
            statistics.mean(all_inter_chunks) if all_inter_chunks else 0.0
        ),
        inter_chunk_p50_ms=(
            statistics.median(all_inter_chunks) if all_inter_chunks else 0.0
        ),
        inter_chunk_p95_ms=(
            sorted(all_inter_chunks)[int(len(all_inter_chunks) * 0.95)]
            if len(all_inter_chunks) >= 2
            else (all_inter_chunks[0] if all_inter_chunks else 0.0)
        ),
        total_time_avg_ms=statistics.mean(total_times),
        tokens_per_sec_avg=statistics.mean(tps_vals),
        rtf_avg=statistics.mean(rtfs),
        peak_memory_gb=max(peak_mems),
    )


def print_trial_detail(trial: TrialResult) -> None:
    """Print detailed chunk-level metrics for a trial."""
    print(
        f'\n  Prompt: "{trial.prompt_text[:60]}..."'
        if len(trial.prompt_text) > 60
        else f'\n  Prompt: "{trial.prompt_text}"'
    )
    print(
        f"  TTFB: {trial.ttfb_ms:.1f}ms | Total: {trial.total_time_ms:.1f}ms | "
        f"Tokens: {trial.total_tokens} | Audio: {trial.audio_duration_s:.2f}s | "
        f"RTF: {trial.real_time_factor:.2f}x | TPS: {trial.tokens_per_second:.1f} | "
        f"Peak Mem: {trial.peak_memory_gb:.2f}GB"
    )

    if trial.chunk_metrics:
        print(
            f"  {'Chunk':>5} | {'Tokens':>6} | {'Samples':>8} | {'Latency':>10} | {'Cumulative':>10}"
        )
        print(f"  {'─'*5} | {'─'*6} | {'─'*8} | {'─'*10} | {'─'*10}")
        for cm in trial.chunk_metrics:
            print(
                f"  {cm.chunk_idx:>5} | {cm.token_count:>6} | {cm.audio_samples:>8} | "
                f"{cm.latency_ms:>8.1f}ms | {cm.cumulative_ms:>8.1f}ms"
            )


def print_summary(summary: BenchmarkSummary) -> None:
    """Print aggregated benchmark summary."""
    print(f"\n{'='*70}")
    print(f"  Summary: '{summary.prompt_key}' ({summary.num_trials} trials)")
    print(f"{'='*70}")
    print(
        f"  TTFB        avg={summary.ttfb_avg_ms:.1f}ms  min={summary.ttfb_min_ms:.1f}ms  "
        f"max={summary.ttfb_max_ms:.1f}ms  std={summary.ttfb_std_ms:.1f}ms"
    )
    print(
        f"  Inter-chunk avg={summary.inter_chunk_avg_ms:.1f}ms  "
        f"p50={summary.inter_chunk_p50_ms:.1f}ms  p95={summary.inter_chunk_p95_ms:.1f}ms"
    )
    print(f"  Total time  avg={summary.total_time_avg_ms:.1f}ms")
    print(f"  Throughput   {summary.tokens_per_sec_avg:.1f} tokens/sec")
    print(f"  RTF          {summary.rtf_avg:.2f}x realtime")
    print(f"  Peak memory  {summary.peak_memory_gb:.2f}GB")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-TTS Benchmark: TTFB & Inter-Chunk Latency"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-6bit",
        help="Model path or HuggingFace repo ID",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default=None,
        help="Speaker voice name (auto-detected from model if not specified)",
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=["short", "medium", "long"],
        choices=list(PROMPTS.keys()),
        help="Which prompts to benchmark",
    )
    parser.add_argument(
        "--custom-prompt",
        type=str,
        default=None,
        help="Custom prompt text (overrides --prompts)",
    )
    parser.add_argument(
        "--num-trials",
        "-n",
        type=int,
        default=3,
        help="Number of trials per prompt",
    )
    parser.add_argument(
        "--streaming-interval",
        type=float,
        default=0.32,
        help="Streaming chunk interval in seconds (0.32s = 4 tokens at 12.5Hz)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum generation tokens",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print per-chunk detail for each trial",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        default=True,
        help="Run a warmup generation before benchmarking (default: True)",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_false",
        dest="warmup",
        help="Skip warmup generation",
    )

    args = parser.parse_args()

    # Load model
    print(f"Loading model: {args.model}")
    from mlx_audio.tts.utils import load

    model = load(args.model)

    # Auto-detect voice if not specified
    voice = args.voice
    if voice is None:
        speakers = getattr(model, "supported_speakers", None)
        if speakers:
            voice = speakers[0]
            print(f"Auto-selected voice: '{voice}' (available: {speakers})")
        else:
            voice = "Chelsie"  # fallback for base models
    print(f"Model loaded successfully.\n")

    # Determine prompts
    if args.custom_prompt:
        prompt_map = {"custom": args.custom_prompt}
    else:
        prompt_map = {k: PROMPTS[k] for k in args.prompts}

    # Warmup
    if args.warmup:
        print("Running warmup generation...")
        warmup_text = "Hello world."
        for _ in model.generate(
            text=warmup_text,
            voice=voice,
            temperature=args.temperature,
            max_tokens=128,
            stream=True,
            streaming_interval=args.streaming_interval,
        ):
            pass
        mx.clear_cache()
        gc.collect()
        print("Warmup complete.\n")

    # Run benchmarks
    all_summaries = []

    for prompt_key, prompt_text in prompt_map.items():
        print(f"\n{'─'*70}")
        print(
            f"Benchmarking: '{prompt_key}' ({len(prompt_text)} chars, {args.num_trials} trials)"
        )
        print(f"{'─'*70}")

        trials = []
        for trial_idx in range(args.num_trials):
            print(f"  Trial {trial_idx + 1}/{args.num_trials}...", end="", flush=True)
            result = run_trial(
                model=model,
                prompt_key=prompt_key,
                prompt_text=prompt_text,
                voice=voice,
                streaming_interval=args.streaming_interval,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            trials.append(result)
            print(
                f" TTFB={result.ttfb_ms:.0f}ms, Total={result.total_time_ms:.0f}ms, "
                f"RTF={result.real_time_factor:.2f}x, TPS={result.tokens_per_second:.1f}"
            )

            if args.verbose:
                print_trial_detail(result)

        summary = summarize_trials(prompt_key, trials)
        all_summaries.append(summary)
        print_summary(summary)

    # Final comparison table
    if len(all_summaries) > 1:
        print(f"\n\n{'='*70}")
        print(f"  Comparison Across Prompts")
        print(f"{'='*70}")
        print(
            f"  {'Prompt':<10} | {'TTFB(ms)':>10} | {'InterChunk':>10} | {'TPS':>8} | {'RTF':>6} | {'Mem(GB)':>8}"
        )
        print(f"  {'─'*10} | {'─'*10} | {'─'*10} | {'─'*8} | {'─'*6} | {'─'*8}")
        for s in all_summaries:
            print(
                f"  {s.prompt_key:<10} | {s.ttfb_avg_ms:>8.1f}ms | {s.inter_chunk_avg_ms:>8.1f}ms | "
                f"{s.tokens_per_sec_avg:>8.1f} | {s.rtf_avg:>5.2f}x | {s.peak_memory_gb:>7.2f}"
            )
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
