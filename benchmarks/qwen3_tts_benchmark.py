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
import os
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
    collect_audio: bool = False,
) -> TrialResult:
    """Run a single generation trial and collect metrics."""
    mx.clear_cache()
    gc.collect()

    chunk_metrics = []
    total_tokens = 0
    total_audio_samples = 0
    chunk_idx = 0
    audio_chunks = [] if collect_audio else None

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

        if collect_audio and result.audio is not None:
            audio_chunks.append(result.audio)

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

    trial = TrialResult(
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

    if collect_audio and audio_chunks:
        import numpy as np

        trial._audio = np.concatenate([np.array(c, copy=False) for c in audio_chunks])
        trial._sample_rate = sample_rate
    return trial


@dataclass
class BatchTrialResult:
    """Result from a single batch benchmark trial."""

    batch_size: int
    prompt_key: str
    prompt_texts: List[str]
    per_seq_ttfb_ms: List[float]  # TTFB per sequence in batch
    total_time_ms: float
    per_seq_tokens: List[int]
    per_seq_audio_samples: List[int]
    per_seq_audio_duration_s: List[float]
    peak_memory_gb: float = 0.0

    @property
    def total_tokens(self) -> int:
        return sum(self.per_seq_tokens)

    @property
    def total_audio_duration_s(self) -> float:
        return sum(self.per_seq_audio_duration_s)

    @property
    def tokens_per_second(self) -> float:
        return (
            self.total_tokens / (self.total_time_ms / 1000)
            if self.total_time_ms > 0
            else 0.0
        )

    @property
    def avg_ttfb_ms(self) -> float:
        return statistics.mean(self.per_seq_ttfb_ms) if self.per_seq_ttfb_ms else 0.0

    @property
    def throughput_ratio(self) -> float:
        """Total audio duration / wall time."""
        return (
            self.total_audio_duration_s / (self.total_time_ms / 1000)
            if self.total_time_ms > 0
            else 0.0
        )


def run_batch_trial(
    model,
    prompt_key: str,
    prompt_texts: List[str],
    voice: str = "Chelsie",
    streaming_interval: float = 2.0,
    max_tokens: int = 4096,
    temperature: float = 0.9,
    sample_rate: int = 24000,
    collect_audio: bool = False,
) -> BatchTrialResult:
    """Run a single batch generation trial and collect per-sequence metrics."""
    mx.clear_cache()
    gc.collect()

    batch_size = len(prompt_texts)
    per_seq_tokens = [0] * batch_size
    per_seq_audio_samples = [0] * batch_size
    per_seq_ttfb = [None] * batch_size
    per_seq_audio_chunks = [[] for _ in range(batch_size)] if collect_audio else None

    mx.reset_peak_memory()
    gen_start = time.perf_counter()

    voices = [voice] * batch_size

    for result in model.batch_generate(
        texts=prompt_texts,
        voices=voices,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
        streaming_interval=streaming_interval,
    ):
        now = time.perf_counter()
        seq_idx = result.sequence_idx

        if per_seq_ttfb[seq_idx] is None:
            per_seq_ttfb[seq_idx] = (now - gen_start) * 1000

        samples = result.audio.shape[0] if result.audio is not None else 0
        per_seq_tokens[seq_idx] += result.token_count
        per_seq_audio_samples[seq_idx] += samples

        if collect_audio and result.audio is not None:
            per_seq_audio_chunks[seq_idx].append(result.audio)

    gen_end = time.perf_counter()
    total_time_ms = (gen_end - gen_start) * 1000

    per_seq_audio_duration_s = [
        s / sample_rate if sample_rate > 0 else 0 for s in per_seq_audio_samples
    ]

    trial = BatchTrialResult(
        batch_size=batch_size,
        prompt_key=prompt_key,
        prompt_texts=prompt_texts,
        per_seq_ttfb_ms=[t or 0.0 for t in per_seq_ttfb],
        total_time_ms=total_time_ms,
        per_seq_tokens=per_seq_tokens,
        per_seq_audio_samples=per_seq_audio_samples,
        per_seq_audio_duration_s=per_seq_audio_duration_s,
        peak_memory_gb=mx.get_peak_memory() / 1e9,
    )

    if collect_audio:
        import numpy as np

        trial._per_seq_audio = []
        for chunks in per_seq_audio_chunks:
            if chunks:
                trial._per_seq_audio.append(
                    np.concatenate([np.array(c, copy=False) for c in chunks])
                )
            else:
                trial._per_seq_audio.append(np.array([], dtype=np.float32))
        trial._sample_rate = sample_rate
    return trial


def save_wav(path: str, audio, sample_rate: int) -> None:
    """Save a numpy audio array to a WAV file."""
    import soundfile as sf

    sf.write(path, audio, sample_rate)
    print(f"  Saved: {path}")


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
    parser.add_argument(
        "--batch-size",
        nargs="+",
        type=int,
        default=None,
        help="Batch sizes to benchmark (e.g., --batch-size 1 2 4). Runs batch_generate() comparison.",
    )
    parser.add_argument(
        "--save-audio",
        type=str,
        default=None,
        help="Directory to save audio from the last trial of each benchmark (e.g., --save-audio ./audio_out)",
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
            is_last = trial_idx == args.num_trials - 1
            print(f"  Trial {trial_idx + 1}/{args.num_trials}...", end="", flush=True)
            result = run_trial(
                model=model,
                prompt_key=prompt_key,
                prompt_text=prompt_text,
                voice=voice,
                streaming_interval=args.streaming_interval,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                collect_audio=bool(args.save_audio) and is_last,
            )
            trials.append(result)
            print(
                f" TTFB={result.ttfb_ms:.0f}ms, Total={result.total_time_ms:.0f}ms, "
                f"RTF={result.real_time_factor:.2f}x, TPS={result.tokens_per_second:.1f}"
            )

            if args.verbose:
                print_trial_detail(result)

        # Save audio from last trial
        if args.save_audio and hasattr(trials[-1], "_audio"):
            os.makedirs(args.save_audio, exist_ok=True)
            path = os.path.join(args.save_audio, f"sequential_{prompt_key}.wav")
            save_wav(path, trials[-1]._audio, trials[-1]._sample_rate)

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

    # Batch benchmarking
    if args.batch_size:
        print(f"\n\n{'='*70}")
        print(f"  Batch Generation Benchmark")
        print(f"{'='*70}")

        # Use the first prompt for batch benchmarking
        prompt_key = list(prompt_map.keys())[0]
        prompt_text = prompt_map[prompt_key]

        batch_results = []

        for bs in args.batch_size:
            print(f"\n  Batch size: {bs}")
            texts = [prompt_text] * bs

            trials = []
            for trial_idx in range(args.num_trials):
                is_last = trial_idx == args.num_trials - 1
                print(
                    f"    Trial {trial_idx + 1}/{args.num_trials}...",
                    end="",
                    flush=True,
                )
                result = run_batch_trial(
                    model=model,
                    prompt_key=prompt_key,
                    prompt_texts=texts,
                    voice=voice,
                    streaming_interval=args.streaming_interval,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    collect_audio=bool(args.save_audio) and is_last,
                )
                trials.append(result)
                print(
                    f" Total={result.total_time_ms:.0f}ms, "
                    f"TPS={result.tokens_per_second:.1f}, "
                    f"AvgTTFB={result.avg_ttfb_ms:.0f}ms, "
                    f"Throughput={result.throughput_ratio:.2f}x, "
                    f"Mem={result.peak_memory_gb:.2f}GB"
                )

            # Save audio from last trial
            if args.save_audio and hasattr(trials[-1], "_per_seq_audio"):
                os.makedirs(args.save_audio, exist_ok=True)
                for seq_idx, audio in enumerate(trials[-1]._per_seq_audio):
                    if len(audio) > 0:
                        path = os.path.join(
                            args.save_audio,
                            f"batch{bs}_{prompt_key}_seq{seq_idx}.wav",
                        )
                        save_wav(path, audio, trials[-1]._sample_rate)

            batch_results.append((bs, trials))

        # Comparison table across batch sizes
        if len(batch_results) > 1:
            print(f"\n{'='*70}")
            print(f"  Batch Size Comparison (prompt: '{prompt_key}')")
            print(f"{'='*70}")
            print(
                f"  {'Batch':>5} | {'TotalTime(ms)':>13} | {'TPS':>8} | "
                f"{'AvgTTFB(ms)':>11} | {'Throughput':>10} | {'Mem(GB)':>8}"
            )
            print(f"  {'─'*5} | {'─'*13} | {'─'*8} | {'─'*11} | {'─'*10} | {'─'*8}")
            for bs, trials in batch_results:
                avg_time = statistics.mean([t.total_time_ms for t in trials])
                avg_tps = statistics.mean([t.tokens_per_second for t in trials])
                avg_ttfb = statistics.mean([t.avg_ttfb_ms for t in trials])
                avg_throughput = statistics.mean([t.throughput_ratio for t in trials])
                peak_mem = max(t.peak_memory_gb for t in trials)
                print(
                    f"  {bs:>5} | {avg_time:>11.1f}ms | {avg_tps:>8.1f} | "
                    f"{avg_ttfb:>9.1f}ms | {avg_throughput:>9.2f}x | {peak_mem:>7.2f}"
                )
            print(f"{'='*70}")


if __name__ == "__main__":
    main()
