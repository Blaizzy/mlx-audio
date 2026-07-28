from __future__ import annotations

import argparse
import json
from typing import Optional

from .runner import SUPPORTED_METRICS, run_seed_tts_eval
from .seed_tts import DEFAULT_DATASET_REPO, DEFAULT_REVISION


def parse_args(argv: Optional[list[str]] = None):
    parser = argparse.ArgumentParser(
        description="Evaluate an STT model on a dataset with chosen metrics."
    )
    parser.add_argument(
        "--model",
        default="mlx-community/whisper-large-v3-turbo",
        help="STT model path or Hugging Face repo.",
    )
    parser.add_argument(
        "--dataset-repo",
        default=DEFAULT_DATASET_REPO,
        help="Hugging Face dataset repo containing Seed-TTS eval audio.",
    )
    parser.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
        help="Dataset revision to stream.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for predictions, summaries, and cached audio.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N English target wavs.",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Optional model language hint. If omitted, no language kwarg is passed.",
    )
    parser.add_argument(
        "--gen-kwargs",
        type=json.loads,
        default=None,
        help="Additional model.generate kwargs as JSON, e.g. '{\"temperature\": 0}'.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse utterances already present in predictions.jsonl.",
    )
    parser.add_argument(
        "--allow-missing-audio",
        action="store_true",
        help="Skip streamed audio rows that do not have matching metadata.",
    )
    parser.add_argument(
        "--audio-cache-dir",
        default=None,
        help="Optional directory for streamed wav cache. Defaults to output-dir/audio-cache.",
    )
    parser.add_argument(
        "--dataset-batch-size",
        type=int,
        default=8,
        help=("Number of streamed dataset samples to materialize before processing. "),
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["wer"],
        choices=SUPPORTED_METRICS,
        help="Metrics to compute. Currently supported: wer.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Pass verbose=True when supported."
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None):
    args = parse_args(argv)
    summary = run_seed_tts_eval(
        model=args.model,
        output_dir=args.output_dir,
        dataset_repo=args.dataset_repo,
        revision=args.revision,
        limit=args.limit,
        language=args.language,
        gen_kwargs=args.gen_kwargs,
        skip_existing=args.skip_existing,
        fail_on_missing_audio=not args.allow_missing_audio,
        audio_cache_dir=args.audio_cache_dir,
        dataset_batch_size=args.dataset_batch_size,
        metrics=args.metrics,
        verbose=args.verbose,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
