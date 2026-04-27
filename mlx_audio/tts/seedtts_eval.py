from __future__ import annotations

import argparse
import inspect
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import mlx.core as mx
import numpy as np

from mlx_audio.audio_io import write as audio_write
from mlx_audio.tts import load as load_tts


@dataclass(frozen=True)
class SeedTTSEvalItem:
    utt_id: str
    prompt_text: str
    prompt_audio_rel: str
    synthesis_text: str
    ground_truth_audio_rel: Optional[str]
    manifest_line: str

    @property
    def output_name(self) -> str:
        return self.utt_id if self.utt_id.endswith(".wav") else f"{self.utt_id}.wav"


def _parse_manifest_line(
    line: str, *, manifest_path: Path, line_number: int
) -> SeedTTSEvalItem:
    parts = [part.strip() for part in line.split("|")]
    if len(parts) not in (4, 5):
        raise ValueError(
            f"Invalid Seed-TTS manifest line at {manifest_path}:{line_number}; "
            f"expected 4 or 5 pipe-delimited fields, got {len(parts)}"
        )

    utt_id, prompt_text, prompt_audio_rel, synthesis_text = parts[:4]
    ground_truth_audio_rel = parts[4] if len(parts) == 5 else None

    if not utt_id:
        raise ValueError(f"Missing utt_id at {manifest_path}:{line_number}")
    if not prompt_audio_rel:
        raise ValueError(f"Missing prompt audio path at {manifest_path}:{line_number}")
    if not synthesis_text:
        raise ValueError(f"Missing synthesis text at {manifest_path}:{line_number}")

    return SeedTTSEvalItem(
        utt_id=utt_id,
        prompt_text=prompt_text,
        prompt_audio_rel=prompt_audio_rel,
        synthesis_text=synthesis_text,
        ground_truth_audio_rel=ground_truth_audio_rel,
        manifest_line=line,
    )


def load_manifest(manifest_path: Path) -> List[SeedTTSEvalItem]:
    try:
        raw_text = manifest_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise SystemExit(f"Manifest file not found: {manifest_path}") from exc

    items: List[SeedTTSEvalItem] = []
    for line_number, raw_line in enumerate(raw_text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        items.append(
            _parse_manifest_line(
                line, manifest_path=manifest_path, line_number=line_number
            )
        )

    if not items:
        raise SystemExit(f"Manifest is empty after parsing: {manifest_path}")
    return items


def resolve_manifest_path(testset_dir: Path, lang: str, meta_file: str) -> Path:
    candidate = Path(meta_file).expanduser()
    if candidate.is_absolute():
        manifest_path = candidate
    else:
        manifest_path = testset_dir / lang / candidate

    if not manifest_path.exists():
        raise SystemExit(f"Manifest file not found: {manifest_path}")
    return manifest_path


def slice_items(
    items: Sequence[SeedTTSEvalItem],
    *,
    start: int,
    end: Optional[int],
    limit: Optional[int],
) -> List[SeedTTSEvalItem]:
    if start < 0:
        raise SystemExit("--start must be >= 0")
    if end is not None and end < start:
        raise SystemExit("--end must be >= --start")
    if limit is not None and limit <= 0:
        raise SystemExit("--limit must be > 0")

    selected = list(items[start:end])
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise SystemExit("No manifest rows selected. Adjust --start/--end/--limit.")
    return selected


def _filter_generate_kwargs(
    func: Any, kwargs: Dict[str, Any]
) -> Tuple[Dict[str, Any], List[str]]:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return kwargs, []

    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return kwargs, []

    accepted = set(signature.parameters.keys())
    filtered = {key: value for key, value in kwargs.items() if key in accepted}
    ignored = sorted(key for key in kwargs if key not in accepted)
    return filtered, ignored


def _join_audio_segments(segments: Iterable[Any]) -> np.ndarray:
    arrays = [np.asarray(segment, dtype=np.float32).reshape(-1) for segment in segments]
    if not arrays:
        raise RuntimeError("Model.generate() produced no audio segments")
    if len(arrays) == 1:
        return arrays[0]
    return np.concatenate(arrays, axis=0)


def _resolve_relative_audio_path(base_dir: Path, value: str) -> str:
    audio_path = Path(value).expanduser()
    if not audio_path.is_absolute():
        audio_path = base_dir / audio_path
    return str(audio_path)


def _format_seconds(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_manifest(path: Path, items: Sequence[SeedTTSEvalItem]) -> None:
    _ensure_parent_dir(path)
    content = "\n".join(item.manifest_line for item in items)
    if content:
        content += "\n"
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    _ensure_parent_dir(path)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Seed-TTS-Eval wavs from a manifest using an MLX-Audio TTS model."
    )
    parser.add_argument(
        "--model", required=True, help="Local model path or Hugging Face repo id"
    )
    parser.add_argument(
        "--testset-dir",
        required=True,
        help="Seed-TTS testset root directory containing zh/ and en/",
    )
    parser.add_argument(
        "--lang", required=True, choices=["zh", "en"], help="Dataset split"
    )
    parser.add_argument(
        "--meta-file",
        default="meta.lst",
        help="Manifest filename under <testset>/<lang>/, or an absolute manifest path",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where <utt_id>.wav files and filtered manifests will be written",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Zero-based start index over parsed manifest rows (default: 0)",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Zero-based exclusive end index over parsed manifest rows",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of rows to process after start/end slicing",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate wavs even if output_dir/<utt_id>.wav already exists",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep going after per-row failures and emit a failures jsonl log",
    )
    parser.add_argument(
        "--disable-ref-audio",
        action="store_true",
        help="Do not pass prompt-wavs reference audio into model.generate()",
    )
    parser.add_argument(
        "--lang-code",
        default=None,
        help="Override the lang_code passed to model.generate() (default: same as --lang)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1200,
        help="max_tokens passed into model.generate() (default: 1200)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.7,
        help="temperature passed into model.generate() (default: 1.7)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.8,
        help="top_p passed into model.generate() (default: 0.8)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=25,
        help="top_k passed into model.generate() (default: 25)",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="repetition_penalty passed into model.generate() (default: 1.0)",
    )
    parser.add_argument(
        "--text-temperature",
        type=float,
        default=1.5,
        help="text_temperature passed into model.generate() (default: 1.5)",
    )
    parser.add_argument(
        "--text-top-p",
        type=float,
        default=1.0,
        help="text_top_p passed into model.generate() (default: 1.0)",
    )
    parser.add_argument(
        "--text-top-k",
        type=int,
        default=50,
        help="text_top_k passed into model.generate() (default: 50)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    testset_dir = Path(args.testset_dir).expanduser()
    if not testset_dir.exists():
        raise SystemExit(f"Testset directory not found: {testset_dir}")

    manifest_path = resolve_manifest_path(testset_dir, args.lang, args.meta_file)
    manifest_dir = manifest_path.parent
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    items = load_manifest(manifest_path)
    selected_items = slice_items(
        items, start=args.start, end=args.end, limit=args.limit
    )

    requested_manifest_path = output_dir / f"{manifest_path.stem}.requested.lst"
    ready_manifest_path = output_dir / f"{manifest_path.stem}.ready.lst"
    failures_log_path = output_dir / f"{manifest_path.stem}.failures.jsonl"
    summary_path = output_dir / f"{manifest_path.stem}.summary.json"

    _write_manifest(requested_manifest_path, selected_items)

    print(f"[seedtts-eval] loading model: {args.model}")
    model = load_tts(args.model)
    model_generate = getattr(model, "generate", None)
    if model_generate is None:
        raise SystemExit("Loaded model does not implement generate(...)")

    use_ref_audio = not bool(args.disable_ref_audio)
    lang_code = args.lang_code or args.lang
    success_items: List[SeedTTSEvalItem] = []
    failed_items: List[Tuple[SeedTTSEvalItem, str]] = []
    ignored_generate_kwargs: Optional[List[str]] = None
    script_start_time = time.time()

    print(
        f"[seedtts-eval] manifest={manifest_path} rows={len(items)} selected={len(selected_items)} "
        f"output_dir={output_dir}"
    )

    for index, item in enumerate(selected_items, start=1):
        output_path = output_dir / item.output_name
        if (
            output_path.exists()
            and output_path.stat().st_size > 0
            and not args.overwrite
        ):
            elapsed = time.time() - script_start_time
            avg_seconds = elapsed / max(index - 1, 1)
            remaining = max(len(selected_items) - index, 0)
            eta_seconds = avg_seconds * remaining
            print(
                f"[{index}/{len(selected_items)}] skip existing {item.output_name} "
                f"elapsed={_format_seconds(elapsed)} eta={_format_seconds(eta_seconds)}"
            )
            success_items.append(item)
            continue

        item_start_time = time.time()
        prompt_audio_path = _resolve_relative_audio_path(
            manifest_dir, item.prompt_audio_rel
        )
        generate_kwargs: Dict[str, Any] = {
            "text": item.synthesis_text,
            "lang_code": lang_code,
            "ref_audio": prompt_audio_path if use_ref_audio else None,
            "max_tokens": int(args.max_tokens),
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "top_k": int(args.top_k),
            "repetition_penalty": float(args.repetition_penalty),
            "text_temperature": float(args.text_temperature),
            "text_top_p": float(args.text_top_p),
            "text_top_k": int(args.text_top_k),
            "verbose": False,
        }

        filtered_generate_kwargs, ignored = _filter_generate_kwargs(
            model_generate, generate_kwargs
        )
        if ignored_generate_kwargs is None:
            ignored_generate_kwargs = ignored
            if ignored_generate_kwargs:
                print(
                    "[seedtts-eval] warning: model.generate() ignored kwargs: "
                    + ", ".join(ignored_generate_kwargs)
                )

        try:
            mx.reset_peak_memory()
            results = list(model_generate(**filtered_generate_kwargs))
            sample_rate = int(getattr(model, "sample_rate", 24000))
            if results:
                sample_rate = int(getattr(results[-1], "sample_rate", sample_rate))
            audio = _join_audio_segments(result.audio for result in results)
            if audio.size == 0:
                raise RuntimeError("Generated audio is empty")

            audio_write(output_path, audio, sample_rate, format="wav")
            success_items.append(item)

            elapsed = time.time() - script_start_time
            item_seconds = time.time() - item_start_time
            avg_seconds = elapsed / max(index, 1)
            remaining = max(len(selected_items) - index, 0)
            eta_seconds = avg_seconds * remaining
            peak_gb = mx.get_peak_memory() / 1e9
            print(
                f"[{index}/{len(selected_items)}] wrote {item.output_name} "
                f"item={item_seconds:.2f}s peak={peak_gb:.2f}GB "
                f"elapsed={_format_seconds(elapsed)} eta={_format_seconds(eta_seconds)}"
            )
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            failed_items.append((item, message))
            print(
                f"[{index}/{len(selected_items)}] FAILED {item.output_name}: {message}"
            )
            if not args.continue_on_error:
                _write_manifest(ready_manifest_path, success_items)
                if failed_items:
                    failures_log_path.write_text(
                        "".join(
                            json.dumps(
                                {
                                    "utt_id": failed_item.utt_id,
                                    "output_name": failed_item.output_name,
                                    "prompt_audio_rel": failed_item.prompt_audio_rel,
                                    "error": error,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                            for failed_item, error in failed_items
                        ),
                        encoding="utf-8",
                    )
                raise
        finally:
            mx.clear_cache()

    _write_manifest(ready_manifest_path, success_items)

    if failed_items:
        failures_log_path.write_text(
            "".join(
                json.dumps(
                    {
                        "utt_id": item.utt_id,
                        "output_name": item.output_name,
                        "prompt_audio_rel": item.prompt_audio_rel,
                        "error": error,
                    },
                    ensure_ascii=False,
                )
                + "\n"
                for item, error in failed_items
            ),
            encoding="utf-8",
        )

    total_elapsed = time.time() - script_start_time
    summary_payload = {
        "model": args.model,
        "testset_dir": str(testset_dir),
        "lang": args.lang,
        "manifest": str(manifest_path),
        "requested_manifest": str(requested_manifest_path),
        "ready_manifest": str(ready_manifest_path),
        "output_dir": str(output_dir),
        "selected": len(selected_items),
        "ready": len(success_items),
        "failed": len(failed_items),
        "elapsed_seconds": total_elapsed,
        "ignored_generate_kwargs": ignored_generate_kwargs or [],
    }
    if failed_items:
        summary_payload["failures_log"] = str(failures_log_path)
    _write_json(summary_path, summary_payload)

    print(
        f"[seedtts-eval] done ready={len(success_items)} failed={len(failed_items)} "
        f"ready_manifest={ready_manifest_path} summary={summary_path}"
    )

    if failed_items:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
