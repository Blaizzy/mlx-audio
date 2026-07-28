from __future__ import annotations

import inspect
import itertools
import json
import time
import wave
from pathlib import Path
from typing import Any, Iterable, Optional, TypeVar, Union

import mlx.nn as nn
from tqdm import tqdm

from mlx_audio.stt.utils import load_model

from .normalize import normalize_for_wer
from .schema import STTEvalSample
from .seed_tts import (
    DEFAULT_DATASET_REPO,
    DEFAULT_REVISION,
    iter_seed_tts_english_samples,
)
from .wer import WERResult, aggregate_wer, compute_wer

T = TypeVar("T")
SUPPORTED_METRICS = ("wer",)


def run_stt_wer_eval(
    model: Union[str, nn.Module],
    samples: Iterable[STTEvalSample],
    output_dir: str | Path,
    *,
    dataset_name: str,
    dataset_revision: Optional[str] = None,
    dataset_split: Optional[str] = None,
    expected_count: Optional[int] = None,
    summary_metadata: Optional[dict[str, Any]] = None,
    limit: Optional[int] = None,
    language: Optional[str] = None,
    gen_kwargs: Optional[dict[str, Any]] = None,
    skip_existing: bool = False,
    dataset_batch_size: int = 1,
    metrics: Optional[Iterable[str]] = None,
    verbose: bool = False,
) -> dict[str, Any]:
    if dataset_batch_size < 1:
        raise ValueError("dataset_batch_size must be at least 1.")
    metrics = _validate_metrics(metrics)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    predictions_path = output_path / "predictions.jsonl"

    model_id = model if isinstance(model, str) else model.__class__.__name__
    model_obj = load_model(model) if isinstance(model, str) else model
    generate_kwargs = gen_kwargs or {}
    existing = _load_existing_predictions(predictions_path) if skip_existing else {}

    records: list[dict[str, Any]] = []
    wer_results: list[WERResult] = []
    skipped = 0
    total_wall_time = 0.0

    if not skip_existing:
        predictions_file = open(predictions_path, "w", encoding="utf-8")
    else:
        predictions_file = open(predictions_path, "a", encoding="utf-8")

    try:
        sample_iterable = _limit_iterable(samples, limit)
        total = expected_count
        if total is None:
            total = limit
        elif limit is not None:
            total = min(total, limit)
        progress = tqdm(sample_iterable, total=total, desc="STT WER", unit="utt")
        for batch in _batched(progress, dataset_batch_size):
            for sample in batch:
                if skip_existing and sample.utt_id in existing:
                    record = existing[sample.utt_id]
                    records.append(record)
                    wer_results.append(_wer_result_from_record(record))
                    total_wall_time += float(record.get("wall_time_sec", 0.0) or 0.0)
                    skipped += 1
                    continue

                started = time.perf_counter()
                try:
                    hypothesis = _transcribe_sample(
                        model_obj=model_obj,
                        audio_path=sample.audio_path,
                        language=language,
                        gen_kwargs=generate_kwargs,
                        verbose=verbose,
                    )
                except Exception:
                    predictions_file.flush()
                    raise
                wall_time = time.perf_counter() - started
                total_wall_time += wall_time

                reference_normalized = normalize_for_wer(sample.reference_text)
                hypothesis_normalized = normalize_for_wer(hypothesis)
                wer_result = compute_wer(reference_normalized, hypothesis_normalized)
                wer_results.append(wer_result)

                record = {
                    "utt_id": sample.utt_id,
                    "audio_path": str(sample.audio_path),
                    "source_path": sample.source_path,
                    "reference": sample.reference_text,
                    "hypothesis": hypothesis,
                    "reference_normalized": reference_normalized,
                    "hypothesis_normalized": hypothesis_normalized,
                    **wer_result.to_dict(),
                    "wall_time_sec": wall_time,
                    "audio_duration_sec": _wav_duration_seconds(sample.audio_path),
                    "metadata": dict(sample.metadata),
                }
                records.append(record)
                predictions_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            predictions_file.flush()
    finally:
        predictions_file.close()

    if not records:
        raise RuntimeError("No Seed-TTS samples were evaluated.")

    if all(not record.get("hypothesis_normalized", "") for record in records):
        raise RuntimeError("All model hypotheses were empty.")

    aggregates = aggregate_wer(wer_results)
    audio_duration_total = sum(
        float(record["audio_duration_sec"])
        for record in records
        if record.get("audio_duration_sec") is not None
    )
    summary = {
        "model": str(model_id),
        "dataset_name": dataset_name,
        "dataset_revision": dataset_revision,
        "dataset_split": dataset_split,
        "num_samples": len(records),
        "num_skipped": skipped,
        "dataset_batch_size": dataset_batch_size,
        "metrics": metrics,
        "generation_kwargs": generate_kwargs,
        "language": language,
        **aggregates,
        "total_wall_time_sec": total_wall_time,
        "total_audio_duration_sec": audio_duration_total,
        "rtf": total_wall_time / audio_duration_total if audio_duration_total else None,
    }
    if summary_metadata:
        summary.update(summary_metadata)

    _write_summary(output_path, summary)
    return summary


def run_seed_tts_eval(
    model: Union[str, nn.Module],
    output_dir: str | Path,
    dataset_repo: str = DEFAULT_DATASET_REPO,
    revision: str = DEFAULT_REVISION,
    limit: Optional[int] = None,
    language: Optional[str] = None,
    gen_kwargs: Optional[dict[str, Any]] = None,
    skip_existing: bool = False,
    fail_on_missing_audio: bool = True,
    audio_cache_dir: Optional[str | Path] = None,
    dataset_batch_size: int = 1,
    metrics: Optional[Iterable[str]] = None,
    verbose: bool = False,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    audio_cache_path = (
        Path(audio_cache_dir) if audio_cache_dir else output_path / "audio-cache"
    )
    samples = iter_seed_tts_english_samples(
        dataset_repo=dataset_repo,
        revision=revision,
        audio_cache_dir=audio_cache_path,
        limit=limit,
        fail_on_missing_audio=fail_on_missing_audio,
    )
    return run_stt_wer_eval(
        model=model,
        samples=samples,
        output_dir=output_path,
        dataset_name="seed-tts",
        dataset_revision=revision,
        dataset_split="en/standard",
        expected_count=limit,
        summary_metadata={
            "dataset_repo": dataset_repo,
            "locale": "en",
            "set": "standard",
        },
        language=language,
        gen_kwargs=gen_kwargs,
        skip_existing=skip_existing,
        dataset_batch_size=dataset_batch_size,
        metrics=metrics,
        verbose=verbose,
    )


def _validate_metrics(metrics: Optional[Iterable[str]]) -> list[str]:
    normalized = [metric.lower() for metric in (metrics or ["wer"])]
    if not normalized:
        raise ValueError("At least one metric must be requested.")

    unsupported = sorted(set(normalized) - set(SUPPORTED_METRICS))
    if unsupported:
        supported = ", ".join(SUPPORTED_METRICS)
        requested = ", ".join(unsupported)
        raise ValueError(f"Unsupported metric(s): {requested}. Supported: {supported}.")

    return list(dict.fromkeys(normalized))


def _limit_iterable(iterable: Iterable[T], limit: Optional[int]) -> Iterator[T]:
    for index, item in enumerate(iterable):
        if limit is not None and index >= limit:
            return
        yield item


def _batched(iterable: Iterable[T], batch_size: int) -> Iterator[list[T]]:
    iterator = iter(iterable)
    while batch := list(itertools.islice(iterator, batch_size)):
        yield batch


def _transcribe_sample(
    model_obj: nn.Module,
    audio_path: Path,
    language: Optional[str],
    gen_kwargs: dict[str, Any],
    verbose: bool,
) -> str:
    signature = inspect.signature(model_obj.generate)
    call_kwargs = dict(gen_kwargs)
    if language is not None:
        call_kwargs["language"] = language
    if "verbose" in signature.parameters:
        call_kwargs["verbose"] = verbose
    call_kwargs = {k: v for k, v in call_kwargs.items() if k in signature.parameters}

    result = model_obj.generate(str(audio_path), **call_kwargs)
    return _extract_text(result)


def _extract_text(result: Any) -> str:
    if hasattr(result, "text"):
        return str(result.text).strip()
    if isinstance(result, str):
        return result.strip()
    if isinstance(result, dict) and "text" in result:
        return str(result["text"]).strip()
    if inspect.isgenerator(result) or hasattr(result, "__iter__"):
        parts: list[str] = []
        for item in result:
            if hasattr(item, "text"):
                parts.append(str(item.text))
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "".join(parts).strip()
    return str(result).strip()


def _load_existing_predictions(predictions_path: Path) -> dict[str, dict[str, Any]]:
    if not predictions_path.exists():
        return {}

    records: dict[str, dict[str, Any]] = {}
    with open(predictions_path, "r", encoding="utf-8") as predictions_file:
        for line in predictions_file:
            if not line.strip():
                continue
            record = json.loads(line)
            utt_id = record.get("utt_id")
            if utt_id:
                records[str(utt_id)] = record
    return records


def _wer_result_from_record(record: dict[str, Any]) -> WERResult:
    return WERResult(
        substitutions=int(record["substitutions"]),
        deletions=int(record["deletions"]),
        insertions=int(record["insertions"]),
        reference_tokens=int(record["reference_tokens"]),
        hypothesis_tokens=int(record["hypothesis_tokens"]),
        wer=float(record["wer"]),
    )


def _wav_duration_seconds(path: Path) -> Optional[float]:
    try:
        with wave.open(str(path), "rb") as wav_file:
            frames = wav_file.getnframes()
            sample_rate = wav_file.getframerate()
            return frames / sample_rate if sample_rate else None
    except Exception:
        return None


def _write_summary(output_path: Path, summary: dict[str, Any]) -> None:
    with open(output_path / "summary.json", "w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, ensure_ascii=False, indent=2)
        summary_file.write("\n")

    with open(output_path / "summary.txt", "w", encoding="utf-8") as summary_file:
        summary_file.write(f"Model: {summary['model']}\n")
        dataset_label = summary["dataset_name"]
        if summary.get("dataset_repo"):
            dataset_label = summary["dataset_repo"]
        if summary.get("dataset_revision"):
            dataset_label = f"{dataset_label}@{summary['dataset_revision']}"
        summary_file.write(f"Dataset: {dataset_label}\n")
        summary_file.write(f"Samples: {summary['num_samples']}\n")
        summary_file.write(f"WER micro: {summary['wer_micro'] * 100:.3f}%\n")
        summary_file.write(f"WER macro: {summary['wer_macro'] * 100:.3f}%\n")
        summary_file.write(
            "Sub/Del/Ins: "
            f"{summary['substitution_rate'] * 100:.3f}% / "
            f"{summary['deletion_rate'] * 100:.3f}% / "
            f"{summary['insertion_rate'] * 100:.3f}%\n"
        )
        if summary["rtf"] is not None:
            summary_file.write(f"RTF: {summary['rtf']:.3f}\n")
