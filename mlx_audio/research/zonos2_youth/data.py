from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .schema import AGE_BANDS, RIGHTS_LANES, ValidationError, validate_dataset_item


@dataclass(frozen=True)
class DatasetItem:
    dataset: str
    release: str
    rights_lane: str
    terms_hash: str
    speaker_id: str
    session_id: str
    recording_id: str
    age_band: str
    language: str
    locale: str
    original_transcript: str
    normalized_transcript: str
    source_audio_hash: str
    normalized_audio_hash: str
    original_sample_rate: int
    codec: str
    effective_bandwidth_hz: float
    clipping: float
    dc_offset: float
    snr_db: float | None
    vad_speech_ratio: float
    silence_ratio: float
    alignment_confidence: float | None
    source_split: str
    generated_split: str
    reference_audio_ids: list[str]
    speaker_embedding_hash: str | None
    prompt_rows_hash: str
    unsheared_targets_hash: str
    sheared_targets_hash: str
    loss_mask_hash: str
    sequence_bucket: str
    preprocessing_fingerprint: str
    quarantine_reason: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        validate_dataset_item(data)
        return data


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def pseudonymize(*parts: str, prefix: str) -> str:
    return f"{prefix}_{stable_hash('|'.join(parts))[:16]}"


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValidationError(f"{path}:{line_no}: invalid JSONL: {exc}") from exc
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def read_table(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        return [dict(row) for row in read_jsonl(path)]
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f, delimiter=delimiter)]


def map_common_voice_age(age: str | None) -> str:
    value = (age or "").strip().lower()
    if value in {"teens", "teen"}:
        return "teen"
    if value in {"child", "children", "kids"}:
        return "child"
    if value in {
        "twenties",
        "thirties",
        "fourties",
        "forties",
        "fifties",
        "sixties",
        "seventies",
        "eighties",
        "nineties",
    }:
        return "adult"
    return "unknown"


def normalize_transcript_for_prompt(text: str) -> str:
    return " ".join((text or "").strip().split())


def common_voice_items(
    rows: Iterable[dict[str, Any]],
    *,
    dataset: str,
    release: str,
    terms_hash: str,
    rights_lane: str = "permissive_release",
    language: str = "en",
    require_validated: bool = True,
    required_age_band: str | None = None,
) -> list[DatasetItem]:
    if rights_lane not in RIGHTS_LANES:
        raise ValidationError(f"invalid rights lane: {rights_lane}")
    if required_age_band is not None and required_age_band not in AGE_BANDS:
        raise ValidationError(f"invalid required age band: {required_age_band}")
    items: list[DatasetItem] = []
    for idx, row in enumerate(rows):
        status = str(row.get("status") or row.get("bucket") or row.get("validated") or "")
        if require_validated and status.lower() not in {"validated", "valid", "1", "true"}:
            continue
        age_band = map_common_voice_age(row.get("age"))
        if required_age_band is not None and age_band != required_age_band:
            continue
        transcript = str(row.get("sentence") or row.get("transcript") or row.get("text") or "")
        normalized = normalize_transcript_for_prompt(transcript)
        if not normalized:
            continue
        client_id = str(row.get("client_id") or row.get("speaker_id") or "unknown")
        path = str(row.get("path") or row.get("audio_path") or f"row-{idx}")
        session = str(row.get("session_id") or row.get("segment_id") or "unknown")
        source_hash = str(row.get("audio_sha256") or stable_hash(path))
        item = DatasetItem(
            dataset=dataset,
            release=release,
            rights_lane=rights_lane,
            terms_hash=terms_hash,
            speaker_id=pseudonymize(dataset, release, client_id, prefix="spk"),
            session_id=pseudonymize(dataset, release, client_id, session, prefix="ses"),
            recording_id=pseudonymize(dataset, release, path, prefix="rec"),
            age_band=age_band,
            language=language,
            locale=str(row.get("locale") or language),
            original_transcript=transcript,
            normalized_transcript=normalized,
            source_audio_hash=source_hash,
            normalized_audio_hash=str(row.get("normalized_audio_sha256") or source_hash),
            original_sample_rate=int(row.get("sample_rate") or 0),
            codec=str(row.get("codec") or "unknown"),
            effective_bandwidth_hz=float(row.get("effective_bandwidth_hz") or 0.0),
            clipping=float(row.get("clipping") or 0.0),
            dc_offset=float(row.get("dc_offset") or 0.0),
            snr_db=(float(row["snr_db"]) if row.get("snr_db") not in {None, ""} else None),
            vad_speech_ratio=float(row.get("vad_speech_ratio") or 0.0),
            silence_ratio=float(row.get("silence_ratio") or 0.0),
            alignment_confidence=(
                float(row["alignment_confidence"])
                if row.get("alignment_confidence") not in {None, ""}
                else None
            ),
            source_split=str(row.get("split") or "unassigned"),
            generated_split="train",
            reference_audio_ids=[],
            speaker_embedding_hash=str(row.get("speaker_embedding_hash") or "")
            or None,
            prompt_rows_hash=stable_hash(normalized),
            unsheared_targets_hash=str(row.get("unsheared_targets_hash") or ""),
            sheared_targets_hash=str(row.get("sheared_targets_hash") or ""),
            loss_mask_hash=str(row.get("loss_mask_hash") or ""),
            sequence_bucket=str(row.get("sequence_bucket") or "unknown"),
            preprocessing_fingerprint=stable_hash("common_voice_items:v1"),
        )
        items.append(item)
    return assign_speaker_disjoint_splits(items)


def assign_speaker_disjoint_splits(items: list[DatasetItem]) -> list[DatasetItem]:
    speakers = sorted({item.speaker_id for item in items})
    split_by_speaker: dict[str, str] = {}
    for idx, speaker in enumerate(speakers):
        bucket = stable_hash(speaker)
        value = int(bucket[:8], 16) % 100
        if value < 80:
            split = "train"
        elif value < 90:
            split = "validation"
        else:
            split = "sealed_test" if idx % 2 else "test"
        split_by_speaker[speaker] = split
    return [
        DatasetItem(**{**item.to_dict(), "generated_split": split_by_speaker[item.speaker_id]})
        for item in items
    ]


def assert_split_isolation(items: Iterable[DatasetItem]) -> None:
    seen: dict[str, str] = {}
    for item in items:
        previous = seen.setdefault(item.speaker_id, item.generated_split)
        if previous != item.generated_split:
            raise ValidationError(f"speaker crosses splits: {item.speaker_id}")


def duplicate_keys(items: Iterable[DatasetItem]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for item in items:
        key = stable_hash(f"{item.normalized_transcript}|{item.source_audio_hash}")
        groups.setdefault(key, []).append(item.recording_id)
    return {key: ids for key, ids in groups.items() if len(ids) > 1}


def attach_reference_pairs(items: list[DatasetItem]) -> list[DatasetItem]:
    by_speaker: dict[str, list[DatasetItem]] = {}
    for item in items:
        by_speaker.setdefault(item.speaker_id, []).append(item)
    paired: list[DatasetItem] = []
    for item in items:
        candidates = [
            other
            for other in by_speaker.get(item.speaker_id, [])
            if other.recording_id != item.recording_id
        ]
        if not candidates:
            paired.append(item)
            continue
        candidates.sort(key=lambda other: (other.session_id == item.session_id, other.recording_id))
        paired.append(
            DatasetItem(
                **{
                    **item.to_dict(),
                    "reference_audio_ids": [candidates[0].recording_id],
                }
            )
        )
    return paired
