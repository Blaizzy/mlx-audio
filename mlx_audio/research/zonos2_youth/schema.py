from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


RIGHTS_LANES = {
    "permissive_release",
    "research_noncommercial",
    "separately_licensed",
    "user_consent_private",
    "blocked_or_unknown",
}

AGE_BANDS = {"child", "teen", "adult", "unknown"}

RESOURCE_CLASSES = {
    "read_only_analysis",
    "cpu_light",
    "cpu_heavy",
    "io_or_network",
    "mlx_metal_inference",
    "mlx_metal_training",
    "integration",
    "independent_review",
}

STATUS_VALUES = {"completed", "partial", "blocked", "failed"}
TASK_STATUS_VALUES = {"planned", "active", *STATUS_VALUES}


class ValidationError(ValueError):
    """Raised when a YouthNaturalLoRA artifact fails structural validation."""


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{name} must be an object")
    return value


def _require_string(value: Any, *, name: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValidationError(f"{name} must be a string")
    if not allow_empty and not value:
        raise ValidationError(f"{name} must not be empty")
    return value


def _require_list(value: Any, *, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValidationError(f"{name} must be a list")
    return value


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValidationError(f"{path}:{line_no}: invalid JSONL: {exc}") from exc
            rows.append(dict(_require_mapping(row, name=f"{path}:{line_no}")))
    return rows


def validate_handoff(record: Mapping[str, Any]) -> None:
    obj = _require_mapping(record, name="handoff")
    _require_string(obj.get("task_id"), name="task_id")
    status = _require_string(obj.get("status"), name="status")
    if status not in STATUS_VALUES:
        raise ValidationError(f"status must be one of {sorted(STATUS_VALUES)}")
    for key in (
        "files_changed",
        "artifacts_created",
        "commands_run",
        "tests_run",
        "tests_not_run",
        "data_accessed",
        "claims",
        "known_risks",
        "integration_notes",
    ):
        _require_list(obj.get(key), name=key)


def validate_dataset_item(record: Mapping[str, Any]) -> None:
    obj = _require_mapping(record, name="dataset item")
    for key in (
        "dataset",
        "release",
        "rights_lane",
        "terms_hash",
        "speaker_id",
        "session_id",
        "recording_id",
        "age_band",
        "language",
        "original_transcript",
        "normalized_transcript",
        "source_audio_hash",
        "generated_split",
    ):
        _require_string(obj.get(key), name=key)
    if obj["rights_lane"] not in RIGHTS_LANES:
        raise ValidationError(f"invalid rights_lane: {obj['rights_lane']}")
    if obj["age_band"] not in AGE_BANDS:
        raise ValidationError(f"invalid age_band: {obj['age_band']}")
    if obj["generated_split"] not in {"train", "validation", "test", "sealed_test"}:
        raise ValidationError("generated_split must be train/validation/test/sealed_test")
    if obj.get("speaker_embedding") is not None:
        raise ValidationError("dataset items must not contain raw speaker_embedding")
    if obj.get("audio_bytes") is not None:
        raise ValidationError("dataset items must not contain raw audio_bytes")


def validate_adapter_manifest(record: Mapping[str, Any]) -> None:
    obj = _require_mapping(record, name="adapter manifest")
    for key in (
        "adapter_name",
        "format_version",
        "base_checkpoint_hash",
        "created_at",
        "rank",
        "alpha",
        "scaling",
        "dtype",
        "target_modules",
        "strength_behavior",
        "lineage",
    ):
        if key not in obj:
            raise ValidationError(f"missing {key}")
    if obj["adapter_name"] != "YouthNaturalLoRA":
        raise ValidationError("adapter_name must be YouthNaturalLoRA")
    if not isinstance(obj["rank"], int) or obj["rank"] <= 0:
        raise ValidationError("rank must be a positive integer")
    _require_list(obj["target_modules"], name="target_modules")
    lineage = _require_mapping(obj["lineage"], name="lineage")
    _require_list(lineage.get("dataset_snapshots", []), name="dataset_snapshots")
    rights_lanes = lineage.get("rights_lanes", [])
    _require_list(rights_lanes, name="lineage.rights_lanes")
    for lane in rights_lanes:
        if lane not in RIGHTS_LANES:
            raise ValidationError(f"invalid lineage rights lane: {lane}")
    if obj.get("release_eligible", False):
        restricted = {
            "research_noncommercial",
            "separately_licensed",
            "blocked_or_unknown",
        }
        if any(lane in restricted for lane in rights_lanes):
            raise ValidationError("release_eligible adapters must not include restricted lanes")


def validate_generation_record(record: Mapping[str, Any]) -> None:
    obj = _require_mapping(record, name="generation record")
    for key in (
        "generation_id",
        "prompt_hash",
        "provided_age_band",
        "voice_profile_id",
        "reference_hashes",
        "rights_lane",
        "base_hash",
        "adapter_strength",
        "sampling",
        "seed",
        "audio_hash",
        "local_audio_path",
        "checkpoint_stage",
        "future_preference_eligible",
    ):
        if key not in obj:
            raise ValidationError(f"missing {key}")
    if obj["rights_lane"] not in RIGHTS_LANES:
        raise ValidationError(f"invalid rights_lane: {obj['rights_lane']}")
    if obj["provided_age_band"] not in AGE_BANDS:
        raise ValidationError(f"invalid provided_age_band: {obj['provided_age_band']}")
    _require_list(obj["reference_hashes"], name="reference_hashes")


def validate_task_ledger(rows: Sequence[Mapping[str, Any]]) -> None:
    seen_active_paths: dict[str, str] = {}
    for idx, row in enumerate(rows):
        obj = _require_mapping(row, name=f"ledger row {idx}")
        task_id = _require_string(obj.get("task_id"), name=f"ledger row {idx}.task_id")
        status = _require_string(obj.get("status"), name=f"{task_id}.status")
        if status not in TASK_STATUS_VALUES:
            raise ValidationError(f"{task_id}: invalid status {status}")
        resource_class = obj.get("resource_class")
        if resource_class is not None and resource_class not in RESOURCE_CLASSES:
            raise ValidationError(f"{task_id}: invalid resource_class {resource_class}")
        if status != "active":
            continue
        for path in obj.get("mutable_paths", []) or []:
            if path in seen_active_paths:
                raise ValidationError(
                    f"active ownership overlap for {path}: "
                    f"{seen_active_paths[path]} and {task_id}"
                )
            seen_active_paths[str(path)] = task_id


def validate_file_ownership(record: Mapping[str, Any]) -> None:
    obj = _require_mapping(record, name="file ownership")
    ownership = _require_mapping(
        obj.get("active_write_ownership"), name="active_write_ownership"
    )
    seen: dict[str, str] = {}
    for task_id, paths in ownership.items():
        _require_list(paths, name=f"{task_id} paths")
        for path in paths:
            path_s = _require_string(path, name=f"{task_id} path")
            if path_s in seen:
                raise ValidationError(
                    f"write ownership overlap for {path_s}: {seen[path_s]} and {task_id}"
                )
            seen[path_s] = str(task_id)


def schema_for(kind: str) -> dict[str, Any]:
    """Return a lightweight JSON Schema document for repository artifacts."""

    schemas: dict[str, dict[str, Any]] = {
        "handoff": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "required": [
                "task_id",
                "status",
                "branch",
                "commit",
                "files_changed",
                "artifacts_created",
                "commands_run",
                "tests_run",
                "tests_not_run",
                "data_accessed",
                "claims",
                "known_risks",
                "integration_notes",
            ],
            "properties": {
                "status": {"enum": sorted(STATUS_VALUES)},
                "files_changed": {"type": "array"},
                "artifacts_created": {"type": "array"},
            },
        },
        "dataset_item": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "required": [
                "dataset",
                "release",
                "rights_lane",
                "terms_hash",
                "speaker_id",
                "session_id",
                "recording_id",
                "age_band",
                "language",
                "original_transcript",
                "normalized_transcript",
                "source_audio_hash",
                "generated_split",
            ],
            "properties": {
                "rights_lane": {"enum": sorted(RIGHTS_LANES)},
                "age_band": {"enum": sorted(AGE_BANDS)},
            },
        },
        "adapter_manifest": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "required": [
                "adapter_name",
                "format_version",
                "base_checkpoint_hash",
                "rank",
                "alpha",
                "target_modules",
                "lineage",
            ],
            "properties": {
                "adapter_name": {"const": "YouthNaturalLoRA"},
                "release_eligible": {"type": "boolean"},
                "lineage": {
                    "type": "object",
                    "required": ["dataset_snapshots", "rights_lanes"],
                    "properties": {
                        "dataset_snapshots": {"type": "array", "items": {"type": "string"}},
                        "rights_lanes": {
                            "type": "array",
                            "items": {"enum": sorted(RIGHTS_LANES)},
                        },
                    },
                },
            },
        },
        "generation_record": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "required": [
                "generation_id",
                "prompt_hash",
                "provided_age_band",
                "reference_hashes",
                "rights_lane",
                "base_hash",
                "adapter_strength",
                "sampling",
                "seed",
                "audio_hash",
                "local_audio_path",
                "checkpoint_stage",
                "future_preference_eligible",
            ],
        },
    }
    try:
        return schemas[kind]
    except KeyError as exc:
        raise ValidationError(f"unknown schema kind: {kind}") from exc
