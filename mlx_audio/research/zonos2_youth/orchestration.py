from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import (
    ValidationError,
    load_json,
    load_jsonl,
    validate_file_ownership,
    validate_handoff,
    validate_task_ledger,
)


def validate_orchestration(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    orchestration = root / "artifacts" / "youth_natural" / "orchestration"
    ledger_path = orchestration / "task_ledger.jsonl"
    ownership_path = orchestration / "file_ownership.json"
    handoff_dir = orchestration / "handoffs"

    ledger_rows = load_jsonl(ledger_path)
    validate_task_ledger(ledger_rows)
    validate_file_ownership(load_json(ownership_path))

    handoff_count = 0
    for path in sorted(handoff_dir.glob("*.json")):
        validate_handoff(load_json(path))
        handoff_count += 1

    accepted_commits = [
        row
        for row in load_jsonl(orchestration / "integration_log.jsonl")
        if row.get("event") in {"accept_subagent_commit", "lead_patch_commit"}
    ]
    return {
        "status": "ok",
        "ledger_rows": len(ledger_rows),
        "handoff_count": handoff_count,
        "accepted_commit_records": len(accepted_commits),
    }


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def assert_no_private_artifact_paths(paths: list[str]) -> None:
    forbidden = ("raw_audio", "speaker_embedding", "private", "minor_audio")
    for path in paths:
        lowered = path.lower()
        if any(token in lowered for token in forbidden):
            raise ValidationError(f"private-looking path is not allowed in Git: {path}")
