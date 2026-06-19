from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .adapter import AdapterManifest, write_adapter_manifest
from .orchestration import append_jsonl


def write_not_run_receipt(
    run_dir: str | Path,
    *,
    command: str,
    reason: str,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    receipt = {
        "status": "not_run",
        "command": command,
        "reason": reason,
        "config": config or {},
    }
    (run_dir / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    append_jsonl(run_dir / "events.jsonl", {"event": "not_run", **receipt})
    return receipt


def synthetic_overfit(run_dir: str | Path, *, config: dict[str, Any] | None = None) -> dict[str, Any]:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = AdapterManifest(
        base_checkpoint_hash="synthetic_fixture",
        target_modules=[
            {"name": "layers.*.attention.wq", "kind": "lora", "rank": 8},
            {
                "name": "layers.*.attention.wkv.value",
                "kind": "value_slice_lora",
                "rank": 8,
                "value_slice": [0, "kv_dim"],
            },
            {"name": "layers.*.attention.wo", "kind": "lora", "rank": 8},
        ],
        lineage={"dataset_snapshots": ["synthetic_fixture"], "stage": "tiny_overfit"},
    )
    write_adapter_manifest(run_dir / "adapter_manifest.json", manifest)
    report = {
        "status": "completed",
        "mode": "synthetic_fixture",
        "claims": [
            "No real model weights loaded.",
            "No private or youth audio accessed.",
            "Manifest/export path exercised with synthetic metadata.",
        ],
        "config": config or {},
    }
    (run_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report

