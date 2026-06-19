from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path

from mlx_audio.research.zonos2_youth.orchestration import (
    assert_no_private_artifact_paths,
    validate_orchestration,
)
from mlx_audio.research.zonos2_youth.schema import (
    ValidationError,
    load_json,
    load_jsonl,
    schema_for,
    validate_adapter_manifest,
    validate_handoff,
    validate_task_ledger,
)


ROOT = Path(__file__).resolve().parents[4]


class TestYouthNaturalOrchestration(unittest.TestCase):
    def test_schema_documents_exist_and_are_json(self):
        for name, kind in {
            "youth_agent_handoff.schema.json": "handoff",
            "youth_dataset_item.schema.json": "dataset_item",
            "youth_adapter_manifest.schema.json": "adapter_manifest",
            "youth_generation_record.schema.json": "generation_record",
        }.items():
            path = ROOT / "research" / "specs" / name
            self.assertTrue(path.exists(), path)
            self.assertEqual(load_json(path), schema_for(kind))

    def test_handoff_schema_validation_accepts_persisted_handoffs(self):
        handoff_dir = ROOT / "artifacts" / "youth_natural" / "orchestration" / "handoffs"
        paths = sorted(handoff_dir.glob("*.json"))
        self.assertGreaterEqual(len(paths), 5)
        for path in paths:
            validate_handoff(load_json(path))

    def test_task_ledger_and_ownership_validate(self):
        result = validate_orchestration(ROOT)
        self.assertEqual(result["status"], "ok")
        self.assertGreaterEqual(result["handoff_count"], 5)

    def test_active_task_overlap_is_rejected(self):
        rows = [
            {
                "task_id": "a",
                "status": "active",
                "resource_class": "cpu_light",
                "mutable_paths": ["x.py"],
            },
            {
                "task_id": "b",
                "status": "active",
                "resource_class": "cpu_light",
                "mutable_paths": ["x.py"],
            },
        ]
        with self.assertRaises(ValidationError):
            validate_task_ledger(rows)

    def test_private_artifact_paths_are_rejected(self):
        with self.assertRaises(ValidationError):
            assert_no_private_artifact_paths(["artifacts/youth_natural/raw_audio/a.wav"])

    def test_integrated_commits_have_acceptance_record_or_are_setup(self):
        rows = load_jsonl(
            ROOT / "artifacts" / "youth_natural" / "orchestration" / "integration_log.jsonl"
        )
        self.assertTrue(any(row.get("event") == "branch_created" for row in rows))

    def test_no_youth_raw_audio_or_embeddings_are_tracked(self):
        tracked = subprocess.check_output(
            ["git", "-C", str(ROOT), "ls-files", "artifacts/youth_natural", "research/youth_natural"],
            text=True,
        ).splitlines()
        forbidden_suffixes = (".wav", ".flac", ".mp3", ".m4a", ".npy", ".npz")
        for path in tracked:
            lowered = path.lower()
            self.assertFalse(lowered.endswith(forbidden_suffixes), path)
            self.assertNotIn("speaker_embedding", lowered)
            self.assertNotIn("raw_audio", lowered)

    def test_tracked_youth_artifacts_do_not_contain_local_absolute_paths(self):
        tracked = subprocess.check_output(
            [
                "git",
                "-C",
                str(ROOT),
                "ls-files",
                "artifacts/youth_natural",
                "research/youth_natural",
                "research/configs",
            ],
            text=True,
        ).splitlines()
        forbidden = ("/Users/", "/var/folders/")
        for rel in tracked:
            path = ROOT / rel
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif"}:
                continue
            text = path.read_text(encoding="utf-8")
            for needle in forbidden:
                self.assertNotIn(needle, text, rel)


class TestYouthNaturalManifests(unittest.TestCase):
    def test_synthetic_adapter_manifest_validates(self):
        path = (
            ROOT
            / "artifacts"
            / "youth_natural"
            / "training_runs"
            / "synthetic-overfit"
            / "adapter_manifest.json"
        )
        self.assertTrue(path.exists())
        validate_adapter_manifest(json.loads(path.read_text(encoding="utf-8")))


if __name__ == "__main__":
    unittest.main()
