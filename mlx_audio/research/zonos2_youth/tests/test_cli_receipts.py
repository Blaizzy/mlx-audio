from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]


class TestYouthNaturalCliReceipts(unittest.TestCase):
    def run_cli(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "mlx_audio.research.zonos2_youth", *args],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )

    def run_cli_from(self, cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
        env = dict(__import__("os").environ)
        env["PYTHONPATH"] = str(ROOT)
        return subprocess.run(
            [sys.executable, "-m", "mlx_audio.research.zonos2_youth", *args],
            cwd=cwd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )

    def test_validate_orchestration_command(self):
        result = self.run_cli("validate-orchestration")
        payload = json.loads(result.stdout)
        self.assertEqual(payload["status"], "ok")

    def test_prepare_data_writes_truthful_not_run_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "config.yaml"
            run_dir = Path(tmp) / "prepare"
            config.write_text(
                json.dumps({"run_dir": str(run_dir), "not_run_reason": "fixture only"}),
                encoding="utf-8",
            )
            result = self.run_cli("prepare-data", "--config", str(config))
            payload = json.loads(result.stdout)
            self.assertEqual(payload["status"], "not_run")
            receipt = json.loads((run_dir / "receipt.json").read_text(encoding="utf-8"))
            self.assertEqual(receipt["reason"], "fixture only")

    def test_overfit_writes_synthetic_manifest_without_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "config.yaml"
            run_dir = Path(tmp) / "overfit"
            config.write_text(json.dumps({"run_dir": str(run_dir)}), encoding="utf-8")
            result = self.run_cli("overfit", "--config", str(config))
            payload = json.loads(result.stdout)
            self.assertEqual(payload["status"], "completed")
            self.assertTrue((run_dir / "adapter_manifest.json").exists())
            tracked_audio = list(run_dir.glob("*.wav"))
            self.assertEqual(tracked_audio, [])

    def test_train_honors_explicit_temp_run_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "config.yaml"
            run_dir = Path(tmp) / "train-youth"
            config.write_text(json.dumps({"run_dir": str(run_dir)}), encoding="utf-8")
            result = self.run_cli("train", "--stage", "youth", "--config", str(config))
            payload = json.loads(result.stdout)
            self.assertEqual(payload["status"], "not_run")
            self.assertTrue((run_dir / "receipt.json").exists())
            self.assertEqual(payload["config"]["run_dir"], str(run_dir))

    def test_relative_train_run_dir_resolves_under_root_from_outside_repo(self):
        with tempfile.TemporaryDirectory() as tmp:
            outside = Path(tmp) / "outside"
            outside.mkdir()
            root = Path(tmp) / "root"
            root.mkdir()
            config = Path(tmp) / "config.yaml"
            config.write_text(json.dumps({"run_dir": "relative-train"}), encoding="utf-8")
            result = self.run_cli_from(
                outside,
                "--root",
                str(root),
                "train",
                "--stage",
                "youth",
                "--config",
                str(config),
            )
            payload = json.loads(result.stdout)
            self.assertEqual(payload["status"], "not_run")
            self.assertTrue((root / "relative-train" / "receipt.json").exists())
            self.assertFalse((outside / "relative-train").exists())


if __name__ == "__main__":
    unittest.main()
