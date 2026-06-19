from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from .orchestration import validate_orchestration
from .rights import default_rights_records, write_rights_report
from .schema import schema_for
from .train import synthetic_overfit, write_not_run_receipt


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _root_from_args(args: argparse.Namespace) -> Path:
    return Path(args.root or _repo_root()).resolve()


def _resolve_output_path(args: argparse.Namespace, path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else _root_from_args(args) / value


def _load_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    text = Path(path).read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path} must be JSON-compatible YAML for this workflow: {exc}")


def cmd_audit(args: argparse.Namespace) -> int:
    root = _root_from_args(args)
    env = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "repo_root": "<repo>",
    }
    try:
        env["git_commit"] = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        env["git_commit"] = "unknown"
    out = root / "artifacts" / "youth_natural" / "environment.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(env, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rights_path = root / "artifacts" / "youth_natural" / "dataset_rights_report.json"
    write_rights_report(rights_path, default_rights_records())
    source_manifest = {
        "format_version": 1,
        "generated_at": "2026-06-19",
        "repository": {
            "path": "<repo>",
            "commit": env.get("git_commit", "unknown"),
            "branch": _git(root, "rev-parse", "--abbrev-ref", "HEAD"),
            "status_short": _git(root, "status", "--short"),
        },
        "models": {
            "base": "Zyphra/ZONOS2",
            "mlx_checkpoint": "mlx-community/Zyphra-ZONOS2",
            "dac": "mlx-community/descript-audio-codec-44khz",
            "speaker_encoder": "marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B",
        },
        "python_packages": {
            name: _package_version(name)
            for name in ("mlx", "mlx-lm", "mlx-audio", "numpy")
        },
        "zonos2_source_paths": [
            "mlx_audio/tts/models/zonos2/config.py",
            "mlx_audio/tts/models/zonos2/prompt.py",
            "mlx_audio/tts/models/zonos2/generation.py",
            "mlx_audio/tts/models/zonos2/model.py",
            "mlx_audio/tts/models/zonos2/speaker_encoder.py",
            "mlx_audio/tts/models/zonos2/convert.py",
        ],
    }
    source_path = root / "artifacts" / "youth_natural" / "source_manifest.json"
    source_path.write_text(
        json.dumps(source_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "environment": str(out),
                "rights_report": str(rights_path),
                "source_manifest": str(source_path),
            },
            indent=2,
        )
    )
    return 0


def _git(root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not_installed"


def cmd_validate(args: argparse.Namespace) -> int:
    root = _root_from_args(args)
    result = validate_orchestration(root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def cmd_emit_schemas(args: argparse.Namespace) -> int:
    root = _root_from_args(args)
    out_dir = root / "research" / "specs"
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "youth_agent_handoff.schema.json": "handoff",
        "youth_dataset_item.schema.json": "dataset_item",
        "youth_adapter_manifest.schema.json": "adapter_manifest",
        "youth_generation_record.schema.json": "generation_record",
    }
    for filename, kind in mapping.items():
        (out_dir / filename).write_text(
            json.dumps(schema_for(kind), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({"schemas": sorted(mapping)}, indent=2))
    return 0


def cmd_prepare_data(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    reason = config.get(
        "not_run_reason",
        "No dataset archive was supplied; synthetic fixtures are used by tests.",
    )
    out = write_not_run_receipt(
        _resolve_output_path(
            args,
            config.get(
                "run_dir",
                "artifacts/youth_natural/training_runs/prepare-data-not-run",
            ),
        ),
        command="prepare-data",
        reason=reason,
        config=config,
    )
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


def cmd_overfit(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    run_dir = _resolve_output_path(
        args,
        config.get("run_dir", "artifacts/youth_natural/training_runs/synthetic-overfit"),
    )
    report = synthetic_overfit(run_dir, config=config)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    reason = (
        "Real MLX training was not run because this command requires an explicit "
        "rights-checked dataset snapshot, sufficient disk, and a selected model tier."
    )
    stage_run_dirs = config.get("stage_run_dirs", {})
    configured_run_dir = config.get("run_dir")
    run_dir = _resolve_output_path(
        args,
        stage_run_dirs.get(args.stage)
        or configured_run_dir
        or f"artifacts/youth_natural/training_runs/{args.stage}-not-run",
    )
    out = write_not_run_receipt(run_dir, command=f"train:{args.stage}", reason=reason, config=config)
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    reason = (
        "Evaluation needs immutable generation manifests from Studio and adapter checkpoints; "
        "none were supplied in this synthetic-only run."
    )
    out = write_not_run_receipt(
        _resolve_output_path(
            args,
            config.get("run_dir", "artifacts/youth_natural/training_runs/evaluate-not-run"),
        ),
        command="evaluate",
        reason=reason,
        config=config,
    )
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


def cmd_export_adapter(args: argparse.Namespace) -> int:
    config = {"checkpoint": args.checkpoint}
    out = write_not_run_receipt(
        _resolve_output_path(args, "artifacts/youth_natural/training_runs/export-adapter-not-run"),
        command="export-adapter",
        reason="No real checkpoint was supplied for export in this run.",
        config=config,
    )
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


def cmd_test_merge(args: argparse.Namespace) -> int:
    config = {"adapter": args.adapter}
    out = write_not_run_receipt(
        _resolve_output_path(args, "artifacts/youth_natural/training_runs/test-merge-not-run"),
        command="test-merge",
        reason="No real adapter was supplied for floating-point merge parity in this run.",
        config=config,
    )
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YouthNaturalLoRA ZONOS2 research workflow")
    parser.add_argument("--root", default=None, help="Repository root override")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("audit").set_defaults(func=cmd_audit)
    sub.add_parser("validate-orchestration").set_defaults(func=cmd_validate)
    sub.add_parser("emit-schemas").set_defaults(func=cmd_emit_schemas)

    prepare = sub.add_parser("prepare-data")
    prepare.add_argument("--config", required=True)
    prepare.set_defaults(func=cmd_prepare_data)

    overfit = sub.add_parser("overfit")
    overfit.add_argument("--config", required=True)
    overfit.set_defaults(func=cmd_overfit)

    train = sub.add_parser("train")
    train.add_argument("--stage", required=True, choices=["youth", "natural"])
    train.add_argument("--config", required=True)
    train.add_argument("--resume", default=None)
    train.set_defaults(func=cmd_train)

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--config", required=True)
    evaluate.set_defaults(func=cmd_evaluate)

    export = sub.add_parser("export-adapter")
    export.add_argument("checkpoint")
    export.set_defaults(func=cmd_export_adapter)

    merge = sub.add_parser("test-merge")
    merge.add_argument("adapter")
    merge.set_defaults(func=cmd_test_merge)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))
