"""Tests for optional dependency groups in pyproject.toml.

These tests verify that pyproject.toml optional dependency groups
are correctly defined and can be resolved by package managers.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

# Find project root (where pyproject.toml lives)
PROJECT_ROOT = Path(__file__).parent.parent.parent


def get_package_manager() -> str:
    """Detect available package manager (uv preferred, fallback to pip)."""
    if shutil.which("uv"):
        return "uv"
    if shutil.which("pip"):
        return "pip"
    pytest.skip("No package manager (uv or pip) available")


def run_dry_run(extra: str = None) -> subprocess.CompletedProcess:
    """Run package manager dry-run for optional extra."""
    pm = get_package_manager()
    pkg = f".[{extra}]" if extra else "."

    if pm == "uv":
        cmd = ["uv", "pip", "install", "--dry-run", pkg]
    else:
        cmd = ["pip", "install", "--dry-run", pkg]

    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )


class TestOptionalDeps:
    """Test that optional dependency groups resolve correctly."""

    def test_core_deps_defined(self):
        """Verify core dependencies are minimal."""
        import tomllib

        with open(PROJECT_ROOT / "pyproject.toml", "rb") as f:
            config = tomllib.load(f)

        deps = config["project"]["dependencies"]
        assert len(deps) == 4, f"Core should have 4 deps, got {len(deps)}: {deps}"
        dep_names = [d.split(">=")[0].split("==")[0] for d in deps]
        assert "mlx" in dep_names
        assert "numpy" in dep_names
        assert "huggingface_hub" in dep_names
        assert "transformers" in dep_names

    def test_stt_extra_defined(self):
        """Verify [stt] extra contains expected deps."""
        import tomllib

        with open(PROJECT_ROOT / "pyproject.toml", "rb") as f:
            config = tomllib.load(f)

        stt_deps = config["project"]["optional-dependencies"]["stt"]
        dep_names = [d.split(">=")[0].split("==")[0] for d in stt_deps]
        # Note: transformers moved to core deps
        assert "tiktoken" in dep_names
        assert "tqdm" in dep_names

    def test_tts_extra_defined(self):
        """Verify [tts] extra contains expected deps."""
        import tomllib

        with open(PROJECT_ROOT / "pyproject.toml", "rb") as f:
            config = tomllib.load(f)

        tts_deps = config["project"]["optional-dependencies"]["tts"]
        dep_names = [d.split(">=")[0].split("==")[0].split("[")[0] for d in tts_deps]
        assert "misaki" in dep_names
        assert "spacy" in dep_names

    def test_sts_extra_uses_self_reference(self):
        """Verify [sts] extra uses self-referencing for DRY deps."""
        import tomllib

        with open(PROJECT_ROOT / "pyproject.toml", "rb") as f:
            config = tomllib.load(f)

        sts_deps = config["project"]["optional-dependencies"]["sts"]
        # Self-references like mlx-audio[stt,tts] are valid pip features for DRY
        has_self_ref = any(dep.startswith("mlx-audio[") for dep in sts_deps)
        assert has_self_ref, "STS should use self-reference for DRY deps"

    def test_all_extra_uses_self_reference(self):
        """Verify [all] extra uses self-referencing for DRY deps."""
        import tomllib

        with open(PROJECT_ROOT / "pyproject.toml", "rb") as f:
            config = tomllib.load(f)

        all_deps = config["project"]["optional-dependencies"]["all"]
        # Self-references are valid pip features for DRY
        has_self_ref = any(dep.startswith("mlx-audio[") for dep in all_deps)
        assert has_self_ref, "All should use self-reference for DRY deps"

    def test_server_extra_defined(self):
        """Verify [server] extra contains expected deps."""
        import tomllib

        with open(PROJECT_ROOT / "pyproject.toml", "rb") as f:
            config = tomllib.load(f)

        server_deps = config["project"]["optional-dependencies"]["server"]
        dep_names = [d.split(">=")[0] for d in server_deps]
        assert "fastapi" in dep_names
        assert "uvicorn" in dep_names

    def test_dev_extra_defined(self):
        """Verify [dev] extra contains expected deps."""
        import tomllib

        with open(PROJECT_ROOT / "pyproject.toml", "rb") as f:
            config = tomllib.load(f)

        dev_deps = config["project"]["optional-dependencies"]["dev"]
        dep_names = [d.split(">=")[0] for d in dev_deps]
        assert "pytest" in dep_names

    def test_core_resolves(self):
        """Verify core install resolves without errors."""
        result = run_dry_run()
        assert result.returncode == 0, f"Core resolve failed: {result.stderr}"

    def test_stt_extra_resolves(self):
        """Verify [stt] extra resolves without errors."""
        result = run_dry_run("stt")
        assert result.returncode == 0, f"STT resolve failed: {result.stderr}"

    def test_tts_extra_resolves(self):
        """Verify [tts] extra resolves without errors."""
        result = run_dry_run("tts")
        assert result.returncode == 0, f"TTS resolve failed: {result.stderr}"

    def test_sts_extra_resolves(self):
        """Verify [sts] extra resolves without errors."""
        result = run_dry_run("sts")
        assert result.returncode == 0, f"STS resolve failed: {result.stderr}"

    def test_server_extra_resolves(self):
        """Verify [server] extra resolves without errors."""
        result = run_dry_run("server")
        assert result.returncode == 0, f"Server resolve failed: {result.stderr}"

    def test_all_extra_resolves(self):
        """Verify [all] extra resolves without errors."""
        result = run_dry_run("all")
        assert result.returncode == 0, f"All resolve failed: {result.stderr}"

    def test_dev_extra_resolves(self):
        """Verify [dev] extra resolves without errors."""
        result = run_dry_run("dev")
        assert result.returncode == 0, f"Dev resolve failed: {result.stderr}"
