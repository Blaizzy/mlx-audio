"""Tests for optional dependency groups.

These tests verify that optional dependency groups are correctly defined
and can be resolved by package managers using importlib.metadata.
"""

import shutil
import subprocess
from importlib.metadata import PackageNotFoundError, metadata
from pathlib import Path

import pytest

# Find project root (where pyproject.toml lives)
PROJECT_ROOT = Path(__file__).parent.parent.parent


def get_package_metadata():
    """Get package metadata via importlib.metadata."""
    try:
        return metadata("mlx-audio")
    except PackageNotFoundError:
        pytest.skip("Package not installed. Run: pip install -e .")


def extract_package_name(req: str) -> str:
    """Extract package name from Requires-Dist string.

    Examples:
        'tiktoken>=0.9.0; extra == "stt"' -> 'tiktoken'
        'misaki[en]>=0.8.2; extra == "tts"' -> 'misaki'
    """
    import re

    # Match package name (alphanumeric, hyphens, underscores) before any version/extra specifier
    match = re.match(r"^([a-zA-Z0-9_-]+)", req)
    return match.group(1).lower() if match else req.lower()


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
        """Verify core dependencies are defined."""
        meta = get_package_metadata()
        requires = meta.get_all("Requires-Dist") or []
        # Filter core deps (no extra marker)
        core_deps = [r for r in requires if "extra ==" not in r]
        assert (
            len(core_deps) >= 3
        ), f"Core should have at least 3 deps, got {len(core_deps)}"

    def test_stt_extra_defined(self):
        """Verify [stt] extra contains expected deps."""
        meta = get_package_metadata()
        requires = meta.get_all("Requires-Dist") or []
        stt_deps = [r for r in requires if 'extra == "stt"' in r]
        dep_names = [extract_package_name(r) for r in stt_deps]
        assert "tiktoken" in dep_names, f"tiktoken not in stt deps: {dep_names}"

    def test_tts_extra_defined(self):
        """Verify [tts] extra contains expected deps."""
        meta = get_package_metadata()
        requires = meta.get_all("Requires-Dist") or []
        tts_deps = [r for r in requires if 'extra == "tts"' in r]
        dep_names = [extract_package_name(r) for r in tts_deps]
        assert "misaki" in dep_names, f"misaki not in tts deps: {dep_names}"

    def test_server_extra_defined(self):
        """Verify [server] extra contains expected deps."""
        meta = get_package_metadata()
        requires = meta.get_all("Requires-Dist") or []
        server_deps = [r for r in requires if 'extra == "server"' in r]
        dep_names = [extract_package_name(r) for r in server_deps]
        assert "fastapi" in dep_names, f"fastapi not in server deps: {dep_names}"
        assert "uvicorn" in dep_names, f"uvicorn not in server deps: {dep_names}"

    def test_dev_extra_defined(self):
        """Verify [dev] extra contains expected deps."""
        meta = get_package_metadata()
        requires = meta.get_all("Requires-Dist") or []
        dev_deps = [r for r in requires if 'extra == "dev"' in r]
        dep_names = [extract_package_name(r) for r in dev_deps]
        assert "pytest" in dep_names, f"pytest not in dev deps: {dep_names}"

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
