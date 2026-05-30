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
        'mistral-common[audio]; extra == "tts"' -> 'mistral-common'
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

    def test_tts_extra_defined(self):
        """Verify [tts] extra contains expected deps."""
        meta = get_package_metadata()
        requires = meta.get_all("Requires-Dist") or []
        tts_deps = [r for r in requires if 'extra == "tts"' in r]
        dep_names = [extract_package_name(r) for r in tts_deps]
        assert (
            "misaki" not in dep_names
        ), f"misaki should not be a shared tts dep: {dep_names}"
        assert (
            "phonemizer-fork" not in dep_names
        ), f"phonemizer-fork should not be a shared tts dep: {dep_names}"

    def test_model_specific_text_deps_not_in_shared_extras(self):
        """Verify model-specific text deps are not pulled into shared extras."""
        meta = get_package_metadata()
        requires = meta.get_all("Requires-Dist") or []
        model_specific_text_deps = {
            "misaki",
            "num2words",
            "spacy",
            "espeakng-loader",
            "phonemizer-fork",
        }

        for extra in ("tts", "sts", "all"):
            extra_deps = [r for r in requires if f'extra == "{extra}"' in r]
            dep_names = {extract_package_name(r) for r in extra_deps}
            unexpected = model_specific_text_deps & dep_names
            assert not unexpected, f"{extra} extra still includes {unexpected}"

    def test_server_extra_defined(self):
        """Verify [server] extra contains expected deps."""
        meta = get_package_metadata()
        requires = meta.get_all("Requires-Dist") or []
        server_deps = [r for r in requires if 'extra == "server"' in r]
        dep_names = [extract_package_name(r) for r in server_deps]
        assert "fastapi" in dep_names, f"fastapi not in server deps: {dep_names}"
        assert "uvicorn" in dep_names, f"uvicorn not in server deps: {dep_names}"
        assert "webrtcvad" in dep_names, f"webrtcvad not in server deps: {dep_names}"
        assert "setuptools" in dep_names, f"setuptools not in server deps: {dep_names}"

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
