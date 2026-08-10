"""mlx-lm is vendored under mlx_audio/lm; only the optional LLM responder may import it."""

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "mlx_audio"
ALLOWED = {"mlx_audio/sts/voice_pipeline.py"}
DYNAMIC = re.compile(r"""(?:importlib\.import_module|__import__)\(\s*["']mlx_lm""")


def _sources():
    for path in SOURCE.rglob("*.py"):
        rel = path.relative_to(ROOT).as_posix()
        if "/tests/" in rel or rel in ALLOWED:
            continue
        yield rel, path


def test_no_mlx_lm_imports_in_source():
    offenders = []
    for rel, path in _sources():
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(n == "mlx_lm" or n.startswith("mlx_lm.") for n in names):
                offenders.append(f"{rel}:{node.lineno}")
    assert not offenders, f"mlx_lm imported outside {sorted(ALLOWED)}: {offenders}"


def test_no_dynamic_mlx_lm_imports_in_source():
    offenders = [
        rel
        for rel, path in _sources()
        if DYNAMIC.search(path.read_text(encoding="utf-8"))
    ]
    assert not offenders, f"dynamic mlx_lm import in: {offenders}"


def test_mlx_lm_is_not_a_core_dependency():
    lines = (ROOT / "pyproject.toml").read_text(encoding="utf-8").splitlines()
    section = None
    core = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped
        elif (
            section == "[project]"
            and "mlx-lm" in stripped
            and not stripped.startswith("#")
        ):
            core.append(stripped)
    assert not core, f"mlx-lm must live in an optional extra, found: {core}"
