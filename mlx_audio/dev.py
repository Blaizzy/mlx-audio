"""Launch MLX-Audio API server and web UI dev server."""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Optional, Sequence


def _iter_ui_candidates(ui_dir: Optional[str]) -> Iterable[Path]:
    if ui_dir:
        yield Path(ui_dir)

    env_dir = os.getenv("MLX_AUDIO_UI_DIR")
    if env_dir:
        yield Path(env_dir)

    cwd = Path.cwd()
    yield cwd / "mlx_audio" / "ui"
    yield cwd / "ui"

    here = Path(__file__).resolve().parent
    yield here / "ui"
    yield here.parent / "mlx_audio" / "ui"


def _resolve_ui_dir(ui_dir: Optional[str]) -> Path:
    seen = set()
    for candidate in _iter_ui_candidates(ui_dir):
        if not candidate:
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "package.json").exists():
            return candidate

    raise FileNotFoundError(
        "Could not find the UI directory. Set MLX_AUDIO_UI_DIR or pass --ui-dir."
    )


def _require_npm() -> str:
    npm = shutil.which("npm")
    if npm is None:
        raise RuntimeError("npm is required to run the web UI. Install Node.js first.")
    return npm


def _spawn(cmd: Sequence[str], cwd: Optional[Path] = None) -> subprocess.Popen:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    return subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        start_new_session=True,
    )


def _terminate(proc: subprocess.Popen, name: str) -> None:
    if proc.poll() is not None:
        return

    try:
        if os.name != "nt":
            os.killpg(proc.pid, signal.SIGTERM)
        else:
            proc.terminate()
    except ProcessLookupError:
        return

    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run MLX-Audio API server and web UI dev server together."
    )
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument(
        "--ui-dir",
        default=None,
        help="Path to the UI directory (defaults to mlx_audio/ui)",
    )
    parser.add_argument(
        "--skip-npm-install",
        action="store_true",
        help="Skip running npm install",
    )

    args = parser.parse_args()

    ui_dir = _resolve_ui_dir(args.ui_dir)
    npm = _require_npm()

    server_cmd = [
        sys.executable,
        "-m",
        "mlx_audio.server",
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]

    server_proc = _spawn(server_cmd)

    try:
        if not args.skip_npm_install:
            subprocess.run([npm, "install"], cwd=ui_dir, check=True)
        ui_proc = _spawn([npm, "run", "dev"], cwd=ui_dir)
    except Exception:
        _terminate(server_proc, "server")
        raise

    print(
        f"Server running on http://{args.host}:{args.port} (API). "
        "UI dev server starting on http://localhost:3000"
    )
    print("Press Ctrl+C to stop.")

    exit_code = 0
    try:
        while True:
            server_ret = server_proc.poll()
            ui_ret = ui_proc.poll()
            if server_ret is not None:
                print(f"Server exited with code {server_ret}.")
                exit_code = server_ret or 1
                break
            if ui_ret is not None:
                print(f"UI dev server exited with code {ui_ret}.")
                exit_code = ui_ret or 1
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        exit_code = 130
    finally:
        _terminate(ui_proc, "ui")
        _terminate(server_proc, "server")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
