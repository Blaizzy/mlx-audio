#!/bin/bash
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This launcher is intended for macOS."
  read -r -p "Press Enter to close..." _
  exit 1
fi

if ! command -v uvx >/dev/null 2>&1; then
  echo "uvx is not installed. Install uv first: brew install uv"
  read -r -p "Press Enter to close..." _
  exit 1
fi

if command -v brew >/dev/null 2>&1; then
  brew_prefix=$(brew --prefix 2>/dev/null || true)
  if [[ -n "$brew_prefix" && "$brew_prefix" != "/opt/homebrew" ]]; then
    echo "Warning: Homebrew prefix is '$brew_prefix'."
    echo "On Apple Silicon, use Homebrew at /opt/homebrew (not the Intel /usr/local)."
    echo "This may cause missing deps if you are on Apple Silicon."
    echo
  fi
fi

echo "Starting MLX-Audio API server and UI dev server..."
uvx --from "mlx-audio[app]" mlx_audio.dev

echo
read -r -p "Press Enter to close..." _
