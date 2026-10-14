#!/bin/zsh
# Drama 3 test UI: http://127.0.0.1:7860
# Usage: examples/drama3_ui.sh [model dir or Hub id ...]
#   default: cguldogan/drama-3-mlx-q4 and -q8 (downloaded on first use)
#   DRAMA3_VOICE=ref.wav DRAMA3_VOICE_TEXT="transcript" preloads a reference voice.
set -e
cd "${0:A:h}/.."
if (( $# == 0 )); then
  set -- cguldogan/drama-3-mlx-q4 cguldogan/drama-3-mlx-q8
fi
VOICE_ARGS=()
if [[ -n "$DRAMA3_VOICE" ]]; then
  VOICE_ARGS=(--voice "$DRAMA3_VOICE" "${DRAMA3_VOICE_TEXT:?set DRAMA3_VOICE_TEXT to the clip transcript}")
fi
# arch -arm64: MLX needs native arm64 Python even when the shell runs under Rosetta.
exec arch -arm64 .venv/bin/python -m mlx_audio.tts.models.fish_omni.app --models "$@" "${VOICE_ARGS[@]}"
