#!/usr/bin/env bash
set -euo pipefail
REPO="mlx-community/Kokoro-82M"
OUT_DIR="./kokoro_voices"
voices=(
  af_alloy af_aoede af_bella af_heart af_jessica af_kore af_nicole af_nova af_river af_sarah af_sky
  am_adam am_echo am_eric am_fenrir am_liam am_michael am_onyx am_puck am_santa
  bf_alice bf_emma bf_isabella bf_lily
  bm_daniel bm_fable bm_george bm_lewis
)
echo "Downloading voices into \$OUT_DIR ..."
for v in "${voices[@]}"; do
  echo "  • \$v"
  huggingface-cli download "$REPO" "voices/\${v}.pt" --local-dir "\$OUT_DIR" \
             --local-dir-use-symlinks False --resume-download
done
echo "✅  Done."
