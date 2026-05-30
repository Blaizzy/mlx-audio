"""End-to-end test for granite_speech_nar via the mlx-audio loader.

Loads the model through the canonical mlx-audio path (load_model -> ModelConfig
-> Model -> sanitize -> load_weights -> post_load_hook), runs generate() on the
example wav, and asserts the transcript is similar to the upstream PyTorch
reference. A similarity floor (rather than exact equality) accommodates the
small CTC-boundary drift that arises across Metal hardware generations and
across kernel-reduction orderings.
"""

from __future__ import annotations

import difflib
import os

import pytest

from mlx_audio.stt.utils import load_model

REPO = os.environ.get(
    "GRANITE_NAR_MLX_REPO",
    "mlx-community/granite-speech-4.1-2b-nar-mlx",
)
WAV_FILENAME = os.environ.get("GRANITE_NAR_TEST_WAV", "multilingual_sample.wav")

# Multilingual (English + French) reference transcript produced by the upstream
# PyTorch model on the same wav. Exercises the model's cross-lingual code-switching
# behavior in a single utterance — a stronger smoke test than English-only.
GOLD_TRANSCRIPT = (
    "for timothy was a spoiled cat, and he allowed no one to interfere. "
    "everybody waited upon him, moving their chairs even, for he was monarch "
    "of the hearth. dinarzade, la nuit suivante appela sa soeur quand il en "
    'fut temps. " si vous ne dormez pas, ma soeur, lui dit-elle, je vous '
    "prie, en attendant le jour qui paraîtra bientôt, de continuer le conte "
    "du pêcheur."
)

# Single-pass NAR decoding sits on CTC argmax boundaries; small reduction-order
# differences in Metal SDPA across M-series generations can flip a handful of
# boundary characters. 0.98 catches real regressions (wrong language, word order,
# missing phrases) while tolerating a few char-level CTC flips.
SIMILARITY_THRESHOLD = 0.98


def test_generate_matches_upstream_transcript():
    """Loads the model via mlx-audio's canonical path and asserts transcript similarity."""
    try:
        from huggingface_hub import hf_hub_download

        # load_model handles model weights/config/tokenizer via snapshot_download,
        # but DEFAULT_ALLOW_PATTERNS excludes audio — fetch the sample wav directly.
        wav_path = hf_hub_download(REPO, WAV_FILENAME)
        model = load_model(REPO)
    except Exception as e:
        pytest.skip(f"could not fetch {REPO} from HF (offline or unavailable): {e}")

    out = model.generate(wav_path)
    similarity = difflib.SequenceMatcher(None, GOLD_TRANSCRIPT, out.text).ratio()
    assert similarity >= SIMILARITY_THRESHOLD, (
        f"\n  similarity: {similarity:.4f} (threshold: {SIMILARITY_THRESHOLD})"
        f"\n  expected: {GOLD_TRANSCRIPT!r}"
        f"\n  got: {out.text!r}"
    )
