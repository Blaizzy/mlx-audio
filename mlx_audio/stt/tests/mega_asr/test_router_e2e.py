from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from mlx_audio.stt.models.mega_asr.convert_router import convert_router_weights
from mlx_audio.stt.utils import load_audio

FIX = Path(__file__).parent / "fixtures"
REFERENCE = json.loads((FIX / "reference.json").read_text())


def _router_weights():
    if not os.environ.get("MEGA_ASR_RUN_HF_TESTS"):
        pytest.skip("set MEGA_ASR_RUN_HF_TESTS=1 to run HuggingFace-download tests")
    hf = pytest.importorskip("huggingface_hub")
    try:
        path = hf.hf_hub_download(
            "zhifeixie/Mega-ASR",
            "audio_quality_router/best_acc_model.safetensors",
        )
    except Exception as exc:
        pytest.skip(f"router weights unavailable (offline?): {exc}")
    return convert_router_weights(path)


@pytest.mark.requires_weights
def test_router_logits_and_decision_match_reference():
    from mlx_audio.stt.models.mega_asr.router import AudioQualityRouter

    router = AudioQualityRouter.from_converted(_router_weights())

    for clip in ("clean", "degraded"):
        waveform = load_audio(str(FIX / f"{clip}.wav"))

        logits = np.array(router.logits(waveform))
        route = router.route(waveform)

        assert logits.shape == (2,)
        assert np.allclose(logits, REFERENCE[f"router_logits_{clip}"], atol=2e-3)
        assert route["use_lora"] is REFERENCE[clip]["use_lora"]
