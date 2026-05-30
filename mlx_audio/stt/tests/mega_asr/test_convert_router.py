import json
import os
from pathlib import Path

import mlx.core as mx
import pytest

FIX = Path(__file__).parent / "fixtures"
SOURCE_KEYS = json.loads((FIX / "router_state_dict_keys.json").read_text())
DROPPED = {k for k in SOURCE_KEYS if k.endswith("num_batches_tracked")}
EXPECTED_KEYS = set(SOURCE_KEYS) - DROPPED


def _router_safetensors():
    if not os.environ.get("MEGA_ASR_RUN_HF_TESTS"):
        pytest.skip("set MEGA_ASR_RUN_HF_TESTS=1 to run HuggingFace-download tests")
    hf = pytest.importorskip("huggingface_hub")
    try:
        return hf.hf_hub_download(
            "zhifeixie/Mega-ASR",
            "audio_quality_router/best_acc_model.safetensors",
        )
    except Exception as exc:
        pytest.skip(f"router weights unavailable (offline?): {exc}")


@pytest.mark.requires_weights
def test_router_weights_remap():
    from mlx_audio.stt.models.mega_asr.convert_router import convert_router_weights
    from mlx_audio.stt.models.mega_asr.router import AudioQualityRouter

    out = convert_router_weights(_router_safetensors())

    assert out
    assert all(isinstance(v, mx.array) for v in out.values())
    assert not any(k.endswith("num_batches_tracked") for k in out)
    assert set(out) == EXPECTED_KEYS
    assert len(out) == 33

    conv_kernels = {
        k: v
        for k, v in out.items()
        if "conv" in k and k.endswith("weight") and v.ndim == 3
    }
    assert set(conv_kernels) == {"frontend.conv.0.weight", "frontend.conv.4.weight"}
    assert tuple(conv_kernels["frontend.conv.0.weight"].shape) == (128, 3, 80)
    assert tuple(conv_kernels["frontend.conv.4.weight"].shape) == (256, 3, 128)

    assert tuple(out["frontend.conv.1.weight"].shape) == (128,)
    assert tuple(out["frontend.conv.5.weight"].shape) == (256,)

    assert tuple(out["transformer.layers.0.self_attn.in_proj_weight"].shape) == (
        768,
        256,
    )
    assert tuple(out["pos_encoder.pe"].shape) == (1, 850, 256)

    router = AudioQualityRouter.from_converted(out)
    logits = router.logits(mx.zeros((16000,), dtype=mx.float32))
    assert tuple(logits.shape) == (2,)
