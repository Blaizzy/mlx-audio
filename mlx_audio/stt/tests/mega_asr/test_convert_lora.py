import os
from pathlib import Path

import mlx.core as mx
import pytest

REPO_ID = "zhifeixie/Mega-ASR"
CONFIG_FILE = "mega-asr-merged/adapter_config.json"
MODEL_FILE = "mega-asr-merged/adapter_model.safetensors"


def _merged_dir() -> Path:
    if not os.environ.get("MEGA_ASR_RUN_HF_TESTS"):
        pytest.skip("set MEGA_ASR_RUN_HF_TESTS=1 to run HuggingFace-download tests")
    hf = pytest.importorskip("huggingface_hub")
    try:
        config_path = hf.hf_hub_download(REPO_ID, CONFIG_FILE)
        hf.hf_hub_download(REPO_ID, MODEL_FILE)
    except Exception as exc:
        pytest.skip(f"Mega-ASR adapter unavailable ({exc!r})")
    return Path(config_path).parent


@pytest.mark.requires_weights
def test_lora_pairs_and_scaling():
    from mlx_audio.stt.models.mega_asr.convert_lora import load_lora_adapter

    adapter = load_lora_adapter(_merged_dir())

    assert len(adapter) == 539

    m = adapter["audio_tower.layers.0.self_attn.q_proj"]
    assert m["A"].shape[0] == m["B"].shape[1]
    assert m["A"].shape[0] == 16
    assert m["A"].dtype == mx.float32
    assert m["B"].dtype == mx.float32
    assert isinstance(m["scaling"], float) and m["scaling"] > 0

    assert any(k.startswith("audio_tower.") for k in adapter)
    assert any(k.startswith("model.layers.") for k in adapter)

    for entry in adapter.values():
        rank = entry["A"].shape[0]
        assert entry["B"].shape[1] == rank
        assert isinstance(entry["scaling"], float) and entry["scaling"] > 0
