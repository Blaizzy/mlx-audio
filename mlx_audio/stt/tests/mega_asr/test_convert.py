from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import mlx.core as mx
import numpy as np
from safetensors.mlx import save_file

from mlx_audio.stt import load
from mlx_audio.stt.models.base import STTOutput
from mlx_audio.stt.models.mega_asr.mega_asr import Model as MegaASRModel


def _tiny_qwen3_config() -> dict[str, object]:
    return {
        "model_type": "qwen3_asr",
        "model_repo": "test/qwen3-asr-tiny",
        "audio_token_id": 151676,
        "audio_start_token_id": 151669,
        "audio_end_token_id": 151670,
        "support_languages": ["english"],
        "audio_config": {
            "num_mel_bins": 32,
            "encoder_layers": 1,
            "encoder_attention_heads": 2,
            "encoder_ffn_dim": 64,
            "d_model": 32,
            "max_source_positions": 64,
            "output_dim": 32,
            "downsample_hidden_size": 8,
            "n_window": 2,
            "n_window_infer": 8,
            "conv_chunksize": 8,
        },
        "text_config": {
            "model_type": "qwen3",
            "vocab_size": 64,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "tie_word_embeddings": True,
        },
    }


def _tiny_base_weights() -> dict[str, mx.array]:
    return {
        "audio_tower.proj1.weight": mx.random.normal((32, 32)),
        "model.embed_tokens.weight": mx.random.normal((64, 32)),
    }


def _write_router_weights(path: Path) -> None:
    d_model, hidden = 32, 16
    save_file(
        {
            "frontend.conv.0.weight": mx.random.normal((hidden, 80, 3)),
            "frontend.conv.0.bias": mx.zeros((hidden,)),
            "frontend.conv.1.weight": mx.ones((hidden,)),
            "frontend.conv.1.bias": mx.zeros((hidden,)),
            "frontend.conv.1.running_mean": mx.zeros((hidden,)),
            "frontend.conv.1.running_var": mx.ones((hidden,)),
            "frontend.conv.4.weight": mx.random.normal((d_model, hidden, 3)),
            "frontend.conv.4.bias": mx.zeros((d_model,)),
            "frontend.conv.5.weight": mx.ones((d_model,)),
            "frontend.conv.5.bias": mx.zeros((d_model,)),
            "frontend.conv.5.running_mean": mx.zeros((d_model,)),
            "frontend.conv.5.running_var": mx.ones((d_model,)),
            "pos_encoder.pe": mx.zeros((1, 100, d_model), dtype=mx.float32),
            "transformer.layers.0.self_attn.in_proj_weight": mx.random.normal(
                (d_model * 3, d_model)
            ),
            "transformer.layers.0.self_attn.in_proj_bias": mx.zeros((d_model * 3,)),
            "transformer.layers.0.self_attn.out_proj.weight": mx.random.normal(
                (d_model, d_model)
            ),
            "transformer.layers.0.self_attn.out_proj.bias": mx.zeros((d_model,)),
            "transformer.layers.0.linear1.weight": mx.random.normal((hidden, d_model)),
            "transformer.layers.0.linear1.bias": mx.zeros((hidden,)),
            "transformer.layers.0.linear2.weight": mx.random.normal((d_model, hidden)),
            "transformer.layers.0.linear2.bias": mx.zeros((d_model,)),
            "transformer.layers.0.norm1.weight": mx.ones((d_model,)),
            "transformer.layers.0.norm1.bias": mx.zeros((d_model,)),
            "transformer.layers.0.norm2.weight": mx.ones((d_model,)),
            "transformer.layers.0.norm2.bias": mx.zeros((d_model,)),
            "transformer.norm.weight": mx.ones((d_model,)),
            "transformer.norm.bias": mx.zeros((d_model,)),
            "pooling.query.weight": mx.random.normal((1, d_model)),
            "pooling.query.bias": mx.zeros((1,)),
            "classifier.0.weight": mx.random.normal((hidden, d_model)),
            "classifier.0.bias": mx.zeros((hidden,)),
            "classifier.3.weight": mx.random.normal((2, hidden)),
            "classifier.3.bias": mx.zeros((2,)),
        },
        str(path),
    )


def _write_lora_adapter(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "adapter_config.json").write_text(
        json.dumps({"r": 4, "lora_alpha": 4, "rank_pattern": {}, "alpha_pattern": {}})
    )
    save_file(
        {
            "base_model.model.thinker.audio_tower.layers.0.self_attn.q_proj.lora_A.weight": mx.random.normal(
                (4, 32)
            ),
            "base_model.model.thinker.audio_tower.layers.0.self_attn.q_proj.lora_B.weight": mx.random.normal(
                (32, 4)
            ),
        },
        str(path / "adapter_model.safetensors"),
    )


def _write_stub_hf_tree(root: Path) -> Path:
    base = root / "Qwen3-ASR-1.7B"
    router = root / "audio_quality_router"
    lora = root / "mega-asr-merged"
    base.mkdir(parents=True)
    router.mkdir(parents=True)

    (base / "config.json").write_text(json.dumps(_tiny_qwen3_config()))
    (base / "tokenizer_config.json").write_text(json.dumps({"model_type": "gpt2"}))
    (base / "vocab.json").write_text(json.dumps({"<|endoftext|>": 0, "hello": 1}))
    (base / "merges.txt").write_text("#version: 0.2\nh e\n")
    (base / "preprocessor_config.json").write_text(
        json.dumps(
            {
                "feature_extractor_type": "WhisperFeatureExtractor",
                "sampling_rate": 16000,
                "feature_size": 80,
                "n_fft": 400,
                "hop_length": 160,
                "chunk_length": 30,
            }
        )
    )
    _write_router_weights(router / "best_acc_model.safetensors")
    _write_lora_adapter(lora)
    return root


def test_convert_produces_loadable_dir(tmp_path, monkeypatch):
    from mlx_audio.stt.models.mega_asr.convert import convert
    from mlx_audio.stt.models.qwen3_asr.qwen3_asr import Qwen3ASRModel

    hf_dir = _write_stub_hf_tree(tmp_path / "hf")
    out_dir = tmp_path / "mlx"

    def fake_base_convert(
        hf_path: str, mlx_path: str, dtype: str = "bfloat16", **_: object
    ):
        assert Path(hf_path) == hf_dir / "Qwen3-ASR-1.7B"
        assert dtype == "bfloat16"
        dest = Path(mlx_path)
        dest.mkdir(parents=True, exist_ok=True)
        for name in [
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "preprocessor_config.json",
        ]:
            (dest / name).write_text((hf_dir / "Qwen3-ASR-1.7B" / name).read_text())
        (dest / "config.json").write_text(json.dumps(_tiny_qwen3_config()))
        save_file(_tiny_base_weights(), str(dest / "model.safetensors"))

    monkeypatch.setattr(
        "mlx_audio.stt.models.mega_asr.convert.base_convert.convert",
        fake_base_convert,
    )
    monkeypatch.setattr(
        Qwen3ASRModel,
        "post_load_hook",
        classmethod(lambda cls, model, model_path: model),
    )

    out = convert(hf_dir, out_dir)

    assert out == out_dir
    assert (out / "config.json").exists()
    assert list(out.glob("model*.safetensors"))
    assert (out / "extras" / "router.safetensors").exists()
    assert (out / "extras" / "lora.safetensors").exists()
    assert not (out / "extras" / "lora").exists()
    assert not list(out.rglob("adapter_config.json"))

    cfg = json.loads((out / "config.json").read_text())
    assert cfg["model_type"] == "mega_asr"
    assert cfg["router_weights"] == "extras/router.safetensors"
    assert cfg["lora_weights"] == "extras/lora.safetensors"
    assert cfg["router_config"] == {
        "d_model": 256,
        "nhead": 4,
        "dim_feedforward": 1024,
        "num_layers": 1,
        "n_mels": 80,
        "max_len": 850,
    }

    model = cast(MegaASRModel, cast(object, load(str(out))))
    assert type(model).__module__.endswith("mega_asr.mega_asr")
    assert hasattr(model, "_router")
    assert hasattr(model, "_deltas")
    assert model._deltas
    module = model._deltas["audio_tower.layers.0.self_attn.q_proj"]
    assert module["A"].shape == (4, 32)
    assert module["B"].shape == (32, 4)
    assert module["scaling"] == 1.0

    logits = model._router.logits(mx.zeros((16000,), dtype=mx.float32))
    assert tuple(logits.shape) == (2,)
    assert np.isfinite(np.array(logits)).all()
