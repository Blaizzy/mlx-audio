"""Test mega_asr weight-loading wiring (Task 4.2)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from safetensors.mlx import save_file

from mlx_audio.stt.models.mega_asr import MegaASRConfig, Model
from mlx_audio.stt.models.mega_asr.router import AudioQualityRouter
from mlx_audio.stt.models.qwen3_asr.qwen3_asr import Qwen3ASRModel


def _tiny_router_weights(tmp: Path) -> Path:
    extras = tmp / "extras"
    extras.mkdir(exist_ok=True)
    dest = extras / "router.safetensors"
    d_model, hidden, nhead = 32, 16, 2
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
        str(dest),
    )
    return dest


def _tiny_lora_adapter(tmp: Path) -> Path:
    lora_dir = tmp / "lora_adapter"
    lora_dir.mkdir()
    r, in_dim, out_dim = 4, 64, 128

    (lora_dir / "adapter_config.json").write_text(
        json.dumps({"r": r, "lora_alpha": r, "rank_pattern": {}, "alpha_pattern": {}})
    )
    save_file(
        {
            "base_model.model.thinker.audio_tower.layers.0.self_attn.q_proj.lora_A.weight": mx.random.normal(
                (r, in_dim)
            ),
            "base_model.model.thinker.audio_tower.layers.0.self_attn.q_proj.lora_B.weight": mx.random.normal(
                (out_dim, r)
            ),
        },
        str(lora_dir / "adapter_model.safetensors"),
    )
    return lora_dir


def _tiny_lora_weights(tmp: Path) -> Path:
    extras = tmp / "extras"
    extras.mkdir(exist_ok=True)
    dest = extras / "lora.safetensors"
    save_file(
        {
            "audio_tower.layers.0.self_attn.q_proj.lora_A": mx.random.normal(
                (4, 64)
            ).astype(mx.float32),
            "audio_tower.layers.0.self_attn.q_proj.lora_B": mx.random.normal(
                (128, 4)
            ).astype(mx.float32),
        },
        str(dest),
    )
    return dest


def _tiny_base_weights(tmp: Path) -> Path:
    dest = tmp / "model.safetensors"
    save_file(
        {"audio_tower.proj1.weight": mx.random.normal((2048, 1024))},
        str(dest),
    )
    return dest


def _mega_config(tmp: Path) -> Path:
    cfg = {
        "model_type": "mega_asr",
        "model_repo": "test/mega-asr",
        "audio_token_id": 151676,
        "audio_start_token_id": 151669,
        "audio_end_token_id": 151670,
        "support_languages": [],
        "audio_config": {
            "num_mel_bins": 128,
            "encoder_layers": 24,
            "encoder_attention_heads": 16,
            "encoder_ffn_dim": 4096,
            "d_model": 1024,
            "dropout": 0.0,
            "attention_dropout": 0.0,
            "activation_function": "gelu",
            "activation_dropout": 0.0,
            "scale_embedding": False,
            "initializer_range": 0.02,
            "max_source_positions": 1500,
            "n_window": 50,
            "output_dim": 2048,
            "n_window_infer": 800,
            "conv_chunksize": 500,
            "downsample_hidden_size": 480,
        },
        "text_config": {
            "model_type": "qwen3",
            "vocab_size": 151936,
            "hidden_size": 2048,
            "intermediate_size": 6144,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "hidden_act": "silu",
            "max_position_embeddings": 65536,
            "initializer_range": 0.02,
            "rms_norm_eps": 1e-6,
            "use_cache": True,
            "tie_word_embeddings": True,
            "rope_theta": 1000000.0,
            "attention_bias": False,
            "attention_dropout": 0.0,
        },
        "router_config": {
            "d_model": 32,
            "nhead": 2,
            "dim_feedforward": 16,
            "num_layers": 1,
            "n_mels": 80,
            "frontend_hidden_dim": 16,
            "classifier_hidden_dim": 16,
            "max_len": 100,
        },
        "lora_config": {"r": 4, "lora_alpha": 4},
        "router_weights": "extras/router.safetensors",
        "lora_weights": "extras/lora.safetensors",
    }
    p = tmp / "config.json"
    with open(p, "w") as f:
        json.dump(cfg, f)
    return p


class TestWeightLoadingWiring:
    def test_model_init_creates_asr_router_deltas(self):
        cfg = MegaASRConfig(
            model_type="mega_asr",
            router_config={"d_model": 32, "num_layers": 1, "nhead": 2},
            lora_config={"r": 4, "lora_alpha": 4},
            router_weights="router.safetensors",
            lora_weights="lora.safetensors",
        )
        model = Model(cfg)
        assert isinstance(model._asr, Qwen3ASRModel)
        assert isinstance(model._router, AudioQualityRouter)
        assert model._deltas == {}
        assert model._lora_active is False

    def test_sanitize_delegates_to_qwen3asr(self):
        raw = {
            "thinker.audio_tower.layers.0.self_attn.q_proj.weight": mx.random.normal(
                (2048, 1024)
            ),
            "lm_head.weight": mx.random.normal((151936, 2048)),
        }
        out = Model.sanitize(raw)
        assert "lm_head.weight" not in out
        assert "audio_tower.layers.0.self_attn.q_proj.weight" in out

    def test_model_quant_predicate_keeps_audio_tower_unquantized(self):
        cfg = MegaASRConfig(
            model_type="mega_asr",
            router_config={"d_model": 32, "num_layers": 1, "nhead": 2},
        )
        model = Model(cfg)
        from mlx_audio.stt.models.qwen3_asr.qwen3_asr import AudioEncoder

        assert cfg.audio_config is not None
        audio_enc = AudioEncoder(cfg.audio_config)
        assert model.model_quant_predicate("audio_tower.layers", audio_enc) is False
        assert model.model_quant_predicate("model.layers.0", audio_enc) is True

    def test_post_load_hook_wires_router_and_lora(self, tmp_path, monkeypatch):
        _tiny_router_weights(tmp_path)
        _tiny_lora_weights(tmp_path)

        cfg = MegaASRConfig(
            model_type="mega_asr",
            router_weights="extras/router.safetensors",
            lora_weights="extras/lora.safetensors",
        )
        model = Model(cfg)

        monkeypatch.setattr(
            Qwen3ASRModel,
            "post_load_hook",
            classmethod(lambda cls, inner_model, model_path: inner_model),
        )

        model = Model.post_load_hook(model, tmp_path)

        assert isinstance(model._asr, Qwen3ASRModel)
        assert isinstance(model._router, AudioQualityRouter)
        assert model._deltas, "deltas must be populated"
        for path, module in model._deltas.items():
            assert set(module) == {
                "A",
                "B",
                "scaling",
            }, f"module {path!r} must keep LoRA factors"
            assert module["A"].ndim == 2, f"module {path!r} A must be 2-D"
            assert module["B"].ndim == 2, f"module {path!r} B must be 2-D"
            assert isinstance(module["scaling"], float)

    def test_load_weights_glob_excludes_router_and_lora(self, tmp_path):
        _tiny_base_weights(tmp_path)
        _tiny_router_weights(tmp_path)
        _tiny_lora_weights(tmp_path)

        from mlx_audio.utils import load_weights

        weights = load_weights(tmp_path)
        weight_keys = set(weights.keys())
        router_keys = {
            k
            for k in weight_keys
            if k.startswith("frontend.") or k.startswith("transformer.")
        }
        lora_keys = {k for k in weight_keys if "lora_" in k}
        assert not router_keys, f"router keys should not be loaded: {router_keys}"
        assert not lora_keys, f"lora keys should not be loaded: {lora_keys}"
        assert "audio_tower.proj1.weight" in weights

    def test_load_lora_adapter_preserves_paths_and_factor_shapes(self, tmp_path):
        lora_dir = _tiny_lora_adapter(tmp_path)

        from mlx_audio.stt.models.mega_asr.convert_lora import load_lora_adapter

        adapter = load_lora_adapter(lora_dir)

        assert set(adapter.keys()) == {"audio_tower.layers.0.self_attn.q_proj"}
        module = adapter["audio_tower.layers.0.self_attn.q_proj"]
        assert module["A"].shape == (4, 64)
        assert module["B"].shape == (128, 4)
        assert module["scaling"] == 1.0

    def test_load_lora_factors_round_trips_flat_mlx_weights(self, tmp_path):
        from mlx_audio.stt.models.mega_asr.convert_lora import load_lora_factors
        from mlx_audio.stt.models.mega_asr.lora import materialize_delta

        path = tmp_path / "lora.safetensors"
        a = mx.arange(12, dtype=mx.float32).reshape(3, 4)
        b = mx.arange(15, dtype=mx.float32).reshape(5, 3)
        scaling = 1.5
        save_file(
            {
                "model.layers.0.self_attn.q_proj.lora_A": a,
                "model.layers.0.self_attn.q_proj.lora_B": (scaling * b).astype(
                    mx.float32
                ),
            },
            str(path),
        )

        adapter = load_lora_factors(path)

        assert set(adapter.keys()) == {"model.layers.0.self_attn.q_proj"}
        module = adapter["model.layers.0.self_attn.q_proj"]
        assert module["A"].shape == (3, 4)
        assert module["B"].shape == (5, 3)
        assert module["scaling"] == 1.0
        expected_delta = np.array((scaling * b) @ a)
        actual_delta = np.array(materialize_delta(module))
        assert np.allclose(actual_delta, expected_delta)
