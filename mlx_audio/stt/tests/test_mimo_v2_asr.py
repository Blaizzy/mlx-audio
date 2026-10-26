import json
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten

from mlx_audio.stt import load
from mlx_audio.stt.models.mimo_v2_asr.asr import Model
from mlx_audio.stt.models.mimo_v2_asr.audio_encoder import (
    AudioEncoder,
    AudioEncoderConfig,
)
from mlx_audio.stt.models.mimo_v2_asr.config import MiMoAudioConfig


@pytest.fixture
def checkpoint(tmp_path):
    config = MiMoAudioConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        audio_channels=2,
        group_size=2,
        speech_vocab_size="8-8",
        speech_zeroemb_idx="7-7",
        delay_pattern=[0, 1],
        input_local_layers=1,
        input_local_dim=8,
        input_local_attn_heads=2,
        input_local_head_dim=4,
        input_local_intermediate_size=16,
        local_layers=1,
        local_dim=8,
        local_attn_heads=2,
        local_ffn_dim=16,
        out_hidden_size=8,
    )
    encoder_config = AudioEncoderConfig(
        n_mels=4,
        d_model=8,
        encoder_layers=1,
        encoder_skip_layer_id=None,
        encoder_attention_heads=2,
        encoder_ffn_dim=16,
        num_quantizers=2,
        codebook_size=[8, 8],
    )
    model = Model(config)
    assert model.audio_encoder is None
    main_weights = {
        key.removeprefix("model."): value
        for key, value in tree_flatten(model.parameters())
    }
    encoder_weights = dict(tree_flatten(AudioEncoder(encoder_config).parameters()))
    main = tmp_path / "checkpoint"
    encoder = tmp_path / "tokenizer"
    main.mkdir()
    encoder.mkdir()
    (main / "config.json").write_text(
        json.dumps({**asdict(config), "model_type": "mimo_v2_asr"})
    )
    (main / "mlx_manifest.json").write_text(
        json.dumps({"audio_tokenizer_dir": "../tokenizer"})
    )
    (encoder / "config.json").write_text(json.dumps(asdict(encoder_config)))
    mx.save_safetensors(str(main / "model.safetensors"), main_weights)
    mx.save_safetensors(str(encoder / "model.safetensors"), encoder_weights)
    return main, encoder, main_weights, encoder_weights


@pytest.mark.parametrize("strict", [False, True])
def test_complete_checkpoints_load_strictly(checkpoint, strict):
    main, _, _, _ = checkpoint
    calls = []
    original = nn.Module.load_weights

    def observed(module, weights, strict=True):
        calls.append((type(module).__name__, strict))
        return original(module, weights, strict=strict)

    with (
        patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=SimpleNamespace(eos_token_id=1, pad_token_id=0),
        ),
        patch.object(nn.Module, "load_weights", observed),
    ):
        model = load(main, lazy=True, strict=strict)
    assert model.audio_encoder is not None
    assert calls == [("Model", True), ("AudioEncoder", True)]


@pytest.mark.parametrize("stage", ["main", "encoder"])
@pytest.mark.parametrize("corruption", ["missing", "extra", "shape"])
def test_invalid_checkpoints_fail(checkpoint, stage, corruption):
    main, encoder, main_weights, encoder_weights = checkpoint
    path = main if stage == "main" else encoder
    weights = dict(main_weights if stage == "main" else encoder_weights)
    key = next(name for name, value in weights.items() if value.ndim == 2)
    if corruption == "missing":
        del weights[key]
    elif corruption == "extra":
        weights["unexpected.weight"] = mx.zeros((1,))
    else:
        weights[key] = mx.zeros((1,))
    mx.save_safetensors(str(path / "model.safetensors"), weights)
    with (
        patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=SimpleNamespace(eos_token_id=1, pad_token_id=0),
        ),
        pytest.raises(
            ValueError,
            match={"missing": "Missing", "extra": "not in model", "shape": "shape"}[
                corruption
            ],
        ),
    ):
        load(main, lazy=True)


def test_audio_encoder_sanitize_preserves_bias_free_keys():
    projection = mx.ones((8, 8))
    conv = mx.arange(192).reshape(8, 8, 3)
    weights = Model._sanitize_audio_encoder_weights(
        {
            "encoder.layers.0.self_attn.k_proj.weight": projection,
            "encoder.conv1.weight": conv,
            "encoder.quantizer.vq.layers.0._codebook.embed": mx.ones((8, 8)),
            "encoder.quantizer.vq.layers.0._codebook.cluster_size": mx.ones((8,)),
            "decoder.conv.weight": mx.ones((8, 3, 8)),
        }
    )
    assert set(weights) == {
        "layers.0.self_attn.k_proj.weight",
        "conv1.weight",
        "quantizer.vq.layers.0.codebook.embed",
    }
    assert weights["layers.0.self_attn.k_proj.weight"] is projection
    assert mx.array_equal(weights["conv1.weight"], conv.transpose(0, 2, 1))
