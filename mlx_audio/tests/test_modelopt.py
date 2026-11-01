"""Numerical and loader contracts for ModelOpt imports (no network access)."""

import json
import struct
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn
from mlx.utils import tree_flatten

from mlx_audio.lm.models.switch_layers import QuantizedSwitchLinear
from mlx_audio.quantization.modelopt import (
    _dequantize_fp8_weight,
    _e4m3_decode_table,
    prepare_modelopt_weights,
    transform_modelopt_weights,
)
from mlx_audio.quantization.nvfp4 import (
    ScaledQuantizedLinear,
    ScaledQuantizedSwitchLinear,
    replace_scaled_quantized_linears,
)


def nvfp4_source(prefix="layer"):
    return {
        prefix + ".weight": mx.full((4, 32), 0x77, mx.uint8),
        prefix + ".weight_scale": mx.array([[56, 64, 72, 80], [120] * 4] * 2, mx.uint8),
        prefix + ".weight_scale_2": mx.array(0.00012715657, mx.float32),
        prefix + ".input_scale": mx.array(0.25),
    }


def test_e4m3_all_bytes_match_torch():
    torch = pytest.importorskip("torch")
    expected = (
        torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float().numpy()
    )
    np.testing.assert_array_equal(np.array(_e4m3_decode_table()), expected)


@pytest.mark.parametrize(
    "shape,scale_shape",
    [
        ((4, 8), ()),
        ((4, 8), (1,)),
        ((4, 8), (4,)),
        ((4, 8), (4, 1)),
        ((3, 4, 8), (3,)),
        ((3, 4, 8), (3, 4)),
        ((3, 4, 8), (3, 4, 1)),
    ],
)
def test_fp8_tensor_channel_and_expert_scales(shape, scale_shape):
    weight = mx.full(shape, 68, mx.uint8)  # E4M3 3.0
    scale = mx.full(scale_shape, 0.25)
    decoded = _dequantize_fp8_weight(weight, scale)
    assert decoded.dtype == mx.bfloat16
    np.testing.assert_array_equal(
        np.array(decoded.astype(mx.float32)), np.full(shape, 0.75)
    )


@pytest.mark.parametrize(
    "algo", ["FP8", "NVFP4", "NVFP4_AWQ", "W4A16_NVFP4", "MIXED_PRECISION"]
)
def test_transform_modelopt_mixed_and_config_normalization(algo):
    weights = nvfp4_source() | {
        "attention.weight": mx.array([[56, 64], [68, 72]], mx.uint8),
        "attention.weight_scale": mx.array([0.5, 0.25], mx.bfloat16),
        "attention.input_scale": mx.array(0.5),
        "attention.k_scale": mx.array(0.125),
        "attention.v_scale": mx.array(0.125),
        "layer.bias": mx.zeros(4),
    }
    config = {"quantization_config": {"quant_method": "modelopt", "quant_algo": algo}}
    transformed = prepare_modelopt_weights(weights, config)
    assert not weights  # consumed tensors do not hold a second full checkpoint
    assert transformed["layer.weight"].dtype == mx.uint32
    assert transformed["layer.scales"].dtype == mx.uint8
    assert transformed["layer.weight_scale_2"].dtype == mx.float32
    assert transformed["attention.weight"].tolist() == [[0.5, 1.0], [0.75, 1.0]]
    assert not any(
        k.endswith((".input_scale", ".k_scale", ".v_scale", ".weight_scale"))
        for k in transformed
    )
    assert (
        config["quantization"]
        == config["quantization_config"]
        == {"group_size": 16, "bits": 4, "mode": "nvfp4"}
    )


def test_fp8_only_normalizes_to_dense_and_is_idempotent():
    weights = {
        "layer.weight": mx.full((4, 32), 56, mx.uint8),
        "layer.weight_scale": mx.array(0.125),
    }
    config = {
        "llm_config": {
            "quantization_config": {"quant_method": "modelopt", "quant_algo": "FP8"}
        }
    }
    converted = prepare_modelopt_weights(weights, config)
    assert converted["layer.weight"].dtype == mx.bfloat16
    assert not config["llm_config"]
    assert "quantization" not in config and "quantization_config" not in config
    assert prepare_modelopt_weights(converted, config) is converted


def test_hf_quant_config_fallback_and_native_skip(tmp_path):
    (tmp_path / "hf_quant_config.json").write_text(
        json.dumps(
            {"producer": {"name": "modelopt"}, "quantization": {"quant_algo": "NVFP4"}}
        )
    )
    config = {}
    weights = prepare_modelopt_weights(nvfp4_source(), config, tmp_path)
    assert config["quantization"]["mode"] == "nvfp4"
    assert prepare_modelopt_weights(weights, config, tmp_path) is weights
    dense = {"layer.weight": mx.ones((4, 32))}
    assert prepare_modelopt_weights(dense, {}, tmp_path) is dense


def test_non_modelopt_passes_through():
    weights = {"layer.weight": mx.ones((4, 32))}
    assert transform_modelopt_weights(weights, {"quant_method": "other"}) == (
        weights,
        None,
    )
    assert (
        prepare_modelopt_weights(weights, {"quantization_config": {"bits": 4}})
        is weights
    )


@pytest.mark.parametrize(
    "mutation,message",
    [
        (lambda w: w.pop("layer.weight"), "Missing ModelOpt NVFP4"),
        (
            lambda w: w.update({"layer.weight": mx.ones((4, 32))}),
            "Invalid ModelOpt NVFP4 tensors",
        ),
        (
            lambda w: w.update({"layer.weight_scale": mx.ones((4, 2), mx.uint8)}),
            "scale shape",
        ),
        (
            lambda w: w.update({"layer.weight_scale_2": mx.ones(2)}),
            "Invalid ModelOpt NVFP4 tensors",
        ),
    ],
)
def test_invalid_nvfp4_import_rejected(mutation, message):
    weights = nvfp4_source()
    mutation(weights)
    with pytest.raises(ValueError, match=message):
        transform_modelopt_weights(
            weights, {"quant_method": "modelopt", "quant_algo": "NVFP4"}
        )


def test_fp8_missing_weight_invalid_scale_and_unknown_algo():
    with pytest.raises(ValueError, match="Missing ModelOpt FP8 weight"):
        transform_modelopt_weights(
            {"x.weight_scale": mx.array(1.0)},
            {"quant_method": "modelopt", "quant_algo": "FP8"},
        )
    with pytest.raises(ValueError, match="scale shape"):
        _dequantize_fp8_weight(mx.zeros((4, 32), mx.uint8), mx.ones(3))
    with pytest.raises(ValueError, match="Unsupported ModelOpt"):
        transform_modelopt_weights(
            {}, {"quant_method": "modelopt", "quant_algo": "INT8"}
        )


def test_awq_nan_scales_zeroed(caplog):
    source = nvfp4_source()
    source["layer.weight_scale"] = mx.full((4, 4), 127, mx.uint8)
    converted, _ = transform_modelopt_weights(
        source, {"quant_method": "modelopt", "quant_algo": "NVFP4_AWQ"}
    )
    assert mx.all(converted["layer.scales"] == 0).item()
    assert "Zeroing 16 NaN NVFP4 AWQ block scales" in caplog.text


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
@pytest.mark.parametrize("bias", [False, True])
def test_scaled_nvfp4_linear_matches_dense_reference(dtype, bias):
    model = nn.Module()
    model.layer = nn.QuantizedLinear(64, 4, bias=bias, mode="nvfp4")
    weights, _ = transform_modelopt_weights(
        nvfp4_source(), {"quant_method": "modelopt", "quant_algo": "NVFP4"}
    )
    if bias:
        weights["layer.bias"] = mx.array([0.25, -0.5, 1, -2], dtype)
    replace_scaled_quantized_linears(model, weights)
    model.load_weights(list(weights.items()))
    x = mx.linspace(0.1, 1.7, 128).reshape(2, 64).astype(dtype)
    decoded = mx.repeat(mx.array([[1, 2, 4, 8], [256] * 4] * 2), 16, axis=-1) * 6
    decoded = decoded.astype(mx.float32) * weights["layer.weight_scale_2"]
    expected = (x.astype(mx.float32) @ decoded.T).astype(dtype)
    if bias:
        expected += weights["layer.bias"]
    actual = model.layer(x)
    assert actual.dtype == dtype
    # The native NVFP4 GPU matmul uses mixed precision internally, even for
    # FP32 inputs. Keep its rounding tolerance separate from scale decoding.
    np.testing.assert_allclose(
        np.array(actual.astype(mx.float32)),
        np.array(expected.astype(mx.float32)),
        atol=1e-5,
        rtol=1e-3 if dtype == mx.float32 else 1e-5,
    )
    assert not tree_flatten(model.trainable_parameters())


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("device", [mx.cpu, mx.gpu])
def test_scaled_nvfp4_experts_keep_per_expert_global_scale(dtype, device):
    with mx.stream(device):
        check_scaled_experts(dtype)


def check_scaled_experts(dtype):
    model = nn.Module()
    model.experts = QuantizedSwitchLinear(
        64, 4, 3, bias=False, group_size=16, mode="nvfp4"
    )
    weights = {
        "experts.weight": mx.full((3, 4, 8), 0x77777777, mx.uint32),
        "experts.scales": mx.full((3, 4, 4), 56, mx.uint8),
        "experts.weight_scale_2": mx.array([0.137, 0.348, 0.00012715657]),
    }
    replace_scaled_quantized_linears(model, weights)
    model.load_weights(list(weights.items()))
    assert isinstance(model.experts, ScaledQuantizedSwitchLinear)
    x = mx.linspace(0.1, 1.7, 128).reshape(2, 1, 1, 64).astype(dtype)
    indices = mx.array([[0, 2], [2, 1]])
    decoded = (
        mx.full((3, 4, 64), 6.0) * weights["experts.weight_scale_2"][:, None, None]
    )
    expected = mx.gather_mm(
        x.astype(mx.float32), decoded.swapaxes(-1, -2), rhs_indices=indices
    ).astype(dtype)
    actual = model.experts(x, indices)
    np.testing.assert_allclose(
        np.array(actual.astype(mx.float32)),
        np.array(expected.astype(mx.float32)),
        atol=2e-5,
        rtol=2e-5,
    )


@pytest.mark.parametrize("experts", [False, True])
@pytest.mark.parametrize("scale_name", ["weight_scale_2", "weight_global_scale"])
def test_nvfp4_awq_input_scales_and_global_scale_conventions(experts, scale_name):
    model = nn.Module()
    model.layer = (
        QuantizedSwitchLinear(64, 4, 3, bias=False, group_size=16, mode="nvfp4")
        if experts
        else nn.QuantizedLinear(64, 4, bias=False, mode="nvfp4")
    )
    shape = (3, 4, 8) if experts else (4, 8)
    pre_scale = mx.linspace(0.5, 1.5, 3 * 64 if experts else 64).reshape(
        (3, 64) if experts else (64,)
    )
    global_scale = (
        mx.array([0.137, 0.348, 0.00012715657]) if experts else mx.array(0.137)
    )
    weights = {
        "layer.weight": mx.full(shape, 0x77777777, mx.uint32),
        "layer.scales": mx.full((*shape[:-1], 4), 56, mx.uint8),
        "layer." + scale_name: global_scale,
        "layer.pre_quant_scale": pre_scale,
    }
    replace_scaled_quantized_linears(model, weights)
    model.load_weights(list(weights.items()))
    factor = 1 / global_scale if scale_name == "weight_global_scale" else global_scale
    decoded = (
        6 * pre_scale[..., None, :] * (factor[:, None, None] if experts else factor)
    )
    decoded = mx.broadcast_to(decoded, (*shape[:-1], 64))
    if experts:
        x = mx.linspace(0.1, 1.7, 128).reshape(2, 1, 1, 64)
        indices = mx.array([[0, 2], [2, 1]])
        expected = mx.gather_mm(x, decoded.swapaxes(-1, -2), rhs_indices=indices)
        actual = model.layer(x, indices)
    else:
        x = mx.linspace(0.1, 1.7, 128).reshape(2, 64)
        expected = x @ decoded.T
        actual = model.layer(x)
    np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=1e-3)
    np.testing.assert_array_equal(
        np.array(model.layer.to_linear().weight.astype(mx.float32)),
        np.array(decoded.astype(mx.bfloat16).astype(mx.float32)),
    )


class Tiny(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.Linear(64, 4, bias=False)

    def __call__(self, x):
        return self.layer(x)


@pytest.mark.parametrize("precision", ["FP8", "NVFP4"])
def test_modelopt_generic_load_and_native_save_roundtrip(
    tmp_path, monkeypatch, precision
):
    from mlx_audio import utils
    from mlx_audio.lm.convert import save_config, save_model

    monkeypatch.setattr(
        utils, "get_model_class", lambda **_: (SimpleNamespace(Model=Tiny), "tiny")
    )
    config = {
        "model_type": "tiny",
        "quantization_config": {"quant_method": "modelopt", "quant_algo": precision},
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    weights = (
        nvfp4_source()
        if precision == "NVFP4"
        else {
            "layer.weight": mx.full((4, 64), 56, mx.uint8),
            "layer.weight_scale": mx.array(0.125),
            "layer.input_scale": mx.array(0.25),
        }
    )
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    model = utils.base_load_model(
        tmp_path, category="stt", model_remapping={}, strict=True
    )
    assert isinstance(
        model.layer, ScaledQuantizedLinear if precision == "NVFP4" else nn.Linear
    )
    x = mx.ones((2, 64), mx.bfloat16)
    expected = model(x)
    converted = tmp_path / "converted"
    save_model(converted, model)
    save_config(model.config, converted / "config.json")
    restored = utils.base_load_model(
        converted, category="stt", model_remapping={}, strict=True
    )
    np.testing.assert_array_equal(
        np.array(restored(x).astype(mx.float32)), np.array(expected.astype(mx.float32))
    )


def test_mlx_load_reads_real_safetensors_fp8_tags(tmp_path):
    from mlx_audio.utils import load_weights

    data = bytes([56, 64, 68, 72])
    header = json.dumps(
        {"weight": {"dtype": "F8_E4M3", "shape": [2, 2], "data_offsets": [0, 4]}}
    ).encode()
    path = tmp_path / "model.safetensors"
    path.write_bytes(struct.pack("<Q", len(header)) + header + data)
    raw = load_weights(tmp_path)
    assert raw["weight"].dtype == mx.uint8
    np.testing.assert_array_equal(np.array(raw["weight"]), [[56, 64], [68, 72]])


def nemo_modelopt_fixture(path, precision):
    """Small checkpoint with the actual split-expert ModelOpt key layout."""
    from dataclasses import asdict

    from mlx_audio.stt.models.nemotron_transcribe import Model
    from mlx_audio.stt.tests.test_nemotron_transcribe import (
        tiny_config,
        tokenizer_fixture,
    )

    config = tiny_config(True)
    config.llm_config.update(
        hidden_size=64,
        intermediate_size=64,
        moe_intermediate_size=64,
        moe_shared_expert_intermediate_size=64,
        mamba_num_heads=8,
    )
    config.perception["output_dim"] = 64
    model = Model(config)
    model.eval()
    model.load_weights(
        [
            (k, v.astype(mx.bfloat16) if model.cast_predicate(k) else v)
            for k, v in tree_flatten(model.parameters())
        ]
    )
    source = {}

    def emit(key, value):
        if key.startswith(("llm.backbone.", "llm.lm_head.")):
            key = key.removeprefix("llm.")
        key = key.replace(".ffn.linear1.", ".ffn.net.0.").replace(
            ".ffn.linear2.", ".ffn.net.3."
        )
        key = key.replace(".preprocessor.", ".preprocessor.featurizer.")
        eligible = (
            key.endswith(".weight")
            and value.ndim == 2
            and not key.endswith(".gate.weight")
            and ".embeddings." not in key
            and key != "lm_head.weight"
            and not key.startswith("ctc_timestamp.")
        )
        if (
            eligible
            and precision == "NVFP4"
            and key.startswith("backbone.")
            and ".layers.0." not in key
        ):
            packed, scales = mx.quantize(
                value.astype(mx.float32), group_size=16, bits=4, mode="nvfp4"
            )
            source[key] = packed.view(mx.uint8)
            prefix = key.removesuffix(".weight")
            source[prefix + ".weight_scale"] = scales
            source[prefix + ".weight_scale_2"] = mx.array(
                0.137 + (len(source) % 3) * 0.1
            )
            source[prefix + ".input_scale"] = mx.array(0.25)
        elif eligible:
            # Constant E4M3 code 1.0 times a known scale isolates remapping.
            source[key] = mx.full(value.shape, 56, mx.uint8)
            prefix = key.removesuffix(".weight")
            source[prefix + ".weight_scale"] = mx.array(0.01)
            source[prefix + ".input_scale"] = mx.array(0.25)
        else:
            if key.endswith((".conv1d.weight", ".subpixel_upsample.weight")):
                value = value.transpose(0, 2, 1)
            elif key == "ctc_timestamp.decoder.upsample.weight":
                value = value.transpose(2, 0, 1)
            elif key == "ctc_timestamp.decoder.decoder_layers.0.weight":
                value = value[..., None]
            source[key] = value

    for key, value in tree_flatten(model.parameters()):
        if ".switch_mlp." in key:
            for i in range(value.shape[0]):
                source_key = key.replace(
                    ".switch_mlp.fc1.", f".experts.{i}.up_proj."
                ).replace(".switch_mlp.fc2.", f".experts.{i}.down_proj.")
                emit(source_key, value[i])
        else:
            emit(key, value)
    cfg = asdict(config)
    cfg.update(
        model_type="nemo_speechlm",
        torch_dtype="bfloat16",
        quantization_config={
            "quant_method": "modelopt",
            "quant_algo": "FP8" if precision == "FP8" else "MIXED_PRECISION",
        },
    )
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(cfg))
    tokenizer_fixture(path)
    tokenizer_fixture(path / "alignment_tokenizer")
    perception = {k: v for k, v in source.items() if k.startswith("perception.")}
    backbone = {k: v for k, v in source.items() if k not in perception}
    mx.save_safetensors(str(path / "model.safetensors"), backbone)
    mx.save_safetensors(str(path / "perception.safetensors"), perception)
    (path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    k: (
                        "perception.safetensors"
                        if k in perception
                        else "model.safetensors"
                    )
                    for k in source
                }
            }
        )
    )
    return config


@pytest.mark.parametrize("precision", ["FP8", "NVFP4"])
def test_nemotron_modelopt_strict_load_features_and_conversion(tmp_path, precision):
    from mlx_audio.convert import convert
    from mlx_audio.lm.models.cache import KVCache
    from mlx_audio.stt.utils import load_model

    source, target = tmp_path / "source", tmp_path / "converted"
    nemo_modelopt_fixture(source, precision)
    model = load_model(source, strict=True)
    assert model.llm.mtp is not None and model.ctc_timestamp is not None
    assert model._alignment_tokenizer is not None
    if precision == "NVFP4":
        layer = model.llm.layers[2].mixer.switch_mlp.fc1
        assert isinstance(layer, ScaledQuantizedSwitchLinear)
        assert layer.global_scale.shape == (4,)
        assert isinstance(model.llm.layers[1].mixer.q_proj, ScaledQuantizedLinear)
    else:
        assert model.llm.layers[2].mixer.switch_mlp.fc1.weight.dtype == mx.bfloat16
    tokens = mx.array([[1, 2, 3]])
    expected = model(tokens)
    assert mx.all(mx.isfinite(expected)).item()
    hidden = model.llm.backbone(tokens)
    draft = model.llm.mtp(model.llm.backbone.embeddings(tokens), hidden, [KVCache()])
    assert mx.all(mx.isfinite(draft)).item()
    result = model.generate(mx.zeros(2000), max_tokens=3, speculative=True)
    assert result.speculative_stats is not None
    convert(str(source), str(target), model_domain="stt", dtype="bfloat16")
    assert not (target / "perception.safetensors").exists()
    config = json.loads((target / "config.json").read_text())
    assert config.get("quantization_config", {}).get("quant_method") != "modelopt"
    restored = load_model(target, strict=True)
    np.testing.assert_allclose(
        np.array(restored(tokens).astype(mx.float32)),
        np.array(expected.astype(mx.float32)),
        atol=1e-6,
    )
    if precision == "NVFP4":
        params = dict(tree_flatten(restored.parameters()))
        assert (
            params["llm.backbone.layers.2.mixer.switch_mlp.fc1.weight"].dtype
            == mx.uint32
        )
        assert (
            params["llm.backbone.layers.2.mixer.switch_mlp.fc1.scales"].dtype
            == mx.uint8
        )
        assert (
            params["llm.backbone.layers.2.mixer.switch_mlp.fc1.weight_scale_2"].dtype
            == mx.float32
        )


def test_dequantize_nvfp4_dense_and_expert_modules():
    from mlx_audio.lm.convert import dequantize_model

    model = nn.Module()
    model.layer = nn.QuantizedLinear(64, 4, bias=False, mode="nvfp4")
    model.experts = QuantizedSwitchLinear(
        64, 4, 3, bias=False, group_size=16, mode="nvfp4"
    )
    weights, _ = transform_modelopt_weights(
        nvfp4_source(), {"quant_method": "modelopt", "quant_algo": "NVFP4"}
    )
    weights.update(
        {
            "experts.weight": mx.full((3, 4, 8), 0x77777777, mx.uint32),
            "experts.scales": mx.full((3, 4, 4), 56, mx.uint8),
            "experts.weight_scale_2": mx.array([0.137, 0.348, 0.00012715657]),
        }
    )
    replace_scaled_quantized_linears(model, weights)
    model.load_weights(list(weights.items()))
    converted = dequantize_model(model)
    assert isinstance(converted.layer, nn.Linear)
    assert converted.experts.weight.shape == (3, 4, 64)
    assert converted.experts.weight.dtype == mx.bfloat16
    expected = (mx.array([0.137, 0.348, 0.00012715657]) * 6).astype(mx.bfloat16)
    np.testing.assert_array_equal(
        np.array(converted.experts.weight[:, 0, 0].astype(mx.float32)),
        np.array(expected.astype(mx.float32)),
    )
