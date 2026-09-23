"""Offline tests for Redux's source format, loading, and inference semantics."""

import json
from unittest.mock import Mock

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mlx_audio.registry import classify_model
from mlx_audio.stt import load
from mlx_audio.stt.models.parakeet.audio import log_mel_spectrogram
from mlx_audio.stt.models.parakeet.convert import convert
from mlx_audio.stt.models.parakeet.redux import ParakeetRedux, repack_ternary


def pack_base3(codes):
    """Independent source-format encoder, including padding within each row."""
    rows, width = codes.shape
    padded = np.pad(codes, ((0, 0), (0, (-width) % 5)))
    return np.sum(
        padded.reshape(rows, -1, 5) * np.array([1, 3, 9, 27, 81]), axis=-1
    ).astype(np.uint8)


@pytest.fixture
def checkpoint(tmp_path):
    rng = np.random.default_rng(12)
    projections = {
        "feed_forward1.linear1": (256, 128),
        "feed_forward1.linear2": (128, 256),
        "feed_forward2.linear1": (256, 128),
        "feed_forward2.linear2": (128, 256),
        "self_attn.q_proj": (128, 128),
        "self_attn.k_proj": (128, 128),
        "self_attn.v_proj": (128, 128),
        "self_attn.o_proj": (128, 128),
        "self_attn.relative_k_proj": (128, 128),
        "conv.pointwise_conv1": (256, 128),
        "conv.pointwise_conv2": (128, 128),
    }
    names = [f"encoder.layers.0.{name}" for name in projections]
    config = {
        "model_type": "parakeet_tdt",
        "blank_token_id": 32,
        "pad_token_id": 2,
        "vocab_size": 33,
        "decoder_hidden_size": 16,
        "num_decoder_layers": 2,
        "durations": [0, 1, 2],
        "max_symbols_per_step": 4,
        "hidden_act": "relu",
        "encoder_config": {
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "num_mel_bins": 8,
            "conv_kernel_size": 3,
            "subsampling_conv_channels": 4,
            "subsampling_conv_kernel_size": 3,
            "subsampling_conv_stride": 2,
            "subsampling_factor": 2,
            "max_position_embeddings": 32,
            "hidden_act": "silu",
        },
        "ternary_group_size": 128,
        "ternary_modules": names,
    }
    vocab = {f"▁word{i}": i for i in range(32)}
    vocab.pop("▁word0")
    vocab["<unk>"] = 0
    vocab.pop("▁word2")
    vocab["<pad>"] = 2
    tokenizer = {
        "model": {"type": "BPE", "vocab": vocab},
        "added_tokens": [{"content": "<blank>", "id": 32}],
    }
    manifest = {
        "format": "thrush-ternary-v2",
        "names": "hf",
        "quant": {"mode": "ternary", "group_size": 128},
        "packing": {"base": 3, "elements_per_byte": 5, "code_offset": 1},
        "quantized_modules": [],
    }
    weights = {}
    for suffix, (rows, columns) in projections.items():
        name = f"encoder.layers.0.{suffix}"
        manifest["quantized_modules"].append(
            {
                "name": name,
                "out_features": rows,
                "in_features": columns,
                "group_size": 128,
                "as_conv1d": suffix.startswith("conv."),
                "has_bias": False,
            }
        )
        codes = rng.integers(0, 3, (rows, columns), dtype=np.uint8)
        weights[name + ".qweight"] = mx.array(pack_base3(codes))
        weights[name + ".scales"] = mx.array(
            rng.uniform(0.001, 0.02, (rows, columns // 128)).astype(np.float16)
        )

    def dense(name, shape):
        weights[name] = mx.array(rng.normal(0, 0.05, shape).astype(np.float32))

    for norm in (
        "norm_feed_forward1",
        "norm_feed_forward2",
        "norm_self_att",
        "norm_conv",
        "norm_out",
        "conv.norm",
    ):
        weights[f"encoder.layers.0.{norm}.weight"] = mx.ones((128,))
        weights[f"encoder.layers.0.{norm}.bias"] = mx.zeros((128,))
    weights["encoder.layers.0.conv.norm.running_mean"] = mx.zeros((128,))
    weights["encoder.layers.0.conv.norm.running_var"] = mx.ones((128,))
    weights["encoder.layers.0.conv.norm.num_batches_tracked"] = mx.array(1)
    dense("encoder.layers.0.conv.depthwise_conv.weight", (128, 1, 3))
    dense("encoder.layers.0.self_attn.bias_u", (4, 32))
    dense("encoder.layers.0.self_attn.bias_v", (4, 32))
    dense("encoder.subsampling.layers.0.weight", (4, 1, 3, 3))
    dense("encoder.subsampling.layers.0.bias", (4,))
    dense("encoder.subsampling.linear.weight", (128, 16))
    dense("encoder.subsampling.linear.bias", (128,))
    dense("decoder.embedding.weight", (33, 16))
    weights["decoder.embedding.weight"][32] = 0
    for i in range(2):
        for stem in ("ih", "hh"):
            dense(f"decoder.lstm.weight_{stem}_l{i}", (64, 16))
            dense(f"decoder.lstm.bias_{stem}_l{i}", (64,))
    for name, shape in (
        ("decoder.decoder_projector", (16, 16)),
        ("encoder_projector", (16, 128)),
        ("joint.head", (36, 16)),
    ):
        dense(name + ".weight", shape)
        dense(name + ".bias", (shape[0],))
    source = tmp_path / "source"
    source.mkdir()
    for name, data in (
        ("config.json", config),
        ("tokenizer.json", tokenizer),
        ("ternary.json", manifest),
    ):
        (source / name).write_text(json.dumps(data))
    mx.save_safetensors(str(source / "model.safetensors"), weights)
    return source, config, weights


@pytest.mark.parametrize("width", [128, 256, 1024, 4096])
def test_lossless_repacking_and_matmul(width):
    rng = np.random.default_rng(width)
    codes = rng.integers(0, 3, (32, width), dtype=np.uint8)
    codes[:3] = np.arange(3)[:, None]  # all-negative, zero, and all-positive rows
    scales = mx.array(rng.uniform(0.001, 0.1, (32, width // 128)).astype(np.float16))
    packed = repack_ternary(mx.array(pack_base3(codes)), width)
    assert packed.dtype == mx.uint32
    assert packed.shape == (32, width // 16)
    dense = (mx.array(codes).astype(mx.float32) - 1) * mx.repeat(scales, 128, -1)
    restored = mx.dequantize(packed, scales, -scales, group_size=128, bits=2)
    np.testing.assert_array_equal(np.array(restored), np.array(dense))
    for length in (1, 17):
        x = mx.array(rng.normal(size=(2, length, width)).astype(np.float32))
        actual = mx.quantized_matmul(x, packed, scales, -scales, group_size=128, bits=2)
        # NumPy is the full-precision oracle. Metal kernels may use reduced
        # precision products, depending on shape; weight reconstruction above
        # must still be bit-exact.
        expected = np.array(x) @ np.array(dense).T
        np.testing.assert_allclose(np.array(actual), expected, rtol=2e-3, atol=2e-3)


def test_invalid_packing():
    with pytest.raises(ValueError, match="uint8"):
        repack_ternary(mx.zeros((1, 26)), 128)
    with pytest.raises(ValueError, match="uint8"):
        repack_ternary(mx.zeros((1, 25), dtype=mx.uint8), 128)
    with pytest.raises(ValueError, match="Invalid ternary byte"):
        repack_ternary(mx.full((1, 26), 243, dtype=mx.uint8), 128)


def test_load_and_key_mapping(checkpoint):
    path, _, weights = checkpoint
    assert classify_model("parakeet_tdt", "arbitrary-name") == "stt"
    model = load(path, strict=True)
    assert isinstance(model, ParakeetRedux)
    assert not model.supports_vad
    params = dict(tree_flatten(model.parameters()))
    np.testing.assert_array_equal(
        params["encoder.pre_encode.conv.0.weight"],
        weights["encoder.subsampling.layers.0.weight"].transpose(0, 2, 3, 1),
    )
    np.testing.assert_array_equal(
        params["decoder.prediction.dec_rnn.lstm.1.bias"],
        weights["decoder.lstm.bias_ih_l1"] + weights["decoder.lstm.bias_hh_l1"],
    )
    np.testing.assert_array_equal(
        params["joint.pred.weight"], weights["decoder.decoder_projector.weight"]
    )
    assert isinstance(model.encoder.layers[0].conv.pointwise_conv1, nn.QuantizedLinear)
    # Direct pointwise application is identical to its dense Conv1d equivalent.
    layer = model.encoder.layers[0].conv.pointwise_conv1
    dense = mx.dequantize(
        layer.weight, layer.scales, layer.biases, group_size=128, bits=2
    )
    x = mx.random.normal((1, 5, 128))
    np.testing.assert_allclose(layer(x), mx.conv1d(x, dense[:, None, :]), atol=1e-5)


@pytest.mark.parametrize("as_string", [False, True])
@pytest.mark.parametrize("model_type", [None, "parakeet_tdt"])
def test_alias_loading_ignores_directory_name(checkpoint, as_string, model_type):
    path, _, _ = checkpoint
    renamed = path.rename(path.parent / "whisper")
    kwargs = {} if model_type is None else {"model_type": model_type}
    model = load(str(renamed) if as_string else renamed, strict=True, **kwargs)
    assert isinstance(model, ParakeetRedux)


@pytest.mark.parametrize("dtype", [None, "float16", "bfloat16"])
def test_convert_reload(checkpoint, tmp_path, monkeypatch, dtype):
    source, _, _ = checkpoint
    before = load(source, strict=True)
    output = convert(str(source), str(tmp_path / "converted"), dtype=dtype)
    assert {p.name for p in output.iterdir()} == {
        "model.safetensors",
        "config.json",
        "tokenizer.json",
        "README.md",
    }
    assert "model_path" not in json.loads((output / "config.json").read_text())
    monkeypatch.setattr(
        "mlx_audio.stt.models.parakeet.redux.repack_ternary",
        Mock(side_effect=AssertionError("Converted weights must not be repacked")),
    )
    after = load(output, strict=True)
    a, b = (
        dict(tree_flatten(before.parameters())),
        dict(tree_flatten(after.parameters())),
    )
    for name, weight in a.items():
        if (
            dtype is None
            or weight.dtype == mx.uint32
            or name.endswith((".scales", ".biases"))
        ):
            np.testing.assert_array_equal(weight, b[name])
        else:
            assert b[name].dtype == getattr(mx, dtype)
    if dtype is None:
        mel = mx.random.normal((1, 10, 8))
        np.testing.assert_array_equal(before.encoder(mel)[0], after.encoder(mel)[0])


def test_missing_weights_rejected_even_with_non_strict_loader(checkpoint):
    path, _, weights = checkpoint
    weights.pop("encoder.layers.0.self_attn.q_proj.qweight")
    mx.save_safetensors(str(path / "model.safetensors"), weights)
    with pytest.raises(ValueError, match="missing=.*linear_q.weight"):
        load(path)


def test_manifest_mismatch(checkpoint):
    path, _, _ = checkpoint
    manifest = json.loads((path / "ternary.json").read_text())
    manifest["quantized_modules"][0]["in_features"] = 256
    (path / "ternary.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest shape/type mismatch"):
        load(path)


def test_masked_encoder_ignores_extra_padding(checkpoint):
    path, _, _ = checkpoint
    model = load(path)
    mel = mx.random.normal((1, 8, 8))
    lengths = mx.array([8])
    short, valid_short = model.encoder(mx.pad(mel, [(0, 0), (0, 1), (0, 0)]), lengths)
    long, valid_long = model.encoder(mx.pad(mel, [(0, 0), (0, 9), (0, 0)]), lengths)
    np.testing.assert_array_equal(valid_short, valid_long)
    np.testing.assert_allclose(short[:, :4], long[:, :4], atol=2e-5)


def test_frontend_masks_last_frame(checkpoint):
    path, _, _ = checkpoint
    model = load(path)
    mel = log_mel_spectrogram(mx.random.normal((641,)), model.preprocessor_config)
    assert mel.shape == (1, 5, 8)
    np.testing.assert_array_equal(mel[:, -1], np.zeros((1, 8)))
    np.testing.assert_allclose(mx.mean(mel[:, :4], axis=1), 0, atol=3e-5)
    with pytest.raises(ValueError, match="320"):
        model.generate(mx.zeros((319,)))


def test_blank_zero_duration_advances_without_committing_state(checkpoint):
    path, _, _ = checkpoint
    model = load(path)
    model.encoder = Mock(return_value=(mx.zeros((1, 3, 128)), mx.array([3])))
    state = mx.ones((2, 1, 16))
    step = Mock(
        side_effect=[
            (mx.array(32), mx.array(0), state, state),
            (mx.array(3), mx.array(1), state * 2, state * 2),
            (mx.array(32), mx.array(0), state * 3, state * 3),
        ]
    )
    model._compiled_tdt_step = step
    result = model.decode(mx.zeros((1, 6, 8)))[0]
    assert result.text.strip() == "word3"
    assert step.call_count == 3
    np.testing.assert_array_equal(step.call_args_list[1].args[2], mx.zeros((2, 1, 16)))
    np.testing.assert_array_equal(step.call_args_list[2].args[2], state * 2)
    assert result.sentences[0].tokens[0].start == pytest.approx(0.02)


def test_chunking_does_not_reprocess_already_covered_tiny_tail(checkpoint):
    path, _, _ = checkpoint
    model = load(path)
    model.decode = Mock(return_value=[Mock(sentences=[])])
    model.generate(mx.zeros((962,)), chunk_duration=0.04, overlap_duration=0.02)
    assert model.decode.call_count == 3


def test_generic_converter_redirects_to_lossless_converter(checkpoint):
    from mlx_audio.convert import Domain, detect_model_domain, get_model_type
    from mlx_audio.stt.models.parakeet import prepare_config

    path, config, _ = checkpoint
    assert detect_model_domain(config, path) == Domain.STT
    assert get_model_type(config, path, Domain.STT) == "parakeet"
    with pytest.raises(ValueError, match="mlx_audio.stt.models.parakeet.convert"):
        prepare_config(config, path)
