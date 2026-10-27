"""Offline standard-loader and inference regression tests for Phonon-2."""

import json
from dataclasses import asdict

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mlx_audio.registry import classify_model
from mlx_audio.stt import load
from mlx_audio.stt.models.parakeet.audio import PreprocessArgs, log_mel_spectrogram
from mlx_audio.stt.models.parakeet.parakeet import ParakeetTDT, ParakeetTDTArgs
from mlx_audio.stt.models.phonon2 import Model, ModelConfig
from mlx_audio.stt.models.phonon2.packed import ENCODER_SUFFIXES, PackedFiveValueLinear
from mlx_audio.utils import from_dict


def packed_weights(sign, mask):
    """Encode synthetic five-value weights directly in the checkpoint layout."""

    def pack(codes):
        codes = codes.astype(np.uint32).reshape(sign.shape[0], -1, 16)
        return np.bitwise_or.reduce(
            codes << (2 * np.arange(16, dtype=np.uint32)), axis=-1
        )

    return pack(sign + 1), pack(sign * mask + 1)


@pytest.fixture
def checkpoint(tmp_path):
    config = {
        "preprocessor": dict(
            sample_rate=16000,
            normalize="per_feature",
            window_size=0.025,
            window_stride=0.01,
            window="hann",
            features=8,
            n_fft=512,
            dither=0,
        ),
        "encoder": dict(
            feat_in=8,
            n_layers=1,
            d_model=128,
            n_heads=4,
            ff_expansion_factor=2,
            subsampling_factor=2,
            self_attention_model="rel_pos",
            subsampling="dw_striding",
            conv_kernel_size=3,
            subsampling_conv_channels=4,
            pos_emb_max_len=128,
            use_bias=False,
        ),
        "decoder": dict(
            blank_as_pad=True,
            vocab_size=8,
            prednet=dict(pred_hidden=16, pred_rnn_layers=1),
        ),
        "joint": dict(
            num_classes=8,
            vocabulary=[f"▁word{i}" for i in range(8)],
            num_extra_outputs=3,
            jointnet=dict(
                joint_hidden=16, activation="relu", encoder_hidden=128, pred_hidden=16
            ),
        ),
        "decoding": dict(
            model_type="tdt", durations=[0, 1, 2], greedy=dict(max_symbols=4)
        ),
    }
    mx.random.seed(7)
    model = ParakeetTDT(from_dict(ParakeetTDTArgs, config))
    projections = {f"encoder.layers.0.{s}.weight" for s in ENCODER_SUFFIXES}
    weights, modules = {}, []
    for name, weight in tree_flatten(model.parameters()):
        if name in projections:
            outputs, inputs = weight.shape[0], int(np.prod(weight.shape[1:]))
            shape = (outputs, inputs)
            sign = np.resize(np.array([-1, 0, 1], np.int8), shape)
            mask = np.resize(
                np.array([True, False, False, False, False, True]), shape
            ) & (sign != 0)
            base, residual = packed_weights(sign, mask)
            module = name.removesuffix(".weight")
            modules.append({"name": module, "shape": list(shape)})
            weights[f"{module}.base_q"] = mx.array(base)
            weights[f"{module}.residual_q"] = mx.array(residual)
            weights[f"{module}.row_lo"] = mx.full((outputs,), 0.125, mx.float16)
            weights[f"{module}.row_hi"] = mx.full((outputs,), 0.375, mx.float16)
        else:
            # Force blank and duration 1 so synthetic audio terminates deterministically.
            if name == "joint.joint_net.2.weight":
                weight = mx.zeros_like(weight)
            elif name == "joint.joint_net.2.bias":
                weight = mx.zeros_like(weight)
                weight[8] = weight[10] = 20
            weights[name] = weight.astype(mx.bfloat16)
    config.update(model_type="phonon2", phonon2_modules=modules)
    path = tmp_path / "mlx-model"
    path.mkdir()
    (path / "config.json").write_text(json.dumps(config))
    mx.save_safetensors(str(path / "model.safetensors"), weights)
    return path


def test_standard_local_loader(checkpoint):
    from mlx_audio.stt.models.parakeet import parakeet

    original_frontend = parakeet.log_mel_spectrogram
    model = load(checkpoint)
    assert isinstance(model, Model)
    assert isinstance(model.config, ModelConfig)
    assert parakeet.log_mel_spectrogram is original_frontend
    assert len(model._packed_modules) == 11
    assert classify_model("phonon2") == "stt"
    assert model.generate(mx.zeros(16000)).text == ""
    config = json.loads((checkpoint / "config.json").read_text())
    assert PreprocessArgs(**config["preprocessor"]) == model.preprocessor_config
    # Verify loaded projections numerically; the blank-output fixture alone
    # cannot detect corrupted encoder weights.
    rng = np.random.default_rng(8)
    for row in config["phonon2_modules"]:
        layer = model
        for part in row["name"].split("."):
            layer = layer[int(part)] if part.isdigit() else getattr(layer, part)
        shape = tuple(row["shape"])
        sign = np.resize(np.array([-1, 0, 1], np.int8), shape)
        high = np.resize(np.array([True, False, False, False, False, True]), shape)
        dense = (sign * np.where(high, 0.375, 0.125)).astype(np.float32)
        inputs = rng.normal(size=(2, 3, shape[1])).astype(np.float32)
        np.testing.assert_allclose(
            np.asarray(layer(mx.array(inputs))),
            inputs @ dense.T,
            atol=2e-5,
            err_msg=row["name"],
        )
    again = load(str(checkpoint), lazy=True)
    assert isinstance(again, Model)
    weights = mx.load(str(checkpoint / "model.safetensors"))
    weights.pop(next(iter(weights)))
    with pytest.raises(ValueError):
        again.load_weights(list(weights.items()), strict=False)


def test_packed_forward_matches_independent_dense_math():
    sign = np.resize(np.array([-1, 0, 1], np.int8), (3, 128))
    mask = (np.indices(sign.shape)[1] % 2 == 0) & (sign != 0)
    a, b = packed_weights(sign, mask)
    layer = PackedFiveValueLinear(128, 3, bias=True)
    layer.base_q, layer.residual_q = mx.array(a), mx.array(b)
    lo = np.array([0.1, 0.2, 0.3], np.float16)
    hi = np.array([0.4, 0.5, 0.6], np.float16)
    layer.row_lo, layer.row_hi = mx.array(lo), mx.array(hi)
    layer.bias = mx.array([1.0, 2.0, 3.0])
    layer.materialize()
    x = np.random.default_rng(8).normal(size=(2, 128)).astype(np.float32)
    dense = sign * np.where(mask, hi[:, None], lo[:, None]).astype(np.float32)
    np.testing.assert_allclose(
        np.asarray(layer(mx.array(x))),
        x @ dense.T + np.array([1.0, 2.0, 3.0]),
        atol=2e-5,
    )


def test_model_config_round_trip_and_nested_types(checkpoint):
    path = checkpoint
    raw = json.loads((path / "config.json").read_text())
    raw["model_path"] = str(path)
    raw["unrelated_metadata"] = "ignored"
    original = json.dumps(raw, sort_keys=True)
    config = ModelConfig.from_dict(raw)
    parakeet = from_dict(ParakeetTDTArgs, raw)
    for name in ("preprocessor", "encoder", "decoder", "joint", "decoding"):
        assert getattr(config, name) == getattr(parakeet, name)
    assert config.model_type == "phonon2"
    assert config.model_path == str(path)
    assert config.phonon2_modules == raw["phonon2_modules"]
    assert ModelConfig.from_dict(asdict(config)) == config
    assert json.dumps(raw, sort_keys=True) == original


def test_inherited_decoder_handles_batch_lengths_and_blank_state(checkpoint):
    model = load(checkpoint)
    assert Model.decode is ParakeetTDT.decode
    model.encoder = lambda mel: (
        mx.array([[[0.0], [1.0], [1.0]], [[0.0], [1.0], [1.0]]]),
        mx.array([3, 2]),
    )
    states = []

    def step(feature, token, h, c):
        states.append(int(h[0, 0, 0]))
        pred = mx.where(feature[0, 0, 0] == 0, model.blank_id, 3).astype(mx.int32)
        return pred, mx.array(1), h + 1, c + 1

    model._compiled_tdt_step = step
    results = model.decode(mx.zeros((2, 6, 8)))
    tokens = [[t for s in result.sentences for t in s.tokens] for result in results]
    assert [len(batch) for batch in tokens] == [2, 1]
    assert [[t.id for t in batch] for batch in tokens] == [[3, 3], [3]]
    assert states == [0, 0, 1, 0, 0]
    assert [[t.start for t in batch] for batch in tokens] == [[0.02, 0.04], [0.02]]


def test_inherited_decoder_bounds_zero_duration_emissions(checkpoint):
    model = load(checkpoint)
    model.encoder = lambda mel: (mx.zeros((1, 2, 1)), mx.array([2]))
    model._compiled_tdt_step = lambda feature, token, h, c: (
        mx.array(3),
        mx.array(0),
        h + 1,
        c + 1,
    )
    result = model.decode(mx.zeros((1, 6, 8)))[0]
    tokens = [t for sentence in result.sentences for t in sentence.tokens]
    assert len(tokens) == 2 * model.max_symbols
    assert [t.start for t in tokens] == [0.0] * model.max_symbols + [
        0.02
    ] * model.max_symbols


def test_missing_projection_and_wrong_shape_rejected(checkpoint):
    config = json.loads((checkpoint / "config.json").read_text())
    config["phonon2_modules"].pop()
    with pytest.raises(ValueError, match="every encoder"):
        Model(ModelConfig.from_dict(config))
    config = json.loads((checkpoint / "config.json").read_text())
    config["phonon2_modules"][0]["shape"] = [1, 128]
    with pytest.raises(ValueError, match="shape mismatch"):
        Model(ModelConfig.from_dict(config))


@pytest.mark.parametrize("samples", [1, 319])
def test_short_audio_and_chunk_tail(checkpoint, samples):
    model = load(checkpoint)
    assert model.generate(mx.zeros(samples)).text == ""
    assert (
        model.generate(
            mx.zeros(8000 + samples), chunk_duration=0.5, overlap_duration=0
        ).text
        == ""
    )
    chunks = list(
        model.generate(
            mx.zeros(8000 + samples),
            stream=True,
            chunk_duration=0.5,
            overlap_duration=0,
        )
    )
    assert chunks[-1].is_final
    assert chunks[-1].audio_position == (8000 + samples) / 16000


@pytest.mark.parametrize("suffix", ["base_q", "residual_q", "row_lo", "row_hi"])
def test_weight_cast_rejected(checkpoint, suffix):
    model = load(checkpoint)
    weights = mx.load(str(checkpoint / "model.safetensors"))
    name = next(k for k in weights if k.endswith("." + suffix))
    weights[name] = weights[name].astype(mx.float32)
    with pytest.raises(ValueError, match="must remain"):
        model.load_weights(weights.items())


@pytest.mark.parametrize(
    "maximum,durations",
    [(None, [0, 1, 2]), (0, [0, 1, 2]), (4, [0, -1, 2]), (4, [0, 1])],
)
def test_invalid_decoder_config_rejected(checkpoint, maximum, durations):
    config = json.loads((checkpoint / "config.json").read_text())
    config["decoding"]["greedy"]["max_symbols"] = maximum
    config["decoding"]["durations"] = durations
    with pytest.raises(ValueError, match="durations"):
        Model(ModelConfig.from_dict(config))


@pytest.mark.parametrize("model_type", ["phonon2", "parakeet_tdt_five_value"])
def test_generic_conversion_guard(checkpoint, tmp_path, model_type):
    from mlx_audio.convert import convert

    config = json.loads((checkpoint / "config.json").read_text())
    config["model_type"] = model_type
    (checkpoint / "config.json").write_text(json.dumps(config))
    output = tmp_path / "converted"
    with pytest.raises(ValueError, match="mlx-community/Phonon-2"):
        convert(hf_path=str(checkpoint), mlx_path=str(output))
    assert not output.exists()


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
def test_phonon2_frontend_matches_fp32_reference(checkpoint, dtype):
    model = load(checkpoint)
    assert not hasattr(model.preprocessor_config, "compute_in_float32")
    waveform = mx.random.normal((36 * 16000,)).astype(dtype)
    expected = log_mel_spectrogram(
        waveform.astype(mx.float32), model.preprocessor_config
    ).astype(dtype)
    actual = model._log_mel_spectrogram(waveform)
    assert bool(mx.all(mx.isfinite(actual)))
    np.testing.assert_array_equal(
        np.asarray(actual.astype(mx.float32)), np.asarray(expected.astype(mx.float32))
    )
    with pytest.raises(ValueError, match="nonempty"):
        model._log_mel_spectrogram(mx.zeros(0))
    with pytest.raises(ValueError, match="mono"):
        model._log_mel_spectrogram(mx.zeros((2, 16000)))


def test_all_generation_paths_use_local_frontend(checkpoint, monkeypatch):
    model = load(checkpoint)
    calls = []
    frontend = model._log_mel_spectrogram

    def record(audio):
        calls.append(audio.size)
        return frontend(audio)

    monkeypatch.setattr(model, "_log_mel_spectrogram", record)
    model.generate(mx.zeros(8001), chunk_duration=0.5, overlap_duration=0)
    assert calls == [8000, 1]
    calls.clear()
    list(
        model.generate(
            mx.zeros(8001), stream=True, chunk_duration=0.5, overlap_duration=0
        )
    )
    assert calls == [8000, 1]
    calls.clear()
    model.generate(mx.zeros(1))
    assert calls == [1]
