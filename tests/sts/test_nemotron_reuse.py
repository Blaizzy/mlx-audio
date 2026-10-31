"""RE-USE recurrence, checkpoint layout, DSP, and STS integration tests."""

import json
from dataclasses import asdict
from pathlib import Path
from types import MethodType

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mlx_audio import audio_io, convert
from mlx_audio.registry import classify_model, model_type_from_config
from mlx_audio.sts import load
from mlx_audio.sts.models.nemotron_reuse import Model, ModelConfig
from mlx_audio.sts.models.nemotron_reuse.mamba import selective_scan
from mlx_audio.sts.models.nemotron_reuse.network import SubpixelConv2d
from mlx_audio.sts.utils import infer_model_type_from_config


def tiny_config():
    return ModelConfig(hid_feature=4, num_tfmamba=1, d_state=3, expand=2)


@pytest.mark.parametrize("device", [mx.cpu, mx.gpu])
def test_selective_scan_matches_independent_numpy_recurrence(device):
    rng = np.random.default_rng(13)
    batch, length, channels, states = 2, 19, 7, 3
    x = rng.normal(size=(batch, length, channels)).astype(np.float32)
    dt = rng.uniform(0, 0.3, size=x.shape).astype(np.float32)
    a = -rng.uniform(0.1, 2, size=(channels, states)).astype(np.float32)
    b, c = rng.normal(size=(2, batch, length, states)).astype(np.float32)
    d = rng.normal(size=channels).astype(np.float32)
    state = np.zeros((batch, channels, states), dtype=np.float64)
    expected = []
    for i in range(length):
        delta = dt[:, i, :, None]
        state = np.exp(delta * a) * state + delta * b[:, i, None] * x[:, i, :, None]
        expected.append((state * c[:, i, None]).sum(-1) + d * x[:, i])
    with mx.stream(device):
        actual = selective_scan(*map(mx.array, (x, dt, a, b, c, d)))
        mx.eval(actual)
    np.testing.assert_allclose(
        np.array(actual), np.stack(expected, 1), atol=2e-6, rtol=2e-5
    )


def test_subpixel_channel_order():
    layer = SubpixelConv2d(2, 3)
    layer.conv.weight = mx.zeros_like(layer.conv.weight)
    layer.conv.bias = mx.arange(6, dtype=mx.float32)
    actual = layer(mx.zeros((1, 1, 2, 2)))
    np.testing.assert_array_equal(
        np.array(actual), np.tile(np.arange(6).reshape(3, 2), (2, 1))[None, None]
    )


def test_source_layout_strict_load_and_native_reload(tmp_path):
    config = tiny_config()
    model = Model(config)
    native = dict(tree_flatten(model.parameters()))
    source = {
        k: (
            v.transpose(0, 3, 1, 2)
            if v.ndim == 4
            else v.transpose(0, 2, 1) if k.endswith("conv1d.weight") else v
        )
        for k, v in native.items()
    }
    raw_config = {"model_cfg": asdict(config), "stft_cfg": {"n_fft": 320}}
    (tmp_path / "config.json").write_text(json.dumps(raw_config))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), source)
    loaded = load(str(tmp_path), strict=True)
    for key, value in tree_flatten(loaded.parameters()):
        np.testing.assert_array_equal(np.array(value), np.array(native[key]))
    sanitized = loaded.sanitize(source)
    assert loaded.sanitize(sanitized) is sanitized
    mx.save_safetensors(str(tmp_path / "model.safetensors"), sanitized)
    load(tmp_path, strict=True)
    mag = mx.ones((1, 17, 11))
    phase = mx.zeros_like(mag)
    output = loaded(mag, phase)
    assert all(x.shape == mag.shape for x in output)
    assert all(bool(mx.all(mx.isfinite(x))) for x in output)


def test_detection_without_repository_name():
    config = {"model_cfg": {"num_tfmamba": 30}, "stft_cfg": {"n_fft": 320}}
    assert model_type_from_config(config) == "nemotron_reuse"
    assert infer_model_type_from_config(config) == "nemotron_reuse"
    assert classify_model("nemotron_reuse") == "sts"
    assert convert.detect_model_domain(config, Path("arbitrary")) == convert.Domain.STS
    assert (
        convert.get_model_type(config, Path("arbitrary"), convert.Domain.STS)
        == "nemotron_reuse"
    )
    assert model_type_from_config({**config, "model_type": "explicit"}) == "explicit"


@pytest.mark.parametrize("rate", [8000, 16000, 22050, 24000, 32000, 44100, 48000])
@pytest.mark.parametrize("frontend", ["native", "stable"])
def test_stft_roundtrip_with_identity_network(rate, frontend):
    # Bypass the network to exercise the real analysis/synthesis path.

    class Identity(Model):
        def __call__(self, magnitude, phase):
            return magnitude, phase

    model = Identity(tiny_config())
    rng = np.random.default_rng(rate)
    nfft, hop, win = model.config.stft_sizes(rate)
    audio = rng.normal(0, 0.1, size=hop * 8 + 7).astype(np.float32)
    actual = np.array(
        model.generate(
            audio, sample_rate=rate, chunk_seconds=None, frontend=frontend
        ).audio
    )
    np.testing.assert_allclose(actual[:-7], audio[:-7], atol=2e-6)
    np.testing.assert_array_equal(actual[-7:], np.full(7, 1e-8, dtype=np.float32))


@pytest.mark.parametrize("frontend", ["native", "stable"])
def test_chunk_overlap_and_channels_preserve_audio(tmp_path, frontend):
    model = Model(tiny_config())
    selected = []

    def passthrough(self, audio, rate, **kwargs):
        selected.append(kwargs["frontend"])
        return audio

    model._enhance_chunk = MethodType(passthrough, model)
    rng = np.random.default_rng(42)
    audio = rng.normal(0, 0.1, size=(3501, 2)).astype(np.float32)
    result = model.generate(
        audio,
        sample_rate=8000,
        chunk_seconds=0.1,
        overlap_seconds=0.025,
        frontend=frontend,
    )
    np.testing.assert_allclose(np.array(result.audio), audio, atol=3e-8)
    path = tmp_path / "input.wav"
    audio_io.write(str(path), audio, 8000)
    result = model.generate(
        path, target_sample_rate=16000, chunk_seconds=None, frontend=frontend
    )
    assert result.audio.shape == (7002, 2)
    assert result.sample_rate == 16000
    assert set(selected) == {frontend}


@pytest.mark.parametrize(
    "audio,kwargs",
    [
        (np.zeros(10), {}),
        (np.zeros(10), {"sample_rate": 7999}),
        (np.zeros(10), {"sample_rate": 48001}),
        (np.array([]), {"sample_rate": 8000}),
        (np.array([float("nan")]), {"sample_rate": 8000}),
        (np.zeros(10), {"sample_rate": 8000, "chunk_seconds": 0}),
        (np.zeros(10), {"sample_rate": 8000, "frontend": "unknown"}),
        (
            np.zeros(10),
            {
                "sample_rate": 8000,
                "chunk_seconds": 0.0002,
                "overlap_seconds": 0.00015,
            },
        ),
        (np.zeros(10), {"sample_rate": 8000, "chunk_seconds": 1, "overlap_seconds": 1}),
    ],
)
def test_invalid_inputs(audio, kwargs):
    with pytest.raises(ValueError):
        Model(tiny_config()).generate(audio, **kwargs)


@pytest.mark.parametrize("length", [1, 79, 160, 161])
@pytest.mark.parametrize("frontend", ["native", "stable"])
def test_short_silent_inputs_are_finite(length, frontend):
    output = Model(tiny_config()).generate(
        np.zeros(length), sample_rate=8000, chunk_seconds=None, frontend=frontend
    )
    assert output.audio.shape == (length,)
    assert bool(mx.all(mx.isfinite(output.audio)))


def test_stft_negative_zero_uses_positive_pi_phase(monkeypatch):
    from mlx_audio.sts.models.nemotron_reuse import model as model_module

    spectrum = np.empty((2, 161), dtype=np.complex64)
    spectrum.real = -1.0
    spectrum.imag = -0.0
    monkeypatch.setattr(model_module, "stft", lambda *a, **kw: mx.array(spectrum))
    phases = []

    class Capture(Model):
        def __call__(self, magnitude, phase):
            phases.append(np.array(phase))
            return magnitude, phase

    Capture(tiny_config()).generate(np.zeros(400), sample_rate=8000, chunk_seconds=None)
    np.testing.assert_allclose(phases[0], np.pi, atol=1e-7)


def test_stable_frontend_matches_direct_dft_and_real_bin_convention():
    from mlx_audio.sts.models.nemotron_reuse.frontend import stable_stft

    n_fft, hop = 32, 8
    audio = np.random.default_rng(19).normal(size=129)
    magnitude, phase, window = stable_stft(audio, n_fft, hop, n_fft, "relu_log1p")
    expected_window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)
    frames = np.lib.stride_tricks.sliding_window_view(
        np.pad(audio, (n_fft // 2, n_fft // 2), mode="reflect"), n_fft
    )[::hop]
    basis = np.exp(
        -2j * np.pi * np.arange(n_fft // 2 + 1)[:, None] * np.arange(n_fft) / n_fft
    )
    spectrum = basis @ (frames * expected_window).T
    spectrum.imag[[0, -1], :] = 0
    spectrum.imag[:, [0, -1]] = 0
    np.testing.assert_allclose(magnitude, np.log1p(np.abs(spectrum)), atol=2e-7)
    np.testing.assert_allclose(phase, np.angle(spectrum), atol=2e-7)
    np.testing.assert_array_equal(window[1:], window[:0:-1])
    assert set(np.unique(phase[[0, -1], :])).issubset(
        {np.float32(0), np.float32(np.pi)}
    )


def test_stable_frontend_silence_is_independent_of_zero_sign():
    from mlx_audio.sts.models.nemotron_reuse.frontend import stable_stft

    positive = stable_stft(np.zeros(400), 320, 40, 320, "relu_log1p")
    negative = stable_stft(-np.zeros(400), 320, 40, 320, "relu_log1p")
    for expected, actual in zip(positive, negative):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(positive[1], 0)
