"""Offline tests; no checkpoints or network access are required.

For numerical tests on hardware with tensor units, launch pytest with
MLX_ENABLE_TF32=0. Torch/Transformers/Diffusers checks skip if unavailable.
"""

import json
import os
from dataclasses import asdict

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mlx_audio.sts.models.dialogue_sidon import Model, ModelConfig
from mlx_audio.sts.models.dialogue_sidon.config import DiffusionConfig, EncoderConfig
from mlx_audio.sts.models.dialogue_sidon.diffusion import DPMSolver
from mlx_audio.sts.models.dialogue_sidon.frontend import (
    extract_features,
    normalize_chunk,
    resample,
)
from mlx_audio.sts.models.dialogue_sidon.model import align_speakers


def small_config():
    return ModelConfig(
        latent_dim=4,
        decoder_channels=32,
        encoder=EncoderConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            conv_depthwise_kernel_size=5,
            left_max_position_embeddings=4,
            right_max_position_embeddings=2,
        ),
        diffusion=DiffusionConfig(hidden_size=16, num_layers=1, num_heads=2),
    )


@pytest.fixture
def tiny_model():
    return Model(small_config())


def test_config_roundtrip():
    config = small_config()
    assert ModelConfig.from_dict(asdict(config)) == config
    with pytest.raises(ValueError, match="normalization"):
        ModelConfig(latent_norm_initialized=True)
    with pytest.raises(ValueError, match="positive std"):
        ModelConfig(
            latent_norm_initialized=True,
            latent_norm_mean=[0] * 64,
            latent_norm_std=[0] * 64,
        )


def test_loader_roundtrip_and_registration(tmp_path):
    from mlx_audio.registry import classify_model
    from mlx_audio.sts import load

    config = small_config()
    model = Model(config)
    mx.save_safetensors(
        str(tmp_path / "model.safetensors"), dict(tree_flatten(model.parameters()))
    )
    (tmp_path / "config.json").write_text(json.dumps(asdict(config)))
    restored = load(tmp_path, strict=True)
    assert isinstance(restored, Model)
    assert classify_model("dialogue_sidon") == "sts"
    assert classify_model("", "mlx-community/DialogueSidon-fp32") == "sts"
    np.testing.assert_array_equal(
        np.array(restored.linear1.weight), np.array(model.linear1.weight)
    )


@pytest.mark.parametrize("length", [1, 159, 1600, 1761])
def test_silence_shapes_and_seed(tiny_model, length):
    wav = mx.zeros(length)
    a = tiny_model.separate(wav, 16000, num_steps=2, seed=123)
    b = tiny_model.separate(wav, 16000, num_steps=2, seed=123)
    assert a.speakers.shape == (2, max(1, round(length * 1.5)))
    assert mx.all(mx.isfinite(a.speakers)).item()
    np.testing.assert_array_equal(np.array(a.speakers), np.array(b.speakers))


def test_seed_does_not_reset_global_random_stream(tiny_model):
    mx.random.seed(55)
    expected = mx.random.normal((3,))
    mx.eval(expected)
    mx.random.seed(55)
    tiny_model.separate(mx.zeros(1600), 16000, num_steps=1, seed=4)
    actual = mx.random.normal((3,))
    np.testing.assert_array_equal(np.array(actual), np.array(expected))


def test_invalid_audio_and_controls(tiny_model):
    for audio in (mx.array([]), mx.array([float("nan")]), mx.zeros((1, 2, 3))):
        with pytest.raises(ValueError):
            tiny_model.separate(audio, 16000)
    with pytest.raises(ValueError, match="sample_rate"):
        tiny_model.separate(mx.zeros(200))
    for kwargs in ({"num_steps": 0}, {"chunk_seconds": 5, "overlap_seconds": 5}):
        with pytest.raises(ValueError):
            tiny_model.separate(mx.zeros(200), 16000, **kwargs)


def test_chunked_stereo_resampling(tiny_model):
    audio = mx.zeros((3601, 2))
    result = tiny_model.separate(
        audio, 24000, num_steps=1, seed=0, chunk_seconds=0.1, overlap_seconds=0.025
    )
    assert result.speakers.shape == (2, 3601)
    assert result.sample_rate == 24000
    assert mx.all(mx.isfinite(result.speakers)).item()


def test_align_speakers_and_silent_ties():
    a = mx.array(np.random.default_rng(0).standard_normal((2, 100)).astype(np.float32))
    np.testing.assert_array_equal(
        np.array(align_speakers(a, a[::-1], 100)), np.array(a)
    )
    np.testing.assert_array_equal(
        np.array(align_speakers(mx.zeros_like(a), a, 100)), np.array(a)
    )


def test_stitching_tracks_speaker_swaps_and_preserves_tail(tiny_model, monkeypatch):
    length = 51200
    expected = (
        np.random.default_rng(4)
        .standard_normal((2, length * 3 // 2))
        .astype(np.float32)
    )
    counter = iter(range(5))

    def predict(*args, **kwargs):
        return next(counter)

    def decode(index):
        start = index * 18000  # 0.75 second hop at the output sample rate.
        chunk = expected[:, start : min(start + 24000, expected.shape[-1])]
        if index % 2:
            chunk = chunk[::-1].copy()
        return mx.array(chunk)[None]

    monkeypatch.setattr(tiny_model, "predict_latents", predict)
    monkeypatch.setattr(tiny_model, "decode_latents", decode)
    result = tiny_model.separate(
        mx.zeros(length), 16000, num_steps=1, chunk_seconds=1, overlap_seconds=0.25
    )
    np.testing.assert_allclose(
        np.array(result.speakers), expected, atol=5e-7, rtol=1e-6
    )


def test_cli_uses_local_metadata_and_writes_two_stems(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    import mlx_audio.sts
    from mlx_audio import audio_io
    from mlx_audio.sts.generate import main

    model_dir = tmp_path / "converted"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type": "dialogue_sidon"}')
    input_path = tmp_path / "input.wav"
    audio_io.write(str(input_path), np.zeros(8), 24000)
    calls = {}

    def separate(path, **kwargs):
        calls.update(kwargs)
        return SimpleNamespace(speakers=mx.zeros((2, 8)), sample_rate=24000)

    monkeypatch.setattr(
        mlx_audio.sts, "load", lambda *a, **kw: SimpleNamespace(separate=separate)
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate",
            "--model",
            str(model_dir),
            "--audio",
            str(input_path),
            "--output-path",
            str(tmp_path / "out.wav"),
            "--seed",
            "9",
        ],
    )
    main()
    assert calls["seed"] == 9 and calls["num_steps"] == 30
    for i in (1, 2):
        data, sr = audio_io.read(str(tmp_path / f"out_speaker_{i}.wav"))
        assert data.shape == (8,) and sr == 24000


def test_frontend_odd_padding_and_short_input():
    # Three fbank frames -> two stacked frames; the last is masked.
    features, mask = extract_features(mx.zeros(720))
    assert features.shape == (1, 2, 160)
    np.testing.assert_array_equal(np.array(mask), [[True, False]])
    assert mx.all(mx.isfinite(features)).item()
    assert normalize_chunk(mx.zeros(1)).shape == (560,)


def precise_test():
    if os.environ.get("MLX_ENABLE_TF32") != "0":
        pytest.skip("Launch with MLX_ENABLE_TF32=0 for full-precision comparisons")


@pytest.mark.parametrize("rate", [8000, 16000, 24000, 44100, 48000])
def test_resample_torchaudio_parity(rate):
    precise_test()
    torch = pytest.importorskip("torch")
    torchaudio = pytest.importorskip("torchaudio")
    audio = np.random.default_rng(0).standard_normal(1761).astype(np.float32) * 0.1
    expected = torchaudio.functional.resample(
        torch.from_numpy(audio), rate, 16000
    ).numpy()
    np.testing.assert_allclose(
        np.array(resample(mx.array(audio), rate)), expected, atol=5e-7, rtol=1e-5
    )


@pytest.mark.parametrize("length", [1040, 1761, 16321])
def test_frontend_torchaudio_parity(length):
    precise_test()
    torch = pytest.importorskip("torch")
    kaldi = pytest.importorskip("torchaudio.compliance.kaldi")
    audio = np.random.default_rng(0).standard_normal(length).astype(np.float32) * 0.1
    expected = kaldi.fbank(torch.from_numpy(audio)[None], num_mel_bins=80, dither=0)
    expected = (expected - expected.mean(0)) / (expected.var(0) + 1e-5).sqrt()
    if len(expected) % 2:
        expected = torch.nn.functional.pad(expected, (0, 0, 0, 1))
    actual, _ = extract_features(mx.array(audio))
    np.testing.assert_allclose(
        np.array(actual), expected.reshape(1, -1, 160).numpy(), atol=4e-5, rtol=1e-4
    )


@pytest.mark.parametrize("steps", [1, 2, 7, 30])
def test_every_solver_step_against_diffusers(steps):
    precise_test()
    torch = pytest.importorskip("torch")
    diffusers = pytest.importorskip("diffusers")
    ref = diffusers.DPMSolverMultistepScheduler(
        beta_schedule="linear",
        prediction_type="v_prediction",
        timestep_spacing="linspace",
    )
    ref.set_timesteps(steps)
    solver = DPMSolver(DiffusionConfig(), steps)
    np.testing.assert_array_equal(solver.timesteps, ref.timesteps.numpy())
    rng = np.random.default_rng(42)
    noise = rng.standard_normal((1, 9, 8)).astype(np.float32)
    expected, actual = torch.from_numpy(noise), mx.array(noise)
    for t in ref.timesteps:
        pred = rng.standard_normal(noise.shape).astype(np.float32)
        expected = ref.step(torch.from_numpy(pred), t, expected).prev_sample
        actual = solver.step(mx.array(pred), actual)
        np.testing.assert_allclose(
            np.array(actual), expected.numpy(), atol=8e-6, rtol=1e-5
        )


def test_encoder_against_transformers_with_padding():
    precise_test()
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from mlx_audio.sts.models.dialogue_sidon.convert import convert_encoder_weights
    from mlx_audio.sts.models.dialogue_sidon.encoder import Encoder

    config = small_config().encoder
    hf_config = transformers.Wav2Vec2BertConfig(
        **asdict(config),
        mask_time_prob=0,
        hidden_dropout=0,
        attention_dropout=0,
        apply_spec_augment=False,
    )
    ref = transformers.Wav2Vec2BertModel(hf_config).eval()
    weights = {
        "ssl_model.base_model.model." + k: v for k, v in ref.state_dict().items()
    }
    weights = {
        k.removeprefix("encoder."): v
        for k, v in convert_encoder_weights(weights).items()
    }
    model = Encoder(config)
    model.load_weights(list(weights.items()), strict=True)
    features = np.random.default_rng(0).standard_normal((2, 11, 160)).astype(np.float32)
    mask = np.ones((2, 11), dtype=np.int64)
    mask[1, -3:] = 0
    with torch.inference_mode():
        expected = ref(
            torch.from_numpy(features), attention_mask=torch.from_numpy(mask)
        ).last_hidden_state
    actual = model(mx.array(features), mx.array(mask))
    np.testing.assert_allclose(np.array(actual), expected.numpy(), atol=2e-5, rtol=1e-4)


def test_lora_merge_and_reject_unpaired():
    torch = pytest.importorskip("torch")
    from mlx_audio.sts.models.dialogue_sidon.convert import merge_lora

    rng = np.random.default_rng(0)
    w, a, b, x = [
        torch.tensor(rng.standard_normal(shape).astype(np.float32))
        for shape in ((5, 3), (2, 3), (5, 2), (7, 3))
    ]
    state = {
        "layer.base_layer.weight": w,
        "layer.lora_A.default.weight": a,
        "layer.lora_B.default.weight": b,
    }
    merged = merge_lora(state)
    torch.testing.assert_close(
        x @ merged["layer.weight"].T, x @ w.T + 0.25 * ((x @ a.T) @ b.T)
    )
    with pytest.raises(ValueError, match="Unpaired"):
        merge_lora({"layer.lora_B.default.weight": b})


@pytest.mark.parametrize("stride", [3, 5, 8])
def test_decoder_transposed_conv_mapping(stride):
    precise_test()
    torch = pytest.importorskip("torch")
    from mlx_audio.codec.models.descript.nn.layers import WNConvTranspose1d
    from mlx_audio.sts.models.dialogue_sidon.convert import convert_decoder_weights

    rng = np.random.default_rng(0)
    v = torch.tensor(rng.standard_normal((6, 4, stride * 2)).astype(np.float32))
    g = torch.tensor(rng.uniform(0.1, 1, (6, 1, 1)).astype(np.float32))
    bias = torch.tensor(rng.standard_normal(4).astype(np.float32))
    prefix = "decoder.model.2.block.1."
    mapped = convert_decoder_weights(
        {prefix + "weight_v": v, prefix + "weight_g": g, prefix + "bias": bias}
    )
    layer = WNConvTranspose1d(
        6, 4, 2 * stride, stride=stride, padding=(stride + 1) // 2
    )
    layer.load_weights(
        [
            (k.removeprefix("decoder.model.layers.2.block.layers.1."), value)
            for k, value in mapped.items()
        ],
        strict=True,
    )
    x = rng.standard_normal((1, 9, 6)).astype(np.float32)
    expected = torch.nn.functional.conv_transpose1d(
        torch.from_numpy(x).transpose(1, 2),
        g * v / torch.linalg.vector_norm(v, dim=(1, 2), keepdim=True),
        bias,
        stride=stride,
        padding=(stride + 1) // 2,
    )
    np.testing.assert_allclose(
        np.array(layer(mx.array(x))),
        expected.transpose(1, 2).numpy(),
        atol=3e-6,
        rtol=1e-5,
    )
