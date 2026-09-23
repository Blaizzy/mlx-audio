"""No-download unit tests and optional full-checkpoint numerical parity."""

import json
import os
from dataclasses import asdict

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mlx_audio.vad.models.nemotron_diarization import Model, ModelConfig
from mlx_audio.vad.models.nemotron_diarization.config import (
    DiarizationModulesConfig,
    EncoderConfig,
    MelConfig,
)
from mlx_audio.vad.models.nemotron_diarization.convert import (
    convert_weights,
    validate_weights,
)


@pytest.fixture
def precise_math():
    # CPU avoids M5 reduced-precision float32 GEMMs without global env changes.
    with mx.stream(mx.cpu):
        yield


def tiny_config(**kwargs):
    return ModelConfig(
        num_speakers=2,
        encoder_config=EncoderConfig(feat_in=8, d_model=32, n_layers=2, n_heads=2),
        modules_config=DiarizationModulesConfig(
            num_speakers=2,
            fc_d_model=32,
            tf_d_model=16,
            chunk_len=3,
            fifo_len=3,
            spkcache_len=8,
            spkcache_update_period=3,
            chunk_right_context=1,
        ),
        processor_config=MelConfig(feature_size=8),
        **kwargs,
    )


def tiny_model(**kwargs):
    mx.random.seed(1)
    model = Model(tiny_config(**kwargs))
    model.eval()
    return model


def torch_forward(weights, cfg, features, lengths):
    """Independent PyTorch evaluation of NeMo's inference equations."""
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    s = {
        k: torch.from_numpy(np.array(v.astype(mx.float32))) for k, v in weights.items()
    }

    def linear(x, prefix):
        return F.linear(x, s[prefix + ".weight"], s.get(prefix + ".bias"))

    def norm(x, prefix):
        return F.layer_norm(
            x, (x.shape[-1],), s[prefix + ".weight"], s[prefix + ".bias"], 1e-5
        )

    ec = cfg.encoder_config
    x = torch.from_numpy(features).transpose(1, 2)
    x = F.pad(x, (0, 0, 0, -x.shape[1] % ec.subsampling_factor))
    b, _, c = x.shape
    x = linear(x.reshape(b, -1, c * ec.subsampling_factor), "encoder.pre_encode.proj")
    if ec.xscaling:
        x = x * ec.d_model**0.5
    if ec.pre_block_norm:
        x = norm(x, "encoder.embed_norm")
    valid = (
        torch.arange(x.shape[1])[None]
        < (torch.tensor(lengths)[:, None] + ec.subsampling_factor - 1)
        // ec.subsampling_factor
    )
    dim = ec.d_model // ec.n_heads
    rot_dim = int(dim * ec.rotary_fraction)
    inv_freq = 1 / (ec.rope_base ** (torch.arange(0, rot_dim, 2).float() / rot_dim))
    freqs = torch.outer(torch.arange(x.shape[1]).float(), inv_freq)
    angles = torch.cat([freqs, freqs], dim=-1)

    def rotate(x):
        r, tail = x[..., :rot_dim], x[..., rot_dim:]
        half = rot_dim // 2
        rotated = torch.cat([-r[..., half:], r[..., :half]], dim=-1)
        return torch.cat([r * angles.cos() + rotated * angles.sin(), tail], dim=-1)

    for i in range(ec.n_layers):
        p = f"encoder.layers.{i}"
        h = norm(x, p + ".norm1")
        q, k, v = (
            linear(h, p + ".attn.w_qkv")
            .reshape(b, -1, 3, ec.n_heads, dim)
            .permute(2, 0, 3, 1, 4)
        )
        if ec.qk_norm:
            q, k = norm(q, p + ".attn.q_norm"), norm(k, p + ".attn.k_norm")
        y = F.scaled_dot_product_attention(
            rotate(q), rotate(k), v, attn_mask=valid[:, None, None]
        )
        x = x + linear(
            y.transpose(1, 2).reshape(b, -1, ec.d_model), p + ".attn.out_proj"
        )
        h = norm(x, p + ".norm2")
        x = x + linear(F.gelu(linear(h, p + ".ffn.linear1")), p + ".ffn.linear2")
    x = norm(x, "encoder.final_norm")
    p = "sortformer_modules"
    x = linear(x, p + ".encoder_proj")
    h = x.shape[-1]
    x = F.conv1d(
        x.transpose(1, 2),
        s[p + ".subpixel_upsample.weight"].permute(0, 2, 1),
        s[p + ".subpixel_upsample.bias"],
        padding=1,
    )
    x = x.transpose(1, 2).reshape(b, -1, h)
    x = F.relu(linear(F.relu(x), p + ".first_hidden_to_hidden"))
    probs = torch.sigmoid(linear(x, p + ".single_hidden_to_spks"))
    valid = torch.arange(probs.shape[1])[None] < torch.tensor(lengths)[:, None]
    return (probs * valid[..., None]).numpy()


def test_config_and_loader_registration(tmp_path):
    from mlx_audio.utils import get_model_category
    from mlx_audio.vad import load

    cfg = tiny_config()
    assert ModelConfig.from_dict(asdict(cfg)) == cfg
    assert get_model_category("nemotron_diarization", []) == "vad"
    model = tiny_model()
    (tmp_path / "config.json").write_text(json.dumps(asdict(cfg)))
    mx.save_safetensors(
        str(tmp_path / "model.safetensors"), dict(tree_flatten(model.parameters()))
    )
    restored = load(tmp_path, strict=True)
    assert isinstance(restored, Model)


@pytest.mark.parametrize("output_factor", [0, -1, 5, True])
def test_invalid_output_stride(output_factor):
    with pytest.raises(ValueError):
        tiny_config(output_subsampling_factor=output_factor)


def test_forward_parity_and_padding(precise_math):
    model = tiny_model()
    features = np.random.default_rng(2).normal(size=(2, 8, 27)).astype(np.float32)
    lengths = [27, 17]
    actual = model(mx.array(features), mx.array(lengths))
    expected = torch_forward(
        dict(tree_flatten(model.parameters())), model.config, features, lengths
    )
    np.testing.assert_allclose(np.array(actual), expected, atol=2e-6, rtol=2e-5)
    assert np.all(np.array(actual)[1, 17:] == 0)


def test_feature_stacking_order(precise_math):
    model = tiny_model()
    x = mx.arange(8 * 11, dtype=mx.float32).reshape(1, 8, 11)
    out, lengths = model.encoder.pre_encode(x, mx.array([11]))
    padded = np.pad(np.array(x).transpose(0, 2, 1), ((0, 0), (0, 5), (0, 0)))
    expected = (
        padded.reshape(1, 2, 64) @ np.array(model.encoder.pre_encode.proj.weight).T
    )
    np.testing.assert_allclose(np.array(out), expected, atol=1e-4)
    assert lengths.item() == 2


def test_frontend_matches_torch_stft_and_streaming_suffix(precise_math):
    torch = pytest.importorskip("torch")
    model = tiny_model()
    audio = np.random.default_rng(3).normal(0, 0.1, 16543).astype(np.float32)
    actual = model.preprocessor(mx.array(audio))
    x = torch.from_numpy(audio)
    x = torch.cat([x[:1], x[1:] - 0.97 * x[:-1]])
    window = torch.from_numpy(np.array(model.preprocessor.window))
    spec = torch.stft(
        x, 512, 160, 400, window, center=True, pad_mode="constant", return_complex=True
    )
    fb = torch.from_numpy(np.array(model.preprocessor.fb[0]))
    expected = torch.log(fb @ spec.abs().square() + 2**-24)[:, : len(audio) // 160]
    np.testing.assert_allclose(
        np.array(actual[0]), expected.numpy(), atol=2e-5, rtol=2e-5
    )
    start, count = 30, 22
    offset = start * 160 - 257
    suffix = model.preprocessor(
        mx.array(audio[offset:]), start, count, offset, len(audio)
    )
    np.testing.assert_allclose(
        np.array(actual[:, :, start : start + count]), np.array(suffix), atol=1e-6
    )


def test_stream_partition_invariance_and_bounded_state():
    model = tiny_model(output_subsampling_factor=3)
    audio = np.random.default_rng(4).normal(0, 0.05, 40173).astype(np.float32)
    whole = model.generate(audio)
    state = model.init_streaming_state()
    outputs = []
    for i in range(0, len(audio), 317):
        result, state = model.feed(audio[i : i + 317], state)
        outputs.append(result.speaker_probs)
        assert state.spkcache.shape[1] <= 8
        assert state.fifo.shape[1] <= 3
        assert state.audio_buffer.shape[0] < 8 * 4 * 160 + 1024
    result, state = model.feed([], state, final=True)
    outputs.append(result.speaker_probs)
    actual = mx.concatenate(outputs)
    assert state.spkcache_compressed
    assert state.finished and state.audio_buffer.size == 0
    assert actual.shape == ((len(audio) // 160 + 2) // 3, 2)
    np.testing.assert_allclose(
        np.array(actual), np.array(whole.speaker_probs), atol=2e-6
    )
    with pytest.raises(ValueError, match="finished"):
        model.feed([], state)


def test_empty_short_and_invalid_audio():
    model = tiny_model()
    for n in [0, 1, 159]:
        result = model.generate(np.zeros(n, np.float32))
        assert result.speaker_probs.shape == (0, 2)
        assert result.segments == []
    result = model.generate(np.zeros(161, np.float32))
    assert result.speaker_probs.shape == (1, 2)
    with pytest.raises(ValueError, match="sample|Hz"):
        model.feed([], model.init_streaming_state(), sample_rate=8000)
    with pytest.raises(ValueError, match="mono"):
        model.feed(np.zeros((10, 2)), model.init_streaming_state())


def test_conversion_bfloat16_layout_and_strict_validation():
    torch = pytest.importorskip("torch")
    source = {
        "encoder.layers.0.ffn.net.0.weight": torch.ones((4, 2), dtype=torch.bfloat16),
        "sortformer_modules.subpixel_upsample.weight": torch.arange(24).reshape(
            4, 2, 3
        ),
        "preprocessor.featurizer.window": torch.ones((400,), dtype=torch.bfloat16),
    }
    weights = convert_weights(source, "float16")
    assert weights["encoder.layers.0.ffn.linear1.weight"].dtype == mx.float16
    assert weights["preprocessor.window"].dtype == mx.float32
    assert weights["sortformer_modules.subpixel_upsample.weight"].shape == (4, 3, 2)
    np.testing.assert_array_equal(
        np.array(weights["sortformer_modules.subpixel_upsample.weight"]),
        source["sortformer_modules.subpixel_upsample.weight"].permute(0, 2, 1).numpy(),
    )
    model = tiny_model()
    weights = dict(tree_flatten(model.parameters()))
    validate_weights(model, weights)
    with pytest.raises(ValueError, match="Missing"):
        validate_weights(model, {})
    weights["unknown.weight"] = mx.ones((2,))
    with pytest.raises(ValueError, match="Unexpected"):
        validate_weights(model, weights)


@pytest.mark.requires_weights
def test_full_checkpoint_parity(precise_math):
    path = os.environ.get("NEMOTRON_DIARIZATION_MODEL")
    if not path:
        pytest.skip(
            "Set NEMOTRON_DIARIZATION_MODEL to a local converted fp32 checkpoint"
        )
    from mlx_audio.vad import load

    model = load(path, strict=True)
    audio = np.random.default_rng(9).normal(0, 0.1, 32017).astype(np.float32)
    features = np.array(model.preprocessor(mx.array(audio)))
    lengths = [features.shape[-1]]
    actual = model(mx.array(features), mx.array(lengths))
    expected = torch_forward(
        dict(tree_flatten(model.parameters())), model.config, features, lengths
    )
    error = np.abs(np.array(actual) - expected)
    print(
        f"Full checkpoint probability error: max={error.max():.8g}, mean={error.mean():.8g}"
    )
    np.testing.assert_allclose(np.array(actual), expected, atol=2e-5, rtol=2e-4)


def test_final_padding_matches_single_window():
    model = tiny_model()
    audio = np.random.default_rng(12).normal(0, 0.1, 17 * 160 + 31).astype(np.float32)
    # NeMo's centered STFT produces 18 physical frames, then pads to 32.
    features = model.preprocessor(mx.array(audio), count=32)
    expected = model(features, mx.array([17]))[0, :17]
    actual = model.generate(audio).speaker_probs
    np.testing.assert_allclose(np.array(actual), np.array(expected), atol=2e-6)


def test_learned_silence_cache_and_frozen_scores():
    model = tiny_model()
    state = model.init_streaming_state()
    # Force all frames to silence. Every compressed slot must use the learned vector.
    head = model.sortformer_modules.single_hidden_to_spks
    head.weight = mx.zeros_like(head.weight)
    head.bias = mx.full(head.bias.shape, -20.0)
    silence = mx.arange(32, dtype=mx.float32)
    model.sortformer_modules.learnable_sil_emb = silence
    features = mx.zeros((1, 8, 32))
    for _ in range(4):
        model.streaming_step(features, state, 24)
    assert state.spkcache_compressed
    np.testing.assert_array_equal(
        np.array(state.spkcache), np.broadcast_to(np.array(silence), (1, 8, 32))
    )
    np.testing.assert_array_equal(np.array(state.spkcache_preds), np.zeros((1, 8, 2)))
    # Once compressed, predictions for cached slots are retained until replacement.
    previous = np.array(state.spkcache_preds)
    head.bias = mx.full(head.bias.shape, 20.0)
    # Avoid overflow so no new compression hides the frozen-score behavior.
    model.config.modules_config.fifo_len = 100
    model.streaming_step(features, state, 24)
    np.testing.assert_array_equal(np.array(state.spkcache_preds), previous)


def test_streaming_presets():
    model = tiny_model()
    expected = {
        "offline": (340, 40),
        "low": (9, 4),
        "very_low": (6, 2),
        "ultra_low": (3, 1),
    }
    for preset, (chunk, right) in expected.items():
        model.set_streaming_config(preset)
        assert model.config.modules_config.chunk_len == chunk
        assert model.config.modules_config.chunk_right_context == right
    with pytest.raises(ValueError, match="Unknown preset"):
        model.set_streaming_config("unknown")


@pytest.mark.parametrize("quantize", [False, True])
def test_local_nemo_conversion_roundtrip(tmp_path, quantize):
    import io
    import tarfile

    torch = pytest.importorskip("torch")
    yaml = pytest.importorskip("yaml")
    from mlx_audio.vad import load
    from mlx_audio.vad.models.nemotron_diarization.convert import convert

    model = tiny_model()
    state = {}
    for key, value in tree_flatten(model.parameters()):
        arr = np.array(value)
        if key == "sortformer_modules.subpixel_upsample.weight":
            arr = arr.transpose(0, 2, 1)
        key = key.replace("preprocessor.", "preprocessor.featurizer.")
        key = key.replace(".ffn.linear1.", ".ffn.net.0.").replace(
            ".ffn.linear2.", ".ffn.net.3."
        )
        key = key.replace(".activity_head.layers.", ".activity_head.")
        state[key] = torch.from_numpy(arr.copy()).to(torch.bfloat16)
    source = {
        "encoder": {
            **asdict(model.config.encoder_config),
            "subsampling": "feature_stacking",
            "self_attention_model": "rope",
        },
        "high_resolution": True,
        "preprocessor": {
            "normalize": "NA",
            "features": 8,
            "sample_rate": 16000,
            "window_size": 0.025,
            "window_stride": 0.01,
        },
        "sortformer_modules": {**asdict(model.config.modules_config), "num_spks": 2},
    }
    checkpoint = io.BytesIO()
    torch.save(state, checkpoint)
    archive = tmp_path / "model.nemo"
    with tarfile.open(archive, "w:gz") as tar:
        for name, data in [
            ("model_config.yaml", yaml.safe_dump(source).encode()),
            ("model_weights.ckpt", checkpoint.getvalue()),
        ]:
            member = tarfile.TarInfo("nested/" + name)
            member.size = len(data)
            tar.addfile(member, io.BytesIO(data))
    output = convert(
        archive,
        tmp_path / "converted",
        model_id="example/model",
        dtype="bfloat16",
        quantize=quantize,
    )
    assert {p.name for p in output.iterdir()} == {
        "config.json",
        "model.safetensors",
        "README.md",
    }
    restored = load(output, strict=True)
    assert restored.dtype == mx.bfloat16
    config = json.loads((output / "config.json").read_text())
    if quantize:
        assert config["quantization"] == {"bits": 8, "group_size": 64, "mode": "affine"}
        assert restored.encoder.pre_encode.proj.weight.dtype == mx.uint32
        assert "8-bit affine" in (output / "README.md").read_text()
    else:
        assert "quantization" not in config
    result = restored.generate(np.zeros(3200, dtype=np.float32))
    assert result.speaker_probs.shape == (20, 2)
    assert np.isfinite(np.array(result.speaker_probs)).all()
    assert 'load("example/model", strict=True)' in (output / "README.md").read_text()
    assert "license: openmdw-1.1" in (output / "README.md").read_text()
