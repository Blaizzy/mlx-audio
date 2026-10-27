"""Offline regression tests for the published Phonon-2.5 container format."""

import hashlib
import io
import json
import tarfile
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
import zstandard
from mlx.utils import tree_flatten

from mlx_audio.registry import classify_model
from mlx_audio.stt import load
from mlx_audio.stt.models.phonon2.container import (
    CONTAINER_FORMAT,
    _five_value,
    _intn,
    read_container,
)
from mlx_audio.stt.models.phonon2.transport import prepare_model_path
from mlx_audio.stt.tests.test_phonon2 import checkpoint as checkpoint  # noqa: PLC0414
from mlx_audio.stt.tests.test_phonon2 import packed_weights


def _encode_five_value(sign, high, lo, hi):
    codes = sign.astype(np.int16) + 1
    codes = np.pad(codes, ((0, 0), (0, -codes.shape[1] % 5)), constant_values=1)
    trits = (codes.reshape(sign.shape[0], -1, 5) * 3 ** np.arange(5)).sum(axis=-1)
    mask = np.packbits(high[sign != 0], bitorder="little")
    return (
        trits.astype(np.uint8).tobytes() + mask.tobytes() + lo.tobytes() + hi.tobytes()
    )


def _encode_intn(codes, scales, bits):
    if bits == 8:
        body = codes.astype(np.int8).tobytes()
    else:
        unsigned = np.pad(codes.reshape(-1).astype(np.int32) + 32, (0, -codes.size % 4))
        body = b"".join(
            sum(int(q) << (6 * j) for j, q in enumerate(group)).to_bytes(3, "little")
            for group in unsigned.reshape(-1, 4)
        )
    return body + scales.tobytes()


def _container_bytes(records):
    index = [
        {"n": name, "k": kind, "shape": list(shape), "b": len(blob)}
        for name, kind, shape, blob in records
    ]
    header = json.dumps({"format": CONTAINER_FORMAT, "index": index}).encode()
    return len(header).to_bytes(8, "little") + header + b"".join(r[3] for r in records)


@pytest.mark.parametrize("bits", [6, 8])
@pytest.mark.parametrize("shape", [(3, 5), (2, 3, 3, 1)])
def test_integer_tables_match_independent_dense_math(bits, shape):
    half = 2 ** (bits - 1)
    codes = np.resize(np.array([-half, -1, 0, 1, half - 1], np.int8), shape)
    scales = np.linspace(0.125, 0.375, shape[0], dtype=np.float16)
    blob = _encode_intn(codes, scales, bits)
    actual = _intn(blob, shape, bits)
    expected = (
        codes.reshape(shape[0], -1).astype(np.float32)
        * scales.astype(np.float32)[:, None]
    )
    np.testing.assert_array_equal(actual, expected.reshape(shape))
    with pytest.raises(ValueError, match="size"):
        _intn(blob[:-1], shape, bits)


@pytest.mark.parametrize("nonzero", [False, True])
def test_five_value_container_preserves_exact_planes_and_palettes(nonzero):
    rng = np.random.default_rng(25)
    shape = (3, 128)  # The final base-3 byte contains padding trits.
    sign = (
        rng.integers(-1, 2, shape, dtype=np.int8)
        if nonzero
        else np.zeros(shape, np.int8)
    )
    high = rng.integers(0, 2, shape).astype(bool) & (sign != 0)
    lo = np.array([0.1, 0.2, 0.3], np.float16)
    hi = np.array([0.4, 0.5, 0.6], np.float16)
    blob = _encode_five_value(sign, high, lo, hi)
    actual = _five_value(blob, shape)
    base, residual = packed_weights(sign, high)
    for key, expected in [
        ("base_q", base),
        ("residual_q", residual),
        ("row_lo", lo),
        ("row_hi", hi),
    ]:
        np.testing.assert_array_equal(actual[key], expected)
        assert actual[key].dtype == expected.dtype
    with pytest.raises(ValueError, match="record"):
        _five_value(blob[:-1], shape)


@pytest.fixture
def container_checkpoint(checkpoint, tmp_path):
    """Express the tiny v2 checkpoint in the upstream v2.5 container layout."""
    config = json.loads((checkpoint / "config.json").read_text())
    weights = mx.load(str(checkpoint / "model.safetensors"))
    records = []
    attention = {
        "linear_q": "q_proj",
        "linear_k": "k_proj",
        "linear_v": "v_proj",
        "linear_out": "o_proj",
        "linear_pos": "relative_k_proj",
        "pos_bias_u": "bias_u",
        "pos_bias_v": "bias_v",
    }

    def source_name(name):
        for src, dst in attention.items():
            name = name.replace(f".self_attn.{src}", f".self_attn.{dst}")
        return name.replace(".conv.batch_norm.", ".conv.norm.")

    for row in config.pop("phonon2_modules"):
        shape = tuple(row["shape"])
        sign = np.resize(np.array([-1, 0, 1], np.int8), shape)
        high = np.resize(np.array([True, False, False, False, False, True]), shape) & (
            sign != 0
        )
        lo = np.full(shape[0], 0.125, np.float16)
        hi = np.full(shape[0], 0.375, np.float16)
        records.append(
            (
                source_name(row["name"]),
                "five_value",
                shape,
                _encode_five_value(sign, high, lo, hi),
            )
        )
    for name, weight in weights.items():
        if name.endswith((".base_q", ".residual_q", ".row_lo", ".row_hi")):
            continue
        value = np.asarray(weight.astype(mx.float32)).astype(np.float16)
        source = source_name(name)
        if name.startswith("encoder.pre_encode.conv."):
            source = name.replace(
                "encoder.pre_encode.conv.", "encoder.subsampling.layers."
            )
            if name.endswith(".weight"):
                value = value.transpose(0, 3, 1, 2)
        elif name.startswith("encoder.pre_encode.out."):
            source = name.replace(
                "encoder.pre_encode.out.", "encoder.subsampling.linear."
            )
        elif name.endswith(".conv.depthwise_conv.weight"):
            value = value.transpose(0, 2, 1)
        elif name.startswith("joint.enc."):
            source = name.replace("joint.enc.", "encoder_projector.")
        elif name.startswith("joint.pred."):
            source = name.replace("joint.pred.", "decoder.decoder_projector.")
        elif name.startswith("joint.joint_net.2."):
            source = name.replace("joint.joint_net.2.", "joint.head.")
        elif name == "decoder.prediction.embed.weight":
            source = "decoder.embedding.weight"
            codes = np.resize(np.array([-32, -1, 0, 1, 31], np.int8), value.shape)
            scales = np.full(value.shape[0], 0.125, np.float16)
            records.append(
                (source, "int6", value.shape, _encode_intn(codes, scales, 6))
            )
            continue
        elif name.startswith("decoder.prediction.dec_rnn.lstm."):
            layer, leaf = name.split(".")[-2:]
            if leaf == "bias":
                for side in ("ih", "hh"):
                    half = value / np.float16(2)
                    records.append(
                        (
                            f"decoder.lstm.bias_{side}_l{layer}",
                            "fp16",
                            half.shape,
                            half.tobytes(),
                        )
                    )
                continue
            side = "ih" if leaf == "Wx" else "hh"
            source = f"decoder.lstm.weight_{side}_l{layer}"
        records.append((source, "fp16", value.shape, value.tobytes()))
    records.append(
        (
            "encoder.layers.0.conv.norm.num_batches_tracked",
            "fp16",
            (),
            np.float16(np.inf).tobytes(),
        )
    )
    blob = _container_bytes(records)
    config.update(
        model_type="parakeet",
        fermion={
            "container_format": CONTAINER_FORMAT,
            "container_sha256": hashlib.sha256(blob).hexdigest(),
        },
    )
    path = tmp_path / "unrelated-name"
    path.mkdir()
    (path / "model.fermion").write_bytes(blob)
    (path / "config.json").write_text(json.dumps(config))
    return path


def test_unpacked_container_loads_with_exact_weights(
    checkpoint, container_checkpoint, tmp_path, monkeypatch
):
    monkeypatch.setenv("MLX_AUDIO_CACHE_DIR", str(tmp_path / "cache"))
    expected = dict(tree_flatten(load(checkpoint).parameters()))
    model = load(container_checkpoint, strict=True)
    actual = dict(tree_flatten(model.parameters()))
    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        if name == "decoder.prediction.embed.weight":
            codes = np.resize(np.array([-32, -1, 0, 1, 31], np.int8), value.shape)
            value = mx.array(codes.astype(np.float32) * 0.125).astype(mx.bfloat16)
        assert actual[name].dtype == value.dtype
        np.testing.assert_array_equal(
            np.asarray(actual[name].astype(mx.float32)),
            np.asarray(value.astype(mx.float32)),
            err_msg=name,
        )
    assert len(model._packed_modules) == 11
    assert model.generate(mx.zeros(16000)).text == ""
    # Materialization is reusable without reading the container again.
    from mlx_audio.stt.models.phonon2 import transport

    monkeypatch.setattr(
        transport, "read_container", lambda _: pytest.fail("cache was not reused")
    )
    assert prepare_model_path(container_checkpoint) == Path(model.config.model_path)


@pytest.fixture
def release_checkpoint(container_checkpoint, tmp_path):
    files = {
        name: (container_checkpoint / name).read_bytes()
        for name in ["config.json", "model.fermion"]
    }
    manifest = {
        "release_format": "phonon-byteplane-tar-zstd-v1",
        "files": [
            {
                "path": name,
                "original_bytes": len(data),
                "original_sha256": hashlib.sha256(data).hexdigest(),
            }
            for name, data in files.items()
        ],
    }
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for name, data in [
            ("bps_manifest.json", json.dumps(manifest).encode()),
            *files.items(),
        ]:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    archive = zstandard.ZstdCompressor().compress(buffer.getvalue())
    path = tmp_path / "release"
    path.mkdir()
    filename = "phonon-2.5.bps.tar.zst"
    config = {
        "model_type": "parakeet_tdt_five_value",
        "model_id": "test/phonon-2.5",
        "languages": ["en", "es"],
        "artifact": {
            "filename": filename,
            "sha256": hashlib.sha256(archive).hexdigest(),
            "unpacked_bytes": sum(map(len, files.values())),
            "container_sha256": hashlib.sha256(files["model.fermion"]).hexdigest(),
        },
    }
    (path / filename).write_bytes(archive)
    (path / "config.json").write_text(json.dumps(config))
    return path


def test_release_loads_and_downloads_only_selected_archive(
    release_checkpoint, tmp_path, monkeypatch
):
    from mlx_audio.stt.models.phonon2 import transport

    monkeypatch.setenv("MLX_AUDIO_CACHE_DIR", str(tmp_path / "cache"))
    config = json.loads((release_checkpoint / "config.json").read_text())
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    (metadata / "config.json").write_text(json.dumps(config))
    calls = []

    def download(repo, **kwargs):
        calls.append((repo, kwargs))
        return str(release_checkpoint)

    monkeypatch.setattr(transport, "snapshot_download", download)
    materialized = prepare_model_path(
        metadata, source="test/release", revision="pinned", force_download=True
    )
    assert calls == [
        (
            "test/release",
            {
                "revision": "pinned",
                "allow_patterns": [config["artifact"]["filename"]],
                "force_download": True,
            },
        )
    ]
    model = load(materialized, strict=True)
    assert model.config.languages == ["en", "es"]
    assert model.generate(mx.zeros(1)).text == ""
    assert not (materialized / "model.fermion").exists()
    assert classify_model("parakeet_tdt_five_value") == "stt"
    # Exercise the full standard-loader path for release configs too.
    assert load(release_checkpoint).config.languages == ["en", "es"]


@pytest.mark.parametrize("kind", ["archive", "container"])
def test_checksums_rejected_before_materializing(
    release_checkpoint, container_checkpoint, tmp_path, monkeypatch, kind
):
    monkeypatch.setenv("MLX_AUDIO_CACHE_DIR", str(tmp_path / "cache"))
    path = release_checkpoint if kind == "archive" else container_checkpoint
    config = json.loads((path / "config.json").read_text())
    filename = config["artifact"]["filename"] if kind == "archive" else "model.fermion"
    with (path / filename).open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        load(path)
    assert not (tmp_path / "cache").exists()


@pytest.mark.parametrize(
    "corruption", ["duplicate", "truncated", "trailing", "unsupported"]
)
def test_malformed_container_rejected(tmp_path, corruption):
    record = ("encoder_projector.bias", "fp16", (2,), np.zeros(2, np.float16).tobytes())
    records = [record, record] if corruption == "duplicate" else [record]
    if corruption == "unsupported":
        records = [(record[0], "int4", *record[2:])]
    blob = _container_bytes(records)
    if corruption == "truncated":
        blob = blob[:-1]
    elif corruption == "trailing":
        blob += b"x"
    path = tmp_path / "model.fermion"
    path.write_bytes(blob)
    with pytest.raises(ValueError):
        read_container(path)
