import hashlib
import io
import json
import tarfile
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mlx_audio.stt.models.phonon.config import ModelConfig
from mlx_audio.stt.models.phonon.packed import (
    PackedTritLinear,
    packed_bytes_per_row,
    unpack_base5_codes,
)
from mlx_audio.stt.models.phonon.phonon import (
    PhononASRModel,
    configure_packed_modules,
)
from mlx_audio.stt.models.phonon.transport import (
    _extract_archive,
    _join_byte_planes,
    _safe_relative_path,
    is_phonon_model,
    prepare_model_path,
)


def _tiny_config() -> ModelConfig:
    return ModelConfig.from_dict(
        {
            "model_type": "qwen3_asr",
            "audio_config": {
                "num_mel_bins": 16,
                "encoder_layers": 1,
                "encoder_attention_heads": 1,
                "encoder_ffn_dim": 256,
                "d_model": 128,
                "output_dim": 128,
                "downsample_hidden_size": 8,
                "max_source_positions": 64,
            },
            "text_config": {
                "vocab_size": 128,
                "hidden_size": 128,
                "intermediate_size": 256,
                "num_hidden_layers": 1,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "head_dim": 128,
            },
        }
    )


def _tiny_manifest(model: PhononASRModel) -> dict:
    suffixes = (
        "mlp.down_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "self_attn.k_proj",
        "self_attn.o_proj",
        "self_attn.q_proj",
        "self_attn.v_proj",
    )
    rows = []
    for suffix in suffixes:
        module = model.model.layers[0]
        for part in suffix.split("."):
            module = getattr(module, part)
        rows.append(
            {
                "name": f"model.layers.0.{suffix}",
                "in_features": int(module.weight.shape[1]),
                "out_features": int(module.weight.shape[0]),
            }
        )
    return {
        "status": "PASS",
        "group_size": 128,
        "bits": 2,
        "modules": rows,
    }


def _pack_base5(symbols: list[int]) -> np.ndarray:
    payloads = []
    for start in range(0, len(symbols), 10):
        group = symbols[start : start + 10] + [0] * (
            10 - len(symbols[start : start + 10])
        )
        payload = sum(symbol * (5**index) for index, symbol in enumerate(group))
        payloads.extend((payload & 255, (payload >> 8) & 255, (payload >> 16) & 255))
    return np.array([payloads], dtype=np.uint8)


def test_base5_unpack_reconstructs_five_values():
    symbols = ([0, 1, 2, 3, 4] * 26)[:128]
    encoded = mx.array(_pack_base5(symbols))
    assert encoded.shape[1] == packed_bytes_per_row(128)

    base_q, residual_q = unpack_base5_codes(encoded, 128)
    base = mx.dequantize(
        base_q,
        mx.array([[2.0]]),
        mx.array([[-2.0]]),
        group_size=128,
        bits=2,
        mode="affine",
    )
    residual = mx.dequantize(
        residual_q,
        mx.array([[0.5]]),
        mx.array([[-0.5]]),
        group_size=128,
        bits=2,
        mode="affine",
    )
    expected_values = np.array([-2.5, -1.5, 0.0, 1.5, 2.5], dtype=np.float32)
    np.testing.assert_allclose(
        np.asarray(base + residual)[0], expected_values[np.array(symbols)]
    )


def test_manifest_replaces_every_decoder_linear():
    model = PhononASRModel(_tiny_config())
    modules = configure_packed_modules(model, _tiny_manifest(model))

    assert len(modules) == 7
    assert isinstance(model.model.layers[0].self_attn.q_proj, PackedTritLinear)
    assert isinstance(model.model.layers[0].mlp.down_proj, PackedTritLinear)


def test_manifest_rejects_partial_decoder_coverage():
    model = PhononASRModel(_tiny_config())
    manifest = _tiny_manifest(model)
    manifest["modules"].pop()

    with pytest.raises(RuntimeError, match="cover the decoder linears exactly"):
        configure_packed_modules(model, manifest)


def test_byte_plane_join_round_trip():
    header = b"hdr"
    payload = bytes(range(1, 9))
    stored = header + payload[0::2] + payload[1::2]
    metadata = {
        "base": len(header),
        "plan": [[0, len(payload), 2]],
        "payload_bytes": len(payload),
        "original_bytes": len(header) + len(payload),
    }

    assert _join_byte_planes(stored, metadata) == header + payload


def test_published_transport_format_extracts(tmp_path: Path):
    import zstandard

    payload = b'{"model_type": "qwen3_asr"}'
    manifest = {
        "release_format": "sttg1a-byteplane-tar-zstd-v1",
        "files": [
            {
                "path": "config.json",
                "original_bytes": len(payload),
                "original_sha256": hashlib.sha256(payload).hexdigest(),
            }
        ],
    }

    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w") as archive:
        for name, data in (
            ("bps_manifest.json", json.dumps(manifest).encode()),
            ("config.json", payload),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))

    packed = tmp_path / "model.bps.tar.zst"
    packed.write_bytes(zstandard.ZstdCompressor().compress(tar_buffer.getvalue()))
    destination = tmp_path / "model"
    _extract_archive(packed, destination, len(payload))

    assert (destination / "config.json").read_bytes() == payload


@pytest.mark.parametrize(
    "name", ["../weights", "/absolute", "a/../../b", r"..\weights"]
)
def test_transport_rejects_unsafe_paths(name):
    with pytest.raises(RuntimeError, match="unsafe path"):
        _safe_relative_path(name)


def test_materialized_phonon_directory_is_used_directly(tmp_path: Path):
    (tmp_path / "config.json").write_text('{"model_type": "qwen3_asr"}')
    (tmp_path / "packed_manifest.json").write_text(
        json.dumps({"format": "sttg1a-test"})
    )
    (tmp_path / "model.safetensors").write_bytes(b"weights")

    config = json.loads((tmp_path / "config.json").read_text())
    assert is_phonon_model(tmp_path, config)
    assert prepare_model_path(tmp_path) == tmp_path.resolve()
