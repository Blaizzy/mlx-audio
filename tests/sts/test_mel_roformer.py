# Copyright (c) 2026 Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

"""Tests for Mel-Band-RoFormer vocal separation model."""

import pytest

from mlx_audio.sts.models.mel_roformer import MelRoFormer, MelRoFormerConfig


class TestMelRoFormerConfig:
    def test_default_hyperparameters(self):
        """Verify the dataclass defaults match the canonical Mel-Band-RoFormer config."""
        config = MelRoFormerConfig()
        assert config.dim == 384
        assert config.depth == 6
        assert config.heads == 8
        assert config.dim_head == 64
        assert config.num_bands == 60
        assert config.n_fft == 2048
        assert config.hop_length == 441
        assert config.sample_rate == 44100

    def test_derived_properties(self):
        config = MelRoFormerConfig()
        assert config.dim_inner == 512  # 8 * 64
        assert config.ff_dim == 1536  # 384 * 4
        assert config.mlp_hidden == 1536
        assert config.freq_bins == 1025  # 2048/2 + 1

    def test_kim_vocal_2_preset(self):
        config = MelRoFormerConfig.kim_vocal_2()
        assert config.depth == 6
        assert config.dim == 384
        assert config.checkpoint_family == "kim_vocal_2"

    def test_viperx_vocals_preset(self):
        config = MelRoFormerConfig.viperx_vocals()
        assert config.depth == 12
        assert config.dim == 384
        assert config.checkpoint_family == "viperx_vocals"

    def test_zfturbo_bs_roformer_preset(self):
        config = MelRoFormerConfig.zfturbo_bs_roformer()
        assert config.depth == 12
        assert config.checkpoint_family == "zfturbo_bs_roformer"

    def test_custom_preset(self):
        # The custom() escape hatch for non-standard variants
        config = MelRoFormerConfig.custom(depth=8, num_bands=48)
        assert config.depth == 8
        assert config.num_bands == 48
        assert config.checkpoint_family == "custom"

    def test_custom_keyword_only(self):
        """custom() requires depth as a keyword argument."""
        with pytest.raises(TypeError):
            # Positional depth should fail
            MelRoFormerConfig.custom(8)


class TestMelRoFormerModel:
    def test_model_creation_kim_vocal_2(self):
        config = MelRoFormerConfig.kim_vocal_2()
        model = MelRoFormer(config)
        assert len(model.layers) == 6
        assert len(model.mask_estimators) == 1

    def test_model_creation_viperx_vocals(self):
        config = MelRoFormerConfig.viperx_vocals()
        model = MelRoFormer(config)
        assert len(model.layers) == 12

    def test_model_creation_zfturbo(self):
        config = MelRoFormerConfig.zfturbo_bs_roformer()
        model = MelRoFormer(config)
        assert len(model.layers) == 12

    @pytest.mark.skip(reason="Forward pass requires GPU")
    def test_forward_shape(self):
        import mlx.core as mx

        config = MelRoFormerConfig.kim_vocal_2()
        model = MelRoFormer(config)
        audio = mx.random.normal((1, 2, 44100))  # 1 second stereo
        output = model(audio)
        assert output.shape == (1, 2, 44100)


class TestSanitize:
    def test_qkv_split(self):
        """Verify sanitize() splits packed to_qkv weights into to_q, to_k, to_v."""
        import mlx.core as mx

        config = MelRoFormerConfig.kim_vocal_2()
        model = MelRoFormer(config)

        # Simulate a PyTorch checkpoint with packed QKV
        inner_dim = config.dim_inner  # 512
        packed = mx.random.normal((inner_dim * 3, config.dim))

        raw_weights = {
            "layers.0.0.layers.0.0.to_qkv.weight": packed,
            "layers.0.0.layers.0.0.to_out.weight": mx.random.normal(
                (config.dim, inner_dim)
            ),
        }

        sanitized = model.sanitize(raw_weights)

        assert "layers.0.0.layers.0.0.to_q.weight" in sanitized
        assert "layers.0.0.layers.0.0.to_k.weight" in sanitized
        assert "layers.0.0.layers.0.0.to_v.weight" in sanitized
        assert "layers.0.0.layers.0.0.to_qkv.weight" not in sanitized
        assert "layers.0.0.layers.0.0.to_out.weight" in sanitized

        assert sanitized["layers.0.0.layers.0.0.to_q.weight"].shape == (
            inner_dim,
            config.dim,
        )
        assert sanitized["layers.0.0.layers.0.0.to_k.weight"].shape == (
            inner_dim,
            config.dim,
        )
        assert sanitized["layers.0.0.layers.0.0.to_v.weight"].shape == (
            inner_dim,
            config.dim,
        )


class TestConvert:
    """Tests for the weight conversion utilities (no GPU, no torch required)."""

    def test_strip_training_keys(self):
        from mlx_audio.sts.models.mel_roformer.convert import _should_strip

        # Training state — should strip
        assert _should_strip("optimizer_states.0.params")
        assert _should_strip("lr_schedulers.0")
        assert _should_strip("ema.foo")
        assert _should_strip("ema_model.bar")
        assert _should_strip("layers.0.bn.num_batches_tracked")

        # Model weights — should keep
        assert not _should_strip("band_split.to_features.0.0.gamma")
        assert not _should_strip("layers.0.0.layers.0.0.to_qkv.weight")
        assert not _should_strip("mask_estimators.0.to_freqs.0.0.0.weight")

    def test_extract_state_dict_plain(self):
        from mlx_audio.sts.models.mel_roformer.convert import _extract_state_dict

        raw = {"key1": "value1", "key2": "value2"}
        result = _extract_state_dict(raw)
        assert result == raw

    def test_extract_state_dict_lightning(self):
        from mlx_audio.sts.models.mel_roformer.convert import _extract_state_dict

        raw = {
            "state_dict": {"foo.weight": "tensor1", "bar.bias": "tensor2"},
            "optimizer_states": "discarded",
        }
        result = _extract_state_dict(raw)
        assert "foo.weight" in result
        assert "bar.bias" in result
        assert "optimizer_states" not in result

    def test_extract_state_dict_model_prefix(self):
        from mlx_audio.sts.models.mel_roformer.convert import _extract_state_dict

        raw = {
            "state_dict": {
                "model.band_split.gamma": "tensor1",
                "model.layers.0.weight": "tensor2",
            }
        }
        result = _extract_state_dict(raw)
        assert "band_split.gamma" in result
        assert "layers.0.weight" in result
        assert "model.band_split.gamma" not in result


class TestLicenseDetection:
    """Tests for the license-aware warnings in convert.py."""

    def test_detects_kim_vocal_2(self):
        from pathlib import Path

        from mlx_audio.sts.models.mel_roformer.convert import _detect_license

        hint = _detect_license(
            Path("/downloads/KimberleyJSN_melbandroformer/model.ckpt")
        )
        assert hint is not None
        substring, preset, license_tag, note = hint
        assert preset == "kim_vocal_2"
        # Kim Vocal 2 was relicensed MIT in April 2026 (previously GPL-3.0)
        assert license_tag == "MIT"

    def test_detects_viperx(self):
        from pathlib import Path

        from mlx_audio.sts.models.mel_roformer.convert import _detect_license

        hint = _detect_license(
            Path("/downloads/TRvlvr_model_repo/mel_band_roformer.ckpt")
        )
        assert hint is not None
        _, preset, license_tag, _ = hint
        assert preset == "viperx_vocals"
        assert license_tag == "undeclared"

    def test_detects_zfturbo(self):
        from pathlib import Path

        from mlx_audio.sts.models.mel_roformer.convert import _detect_license

        hint = _detect_license(Path("/downloads/ZFTurbo_release/model.ckpt"))
        assert hint is not None
        _, preset, license_tag, _ = hint
        assert preset == "zfturbo_bs_roformer"
        assert "MIT" in license_tag

    def test_no_match_returns_none(self):
        from pathlib import Path

        from mlx_audio.sts.models.mel_roformer.convert import _detect_license

        hint = _detect_license(Path("/tmp/unknown_checkpoint.ckpt"))
        assert hint is None


class TestContentAddressing:
    """Tests for the content-addressed output naming in convert.py."""

    def test_content_addressed_name(self, tmp_path):
        from mlx_audio.sts.models.mel_roformer.convert import _content_addressed_name

        input_path = tmp_path / "my_checkpoint.ckpt"
        digest = "abc123def456" + "0" * 52  # 64-char hex

        # Dtype is included so multi-precision conversions of the same source
        # don't collide.
        name = _content_addressed_name(input_path, digest, "bfloat16")
        assert name == "my_checkpoint.abc123de.bfloat16"

        name_fp32 = _content_addressed_name(input_path, digest, "float32")
        assert name_fp32 == "my_checkpoint.abc123de.float32"

    def test_sha256_of_file(self, tmp_path):
        from mlx_audio.sts.models.mel_roformer.convert import _sha256_of_file

        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"hello world")

        digest = _sha256_of_file(test_file)
        # Known SHA-256 of "hello world"
        assert (
            digest == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        )

    def test_sha256_chunked_consistency(self, tmp_path):
        """Hash should be the same regardless of read chunk size."""
        from mlx_audio.sts.models.mel_roformer.convert import _sha256_of_file

        test_file = tmp_path / "larger.bin"
        test_file.write_bytes(b"x" * (1 << 21))  # 2 MB, larger than default chunk

        digest1 = _sha256_of_file(test_file, chunk_size=1 << 20)
        digest2 = _sha256_of_file(test_file, chunk_size=1 << 10)
        assert digest1 == digest2


class TestConfigSerialization:
    """Round-trip config → dict → config for companion config.json."""

    def test_roundtrip(self):
        from dataclasses import fields

        from mlx_audio.sts.models.mel_roformer.convert import _config_to_dict

        config = MelRoFormerConfig.kim_vocal_2()
        data = _config_to_dict(config)

        # All dataclass fields should be in the dict
        for f in fields(MelRoFormerConfig):
            assert f.name in data

        # Reconstructing from the dict should give an equivalent config
        rebuilt = MelRoFormerConfig(**data)
        assert rebuilt.depth == config.depth
        assert rebuilt.dim == config.dim
        assert rebuilt.checkpoint_family == config.checkpoint_family


class TestPresetResolution:
    """Tests for the CLI preset resolver."""

    def test_resolve_known_presets(self):
        from mlx_audio.sts.models.mel_roformer.convert import _resolve_preset

        for name in ["kim_vocal_2", "viperx_vocals", "zfturbo_bs_roformer"]:
            config = _resolve_preset(name)
            assert config.checkpoint_family == name

    def test_resolve_unknown_preset_raises(self):
        from mlx_audio.sts.models.mel_roformer.convert import _resolve_preset

        with pytest.raises(ValueError, match="Unknown preset"):
            _resolve_preset("nonexistent_preset")
