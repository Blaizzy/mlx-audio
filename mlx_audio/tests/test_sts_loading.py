import sys
import types

import pytest


def test_get_model_category_detects_sts():
    from mlx_audio.utils import get_model_category

    category = get_model_category("lfm_audio", ["lfm2.5", "audio", "1.5b", "4bit"])
    assert category == "sts"


def test_get_model_name_parts_splits_underscore_sts_repo_names():
    from mlx_audio.utils import get_model_name_parts

    parts = get_model_name_parts("starkdmi/MossFormer2_SE_48K_MLX-4bit")

    assert "mossformer2_se_48k_mlx" in parts
    assert "mossformer2" in parts
    assert "se" in parts
    assert "4bit" in parts


def test_get_model_name_parts_combines_dash_separated_sts_repo_names():
    from mlx_audio.utils import get_model_name_parts

    parts = get_model_name_parts("mlx-community/sam-audio-large")

    assert "sam" in parts
    assert "audio" in parts
    assert "sam_audio" in parts
    assert "samaudio" in parts


def test_top_level_load_model_routes_sts(monkeypatch):
    import mlx_audio.utils as utils

    sentinel = object()
    empty_utils = types.SimpleNamespace(MODEL_REMAPPING={}, load_model=lambda _: None)
    sts_utils = types.SimpleNamespace(
        MODEL_REMAPPING={"lfm_audio": "lfm_audio"},
        load_model=lambda model_name: sentinel,
    )

    monkeypatch.setattr(
        utils,
        "load_config",
        lambda model_name: {"model_type": "lfm_audio"},
    )
    monkeypatch.setattr(utils, "_get_tts_utils", lambda: empty_utils)
    monkeypatch.setattr(utils, "_get_stt_utils", lambda: empty_utils)
    monkeypatch.setattr(utils, "_get_sts_utils", lambda: sts_utils)
    monkeypatch.setattr(utils, "_get_lid_utils", lambda: empty_utils)
    monkeypatch.setattr(utils, "_get_vad_utils", lambda: empty_utils)

    result = utils.load_model("mlx-community/LFM2.5-Audio-1.5B-4bit")
    assert result is sentinel


def test_sts_load_model_dispatches_lfm_audio(monkeypatch):
    import mlx_audio.sts.utils as sts_utils

    sentinel = object()
    captured = {}

    monkeypatch.setattr(
        sts_utils,
        "load_config",
        lambda model_path, **kwargs: {"model_type": "lfm_audio"},
    )

    def fake_base_load_model(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(sts_utils, "base_load_model", fake_base_load_model)

    result = sts_utils.load_model("mlx-community/LFM2.5-Audio-1.5B-4bit")
    assert result is sentinel
    assert captured["model_path"] == "mlx-community/LFM2.5-Audio-1.5B-4bit"
    assert captured["category"] == "sts"
    assert captured["model_remapping"] is sts_utils.MODEL_REMAPPING


def test_sts_load_model_dispatches_mossformer2_se_via_base_load_model(monkeypatch):
    import mlx_audio.sts.utils as sts_utils

    sentinel = object()
    captured = {}

    def raise_missing_config(model_path, **kwargs):
        raise FileNotFoundError("config missing at repo root")

    monkeypatch.setattr(sts_utils, "load_config", raise_missing_config)

    def fake_base_load_model(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(sts_utils, "base_load_model", fake_base_load_model)

    result = sts_utils.load_model("starkdmi/MossFormer2_SE_48K_MLX-4bit")
    assert result is sentinel
    assert captured["model_path"] == "starkdmi/MossFormer2_SE_48K_MLX-4bit"
    assert captured["model_type"] == "mossformer2_se"
    assert captured["category"] == "sts"


def test_sts_load_model_dispatches_deepfilternet_via_base_load_model(monkeypatch):
    import mlx_audio.sts.utils as sts_utils

    sentinel = object()
    captured = {}
    resolved_path = "/tmp/deepfilter/v3"

    def raise_missing_config(model_path, **kwargs):
        raise FileNotFoundError("config missing at repo root")

    monkeypatch.setattr(sts_utils, "load_config", raise_missing_config)

    def fake_resolve_model_path(model_path, **kwargs):
        assert model_path == "mlx-community/DeepFilterNet-mlx"
        return resolved_path

    def fake_base_load_model(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        sts_utils, "_resolve_deepfilternet_model_path", fake_resolve_model_path
    )
    monkeypatch.setattr(sts_utils, "base_load_model", fake_base_load_model)

    result = sts_utils.load_model("mlx-community/DeepFilterNet-mlx")
    assert result is sentinel
    assert captured["model_path"] == resolved_path
    assert captured["model_type"] == "deepfilternet"
    assert captured["category"] == "sts"


def test_sts_load_model_dispatches_sam_audio_via_base_load_model(monkeypatch):
    import mlx_audio.sts.utils as sts_utils

    sentinel = object()
    captured = {}

    monkeypatch.setattr(
        sts_utils,
        "load_config",
        lambda model_path, **kwargs: {
            "audio_codec": {},
            "text_encoder": {},
            "transformer": {},
        },
    )

    def fake_base_load_model(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(sts_utils, "base_load_model", fake_base_load_model)

    result = sts_utils.load_model("mlx-community/sam-audio-large")
    assert result is sentinel
    assert captured["model_path"] == "mlx-community/sam-audio-large"
    assert captured["model_type"] == "sam_audio"


def test_sts_load_model_dispatches_moshi_via_from_pretrained(monkeypatch):
    import mlx_audio.sts.utils as sts_utils

    sentinel = object()

    class StubMoshiSTSModel:
        @classmethod
        def from_pretrained(
            cls,
            model_name_or_path,
            quantized=None,
            revision=None,
            force_download=False,
        ):
            assert model_name_or_path == "kyutai/moshiko-mlx-q4"
            assert quantized == 4
            assert revision is None
            assert force_download is False
            return sentinel

    stub_module = types.ModuleType("mlx_audio.sts.models.moshi")
    stub_module.MoshiSTSModel = StubMoshiSTSModel

    def raise_missing_config(model_path, **kwargs):
        raise FileNotFoundError("config missing at repo root")

    monkeypatch.setattr(sts_utils, "load_config", raise_missing_config)
    monkeypatch.delitem(sys.modules, "mlx_audio.sts.models.moshi", raising=False)
    monkeypatch.setitem(sys.modules, "mlx_audio.sts.models.moshi", stub_module)

    result = sts_utils.load_model("kyutai/moshiko-mlx-q4")
    assert result is sentinel


def test_top_level_load_model_routes_sts_from_name_when_config_missing(monkeypatch):
    import mlx_audio.utils as utils

    sentinel = object()
    empty_utils = types.SimpleNamespace(MODEL_REMAPPING={}, load_model=lambda _: None)
    sts_utils = types.SimpleNamespace(
        MODEL_REMAPPING={
            "deepfilternet": "deepfilternet",
            "moshiko": "moshi",
            "mossformer2": "mossformer2_se",
            "mossformer2_se": "mossformer2_se",
            "sam_audio": "sam_audio",
        },
        load_model=lambda model_name: sentinel,
        infer_model_type_from_config=lambda config: None,
    )

    def raise_missing_config(model_name):
        raise FileNotFoundError("config missing at repo root")

    monkeypatch.setattr(utils, "load_config", raise_missing_config)
    monkeypatch.setattr(utils, "_get_tts_utils", lambda: empty_utils)
    monkeypatch.setattr(utils, "_get_stt_utils", lambda: empty_utils)
    monkeypatch.setattr(utils, "_get_sts_utils", lambda: sts_utils)
    monkeypatch.setattr(utils, "_get_lid_utils", lambda: empty_utils)
    monkeypatch.setattr(utils, "_get_vad_utils", lambda: empty_utils)

    assert utils.load_model("mlx-community/DeepFilterNet-mlx") is sentinel
    assert utils.load_model("kyutai/moshiko-mlx-q4") is sentinel
    assert utils.load_model("starkdmi/MossFormer2_SE_48K_MLX") is sentinel


def test_top_level_load_model_routes_sts_from_sts_config_heuristics(monkeypatch):
    import mlx_audio.utils as utils

    sentinel = object()
    empty_utils = types.SimpleNamespace(MODEL_REMAPPING={}, load_model=lambda _: None)
    sts_utils = types.SimpleNamespace(
        MODEL_REMAPPING={"sam_audio": "sam_audio"},
        load_model=lambda model_name: sentinel,
        infer_model_type_from_config=lambda config: "sam_audio",
    )

    monkeypatch.setattr(
        utils,
        "load_config",
        lambda model_name: {
            "audio_codec": {},
            "text_encoder": {},
            "transformer": {},
        },
    )
    monkeypatch.setattr(utils, "_get_tts_utils", lambda: empty_utils)
    monkeypatch.setattr(utils, "_get_stt_utils", lambda: empty_utils)
    monkeypatch.setattr(utils, "_get_sts_utils", lambda: sts_utils)
    monkeypatch.setattr(utils, "_get_lid_utils", lambda: empty_utils)
    monkeypatch.setattr(utils, "_get_vad_utils", lambda: empty_utils)

    assert utils.load_model("local-sam-model") is sentinel


def test_sts_load_model_rejects_unsupported_generic_model(monkeypatch):
    import mlx_audio.sts.utils as sts_utils

    monkeypatch.setattr(
        sts_utils,
        "load_config",
        lambda model_path, **kwargs: {"model_type": "unsupported_sts_model"},
    )

    with pytest.raises(ValueError, match="generic STS loader yet"):
        sts_utils.load_model("kyutai/moshiko-mlx-q4")
