import importlib
import inspect
from types import MethodType, SimpleNamespace

import mlx.core as mx
import pytest

from mlx_audio.stt.generate import generate_transcription
from mlx_audio.stt.models.fun_asr_nano.config import FunASRNanoConfig
from mlx_audio.stt.models.fun_asr_nano.convert import _convert_weights
from mlx_audio.stt.models.fun_asr_nano.fun_asr_nano import FunASRNano


def test_config_parses_upstream_names():
    config = FunASRNanoConfig.from_dict(
        {
            "audio_encoder_conf": {"output_size": 16, "sanm_shfit": 2},
        }
    )

    assert config.audio_encoder_conf.output_size == 16
    assert config.audio_encoder_conf.sanm_shift == 2


def test_language_mapping_uses_current_model_iso_hints():
    assert FunASRNano._map_language("zh") == "中文"
    assert FunASRNano._map_language("yue") == "中文"
    assert FunASRNano._map_language("en") == "英文"
    assert FunASRNano._map_language("ja") == "日文"
    assert FunASRNano._map_language("auto") is None
    assert FunASRNano._map_language("中文") == "中文"

    with pytest.raises(ValueError, match="Unsupported ISO language"):
        FunASRNano._map_language("ko")


def test_shared_context_api_is_exposed_for_fun_asr_hotwords():
    assert "context" in inspect.signature(FunASRNano.generate).parameters
    assert "context" in inspect.signature(FunASRNano.stream_generate).parameters

    assert FunASRNano._resolve_hotwords(None, " MLX, Apple Silicon ") == [
        "MLX, Apple Silicon"
    ]


@pytest.mark.parametrize(
    ("hotwords", "context", "expected"),
    [
        (None, None, None),
        (None, "", None),
        (None, "   ", None),
        ([], None, None),
        (["", "   "], None, None),
        ([], "MLX", ["MLX"]),
        (["", " MLX ", "Apple Silicon"], None, ["MLX", "Apple Silicon"]),
        (["MLX"], "", ["MLX"]),
        (["MLX"], "   ", ["MLX"]),
    ],
)
def test_shared_context_api_ignores_empty_values(hotwords, context, expected):
    assert FunASRNano._resolve_hotwords(hotwords, context) == expected


def test_hotword_iterators_are_materialized_for_chunk_reuse():
    hotwords = (word for word in ["MLX", "Apple Silicon"])

    assert FunASRNano._resolve_hotwords(hotwords, None) == ["MLX", "Apple Silicon"]


def test_cli_context_reaches_each_fun_asr_audio_chunk(tmp_path):
    model = FunASRNano.__new__(FunASRNano)
    model.config = SimpleNamespace(
        default_max_tokens=10, frontend_conf=SimpleNamespace(fs=16_000)
    )
    captured_hotwords = []

    def fake_generate_chunk(self, audio, **kwargs):
        captured_hotwords.append(kwargs["hotwords"])
        return "chunk", 1, 1

    model._generate_single_chunk = MethodType(fake_generate_chunk, model)

    result = generate_transcription(
        model=model,
        audio=mx.zeros((32_000,)),
        output_path=str(tmp_path / "transcript"),
        context=" MLX, Apple Silicon ",
        language="en",
        chunk_duration=1.0,
    )

    assert result.text == "chunk chunk"
    assert captured_hotwords == [
        ["MLX, Apple Silicon"],
        ["MLX, Apple Silicon"],
    ]


def test_stream_context_reaches_fun_asr_prompt(monkeypatch):
    model = FunASRNano.__new__(FunASRNano)
    captured_hotwords = []

    def fake_build_inputs(self, audio, **kwargs):
        captured_hotwords.append(kwargs["hotwords"])
        return mx.array([[1]]), mx.zeros((1, 1, 1))

    def fake_generate_step(**kwargs):
        yield 151643, None

    model._build_inputs_embeds = MethodType(fake_build_inputs, model)
    generate_module = importlib.import_module("mlx_lm.generate")
    monkeypatch.setattr(generate_module, "generate_step", fake_generate_step)

    assert list(model.stream_generate(mx.zeros((1,)), context=" MLX ")) == []
    assert captured_hotwords == [["MLX"]]


def test_shared_context_api_rejects_ambiguous_hotwords():
    with pytest.raises(ValueError, match="either hotwords or context"):
        FunASRNano._resolve_hotwords(["MLX"], "Apple Silicon")


def test_converter_transposes_fsmn_and_skips_tied_lm_head():
    torch = pytest.importorskip("torch")
    state = {
        "audio_encoder.encoders0.0.self_attn.fsmn_block.weight": torch.arange(
            6, dtype=torch.float32
        ).reshape(2, 1, 3),
        "llm.lm_head.weight": torch.zeros((4, 4), dtype=torch.float32),
    }

    weights = _convert_weights(state, dtype="float32")

    assert "llm.lm_head.weight" not in weights
    assert weights["audio_encoder.encoders0.0.self_attn.fsmn_block.weight"].shape == (
        2,
        3,
        1,
    )
