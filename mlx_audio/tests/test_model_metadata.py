"""Tests for the machine-readable model metadata endpoint and provider hook.

The endpoint ``GET /v1/models/{model_id}/metadata`` must only serialize what
the model/provider implementation reports; nothing about a model's capabilities
may be hardcoded in the HTTP layer.
"""

import pytest

pytest.importorskip("multipart", reason="python-multipart is required for server tests")

from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from mlx_audio.model_metadata import (
    ModelCapabilities,
    ModelLimits,
    ModelMetadata,
    ModelRuntime,
    metadata_for_model,
    schema_for_callable,
)
from mlx_audio.server import ModelProvider, app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def mock_model_provider():
    with patch(
        "mlx_audio.server.model_provider", new_callable=AsyncMock
    ) as mock_provider:
        yield mock_provider


def _full_metadata(model_id="test-model"):
    return ModelMetadata(
        id=model_id,
        capabilities=ModelCapabilities(
            tools=True,
            streaming=True,
            vision=False,
            audio_input=True,
            audio_output=False,
            structured_output=True,
        ),
        limits=ModelLimits(context_window=32768, max_output_tokens=4096),
        parameters={
            "type": "object",
            "properties": {
                "temperature": {"type": "number"},
                "top_p": {"type": "number"},
            },
        },
        runtime=ModelRuntime(name="mlx-audio", version="0.5.0", protocol="openai"),
    )


def _provider_with(models):
    """A real ModelProvider pre-loaded with the given model objects."""
    provider = ModelProvider()
    provider.models = dict(models)
    return provider


class _BareModel:
    """A loaded model that does not implement ``get_metadata()``."""


# ---------------------------------------------------------------------------
# Endpoint behavior
# ---------------------------------------------------------------------------


def test_model_metadata_valid_response(client, mock_model_provider):
    mock_model_provider.get_metadata = AsyncMock(return_value=_full_metadata())
    response = client.get("/v1/models/test-model/metadata")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    body = response.json()
    assert body["id"] == "test-model"
    assert body["capabilities"] == {
        "tools": True,
        "streaming": True,
        "vision": False,
        "audio_input": True,
        "audio_output": False,
        "structured_output": True,
    }
    assert body["limits"] == {"context_window": 32768, "max_output_tokens": 4096}
    assert body["parameters"] == {
        "type": "object",
        "properties": {
            "temperature": {"type": "number"},
            "top_p": {"type": "number"},
        },
    }
    assert body["runtime"] == {
        "name": "mlx-audio",
        "version": "0.5.0",
        "protocol": "openai",
    }


def test_model_metadata_unknown_model_returns_404(client, mock_model_provider):
    mock_model_provider.get_metadata = AsyncMock(return_value=None)
    response = client.get("/v1/models/not-loaded/metadata")
    assert response.status_code == 404
    assert "not-loaded" in response.json()["detail"]


def test_model_metadata_accepts_hf_repo_ids(client, mock_model_provider):
    """Model ids containing slashes (HuggingFace repo ids) must work, both raw
    and percent-encoded."""
    seen = {}

    async def get_metadata(model_name):
        seen["model"] = model_name
        return ModelMetadata(id=model_name)

    mock_model_provider.get_metadata = AsyncMock(side_effect=get_metadata)
    for path in (
        "/v1/models/mlx-community/whisper-large-v3/metadata",
        "/v1/models/mlx-community%2Fwhisper-large-v3/metadata",
    ):
        response = client.get(path)
        assert response.status_code == 200
        assert response.json()["id"] == "mlx-community/whisper-large-v3"
    assert seen["model"] == "mlx-community/whisper-large-v3"


def test_model_without_metadata_returns_conservative_defaults(client, monkeypatch):
    """A model that does not implement get_metadata() still yields a valid
    response, but capabilities are unknown (null) rather than guessed."""
    monkeypatch.setattr(
        "mlx_audio.server.model_provider", _provider_with({"bare": _BareModel()})
    )
    response = client.get("/v1/models/bare/metadata")
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == "bare"
    assert body["capabilities"] == {
        "tools": None,
        "streaming": None,
        "vision": None,
        "audio_input": None,
        "audio_output": None,
        "structured_output": None,
    }
    assert body["limits"] == {"context_window": None, "max_output_tokens": None}
    assert body["parameters"] is None
    assert body["runtime"]["name"] == "mlx-audio"
    assert body["runtime"]["version"] is not None
    assert body["runtime"]["protocol"] == "openai"


def test_model_metadata_serializes_partial_capabilities_and_limits(
    client, mock_model_provider
):
    """Unreported capabilities serialize as null, never as false."""
    mock_model_provider.get_metadata = AsyncMock(
        return_value=ModelMetadata(
            id="m",
            capabilities=ModelCapabilities(tools=False, streaming=True),
            limits=ModelLimits(context_window=8192),
        )
    )
    body = client.get("/v1/models/m/metadata").json()
    assert body["capabilities"] == {
        "tools": False,
        "streaming": True,
        "vision": None,
        "audio_input": None,
        "audio_output": None,
        "structured_output": None,
    }
    assert body["limits"] == {"context_window": 8192, "max_output_tokens": None}


def test_model_metadata_parameters_json_schema(client, mock_model_provider):
    schema = {
        "type": "object",
        "properties": {
            "temperature": {"type": "number", "minimum": 0.0, "maximum": 2.0},
            "voice": {"type": "string"},
            "stream": {"type": "boolean"},
        },
        "required": ["input"],
    }
    mock_model_provider.get_metadata = AsyncMock(
        return_value=ModelMetadata(id="m", parameters=schema)
    )
    body = client.get("/v1/models/m/metadata").json()
    assert body["parameters"] == schema


def test_model_metadata_multiple_models(client, mock_model_provider):
    async def get_metadata(model_name):
        if model_name == "tts-model":
            return ModelMetadata(
                id=model_name,
                capabilities=ModelCapabilities(audio_output=True, streaming=True),
            )
        if model_name == "stt-model":
            return ModelMetadata(
                id=model_name,
                capabilities=ModelCapabilities(audio_input=True, streaming=False),
            )
        return None

    mock_model_provider.get_metadata = AsyncMock(side_effect=get_metadata)
    tts_body = client.get("/v1/models/tts-model/metadata").json()
    stt_body = client.get("/v1/models/stt-model/metadata").json()
    assert tts_body["capabilities"]["audio_output"] is True
    assert tts_body["capabilities"]["audio_input"] is None
    assert stt_body["capabilities"]["audio_input"] is True
    assert stt_body["capabilities"]["audio_output"] is None


def test_partially_constructed_metadata_serializes_unknowns(
    client, mock_model_provider
):
    """None sections (capabilities/limits/runtime) serialize as unknown rather
    than raising a 500 from the endpoint."""
    mock_model_provider.get_metadata = AsyncMock(
        return_value=ModelMetadata(id="m", capabilities=None, limits=None, runtime=None)
    )
    response = client.get("/v1/models/m/metadata")
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == "m"
    assert body["capabilities"]["tools"] is None
    assert body["limits"]["context_window"] is None
    assert body["runtime"]["name"] == "mlx-audio"


def test_metadata_endpoint_echoes_provider_without_hardcoding(
    client, mock_model_provider
):
    """The endpoint is a pure serializer: it must echo exactly what the
    provider reports, including capabilities the endpoint has no knowledge of."""
    exotic = ModelMetadata(
        id="weird",
        capabilities=ModelCapabilities(vision=True),
        limits=ModelLimits(),
        parameters=None,
        runtime=ModelRuntime(name="custom-runtime", version="9.9", protocol="custom"),
    )
    mock_model_provider.get_metadata = AsyncMock(return_value=exotic)
    body = client.get("/v1/models/weird/metadata").json()
    assert body == exotic.to_dict()
    assert body["capabilities"]["vision"] is True
    assert body["capabilities"]["tools"] is None


def test_new_model_only_needs_metadata_implementation(client, monkeypatch):
    """Adding a model with different capabilities requires only implementing
    ``get_metadata()`` on the model -- the metadata endpoint is untouched and no
    model-specific conditional is added anywhere."""

    class TtsModel:
        def get_metadata(self):
            return ModelMetadata(
                id="internal-name",  # normalized to the requested id by the provider
                capabilities=ModelCapabilities(
                    tools=True, streaming=True, audio_output=True
                ),
                limits=ModelLimits(context_window=32768, max_output_tokens=4096),
                # Property annotations are merged into the parameters schema.
                parameters={
                    "voice": {"type": "string", "enum": ["serena", "vivian"]}
                },
                runtime=ModelRuntime(name="my-tts-runtime", version="1.2.3"),
            )

    class SttModel:
        def get_metadata(self):
            return ModelMetadata(
                id="internal-name",
                capabilities=ModelCapabilities(streaming=False, audio_input=True),
                limits=ModelLimits(context_window=8192),
            )

    monkeypatch.setattr(
        "mlx_audio.server.model_provider",
        _provider_with({"my-tts": TtsModel(), "my-stt": SttModel()}),
    )

    tts_body = client.get("/v1/models/my-tts/metadata").json()
    assert tts_body["id"] == "my-tts"
    assert tts_body["capabilities"]["tools"] is True
    assert tts_body["capabilities"]["audio_output"] is True
    assert tts_body["capabilities"]["audio_input"] is None  # not claimed
    assert tts_body["limits"] == {
        "context_window": 32768,
        "max_output_tokens": 4096,
    }
    assert tts_body["parameters"] == {
        "type": "object",
        "properties": {"voice": {"type": "string", "enum": ["serena", "vivian"]}},
    }
    assert tts_body["runtime"] == {
        "name": "my-tts-runtime",
        "version": "1.2.3",
        "protocol": "openai",  # serving runtime's protocol fills the gap
    }

    stt_body = client.get("/v1/models/my-stt/metadata").json()
    assert stt_body["id"] == "my-stt"
    assert stt_body["capabilities"]["audio_input"] is True
    assert stt_body["capabilities"]["audio_output"] is None
    assert stt_body["capabilities"]["streaming"] is False


# ---------------------------------------------------------------------------
# metadata_for_model dispatch
# ---------------------------------------------------------------------------


def test_metadata_for_model_uses_duck_typed_hook():
    class Model:
        def get_metadata(self):
            return ModelMetadata(
                id="internal-name",
                capabilities=ModelCapabilities(streaming=True),
            )

    metadata = metadata_for_model(Model(), model_id="requested-name")
    assert isinstance(metadata, ModelMetadata)
    assert metadata.id == "requested-name"
    assert metadata.capabilities.streaming is True


def test_metadata_for_model_falls_back_when_hook_missing_or_invalid():
    class NoHook:
        pass

    class NonCallableHook:
        get_metadata = "not a callable"

    class InvalidHook:
        def get_metadata(self):
            return {"tools": True}  # not a ModelMetadata

    for model in (NoHook(), NonCallableHook(), InvalidHook()):
        metadata = metadata_for_model(model, model_id="m")
        assert isinstance(metadata, ModelMetadata)
        assert metadata.id == "m"
        assert metadata.capabilities.tools is None
        assert metadata.limits.context_window is None
        assert metadata.runtime.name == "mlx-audio"


def test_metadata_for_model_id_is_normalized_to_requested_id():
    class Model:
        def get_metadata(self):
            return ModelMetadata(
                id="internal-name",
                capabilities=ModelCapabilities(audio_input=True),
            )

    metadata = metadata_for_model(Model(), model_id="hf-repo/model-name")
    assert metadata.id == "hf-repo/model-name"


def test_metadata_for_model_merges_serving_runtime():
    """Model-provided metadata with an empty runtime gets the serving runtime's
    version/protocol; an explicitly overridden runtime is preserved."""

    class DefaultRuntimeModel:
        def get_metadata(self):
            return ModelMetadata(id="m", capabilities=ModelCapabilities(streaming=True))

    merged = metadata_for_model(DefaultRuntimeModel(), model_id="m")
    assert merged.runtime.name == "mlx-audio"
    assert merged.runtime.version is not None
    assert merged.runtime.protocol == "openai"

    class CustomRuntimeModel:
        def get_metadata(self):
            return ModelMetadata(
                id="m",
                runtime=ModelRuntime(name="my-runtime", version="9.9"),
            )

    custom = metadata_for_model(CustomRuntimeModel(), model_id="m")
    assert custom.runtime.name == "my-runtime"
    assert custom.runtime.version == "9.9"
    assert custom.runtime.protocol == "openai"  # still filled from serving runtime


def _model_with_module(module: str, namespace=None):
    """A minimal loaded model whose concrete class lives under ``module``."""
    cls = type("Model", (), namespace or {})
    cls.__module__ = module
    return cls()


def test_metadata_for_model_infers_audio_shape_from_kind():
    """Audio I/O is inferred from the model's kind (its module path) -- no
    per-model declaration or hardcoded table in the HTTP layer."""
    tts = metadata_for_model(
        _model_with_module("mlx_audio.tts.models.kokoro.kokoro"), model_id="kokoro"
    )
    assert tts.capabilities.audio_output is True
    assert tts.capabilities.audio_input is None  # TTS kind: may or may not clone
    assert tts.capabilities.tools is None  # never guessed from a kind

    stt = metadata_for_model(
        _model_with_module("mlx_audio.stt.models.whisper.whisper"), model_id="whisper"
    )
    assert stt.capabilities.audio_input is True
    assert stt.capabilities.audio_output is False

    sts = metadata_for_model(
        _model_with_module("mlx_audio.sts.models.moshi.moshi"), model_id="moshi"
    )
    assert sts.capabilities.audio_input is True
    assert sts.capabilities.audio_output is True

    lid = metadata_for_model(
        _model_with_module("mlx_audio.lid.models.ecapa_tdnn.ecapa_tdnn"),
        model_id="lid",
    )
    assert lid.capabilities.audio_input is True
    assert lid.capabilities.audio_output is False


def test_metadata_for_model_derives_streaming_and_ref_audio_from_signature():
    """Streaming / reference-audio support come from the model's own generate()
    signature; absent parameters stay unknown."""

    def streaming_with_ref(text, *, stream=False, ref_audio=None):
        ...

    def plain(text):
        ...

    streamed = metadata_for_model(
        _model_with_module("mlx_audio.tts.models.fake.fake", {"generate": streaming_with_ref}),
        model_id="m",
    )
    assert streamed.capabilities.streaming is True
    assert streamed.capabilities.audio_input is True  # ref_audio param

    plain_model = metadata_for_model(
        _model_with_module("mlx_audio.tts.models.fake.fake", {"generate": plain}),
        model_id="m",
    )
    assert plain_model.capabilities.streaming is None
    assert plain_model.capabilities.audio_input is None


def test_metadata_for_model_declared_fields_override_inferred():
    """Non-None declared fields win; everything else falls back to inference."""

    class Model:
        def get_metadata(self):
            return ModelMetadata(
                capabilities=ModelCapabilities(audio_input=False, tools=False),
                limits=ModelLimits(context_window=32768),
            )

    Model.__module__ = "mlx_audio.tts.models.fake.fake"

    meta = metadata_for_model(Model(), model_id="m")
    assert meta.capabilities.audio_output is True  # inferred from TTS kind
    assert meta.capabilities.audio_input is False  # declared override
    assert meta.capabilities.tools is False
    assert meta.limits.context_window == 32768
    assert meta.id == "m"


def test_metadata_for_model_limits_only_hook_keeps_inferred_capabilities():
    """A model that only declares limits still inherits its kind's audio shape."""

    class Model:
        def get_metadata(self):
            return ModelMetadata(limits=ModelLimits(context_window=4096))

    Model.__module__ = "mlx_audio.sts.models.moshi.moshi"

    meta = metadata_for_model(Model(), model_id="moshi")
    assert meta.capabilities.audio_input is True
    assert meta.capabilities.audio_output is True
    assert meta.limits.context_window == 4096


def test_schema_for_callable_introspects_types_defaults_and_required():
    """Type hints become JSON Schema types, defaults are serialized, and
    parameters without defaults become ``required``."""

    def generate(text: str, speed: float = 1.0, max_tokens: int = 4096, verbose: bool = False):
        ...

    schema = schema_for_callable(generate)
    assert schema["type"] == "object"
    assert schema["required"] == ["text"]
    assert schema["properties"]["text"] == {"type": "string"}
    assert schema["properties"]["speed"] == {"type": "number", "default": 1.0}
    assert schema["properties"]["max_tokens"] == {"type": "integer", "default": 4096}
    assert schema["properties"]["verbose"] == {"type": "boolean", "default": False}


def test_schema_for_callable_handles_optional_union_and_literal():
    from typing import Literal, Optional, Union

    def generate(
        text: str,
        voice: Optional[str] = None,
        task: Literal["transcribe", "translate"] = "transcribe",
        audio: Union[str, list[int]] = None,
    ):
        ...

    schema = schema_for_callable(generate)
    props = schema["properties"]
    assert props["voice"] == {"type": "string", "default": None}
    assert props["task"] == {"enum": ["transcribe", "translate"], "default": "transcribe"}
    assert props["audio"] == {"type": ["string", "array"], "default": None}
    assert schema["required"] == ["text"]


def test_metadata_for_model_attaches_introspected_parameters():
    """A model's entry-point signature yields a parameters schema automatically,
    with no get_metadata() hook needed."""

    class Model:
        def generate(self, text: str, speed: float = 1.0):
            ...

    Model.__module__ = "mlx_audio.tts.models.fake.fake"

    meta = metadata_for_model(Model(), model_id="fake")
    assert meta.parameters == {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "speed": {"type": "number", "default": 1.0},
        },
        "required": ["text"],
    }



def test_qwen3_tts_hook_declares_variant_dependent_audio_input():
    """The qwen3_tts hook reports audio_input only for the base variant;
    CustomVoice/VoiceDesign ignore ref_audio. Skipped where the model module
    cannot be imported (no real mlx available)."""
    try:
        from mlx_audio.tts.models.qwen3_tts.config import (
            ModelConfig,
            Qwen3TTSTalkerConfig,
        )
        from mlx_audio.tts.models.qwen3_tts.qwen3_tts import Model
    except (ImportError, AttributeError) as exc:
        pytest.skip(f"qwen3_tts model module unavailable: {exc}")

    model = Model.__new__(Model)  # bypass heavy __init__; only config is needed
    model.config = ModelConfig(
        tts_model_type="custom_voice",
        talker_config=Qwen3TTSTalkerConfig(
            spk_id={"serena": [0], "vivian": [1], "ryan": [2]},
            codec_language_id={"English": [0], "Chinese": [1], "English-dialect": [2]},
        ),
    )
    metadata = model.get_metadata()
    assert isinstance(metadata, ModelMetadata)
    assert metadata.capabilities.audio_output is True
    assert metadata.capabilities.streaming is True
    assert metadata.capabilities.tools is False
    assert metadata.capabilities.audio_input is False  # custom_voice: no ref audio
    assert metadata.parameters == {
        "voice": {"enum": ["serena", "vivian", "ryan"]},
        "lang_code": {"enum": ["auto", "English", "Chinese"]},  # dialect excluded
    }

    model.config = ModelConfig(tts_model_type="voice_design")
    assert model.get_metadata().capabilities.audio_input is False

    model.config = ModelConfig(tts_model_type="base")
    assert model.get_metadata().capabilities.audio_input is True

def test_sts_entry_point_discovered_without_generate():
    """STS families (e.g. LFM2Audio) expose generate_interleaved instead of
    generate; parameters must still be introspected."""

    class Model:
        def generate_interleaved(self, max_new_tokens: int = 512, temperature: float = 1.0):
            ...

    Model.__module__ = "mlx_audio.sts.models.lfm_audio.model"

    meta = metadata_for_model(Model(), model_id="lfm")
    assert meta.capabilities.audio_input is True
    assert meta.capabilities.audio_output is True
    assert meta.parameters == {
        "type": "object",
        "properties": {
            "max_new_tokens": {"type": "integer", "default": 512},
            "temperature": {"type": "number", "default": 1.0},
        },
    }

def test_declared_parameters_merge_as_property_annotations():
    """A hook's parameters dict is treated as property annotations and merged
    into the introspected schema: declared fragments win per-property, while
    introspected type/default survive and untouched properties stay as-is."""

    class Model:
        def generate(self, text: str, voice: str = None, lang_code: str = "auto"):
            ...

        def get_metadata(self):
            return ModelMetadata(
                parameters={
                    "voice": {"enum": ["serena", "vivian"]},
                    "lang_code": {"enum": ["auto", "English"]},
                }
            )

    Model.__module__ = "mlx_audio.tts.models.fake.fake"

    meta = metadata_for_model(Model(), model_id="m")
    props = meta.parameters["properties"]
    assert props["voice"] == {
        "type": "string",
        "default": None,
        "enum": ["serena", "vivian"],
    }
    assert props["lang_code"] == {
        "type": "string",
        "default": "auto",
        "enum": ["auto", "English"],
    }
    assert props["text"] == {"type": "string"}  # untouched
    assert meta.parameters["required"] == ["text"]


def test_declared_parameters_annotations_without_introspected_schema():
    """When there is no introspected schema, declared property annotations are
    wrapped into a minimal object schema."""

    class Model:
        def get_metadata(self):
            return ModelMetadata(parameters={"task": {"enum": ["a", "b"]}})

    Model.__module__ = "mlx_audio.tts.models.fake.fake"

    meta = metadata_for_model(Model(), model_id="m")
    assert meta.parameters == {
        "type": "object",
        "properties": {"task": {"enum": ["a", "b"]}},
    }
