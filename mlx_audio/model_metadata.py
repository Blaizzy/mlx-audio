"""Standardized, machine-readable model metadata for OpenAI-compatible clients.

Model metadata is defined by the model/provider implementation, never by the
HTTP API layer. The serving runtime builds a grounded *inferred* baseline and
then lets each model correct or extend it.

The baseline is never guessed: it comes from structural facts about the loaded
model itself --

* the model's *kind* (``tts`` / ``stt`` / ``sts`` / ``lid`` / ``vad``), read
  from the concrete class's module path (``mlx_audio.<kind>.models...``),
  determines the audio I/O shape (a TTS model emits audio, an STT model
  consumes audio, …);
* the model's own inference entry point (``generate()`` / ``predict()`` /
  ``detect()`` / …) determines streaming support (a ``stream`` parameter) and
  reference-audio input (a ``ref_audio`` parameter); and
* that same entry point's signature is introspected into the ``parameters``
  JSON Schema (type hints drive ``type``, defaults are serialized, parameters
  without defaults become ``required``) -- no hand-written JSON Schema.

Capabilities that are not entailed by those facts (tool calling, vision,
structured output) are reported as ``None`` = unknown.

A model may correct or extend this baseline by exposing a duck-typed
``get_metadata()`` method returning a :class:`ModelMetadata`; any non-``None``
field it declares overrides the inferred value, and the ``id`` / runtime
section are normalized to the serving runtime. This matches the codebase's
existing capability hooks such as ``supports_tts_batch`` / ``batch_generate`` /
``create_streaming_session``.

    class Model(nn.Module):
        def get_metadata(self) -> ModelMetadata:
            return ModelMetadata(
                limits=ModelLimits(context_window=32768),
                # Annotate valid values directly on the parameters schema:
                # each entry is merged into the matching introspected property
                # (type/default stay introspected, the annotation adds enum).
                parameters={"voice": {"enum": [...]}, "lang_code": {"enum": [...]}},
            )

The inferred audio capabilities and introspected parameters are still applied;
only ``context_window`` and the parameter annotations are declared here.
Models that have no extra facts to declare need no hook at all.
"""

from __future__ import annotations

import inspect
import math
import types
from dataclasses import dataclass, field, replace
from typing import (
    Literal,
    Optional,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)


@dataclass
class ModelCapabilities:
    """Reported model capabilities.

    Each field is tri-state:

    * ``True`` -- the model/runtime supports the capability;
    * ``False`` -- the model/runtime does not support it;
    * ``None`` -- unknown; the implementation did not report it.
    """

    tools: Optional[bool] = None
    streaming: Optional[bool] = None
    vision: Optional[bool] = None
    audio_input: Optional[bool] = None
    audio_output: Optional[bool] = None
    structured_output: Optional[bool] = None

    def to_dict(self) -> dict:
        return {
            "tools": self.tools,
            "streaming": self.streaming,
            "vision": self.vision,
            "audio_input": self.audio_input,
            "audio_output": self.audio_output,
            "structured_output": self.structured_output,
        }


@dataclass
class ModelLimits:
    context_window: Optional[int] = None
    max_output_tokens: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "context_window": self.context_window,
            "max_output_tokens": self.max_output_tokens,
        }


@dataclass
class ModelRuntime:
    name: str = "mlx-audio"
    version: Optional[str] = None
    protocol: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "protocol": self.protocol,
        }


@dataclass
class ModelMetadata:
    #: Normalized to the requested model id by ``metadata_for_model``; models
    #: that only declare limits/parameters may leave it unset.
    id: Optional[str] = None
    capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)
    limits: ModelLimits = field(default_factory=ModelLimits)
    #: JSON Schema for the model's request parameters. When a ``get_metadata``
    #: hook declares this, it is treated as *property annotations* (a dict of
    #: parameter name -> schema fragment, e.g. ``{"voice": {"enum": [...]}}``)
    #: and deep-merged into the introspected signature schema, so ``enum``
    #: constraints etc. live here -- the single source of truth for requests.
    parameters: Optional[dict] = None
    runtime: ModelRuntime = field(default_factory=ModelRuntime)

    def to_dict(self) -> dict:
        # Guard against partially constructed metadata: a ``None`` section is
        # serialized as unknown rather than raising.
        capabilities = self.capabilities or ModelCapabilities()
        limits = self.limits or ModelLimits()
        runtime = self.runtime or default_runtime()
        return {
            "id": self.id,
            "capabilities": capabilities.to_dict(),
            "limits": limits.to_dict(),
            "parameters": self.parameters,
            "runtime": runtime.to_dict(),
        }


#: Audio I/O shape entailed by each model kind. A TTS model always emits audio
#: but only *consumes* it when it clones voices (left unknown here, derived from
#: the entry-point signature); STT / LID / VAD consume audio and never emit
#: it; STS does both. Tool calling / vision / structured output are never
#: entailed by a kind and stay unknown unless a model declares them.
_KIND_CAPABILITIES = {
    "tts": {"audio_output": True},
    "stt": {"audio_input": True, "audio_output": False},
    "sts": {"audio_input": True, "audio_output": True},
    "lid": {"audio_input": True, "audio_output": False},
    "vad": {"audio_input": True, "audio_output": False},
}

_CAPABILITY_FIELDS = (
    "tools",
    "streaming",
    "vision",
    "audio_input",
    "audio_output",
    "structured_output",
)

#: Candidate entry points, in priority order, used to introspect a model's
#: parameters and to derive streaming / reference-audio capabilities. The STS
#: ``generate_*`` methods cover STS families (e.g. LFM2Audio) that have no
#: single ``generate`` method.
_PRIMARY_METHODS = (
    "generate",
    "generate_interleaved",
    "generate_sequential",
    "generate_from_chat_state",
    "predict",
    "detect",
    "detect_language",
    "transcribe",
    "classify",
    "process",
)

_JSON_PRIMITIVES = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    tuple: "array",
    set: "array",
    frozenset: "array",
    dict: "object",
}

_UNSET = object()


def default_runtime() -> ModelRuntime:
    """Runtime info describing the mlx-audio serving runtime."""
    try:
        from mlx_audio.version import __version__
    except ImportError:  # pragma: no cover - the version module always ships
        __version__ = None
    return ModelRuntime(name="mlx-audio", version=__version__, protocol="openai")


def metadata_for_model(model, model_id: str) -> ModelMetadata:
    """Return metadata for ``model`` as identified by ``model_id``.

    Capabilities are inferred from the model's kind (its module path) and its
    entry-point signature, and parameters are introspected from that signature.
    A duck-typed ``get_metadata()`` hook -- when present and returning a
    :class:`ModelMetadata` -- overlays any non-``None`` field it declares
    (limits, parameters, capability corrections). The ``id`` is always
    normalized to ``model_id`` and the runtime section filled with the serving
    runtime's identity.
    """
    inferred = _inferred_metadata(model, model_id)
    get_metadata = getattr(model, "get_metadata", None)
    if callable(get_metadata):
        declared = get_metadata()
        if isinstance(declared, ModelMetadata):
            inferred = _finalize(_overlay(inferred, declared), model_id)
    return inferred


def schema_for_callable(func) -> Optional[dict]:
    """Build a JSON Schema for a callable's parameters from its signature.

    Type hints drive each property's ``type`` (``str`` → ``"string"``, ``int``
    → ``"integer"``, ``Optional[X]`` → X, unions become a type list, ``Literal``
    becomes ``enum``); serializable defaults are emitted as ``default`` and
    parameters without defaults are listed under ``required``. ``*args`` /
    ``**kwargs`` / ``self`` are skipped. Returns ``None`` when nothing can be
    introspected.
    """
    if not callable(func):
        return None
    func = getattr(func, "__func__", func)  # unwrap bound methods for hints
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return None
    try:
        hints = get_type_hints(func)
    except Exception:  # pragma: no cover - unresolvable forward refs degrade
        hints = {}

    properties = {}
    required = []
    for name, param in signature.parameters.items():
        if name in ("self", "cls"):
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        prop = _property_schema(hints.get(name, param.annotation))
        if param.default is inspect.Parameter.empty:
            required.append(name)
        else:
            default = _json_default(param.default)
            if default is not _UNSET:
                prop["default"] = default
        properties[name] = prop

    if not properties:
        return None
    schema = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def _property_schema(annotation) -> dict:
    """A JSON Schema fragment for a single parameter's type annotation."""
    if annotation is not None and annotation is not inspect.Parameter.empty:
        origin = get_origin(annotation)
        if origin is Literal:
            return {"enum": list(get_args(annotation))}
    type_name = _type_name(annotation)
    if isinstance(type_name, list):
        return {"type": type_name}
    if type_name:
        return {"type": type_name}
    return {}


def _type_name(annotation):
    """Best-effort JSON Schema type name(s) for a resolved annotation.

    Returns a single type name (``"string"``), a list of names for unions, or
    ``None`` when the type is unknown.
    """
    if annotation is None or annotation is inspect.Parameter.empty:
        return None
    if annotation in _JSON_PRIMITIVES:
        return _JSON_PRIMITIVES[annotation]
    if isinstance(annotation, str):
        return _type_name_for_string(annotation)

    origin = get_origin(annotation)
    args = get_args(annotation)

    # Optional[X] / Union[A, B, None] / ``X | Y``
    if origin in (Union, types.UnionType):
        names = []
        for arg in args:
            if arg is type(None):
                continue
            resolved = _type_name(arg)
            if isinstance(resolved, list):
                names.extend(resolved)
            elif resolved:
                names.append(resolved)
        unique = []
        for name in names:
            if name not in unique:
                unique.append(name)
        if len(unique) == 1:
            return unique[0]
        return unique or None

    if origin in (list, tuple, set, frozenset):
        return "array"
    if origin is dict:
        return "object"

    name = getattr(annotation, "__name__", "") or ""
    lowered = name.lower()
    if lowered in ("path", "posixpath", "windowspath", "purepath"):
        return "string"
    if lowered in ("ndarray", "array"):
        return "array"
    return None


def _type_name_for_string(annotation: str):
    """Resolve a raw (string) forward-ref annotation without evaluating it."""
    annotation = annotation.strip()
    if annotation.startswith("Optional[") and annotation.endswith("]"):
        return _type_name_for_string(annotation[len("Optional[") : -1])
    if annotation.startswith("Union[") and annotation.endswith("]"):
        return _union_names(annotation[len("Union[") : -1])
    if "|" in annotation:
        return _union_names(annotation)
    return {
        "str": "string",
        "int": "integer",
        "float": "number",
        "bool": "boolean",
        "list": "array",
        "tuple": "array",
        "set": "array",
        "dict": "object",
        "Path": "string",
        "None": None,
        "NoneType": None,
    }.get(annotation)


def _union_names(inner: str):
    names = []
    for part in inner.split(",") if "," in inner else inner.split("|"):
        resolved = _type_name_for_string(part.strip())
        if isinstance(resolved, list):
            names.extend(resolved)
        elif resolved:
            names.append(resolved)
    unique = []
    for name in names:
        if name not in unique:
            unique.append(name)
    if len(unique) == 1:
        return unique[0]
    return unique or None


def _json_default(value):
    """Convert a default value to a JSON-serializable form, or ``_UNSET``."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return _UNSET
        return value
    if isinstance(value, (list, tuple)):
        items = []
        for item in value:
            converted = _json_default(item)
            if converted is _UNSET:
                return _UNSET
            items.append(converted)
        return items
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                return _UNSET
            converted = _json_default(item)
            if converted is _UNSET:
                return _UNSET
            result[key] = converted
        return result
    return _UNSET


def _inferred_metadata(model, model_id: str) -> ModelMetadata:
    entry = _primary_callable(model)
    return ModelMetadata(
        id=model_id,
        capabilities=_inferred_capabilities(model, model_id, entry),
        parameters=schema_for_callable(entry),
        runtime=default_runtime(),
    )


def _inferred_capabilities(model, model_id: str, entry=None) -> ModelCapabilities:
    """Grounded baseline capabilities: kind shape + entry-point signature."""
    capabilities = ModelCapabilities()
    kind = _model_kind(model, model_id)
    for name, value in _KIND_CAPABILITIES.get(kind, {}).items():
        setattr(capabilities, name, value)

    if entry is None:
        entry = _primary_callable(model)
    if callable(entry):
        try:
            params = inspect.signature(entry).parameters
        except (TypeError, ValueError):
            params = {}
        if capabilities.streaming is None and "stream" in params:
            capabilities.streaming = True
        if capabilities.audio_input is None and "ref_audio" in params:
            capabilities.audio_input = True
    return capabilities


def _primary_callable(model):
    """The model's user-facing inference method, or ``None``."""
    for name in _PRIMARY_METHODS:
        method = getattr(model, name, None)
        if callable(method):
            return method
    return None


def _model_kind(model, model_id: str) -> Optional[str]:
    """The audio kind of a loaded model, from its concrete class module path.

    Every model architecture lives under ``mlx_audio/<kind>/models/<family>``,
    so the second module segment of the concrete class is the kind. Falls back
    to the import-free registry classification when the module path does not
    follow that layout (e.g. test doubles or dynamically built models).
    """
    module = type(model).__module__.split(".")
    if len(module) >= 2 and module[0] == "mlx_audio" and module[1] in _KIND_CAPABILITIES:
        return module[1]
    from mlx_audio.registry import classify_model

    try:
        model_type = getattr(model, "model_type", "") or ""
    except Exception:  # pragma: no cover - defensive; model_type is trivial
        model_type = ""
    if not isinstance(model_type, str):
        model_type = ""
    return classify_model(model_type, model_id)


def _overlay(inferred: ModelMetadata, declared: ModelMetadata) -> ModelMetadata:
    """Overlay declared metadata onto the inferred baseline.

    A declared ``None`` never overrides an inferred value (a model that only
    declares limits still inherits its kind's audio shape and introspected
    parameters); only explicit ``True``/``False``/values win.
    """
    capabilities = ModelCapabilities()
    for name in _CAPABILITY_FIELDS:
        declared_value = (
            getattr(declared.capabilities, name) if declared.capabilities else None
        )
        inferred_value = getattr(inferred.capabilities, name)
        setattr(
            capabilities,
            name,
            declared_value if declared_value is not None else inferred_value,
        )
    return ModelMetadata(
        id=declared.id,
        capabilities=capabilities,
        limits=declared.limits if declared.limits is not None else inferred.limits,
        parameters=_merge_parameters(inferred.parameters, declared.parameters),
        runtime=declared.runtime,
    )


def _finalize(metadata: ModelMetadata, model_id: str) -> ModelMetadata:
    """Normalize ``id`` and fill the serving runtime identity into metadata."""
    metadata = _with_serving_runtime(metadata)
    if metadata.id != model_id:
        return replace(metadata, id=model_id)
    return metadata


def _merge_parameters(base: Optional[dict], annotations: Optional[dict]) -> Optional[dict]:
    """Merge declared property annotations into the introspected parameters
    schema.

    ``annotations`` is a dict of parameter name -> JSON Schema fragment (e.g.
    ``{"voice": {"enum": [...]}}``). Each fragment is merged into the matching
    introspected property (declared keys win, introspected ``type``/``default``
    survive), so the base schema stays the single source of truth and models
    only annotate valid values. Unknown properties are added as-is; nothing is
    mutated.
    """
    if not annotations:
        return base
    if base is None or not isinstance(base.get("properties"), dict):
        return {"type": "object", "properties": dict(annotations)}
    properties = dict(base["properties"])
    for name, fragment in annotations.items():
        if not isinstance(fragment, dict):
            continue
        existing = properties.get(name)
        if isinstance(existing, dict):
            properties[name] = {**existing, **fragment}
        else:
            properties[name] = fragment
    return {**base, "properties": properties}


def _with_serving_runtime(metadata: ModelMetadata) -> ModelMetadata:
    """Fill the serving runtime's identity (version / protocol) into metadata.

    The runtime section describes who is serving the model; the model only
    overrides fields it actually knows about (e.g. a custom runtime name).
    """
    defaults = default_runtime()
    runtime = metadata.runtime
    if runtime is None:
        return replace(metadata, runtime=defaults)
    return replace(
        metadata,
        runtime=ModelRuntime(
            name=runtime.name or defaults.name,
            version=runtime.version or defaults.version,
            protocol=runtime.protocol or defaults.protocol,
        ),
    )
