from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, cast

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.tts.models.base import BaseModelArgs, GenerationResult
from mlx_audio.utils import get_model_path

DOTS_MODEL_TYPES = {"dots", "dots_tts", "dots_tts_mlx"}
DOTS_DEFAULT_SUBDIR = "int4"


def _load_config_if_exists(model_path: Path) -> dict:
    config_path = model_path / "config.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def _looks_like_dots_checkpoint(
    model_path: Path, config: Optional[dict] = None
) -> bool:
    config = config or {}
    model_type = config.get("model_type") or config.get("architecture")
    if isinstance(model_type, str) and model_type.lower() in DOTS_MODEL_TYPES:
        return True
    required = {
        "config.json",
        "llm_config.json",
        "core.safetensors",
        "vocoder.safetensors",
        "speaker.safetensors",
    }
    if not model_path.exists() or not model_path.is_dir():
        return False
    names = {item.name for item in model_path.iterdir()}
    return required.issubset(names)


def _available_variant_dirs(model_root: Path) -> list[str]:
    if not model_root.exists() or not model_root.is_dir():
        return []
    variants = []
    for item in sorted(model_root.iterdir()):
        if item.is_dir() and _looks_like_dots_checkpoint(
            item, _load_config_if_exists(item)
        ):
            variants.append(item.name)
    return variants


def _resolve_checkpoint_path(
    model_path: str | Path, subdir: Optional[str] = None, **kwargs
) -> Path:
    resolved_subdir = subdir or DOTS_DEFAULT_SUBDIR
    if isinstance(model_path, str):
        allow_patterns = kwargs.get("allow_patterns")
        if allow_patterns is None:
            allow_patterns = [f"{resolved_subdir}/*", f"{resolved_subdir}/**"]
        resolved_path = get_model_path(
            model_path,
            revision=kwargs.get("revision", None),
            force_download=kwargs.get("force_download", False),
            allow_patterns=allow_patterns,
        )
    else:
        resolved_path = Path(model_path).expanduser()

    if _looks_like_dots_checkpoint(
        resolved_path, _load_config_if_exists(resolved_path)
    ):
        return resolved_path

    candidate = resolved_path / resolved_subdir
    if _looks_like_dots_checkpoint(candidate, _load_config_if_exists(candidate)):
        return candidate

    available_variants = _available_variant_dirs(resolved_path)
    if available_variants:
        available = ", ".join(available_variants)
        raise FileNotFoundError(
            f"Dots checkpoint variant '{resolved_subdir}' not found under {resolved_path}. "
            f"Available variants: {available}"
        )

    raise FileNotFoundError(
        f"Could not resolve a Dots checkpoint from {resolved_path}. Expected either a "
        "checkpoint directory or a multi-variant repo root containing a "
        f"'{resolved_subdir}' subdirectory."
    )


@dataclass
class ModelConfig(BaseModelArgs):
    model_type: str = "dots"
    model_path: str = ""
    sample_rate: int = 48000
    dtype: str = "bfloat16"
    use_long_text: bool = False


def _resolve_dtype(dtype_name: str) -> mx.Dtype:
    mapping = {
        "float16": mx.float16,
        "bfloat16": mx.bfloat16,
        "float32": mx.float32,
    }
    return mapping.get(dtype_name, mx.bfloat16)


def _duration_string(samples: int, sample_rate: int) -> str:
    if sample_rate <= 0:
        return "00:00:00.000"
    total_seconds = samples / sample_rate
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    millis = int((total_seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def _coerce_runtime_results(result: Any) -> list[Any]:
    if isinstance(result, GenerationResult):
        return [result]
    if isinstance(result, dict):
        return [result]
    if isinstance(result, Iterable) and not isinstance(result, (str, bytes)):
        return list(result)
    raise TypeError(f"Unsupported dots runtime result type: {type(result)}")


class Model(nn.Module):
    preserve_ref_audio_path = True
    supports_multiple_references = False

    def __init__(self, config: ModelConfig | dict):
        super().__init__()
        self.config = (
            config if isinstance(config, ModelConfig) else ModelConfig.from_dict(config)
        )
        self.sample_rate = int(self.config.sample_rate)
        self._runtime = None

    def load_runtime(self) -> "Model":
        runtime = cast(
            Any, getattr(self, "_runtime", None) or getattr(self, "runtime", None)
        )
        if runtime is None:
            from .loader import from_pretrained

            loaded = from_pretrained(
                self.config.model_path,
                dtype=_resolve_dtype(self.config.dtype),
            )
            runtime = loaded.model
        self._runtime = runtime
        sample_rate = getattr(runtime, "sample_rate", None)
        if sample_rate is not None:
            self.sample_rate = int(sample_rate)
        return self

    def eval(self):
        return self

    def generate(
        self,
        text: str,
        ref_audio: str | Path | None = None,
        ref_text: str | None = None,
        lang_code: str | None = None,
        language: str | None = None,
        long: bool | None = None,
        **kwargs,
    ):
        self.load_runtime()
        if language is None and lang_code:
            language = str(lang_code).upper()

        use_long_text = bool(
            long if long is not None else getattr(self.config, "use_long_text", False)
        )
        runtime_generate = getattr(
            self._runtime,
            "generate_long" if use_long_text else "generate",
        )

        runtime_kwargs = dict(kwargs)
        runtime_kwargs.pop("voice", None)
        runtime_kwargs.pop("speed", None)
        runtime_kwargs.pop("temperature", None)
        runtime_kwargs.pop("verbose", None)
        runtime_kwargs.pop("stream", None)
        runtime_kwargs.pop("streaming_interval", None)
        runtime_kwargs.pop("instruct", None)
        runtime_kwargs.pop("use_zero_spk_emb", None)
        runtime_kwargs.pop("prompt", None)
        runtime_kwargs.pop("sigma", None)
        runtime_kwargs.pop("subdir", None)

        max_tokens = runtime_kwargs.pop("max_tokens", None)
        if max_tokens is not None and "max_generate_length" not in runtime_kwargs:
            runtime_kwargs["max_generate_length"] = max_tokens

        cfg_scale = runtime_kwargs.pop("cfg_scale", None)
        if cfg_scale is not None and "guidance_scale" not in runtime_kwargs:
            runtime_kwargs["guidance_scale"] = cfg_scale

        ddpm_steps = runtime_kwargs.pop("ddpm_steps", None)
        if ddpm_steps is not None and "num_steps" not in runtime_kwargs:
            runtime_kwargs["num_steps"] = ddpm_steps

        start_time = time.perf_counter()
        result = runtime_generate(
            text=text,
            prompt_audio=ref_audio,
            prompt_text=ref_text,
            language=language,
            **runtime_kwargs,
        )
        elapsed_time = time.perf_counter() - start_time

        for index, item in enumerate(_coerce_runtime_results(result)):
            if isinstance(item, GenerationResult):
                yield item
                continue

            audio = item["audio"]
            shape = getattr(audio, "shape", ())
            if len(shape) > 1 and shape[0] == 1:
                audio = audio[0]

            sample_rate = int(item.get("sample_rate", self.sample_rate))
            samples = int(item.get("samples", getattr(audio, "shape", [0])[-1]))
            processing_time = float(item.get("processing_time_seconds", elapsed_time))
            prompt = dict(item.get("prompt", {}))
            if "tokens-per-sec" not in prompt:
                prompt["tokens-per-sec"] = (
                    int(item.get("token_count", item.get("num_patches", 0)))
                    / processing_time
                    if processing_time > 0
                    else 0.0
                )

            audio_samples = dict(item.get("audio_samples", {}))
            audio_samples.setdefault("samples", samples)
            if "samples-per-sec" not in audio_samples:
                audio_samples["samples-per-sec"] = (
                    samples / processing_time if processing_time > 0 else 0.0
                )

            yield GenerationResult(
                audio=audio,
                samples=samples,
                sample_rate=sample_rate,
                segment_idx=int(item.get("segment_idx", index)),
                token_count=int(item.get("token_count", item.get("num_patches", 0))),
                audio_duration=item.get(
                    "audio_duration",
                    _duration_string(samples, sample_rate),
                ),
                real_time_factor=(
                    (samples / sample_rate / processing_time)
                    if processing_time > 0 and sample_rate > 0
                    else 0.0
                ),
                prompt=prompt,
                audio_samples=audio_samples,
                processing_time_seconds=processing_time,
                peak_memory_usage=float(
                    item.get("peak_memory_usage", mx.get_peak_memory() / 1e9)
                ),
            )


def load_model(
    model_path: str | Path,
    lazy: bool = False,
    strict: bool = True,
    subdir: Optional[str] = None,
    **kwargs,
) -> Model:
    del strict
    resolved_path = _resolve_checkpoint_path(model_path, subdir=subdir, **kwargs)
    config_path = resolved_path / "config.json"
    config_data = {}
    if config_path.exists():
        config_data = json.loads(config_path.read_text(encoding="utf-8"))
    config_data.setdefault("model_type", "dots")
    config_data.setdefault("sample_rate", 48000)
    config_data["model_path"] = str(resolved_path)
    model = Model(ModelConfig.from_dict(config_data))
    if not lazy:
        model.load_runtime()
    return model


__all__ = ["DOTS_MODEL_TYPES", "Model", "ModelConfig", "load_model"]
