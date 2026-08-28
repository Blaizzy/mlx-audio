"""MLX runtime for the Fermion Research Phonon-1 STT family.

The model architecture is Qwen3-ASR. Phonon's learned decoder uses a custom
five-value representation stored as exact base-5 codes and evaluated as two
native MLX affine quantized matrix multiplications.
"""

# Portions copyright 2026 Fermion Research, licensed under Apache-2.0.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.stt.models.qwen3_asr.qwen3_asr import (
    Qwen3ASRModel,
    Model as Qwen3Model,
)

from .config import ModelConfig
from .packed import PackedTritLinear


def _resolve_module(root: nn.Module, dotted_name: str) -> nn.Module:
    current: Any = root
    for part in dotted_name.split("."):
        current = current[int(part)] if part.isdigit() else getattr(current, part)
    return current


def _parent_and_leaf(root: nn.Module, dotted_name: str) -> tuple[Any, str]:
    parts = dotted_name.split(".")
    parent: Any = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    return parent, parts[-1]


def _decoder_linear_names(num_layers: int) -> set[str]:
    names = set()
    suffixes = (
        "mlp.down_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "self_attn.k_proj",
        "self_attn.o_proj",
        "self_attn.q_proj",
        "self_attn.v_proj",
    )
    for index in range(num_layers):
        names.update(f"model.layers.{index}.{suffix}" for suffix in suffixes)
    return names


def _linear_shape(module: nn.Module) -> tuple[int, int]:
    weight = getattr(module, "weight", None)
    if weight is None or weight.ndim != 2:
        raise RuntimeError("Phonon manifest target is not a dense linear")
    return int(weight.shape[1]), int(weight.shape[0])


def configure_packed_modules(
    model: Qwen3ASRModel, manifest: dict[str, Any]
) -> list[PackedTritLinear]:
    if manifest.get("status") != "PASS":
        raise RuntimeError("Phonon packed manifest is not marked PASS")
    if manifest.get("group_size") != 128 or manifest.get("bits") != 2:
        raise RuntimeError("unsupported Phonon packed decoder layout")

    rows = manifest.get("modules", [])
    expected_names = _decoder_linear_names(model.config.text_config.num_hidden_layers)
    row_names = [str(row.get("name", "")) for row in rows]
    if len(row_names) != len(set(row_names)) or set(row_names) != expected_names:
        raise RuntimeError("Phonon manifest does not cover the decoder linears exactly")

    decoder_metadata = manifest.get("decoder_metadata")
    slim_metadata = decoder_metadata is not None
    if slim_metadata and decoder_metadata.get("format") != "broadcast-scales-v1":
        raise RuntimeError("unsupported Phonon decoder metadata format")
    decoder_codes = manifest.get("decoder_codes")
    quint5_codes = decoder_codes is not None
    if quint5_codes and decoder_codes.get("format") != "ten-base5-per-24bit-v1":
        raise RuntimeError("unsupported Phonon decoder code format")

    packed_modules = []
    for row in rows:
        name = str(row["name"])
        original = _resolve_module(model, name)
        in_features, out_features = _linear_shape(original)
        if (int(row["in_features"]), int(row["out_features"])) != (
            in_features,
            out_features,
        ):
            raise RuntimeError(f"Phonon shape mismatch for {name}")
        packed = PackedTritLinear(
            in_features,
            out_features,
            group_size=128,
            slim_metadata=slim_metadata,
            quint5_codes=quint5_codes,
        )
        parent, leaf = _parent_and_leaf(model, name)
        setattr(parent, leaf, packed)
        packed_modules.append(packed)

    hybrid = manifest.get("hybrid_quantization")
    if hybrid is None:
        return packed_modules
    if hybrid.get("format") != "mlx-native-affine-v1":
        raise RuntimeError("unsupported Phonon hybrid quantization format")

    embedding = hybrid.get("embedding")
    if embedding:
        name = str(embedding.get("name", ""))
        if name != "model.embed_tokens":
            raise RuntimeError("unsupported Phonon embedding target")
        original = _resolve_module(model, name)
        weight = getattr(original, "weight", None)
        expected_shape = (
            int(embedding["num_embeddings"]),
            int(embedding["dims"]),
        )
        if weight is None or tuple(weight.shape) != expected_shape:
            raise RuntimeError("Phonon embedding shape mismatch")
        parent, leaf = _parent_and_leaf(model, name)
        setattr(
            parent,
            leaf,
            nn.QuantizedEmbedding(
                *expected_shape,
                group_size=int(embedding["group_size"]),
                bits=int(embedding["bits"]),
                mode=str(embedding["mode"]),
            ),
        )

    seen_audio: set[str] = set()
    for row in hybrid.get("audio_linears", []):
        name = str(row.get("name", ""))
        if not name.startswith("audio_tower.") or name in seen_audio:
            raise RuntimeError("invalid Phonon audio linear target")
        seen_audio.add(name)
        original = _resolve_module(model, name)
        in_features, out_features = _linear_shape(original)
        if (int(row["in_features"]), int(row["out_features"])) != (
            in_features,
            out_features,
        ):
            raise RuntimeError(f"Phonon shape mismatch for {name}")
        has_bias = getattr(original, "bias", None) is not None
        if bool(row["bias"]) != has_bias:
            raise RuntimeError(f"Phonon bias mismatch for {name}")
        parent, leaf = _parent_and_leaf(model, name)
        setattr(
            parent,
            leaf,
            nn.QuantizedLinear(
                in_features,
                out_features,
                bias=has_bias,
                group_size=int(row["group_size"]),
                bits=int(row["bits"]),
                mode=str(row["mode"]),
            ),
        )
    return packed_modules


class PhononASRModel(Qwen3ASRModel):
    """Qwen3-ASR with Phonon's exact prompt-head-elision decode path."""

    def stream_generate(
        self,
        audio,
        *,
        max_tokens: int = 8192,
        sampler: Optional[Callable[[mx.array], mx.array]] = None,
        logits_processors: Optional[List[Callable]] = None,
        language: Optional[str] = None,
        prefill_step_size: int = 2048,
        verbose: bool = False,
        system_prompt: str | None = None,
    ) -> Generator[tuple[int, mx.array], None, None]:
        """Generate without projecting unused audio-prefix hidden states."""

        del verbose
        if prefill_step_size <= 0:
            raise ValueError("prefill_step_size must be positive")
        if not hasattr(self, "_tokenizer") or not hasattr(self, "_feature_extractor"):
            raise RuntimeError(
                "Tokenizer/FeatureExtractor not initialized. Call post_load_hook first."
            )
        sampler = sampler or (lambda logits: mx.argmax(logits, axis=-1))

        features, feature_mask, num_audio_tokens = self._preprocess_audio(audio)
        audio_features = self.get_audio_features(features, feature_mask)
        input_ids = self._build_prompt(num_audio_tokens, language, system_prompt)
        inputs = self._build_inputs_embeds(input_ids, audio_features)[0]
        prompt = input_ids[0]
        del features, feature_mask, audio_features

        if max_tokens <= 0:
            return
        cache = self.make_cache()
        remaining = inputs.shape[0] - 1
        consumed = 0
        while remaining > 0:
            count = min(prefill_step_size, remaining)
            self.model(
                inputs_embeds=inputs[consumed : consumed + count][None], cache=cache
            )
            mx.eval([entry.state for entry in cache])
            consumed += count
            remaining -= count
            mx.clear_cache()

        hidden = self.model(inputs_embeds=inputs[consumed:][None], cache=cache)
        logits = (
            self.lm_head(hidden)[:, -1, :]
            if self.lm_head is not None
            else self.model.embed_tokens.as_linear(hidden)[:, -1, :]
        )
        history = prompt[-1:]

        def process(current_logits: mx.array) -> mx.array:
            if logits_processors:
                for processor in logits_processors:
                    current_logits = processor(history, current_logits)
            return current_logits - mx.logsumexp(current_logits, keepdims=True)

        logprobs = process(logits)
        token = sampler(logprobs)
        mx.async_eval(token, logprobs)
        eos_ids = self._eos_token_ids()

        for step in range(max_tokens):
            history = mx.concatenate((history, token.reshape((-1,))))
            next_logits = self._forward_with_embeds(
                self.model.embed_tokens(token.reshape((1, 1))), cache
            )[:, -1, :]
            next_logprobs = process(next_logits)
            next_token = sampler(next_logprobs)
            mx.async_eval(next_token, next_logprobs)

            token_id = int(token.item())
            if token_id in eos_ids:
                break
            yield token_id, logprobs.squeeze(0)
            token, logprobs = next_token, next_logprobs
            if step and step % 256 == 0:
                mx.clear_cache()


class Model(Qwen3Model):
    """Loader wrapper that installs Phonon's packed layers before weight load."""

    def __init__(self, config: ModelConfig):
        self._model = PhononASRModel(config)
        self.config = self._model.config
        model_path = Path(config.model_path) if config.model_path else None
        if model_path is None:
            raise ValueError("Phonon model_path is required")
        manifest = json.loads((model_path / "packed_manifest.json").read_text())
        self._packed_modules = configure_packed_modules(self._model, manifest)
        marker_path = model_path / ".mlx_audio_phonon.json"
        if marker_path.is_file():
            marker = json.loads(marker_path.read_text())
            self.config.model_repo = marker.get("model_id")
            self.config.phonon_profile = marker.get("profile")

    def load_weights(self, weights, strict: bool = False):
        # Published manifests and weights are a single checksummed unit. A
        # partial load would silently invalidate the model's accuracy contract.
        result = self._model.load_weights(weights, strict=True)
        for module in self._packed_modules:
            module.materialize()
        return result

    @classmethod
    def sanitize(cls, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        return PhononASRModel.sanitize(weights)
