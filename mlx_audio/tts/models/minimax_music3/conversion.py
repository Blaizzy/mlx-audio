"""Convert the official modular Diffusers checkpoint into MLX tensors.

Adapted and modified from mikolaj92/minimax-music3-mlx under Apache-2.0.
See LICENSE and NOTICE.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Mapping

import mlx.core as mx

from .config import ModelConfig

_COMPONENTS = (
    "language_model",
    "rvq_depth_decoder",
    "condition_encoder",
    "transformer",
    "vocoder",
)
_CONV_TRANSPOSE_RE = re.compile(
    r"(conv_t\d+|conv_transpose|convtr[^a-z]|deconv)", re.IGNORECASE
)


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"MiniMax Music 3 component config missing: {path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def prepare_config(config: dict, model_path: Path) -> dict:
    """Flatten the five official component configs into ``ModelConfig``."""
    lm = _read_json(model_path / "language_model" / "config.json")
    depth = _read_json(model_path / "rvq_depth_decoder" / "config.json")
    condition = _read_json(model_path / "condition_encoder" / "config.json")
    transformer = _read_json(model_path / "transformer" / "config.json")
    vocoder = _read_json(model_path / "vocoder" / "config.json")

    rope = lm.get("rope_parameters") or {}
    merged = ModelConfig(
        hidden_size=int(lm["hidden_size"]),
        vocab_size=int(lm["vocab_size"]),
        num_hidden_layers=int(lm["num_hidden_layers"]),
        intermediate_size=int(lm["intermediate_size"]),
        num_attention_heads=int(lm["num_attention_heads"]),
        num_key_value_heads=int(lm["num_key_value_heads"]),
        head_dim=int(lm["head_dim"]),
        max_position_embeddings=int(lm["max_position_embeddings"]),
        rms_norm_eps=float(lm["rms_norm_eps"]),
        rope_theta=float(rope.get("rope_theta", 1_000_000.0)),
        tie_word_embeddings=bool(lm.get("tie_word_embeddings", False)),
        audio_vocab_size=int(depth["audio_vocab_size"]),
        num_codebooks=int(depth["num_codebooks"]),
        depth_num_layers=int(depth["num_layers"]),
        depth_num_heads=int(depth["num_attention_heads"]),
        depth_intermediate_size=int(depth["intermediate_size"]),
        depth_max_position_embeddings=int(depth.get("max_position_embeddings", 16)),
        condition_out_dim=int(condition["out_dim"]),
        num_condition_layers=int(condition["num_condition_layers"]),
        input_sampling_rate=int(condition.get("input_sampling_rate", 24_000)),
        input_hop_length=int(condition.get("input_hop_length", 960)),
        output_sampling_rate=int(condition.get("output_sampling_rate", 44_100)),
        output_hop_length=int(condition.get("output_hop_length", 512)),
        dit_in_channels=int(transformer["in_channels"]),
        dit_num_layers=int(transformer["num_layers"]),
        dit_num_heads=int(transformer["num_attention_heads"]),
        dit_head_dim=int(transformer["attention_head_dim"]),
        dit_ff_inner_dim=int(transformer["ff_inner_dim"]),
        dit_rotary_dim=int(transformer["rotary_dim"]),
        dit_fourier_dim=int(transformer["fourier_embedding_dim"]),
        vocoder_input_dim=int(vocoder["decoder_input_dim"]),
        vocoder_hidden_dim=int(vocoder["decoder_hidden_dim"]),
        vocoder_upsampling_ratios=tuple(
            int(value) for value in vocoder["upsampling_ratios"]
        ),
        sample_rate=int(vocoder["sampling_rate"]),
    )

    if int(depth["hidden_size"]) != merged.hidden_size:
        raise ValueError(
            "RVQ depth decoder hidden size does not match the language model"
        )
    if int(condition["condition_hidden_dim"]) != merged.hidden_size:
        raise ValueError(
            "Condition encoder hidden size does not match the language model"
        )
    if int(transformer["condition_dim"]) != merged.condition_out_dim:
        raise ValueError(
            "Transformer condition size does not match the condition encoder"
        )
    if int(vocoder["latent_channels"]) != merged.dit_in_channels:
        raise ValueError("Vocoder latent size does not match the flow transformer")

    result = dict(config)
    result.update(merged.to_dict())
    result["architectures"] = ["MiniMaxMusic3ForConditionalGeneration"]
    result["model_type"] = "minimax_music3"
    source_dtype = lm.get("dtype") or lm.get("torch_dtype")
    if source_dtype:
        result["torch_dtype"] = source_dtype
    return result


def _fuse_weight_norm_pairs(
    state: Mapping[str, mx.array],
) -> dict[str, mx.array]:
    """Collapse PyTorch ``weight_g``/``weight_v`` parameters before remapping."""
    output: dict[str, mx.array] = {}
    consumed: set[str] = set()
    for key, value in state.items():
        if key in consumed:
            continue
        if key.endswith(".weight_v"):
            prefix = key[: -len(".weight_v")]
            g_key = f"{prefix}.weight_g"
            if g_key in state:
                value_32 = value.astype(mx.float32)
                axes = tuple(range(1, value.ndim))
                norm = mx.sqrt(mx.sum(value_32**2, axis=axes, keepdims=True))
                norm = mx.maximum(norm, mx.array(1e-12, dtype=mx.float32))
                output[f"{prefix}.weight"] = (
                    state[g_key].astype(mx.float32) * value_32 / norm
                )
                consumed.update((key, g_key))
                continue
        if key.endswith(".weight_g"):
            v_key = f"{key[: -len('.weight_g')]}.weight_v"
            if v_key in state:
                continue
        output[key] = value
    return output


def _sanitize_key(key: str) -> str | None:
    if "rotary_emb" in key or key.endswith(".inv_freq"):
        return None
    if "transformer_blocks" in key and (
        ".to_out.1." in key or key.endswith(".to_out.1")
    ):
        return None
    if "transformer_blocks" in key and key.endswith((".to_out.weight", ".to_out.bias")):
        suffix = ".weight" if key.endswith(".weight") else ".bias"
        return f"{key[: -len(suffix)]}.0{suffix}"
    return key


def _remap_tensor(key: str, value: mx.array) -> mx.array:
    if value.ndim != 3 or not key.endswith(".weight"):
        return value
    if _CONV_TRANSPOSE_RE.search(key):
        return value.transpose(1, 2, 0)
    return value.transpose(0, 2, 1)


def _load_component(path: Path) -> dict[str, mx.array]:
    files = sorted(path.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No safetensors found in {path}")
    output: dict[str, mx.array] = {}
    for file in files:
        for key, value in mx.load(str(file)).items():
            if key in output:
                raise ValueError(f"Duplicate tensor {key!r} in {path}")
            output[key] = value
    return output


def load_source_weights(model_path: Path) -> dict[str, mx.array]:
    """Load, remap, and namespace the official component weight shards."""
    output: dict[str, mx.array] = {}
    for component in _COMPONENTS:
        state = _load_component(model_path / component)
        if component == "vocoder":
            state = _fuse_weight_norm_pairs(state)
        for source_key, value in state.items():
            key = _sanitize_key(source_key)
            if key is None:
                continue
            full_key = f"{component}.{key}"
            if full_key in output:
                raise ValueError(f"Duplicate converted tensor {full_key!r}")
            output[full_key] = _remap_tensor(key, value)
    return output


def copy_supporting_files(source: Path, destination: Path) -> None:
    """Copy runtime metadata while excluding the original large checkpoints."""
    for name in ("LICENSE", "LICENSE.md", "README.md"):
        item = source / name
        if item.is_file():
            shutil.copy2(item, destination / item.name)

    def ignore_weights(_: str, names: list[str]) -> set[str]:
        return {
            name
            for name in names
            if name.endswith((".safetensors", ".bin", ".pt", ".pth"))
        }

    for name in ("tokenizer", "scheduler"):
        item = source / name
        if item.is_dir():
            shutil.copytree(
                item,
                destination / name,
                dirs_exist_ok=True,
                ignore=ignore_weights,
            )


__all__ = ["copy_supporting_files", "load_source_weights", "prepare_config"]
