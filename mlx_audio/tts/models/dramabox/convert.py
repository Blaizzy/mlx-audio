from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
from huggingface_hub import snapshot_download

SOURCE_REPO = "ResembleAI/Dramabox"
DEFAULT_TEXT_ENCODER = "mlx-community/gemma-3-12b-it-8bit"
DEFAULT_ALLOW_PATTERNS = [
    "config.json",
    "README.md",
    "LICENSE",
    "dramabox-dit-v1.safetensors",
    "dramabox-audio-components.safetensors",
    "assets/silence_latent_frame.pt",
]


def _rename_key(key: str) -> str:
    if key.startswith("model.diffusion_model.audio_embeddings_connector."):
        return key.replace(
            "model.diffusion_model.audio_embeddings_connector.",
            "text_conditioner.audio_connector.",
            1,
        )
    if key.startswith("model.diffusion_model."):
        return "transformer." + key.removeprefix("model.diffusion_model.")
    if key.startswith("text_embedding_projection.audio_aggregate_embed."):
        return key.replace(
            "text_embedding_projection.audio_aggregate_embed.",
            "text_conditioner.feature_extractor.audio_aggregate_embed.",
            1,
        )
    if key.startswith("text_embedding_projection."):
        return "text_conditioner." + key
    if key.startswith("audio_vae."):
        return key
    if key.startswith("vocoder."):
        return key
    return key


def _transpose_weight(key: str, value: mx.array) -> mx.array:
    conv1d_kernels = {3, 4, 7, 11, 12}
    if (
        key.endswith(".weight")
        and key.startswith("audio_vae.")
        and value.ndim == 4
        and value.shape[-1] in (1, 3)
        and value.shape[-2] in (1, 3)
    ):
        return value.transpose(0, 2, 3, 1)
    if key.endswith(".weight") and key.startswith("vocoder."):
        if (
            ".ups." in key
            and value.ndim == 3
            and value.shape[-1] in conv1d_kernels
            and value.shape[1] not in conv1d_kernels
        ):
            return value.transpose(1, 2, 0)
        if (
            value.ndim == 3
            and value.shape[-1] in conv1d_kernels
            and value.shape[1] not in conv1d_kernels
        ):
            return value.transpose(0, 2, 1)
    return value


def sanitize_weights(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    sanitized = {}
    for key, value in weights.items():
        if key.startswith(
            ("vae.per_channel_statistics.", "audio_vae.per_channel_statistics.")
        ):
            stat_name = key.split(".")[-1].replace("-", "_")
            for module_name in ("encoder", "decoder"):
                mapped_key = (
                    f"audio_vae.{module_name}.per_channel_statistics.{stat_name}"
                )
                sanitized[mapped_key] = value
            continue

        mapped_key = _rename_key(key)
        sanitized[mapped_key] = _transpose_weight(mapped_key, value)
    return sanitized


def shard_file_name(source_name: str) -> str:
    return {
        "dramabox-dit-v1.safetensors": "model-00001-of-00002.safetensors",
        "dramabox-audio-components.safetensors": "model-00002-of-00002.safetensors",
    }[source_name]


def convert(
    source: str | Path = SOURCE_REPO,
    output: str | Path = "dramabox-mlx",
) -> Path:
    source_path = Path(source)
    if not source_path.exists():
        source_path = Path(
            snapshot_download(
                repo_id=str(source),
                allow_patterns=DEFAULT_ALLOW_PATTERNS,
            )
        )

    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    config = json.loads((source_path / "config.json").read_text())
    config["model_type"] = "dramabox-tts"
    config["text_encoder"] = DEFAULT_TEXT_ENCODER
    config.pop("mlx_text_encoder", None)
    config.setdefault("inference_defaults", {})
    config["inference_defaults"]["rescale_scale"] = "auto"
    config["mlx_audio"] = {
        "source_repo": SOURCE_REPO,
        "weight_format": "split_safetensors",
        "watermarking": "skipped",
    }
    (output_path / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    source_weight_paths = [
        source_path / "dramabox-dit-v1.safetensors",
        source_path / "dramabox-audio-components.safetensors",
    ]
    missing = [str(path) for path in source_weight_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Dramabox weight files: {missing}")

    for source_weight_path in source_weight_paths:
        weights = sanitize_weights(mx.load(str(source_weight_path)))
        mx.save_safetensors(
            str(output_path / shard_file_name(source_weight_path.name)), weights
        )
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert Dramabox weights to MLX.")
    parser.add_argument("--source", default=SOURCE_REPO)
    parser.add_argument("--output", default="dramabox-mlx")
    args = parser.parse_args()
    print(convert(args.source, args.output))
