from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mlx.core as mx

from mlx_audio.codec.models.nemotron_voicechat import NemotronVoiceChatCodec
from mlx_audio.utils import load_weights

from .config import ModelConfig
from .model import sanitize_weights

QUANTIZED_PREFIXES = (
    "stt_model.llm.",
    "stt_model.embed_tokens.",
    "stt_model.lm_head.",
    "stt_model.function_head.",
)


def _quantize(
    weights: dict[str, mx.array], group_size: int, bits: int
) -> dict[str, mx.array]:
    quantized: dict[str, mx.array] = {}
    for key, value in weights.items():
        if (
            key.startswith(QUANTIZED_PREFIXES)
            and key.endswith(".weight")
            and value.ndim == 2
            and value.shape[-1] % group_size == 0
        ):
            packed, scales, biases = mx.quantize(
                value, group_size=group_size, bits=bits
            )
            stem = key[: -len(".weight")]
            quantized[key] = packed
            quantized[f"{stem}.scales"] = scales
            quantized[f"{stem}.biases"] = biases
            mx.eval(packed, scales, biases)
        else:
            quantized[key] = value
            mx.eval(value)
    return quantized


def convert(
    source: str | Path,
    output: str | Path,
    *,
    quantize: bool = False,
    group_size: int = 64,
    bits: int = 4,
) -> Path:
    source = Path(source).expanduser()
    output = Path(output).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    with open(source / "config.json", encoding="utf-8") as config_file:
        config = json.load(config_file)

    parsed = ModelConfig.from_dict(config).config
    config["model_type"] = "nemotron_voicechat"
    llm_config = {
        key: value
        for key, value in parsed.llm.__dict__.items()
        if value is not None and not key.startswith("_")
    }
    if math.isinf(llm_config.get("time_step_limit", (0, 0))[1]):
        llm_config.pop("time_step_limit")
    config.setdefault("mlx_audio", {})["llm_config"] = llm_config
    config["mlx_audio"]["char_vocab_size"] = parsed.tts.char_vocab_size
    config["mlx_audio"]["prepared_weights"] = True

    weights = sanitize_weights(
        load_weights(source), NemotronVoiceChatCodec(parsed.codec)
    )
    if quantize:
        weights = _quantize(weights, group_size, bits)
        config["quantization"] = {
            "group_size": group_size,
            "bits": bits,
            "mode": "affine",
        }

    with open(output / "config.json", "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, indent=2)
    mx.save_safetensors(str(output / "model.safetensors"), weights)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare NVIDIA NemotronLabs VoiceChat weights for MLX Audio"
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("-q", "--quantize", action="store_true")
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--bits", type=int, default=4)
    args = parser.parse_args()
    convert(
        args.source,
        args.output,
        quantize=args.quantize,
        group_size=args.group_size,
        bits=args.bits,
    )


if __name__ == "__main__":
    main()
