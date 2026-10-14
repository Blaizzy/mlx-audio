"""Convert the original Drama 3 checkpoint (fishaudio/drama-3) to MLX.

Streams one backbone layer at a time, so the ~70 GB bf16 equivalent never sits
in memory: block-FP8 experts are dequantized, stacked into SwitchGLU layout and
quantized; backbone RMSNorm weights are stored pre-shifted (1 + w); the
lm_head keeps only the rows the main token is sampled from (eos + semantic band);
codec.pth becomes codec.safetensors.

    python -m mlx_audio.tts.models.fish_omni.convert \
        --source fishaudio/drama-3 --output ~/models/drama-3-mlx-q4 --expert-bits 4
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Optional

import mlx.core as mx

FP8_BLOCK = 128
SOURCE_BACKBONE = "model.language_model."
COPY_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "LICENSE",
)
SHARD_BYTES = 5 << 30
NOTICE = (
    "These Fish Audio Materials are licensed under the Fish Audio Community License, "
    "Version 1.0. Copyright © 39 AI, INC. All Rights Reserved.\n\n"
    "Built with Fish Audio. Modified: converted to MLX safetensors; block-FP8 "
    "experts dequantized and re-quantized with MLX affine quantization; lm_head "
    "rows outside eos and the semantic band removed; backbone RMSNorm weights "
    "pre-shifted by one.\n"
)
BACKBONE_NORMS = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
)
TOWER_NORMS = (
    "enorm.weight",
    "hnorm.weight",
    "final_layernorm.weight",
    "attention_norm.weight",
    "ffn_norm.weight",
    "attention.q_norm.weight",
    "attention.k_norm.weight",
)
# Small or precision-sensitive matrices stay bf16 whatever --dense-bits says.
KEEP_DENSE = (
    "linear_attn.in_proj_a.weight",
    "linear_attn.in_proj_b.weight",
    "mlp.gate.weight",
    "mlp.shared_expert_gate.weight",
)


def dequantize_fp8(weight: mx.array, scale_inv: mx.array) -> mx.array:
    """Block-FP8 [out, in] (stored as uint8) with fp32 scales [out/128, in/128]."""
    rows, cols = weight.shape
    blocks = mx.from_fp8(weight, dtype=mx.float32).reshape(
        rows // FP8_BLOCK, FP8_BLOCK, cols // FP8_BLOCK, FP8_BLOCK
    )
    scaled = blocks * scale_inv.astype(mx.float32)[:, None, :, None]
    return scaled.reshape(rows, cols)


class ShardWriter:
    def __init__(self, output: Path):
        self.output = output
        self.pending: Dict[str, mx.array] = {}
        self.pending_bytes = 0
        self.shards = []
        self.weight_map: Dict[str, str] = {}

    def add(self, key: str, value: mx.array) -> None:
        mx.eval(value)
        self.pending[key] = value
        self.pending_bytes += value.nbytes
        if self.pending_bytes >= SHARD_BYTES:
            self.flush()

    def flush(self) -> None:
        if not self.pending:
            return
        name = f"model-{len(self.shards) + 1:05d}.safetensors"
        mx.save_safetensors(
            str(self.output / name), self.pending, metadata={"format": "mlx"}
        )
        self.shards.append((name, list(self.pending)))
        self.pending, self.pending_bytes = {}, 0

    def finish(self) -> int:
        self.flush()
        total = len(self.shards)
        total_bytes = 0
        for index, (name, keys) in enumerate(self.shards, 1):
            final = f"model-{index:05d}-of-{total:05d}.safetensors"
            (self.output / name).rename(self.output / final)
            total_bytes += (self.output / final).stat().st_size
            self.weight_map.update({k: final for k in keys})
        with open(self.output / "model.safetensors.index.json", "w") as f:
            json.dump(
                {
                    "metadata": {"total_size": total_bytes},
                    "weight_map": dict(sorted(self.weight_map.items())),
                },
                f,
                indent=2,
            )
        return total_bytes


class Converter:
    def __init__(
        self,
        source: Path,
        output: Path,
        *,
        expert_bits: Optional[int],
        embed_bits: Optional[int] = None,
        dense_bits: Optional[int] = None,
        tower_bits: Optional[int] = None,
        group_size: int = 64,
    ):
        self.source = source
        self.output = output
        self.expert_bits = expert_bits
        self.embed_bits = embed_bits
        self.dense_bits = dense_bits
        self.tower_bits = tower_bits
        self.group_size = group_size
        with open(source / "config.json") as f:
            self.config = json.load(f)
        with open(source / "model.safetensors.index.json") as f:
            self.weight_map = json.load(f)["weight_map"]
        self.shards: Dict[str, Dict[str, mx.array]] = {}
        self.writer = ShardWriter(output)
        self.overrides: Dict[str, dict] = {}
        self.consumed = set()

    def get(self, key: str) -> mx.array:
        shard = self.weight_map[key]
        if shard not in self.shards:
            self.shards[shard] = mx.load(str(self.source / shard))
        self.consumed.add(key)
        return self.shards[shard][key]

    def put(self, key: str, value: mx.array, bits: Optional[int] = None) -> None:
        """Write a dense tensor, or quantize a `.weight` matrix to `bits`."""
        if bits is None or value.ndim < 2 or value.shape[-1] % self.group_size:
            self.writer.add(key, value.astype(mx.bfloat16))
            return
        weight, scales, biases = mx.quantize(
            value.astype(mx.bfloat16), group_size=self.group_size, bits=bits
        )
        prefix = key[: -len(".weight")]
        self.writer.add(f"{prefix}.weight", weight)
        self.writer.add(f"{prefix}.scales", scales)
        self.writer.add(f"{prefix}.biases", biases)
        self.overrides[prefix] = {"group_size": self.group_size, "bits": bits}

    def convert_backbone_layer(self, layer: int) -> None:
        src = f"{SOURCE_BACKBONE}layers.{layer}."
        dst = f"backbone.layers.{layer}."
        for key in [k for k in self.weight_map if k.startswith(src)]:
            name = key[len(src) :]
            if ".mlp.experts." in f".{name}":
                continue
            value = self.get(key)
            if name in BACKBONE_NORMS:
                self.put(dst + name, value.astype(mx.float32) + 1.0)
            elif name == "linear_attn.conv1d.weight":
                self.put(dst + name, value.moveaxis(2, 1))
            elif (
                name.endswith(".weight") and value.ndim == 2 and name not in KEEP_DENSE
            ):
                self.put(dst + name, value, self.dense_bits)
            else:
                self.put(dst + name, value)

        num_experts = self.config["text_config"]["num_experts"]
        for projection in ("gate_proj", "up_proj", "down_proj"):
            stacked = mx.stack(
                [
                    dequantize_fp8(
                        self.get(f"{src}mlp.experts.{e}.{projection}.weight"),
                        self.get(f"{src}mlp.experts.{e}.{projection}.weight_scale_inv"),
                    )
                    for e in range(num_experts)
                ]
            )
            self.put(
                f"{dst}mlp.switch_mlp.{projection}.weight", stacked, self.expert_bits
            )

    def convert_tower(self) -> None:
        prefixes = {
            "audio_decoder.codebook_embeddings.": "tower.codebook_embeddings.",
            "audio_decoder_mtp_shared.layers.": "tower.shared_layers.",
            "audio_decoder_mtp.layers.": "tower.residual_layers.",
            "audio_decoder_mtp.heads.": "tower.codebook_heads.",
        }
        for key in sorted(self.weight_map):
            old = next((p for p in prefixes if key.startswith(p)), None)
            if old is None:
                continue
            name = prefixes[old] + key[len(old) :]
            value = self.get(key)
            # Tower norms stay zero-centered (see tower.ZeroCenteredRMSNorm).
            if name.endswith(TOWER_NORMS) or "embedding" in name:
                self.put(name, value)
            else:
                self.put(name, value, self.tower_bits)

    def convert(self, num_layers: Optional[int] = None) -> None:
        text = self.config["text_config"]
        layers = text["num_hidden_layers"] if num_layers is None else num_layers

        self.put(
            "backbone.embed_tokens.weight",
            self.get(f"{SOURCE_BACKBONE}embed_tokens.weight"),
            self.embed_bits,
        )
        self.put(
            "backbone.norm.weight",
            self.get(f"{SOURCE_BACKBONE}norm.weight").astype(mx.float32) + 1.0,
        )
        eos = self.config["eos_token_id"]
        begin = self.config["semantic_start_token_id"]
        end = self.config["semantic_end_token_id"]
        assert eos < begin, "lm_head rows assume eos sorts before the semantic band"
        lm_head = self.get("lm_head.weight")
        # Only 4097 rows remain (16 MB), so the main-token logits stay bf16.
        self.put(
            "lm_head.weight",
            mx.concatenate([lm_head[eos : eos + 1], lm_head[begin : end + 1]]),
        )
        for layer in range(layers):
            print(f"backbone layer {layer + 1}/{layers}", flush=True)
            self.convert_backbone_layer(layer)
            # Drop the lazily loaded shards between layers to keep memory flat.
            self.shards.clear()
        self.convert_tower()
        total_bytes = self.writer.finish()

        if num_layers is None:
            skipped = set(self.weight_map) - self.consumed
            if skipped:
                raise ValueError(
                    f"{len(skipped)} checkpoint tensors unused: "
                    f"{sorted(skipped)[:8]}"
                )

        config = dict(self.config)
        config.pop("quantization_config", None)
        config["text_config"] = dict(text, num_hidden_layers=layers)
        config["text_config"]["layer_types"] = text["layer_types"][:layers]
        config["text_config"]["mlp_layer_types"] = text["mlp_layer_types"][:layers]
        config["sample_rate"] = 44100
        if self.overrides:
            # Every quantized module is listed; the top-level bits is unused.
            config["quantization"] = {
                "group_size": self.group_size,
                "bits": 8,
                "mode": "affine",
                **self.overrides,
            }
        with open(self.output / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        print(f"wrote {total_bytes / 1e9:.1f} GB of model weights")


def convert_codec(source: Path, output: Path) -> None:
    import torch

    state = torch.load(source / "codec.pth", map_location="cpu", weights_only=True)
    if "state_dict" in state:
        state = state["state_dict"]
    if any("generator." in k for k in state):
        state = {
            k.replace("generator.", ""): v
            for k, v in state.items()
            if "generator." in k
        }
    arrays = {
        k: mx.array(v.float().numpy() if v.dtype == torch.bfloat16 else v.numpy())
        for k, v in state.items()
        if isinstance(v, torch.Tensor)
    }
    mx.save_safetensors(str(output / "codec.safetensors"), arrays)


def resolve_source(source: str) -> Path:
    path = Path(source).expanduser()
    if path.exists():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=source))


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Drama 3 to MLX.")
    parser.add_argument("--source", default="fishaudio/drama-3")
    parser.add_argument("--output", required=True)
    # Defaults from parity against the reference (see README): the always-on
    # dense paths feed the precision-sensitive tower, so they stay bf16.
    parser.add_argument(
        "--expert-bits", type=int, default=4, help="routed experts; 0 keeps bf16"
    )
    parser.add_argument(
        "--embed-bits", type=int, default=8, help="token embedding; 0 keeps bf16"
    )
    parser.add_argument(
        "--dense-bits",
        type=int,
        default=0,
        help="attention projections and shared experts; 0 keeps bf16",
    )
    parser.add_argument(
        "--tower-bits", type=int, default=0, help="codebook tower; 0 keeps bf16"
    )
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument(
        "--layers", type=int, default=None, help="first N backbone layers (tests)"
    )
    parser.add_argument("--skip-codec", action="store_true")
    args = parser.parse_args()

    source = resolve_source(args.source)
    output = Path(args.output).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    # The loader globs *.safetensors: shards of an earlier run must not linger.
    for stale in output.glob("model-*.safetensors"):
        stale.unlink()
    Converter(
        source,
        output,
        expert_bits=args.expert_bits or None,
        embed_bits=args.embed_bits or None,
        dense_bits=args.dense_bits or None,
        tower_bits=args.tower_bits or None,
        group_size=args.group_size,
    ).convert(args.layers)
    for name in COPY_FILES:
        if (source / name).exists():
            # copyfile, not copy: HF cache blobs are read-only.
            (output / name).unlink(missing_ok=True)
            shutil.copyfile(source / name, output / name)
    (output / "NOTICE").write_text(NOTICE)
    if not args.skip_codec:
        convert_codec(source, output)
    print(f"done: {output}  (peak memory {mx.get_peak_memory() / 1e9:.1f} GB)")


if __name__ == "__main__":
    main()
