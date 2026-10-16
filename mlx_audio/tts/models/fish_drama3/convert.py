"""Convert fishaudio/drama-3 (block-FP8 experts) into an MLX checkpoint.

Every routed expert is dequantized from 128x128 block-FP8 and stacked per layer
for SwitchGLU, so no FP8 work is left for inference. By default every weight
stays bfloat16; --expert-bits, --dense-bits and --tower-bits quantize the
experts, the dense projections and the audio tower with MLX affine quantization. Shards are streamed and each layer's experts are written as soon
as they are complete, which keeps peak memory near one layer.

    python -m mlx_audio.tts.models.fish_drama3.convert \\
        --src fishaudio/drama-3 --dst drama-3-mlx

Converting the codec needs PyTorch to read codec.pth.
"""

import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Optional

import mlx.core as mx

FP8_BLOCK = 128
GROUP_SIZE = 64
SHARD_BYTES = 4 << 30
SOURCE_PREFIX = "model.language_model."
EXPERT = re.compile(
    r"model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate|up|down)_proj\.(weight|weight_scale_inv)"
)
# Zero-centered backbone norms; stored as 1 + w in float32.
SHIFTED_NORMS = re.compile(
    r"(input_layernorm|post_attention_layernorm|q_norm|k_norm)\.weight$"
    r"|^model\.language_model\.norm\.weight$"
)
# Large backbone projections. The router, the shared-expert gate, and the small
# gated-delta a/b projections stay bfloat16: they set routing and decay, and
# barely matter for memory bandwidth.
DENSE = re.compile(
    r"(self_attn\.[qkvo]_proj|linear_attn\.in_proj_(qkv|z)|linear_attn\.out_proj"
    r"|shared_expert\.(gate|up|down)_proj)\.weight$"
)
TOWER = re.compile(r"(wqkv|wo|w1|w2|w3|eh_proj|codebook_heads\.\d+)\.weight$")
# Every tower norm is zero-centered, including the q/k norms the exported config
# reports as disabled (attention_qk_norm=false is an exporter default).
TOWER_NORMS = re.compile(
    r"(enorm|hnorm|final_layernorm|attention_norm|ffn_norm|q_norm|k_norm)\.weight$"
)
TOWER_PREFIXES = {
    "audio_decoder.codebook_embeddings.": "tower.codebook_embeddings.",
    "audio_decoder_mtp_shared.layers.": "tower.shared_layers.",
    "audio_decoder_mtp.layers.": "tower.residual_layers.",
    "audio_decoder_mtp.heads.": "tower.codebook_heads.",
}
COPIED_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "LICENSE",
)


def dequantize_fp8(weight: mx.array, scale_inv: mx.array) -> mx.array:
    """Block-FP8 bytes [out, in] with float32 scales [out/128, in/128] -> bf16."""
    rows, cols = weight.shape
    dense = mx.from_fp8(weight, dtype=mx.float32)
    dense = dense.reshape(rows // FP8_BLOCK, FP8_BLOCK, cols // FP8_BLOCK, FP8_BLOCK)
    dense = dense * scale_inv.astype(mx.float32)[:, None, :, None]
    return dense.reshape(rows, cols).astype(mx.bfloat16)


class ShardWriter:
    """Collects output tensors, quantizing on the way, into safetensors shards."""

    def __init__(self, dst: Path):
        self.dst = dst
        self.pending: dict[str, mx.array] = {}
        self.pending_bytes = 0
        self.count = 0
        self.quantization: dict[str, dict] = {}

    def add(self, name: str, value: mx.array, bits: Optional[int] = None) -> None:
        items = {name: value}
        if bits:
            path = name.removesuffix(".weight")
            weight, scales, biases = mx.quantize(value, GROUP_SIZE, bits)
            items = {name: weight, f"{path}.scales": scales, f"{path}.biases": biases}
            self.quantization[path] = {"group_size": GROUP_SIZE, "bits": bits}
        mx.eval(list(items.values()))
        self.pending.update(items)
        self.pending_bytes += sum(v.nbytes for v in items.values())
        if self.pending_bytes >= SHARD_BYTES:
            self.flush()

    def flush(self) -> None:
        if self.pending:
            self.count += 1
            shard = self.dst / f"model-{self.count:05d}.safetensors"
            mx.save_safetensors(str(shard), self.pending, metadata={"format": "mlx"})
            print(f"wrote {shard.name}", flush=True)
            self.pending, self.pending_bytes = {}, 0


def convert(
    src: Path,
    dst: Path,
    expert_bits: int = 0,
    dense_bits: int = 0,
    tower_bits: int = 0,
    codec: bool = True,
) -> None:
    """Writes the MLX checkpoint into dst, which must not exist or be empty:
    the loader reads every safetensors file there, so a leftover shard from an
    earlier run would silently replace converted weights."""
    if dst.exists() and any(dst.iterdir()):
        raise FileExistsError(f"{dst} is not empty")
    dst.mkdir(parents=True, exist_ok=True)
    config = json.loads((src / "config.json").read_text())
    num_experts = config["text_config"]["num_experts"]
    band = slice(config["semantic_start_token_id"], config["semantic_end_token_id"] + 1)
    eos = config["eos_token_id"]

    writer = ShardWriter(dst)
    experts: dict[int, dict[tuple, mx.array]] = defaultdict(dict)

    def write_experts(layer: int) -> None:
        parts = experts.pop(layer)
        for proj in ("gate", "up", "down"):
            stacked = mx.stack(
                [
                    dequantize_fp8(
                        parts[(e, proj, "weight")], parts[(e, proj, "weight_scale_inv")]
                    )
                    for e in range(num_experts)
                ]
            )
            name = f"backbone.model.layers.{layer}.mlp.switch_mlp.{proj}_proj.weight"
            writer.add(name, stacked, expert_bits)

    index = json.loads((src / "model.safetensors.index.json").read_text())
    for shard in sorted(set(index["weight_map"].values())):
        tensors = mx.load(str(src / shard))
        for name in sorted(tensors):
            value = tensors[name]
            if match := EXPERT.fullmatch(name):
                layer, expert = int(match[1]), int(match[2])
                experts[layer][(expert, match[3], match[4])] = value
                if len(experts[layer]) == 6 * num_experts:
                    write_experts(layer)
            elif name.startswith(SOURCE_PREFIX):
                out = "backbone.model." + name.removeprefix(SOURCE_PREFIX)
                if name.endswith("conv1d.weight"):
                    value = value.swapaxes(1, 2)  # [C, 1, K] -> [C, K, 1]
                if SHIFTED_NORMS.search(name):
                    value = value.astype(mx.float32) + 1.0
                writer.add(out, value, dense_bits if DENSE.search(name) else None)
            elif name == "lm_head.weight":
                # Decoding only samples EOS or a semantic token, so the head
                # keeps those rows: [semantic band..., EOS].
                rows = mx.concatenate([value[band], value[eos : eos + 1]])
                writer.add("semantic_head.weight", rows, dense_bits)
            elif old := next((p for p in TOWER_PREFIXES if name.startswith(p)), None):
                out = TOWER_PREFIXES[old] + name.removeprefix(old)
                if TOWER_NORMS.search(out):
                    value = value.astype(mx.float32) + 1.0
                writer.add(out, value, tower_bits if TOWER.search(out) else None)
            else:
                raise ValueError(f"unexpected tensor {name}")
        del tensors
        mx.clear_cache()
    if experts:
        raise ValueError(f"incomplete experts in layers {sorted(experts)}")
    writer.flush()

    config.pop("quantization_config", None)
    if writer.quantization:
        # Top-level values are the loader's defaults; every quantized module
        # also has its own entry.
        config["quantization"] = {
            "group_size": GROUP_SIZE,
            "bits": dense_bits or tower_bits or expert_bits,
            **writer.quantization,
        }
    config["model_type"] = "fish_drama3"
    (dst / "config.json").write_text(json.dumps(config, indent=2))
    for name in COPIED_FILES:
        if (src / name).exists():
            shutil.copy(src / name, dst / name)
    if codec:
        convert_codec(src / "codec.pth", dst / "codec.safetensors")


def convert_codec(src: Path, dst: Path) -> None:
    """codec.pth (a PyTorch state dict) -> codec.safetensors for fish_s1_dac,
    checking that every learned codec parameter is present."""
    import numpy as np
    import torch
    from mlx.utils import tree_flatten

    from mlx_audio.codec.models.fish_s1_dac.fish_s1_dac import build_ae

    state = torch.load(src, map_location="cpu", weights_only=True)
    state = state.get("state_dict", state)
    # Training checkpoints also hold the discriminator.
    if any(key.startswith("generator.") for key in state):
        state = {
            key.removeprefix("generator."): value
            for key, value in state.items()
            if key.startswith("generator.")
        }
    # The square causal masks are buffers the codec builds itself.
    weights = {
        key: mx.array(value.float().numpy().astype(np.float32))
        for key, value in state.items()
        if not key.endswith("causal_mask")
    }

    codec = build_ae()
    loaded = codec.sanitize(weights)
    # Weight-norm convs rebuild conv.conv.{weight,bias} from weight_g/weight_v.
    derived = re.compile(r"(.*)\.conv\.conv\.(weight|bias)$")
    missing = [
        name
        for name, _ in tree_flatten(codec.parameters())
        if name not in loaded
        and not name.endswith("causal_mask")
        and not ((m := derived.match(name)) and f"{m[1]}.weight_g" in loaded)
    ]
    if missing:
        raise ValueError(f"codec.pth lacks {len(missing)} parameters: {missing[:5]}")
    mx.save_safetensors(str(dst), weights)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--src", type=Path, required=True, help="drama-3 directory")
    parser.add_argument("--dst", type=Path, required=True, help="new, empty directory")
    parser.add_argument("--expert-bits", type=int, default=0, help="0 keeps bf16")
    parser.add_argument("--dense-bits", type=int, default=0, help="0 keeps bf16")
    # The tower samples nine codebooks per frame from its own heads; at 4 bits
    # their top-1 agreement with bf16 dropped from about 88% to 40%, so quantize
    # it to 8 at most.
    parser.add_argument("--tower-bits", type=int, default=0, help="0 keeps bf16")
    args = parser.parse_args()
    convert(args.src, args.dst, args.expert_bits, args.dense_bits, args.tower_bits)


if __name__ == "__main__":
    main()
