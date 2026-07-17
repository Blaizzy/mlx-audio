"""Offline quantizer: convert an MLX fp32 dots.tts weights dir to a bf16/int8/int4 variant.

Stage-1 scope: quantize ONLY the ``llm.*`` (Qwen2.5) tensors via mlx-lm's ``nn.quantize``;
keep DiT (``velocity_field_predictor``), ``patch_encoder``, the projections, the vocoder,
and the speaker at bf16. Writes a self-contained output dir
(``{core,vocoder,speaker}.safetensors`` + ``config.json`` [with a ``quantization`` block for
int8/int4] + ``llm_config.json`` + ``latent_stats.npz`` + ``tokenizer/``).

Dev-only build tool, mirroring ``convert.py``. Pure MLX (``mlx`` + ``mlx_lm``); no the expected format.
Run: ``python -m dots_tts_mlx.quantize --src weights/dots_tts_mlx --out weights/dots_tts_mlx_int4 --bits 4``
LOCAL artifacts only.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.models.qwen2 import Model as Qwen2Model
from mlx_lm.models.qwen2 import ModelArgs as Qwen2Args

_DOTS_MEMORY_LIMIT_GB = int(os.environ.get("MLX_AUDIO_DOTS_MEMORY_LIMIT_GB", "20"))
if _DOTS_MEMORY_LIMIT_GB > 0:
    mx.set_memory_limit(int(_DOTS_MEMORY_LIMIT_GB * (1 << 30)))

_LLM_PREFIX = "llm."


def _qwen_args(llm_cfg: dict) -> Qwen2Args:
    # MUST stay in sync with dots_tts_mlx.llm.DotsLLM.from_core's Qwen2Args build:
    # the quantized skeleton written here is loaded by that path, so any divergence
    # in the args silently breaks quantized loading.
    return Qwen2Args(
        model_type=llm_cfg.get("model_type", "qwen2"),
        hidden_size=llm_cfg["hidden_size"],
        num_hidden_layers=llm_cfg["num_hidden_layers"],
        intermediate_size=llm_cfg["intermediate_size"],
        num_attention_heads=llm_cfg["num_attention_heads"],
        rms_norm_eps=llm_cfg["rms_norm_eps"],
        vocab_size=llm_cfg["vocab_size"],
        num_key_value_heads=llm_cfg["num_key_value_heads"],
        max_position_embeddings=llm_cfg.get("max_position_embeddings", 32768),
        rope_theta=llm_cfg.get("rope_theta", 1000000.0),
        rope_traditional=llm_cfg.get("rope_traditional", False),
        rope_scaling=llm_cfg.get("rope_scaling"),
        tie_word_embeddings=llm_cfg.get("tie_word_embeddings", True),
    )


def _cast_quant(v: mx.array) -> mx.array:
    """Keep packed uint32 weights as-is; cast float scales/biases to bf16."""
    return v if v.dtype == mx.uint32 else v.astype(mx.bfloat16)


def quantize_dir(src, out, bits: int, group_size: int = 64) -> dict:
    """Write a bf16/int8/int4 variant of the converted weights dir ``src`` to ``out``.

    ``src`` must be the **fp32 convert output** (``python -m dots_tts_mlx.convert``), not an
    already-quantized variant — the bf16 path casts every tensor to bf16 and would corrupt
    packed weights if fed a quantized dir.
    """
    src, out = Path(src), Path(out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((src / "config.json").read_text())
    raw = mx.load(str(src / "core.safetensors"))

    if bits == 16:
        core_out = {k: v.astype(mx.bfloat16) for k, v in raw.items()}
        cfg.pop("quantization", None)
    elif bits in (8, 4):
        llm_cfg = json.loads((src / "llm_config.json").read_text())
        model = Qwen2Model(_qwen_args(llm_cfg))
        llm_w = {
            k[len(_LLM_PREFIX) :]: v.astype(mx.float32)
            for k, v in raw.items()
            if k.startswith(_LLM_PREFIX)
        }
        llm_w = model.sanitize(llm_w)
        model.load_weights(list(llm_w.items()))
        nn.quantize(model, group_size=group_size, bits=bits)
        core_out = {
            k: v.astype(mx.bfloat16)
            for k, v in raw.items()
            if not k.startswith(_LLM_PREFIX)
        }
        core_out.update(
            {
                f"{_LLM_PREFIX}{k}": _cast_quant(v)
                for k, v in tree_flatten(model.parameters())
            }
        )
        cfg["quantization"] = {
            "bits": bits,
            "group_size": group_size,
            "components": ["llm"],
        }
    else:
        raise ValueError(f"--bits must be 16, 8, or 4 (got {bits})")

    mx.save_safetensors(str(out / "core.safetensors"), core_out)
    for name in ("vocoder.safetensors", "speaker.safetensors"):
        w = mx.load(str(src / name))
        mx.save_safetensors(
            str(out / name), {k: v.astype(mx.bfloat16) for k, v in w.items()}
        )

    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    shutil.copy2(src / "llm_config.json", out / "llm_config.json")
    shutil.copy2(src / "latent_stats.npz", out / "latent_stats.npz")
    tok = out / "tokenizer"
    tok.mkdir(exist_ok=True)
    for sp in (src / "tokenizer").iterdir():
        if sp.is_file():
            shutil.copy2(sp, tok / sp.name)

    return {"out": str(out), "bits": bits, "group_size": group_size}


def _main() -> None:
    ap = argparse.ArgumentParser(
        description="Quantize a converted dots.tts MLX weights dir."
    )
    ap.add_argument(
        "--src", default="weights/dots_tts_mlx", help="converted fp32 weights dir"
    )
    ap.add_argument(
        "--out", required=True, help="output dir (e.g. weights/dots_tts_mlx_int4)"
    )
    ap.add_argument("--bits", type=int, choices=(16, 8, 4), required=True)
    ap.add_argument("--group-size", type=int, default=64)
    args = ap.parse_args()
    summary = quantize_dir(
        args.src, args.out, bits=args.bits, group_size=args.group_size
    )
    print(f"[quantize] wrote {summary}", flush=True)


if __name__ == "__main__":
    _main()
