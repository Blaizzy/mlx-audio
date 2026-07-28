"""Convert an NVIDIA NeMo ``.nemo`` Nemotron ASR checkpoint to MLX format.

Produces a directory with ``config.json``, ``model.safetensors``, the SentencePiece
``tokenizer.model``, and ``vocab.txt`` — loadable via ``mlx_audio.stt.load(path)``.

Usage:
    python -m mlx_audio.stt.models.nemotron_asr.convert \
        --nemo-path nvidia/nemotron-3.5-asr-streaming-0.6b \
        --mlx-path  ./nemotron-3.5-asr-streaming-0.6b-mlx

``--nemo-path`` accepts either a local ``.nemo`` file or a HuggingFace repo id
(the ``.nemo`` is downloaded automatically). Requires ``torch`` + ``pyyaml``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np


# --------------------------------------------------------------------------- io
def _locate_nemo(nemo_path: str) -> Path:
    p = Path(nemo_path).expanduser()
    if p.exists():
        return p
    # Treat as a HF repo id and pull the single .nemo file.
    from huggingface_hub import HfApi, hf_hub_download

    files = [f for f in HfApi().list_repo_files(nemo_path) if f.endswith(".nemo")]
    if not files:
        raise FileNotFoundError(f"No .nemo file found in HF repo '{nemo_path}'")
    return Path(hf_hub_download(nemo_path, files[0]))


def _extract_nemo(nemo_file: Path, dest: Path) -> None:
    # .nemo is a (gzip) tar archive.
    with tarfile.open(nemo_file, "r:*") as tar:
        tar.extractall(dest)


# ----------------------------------------------------------------------- config
def _build_config(nemo_cfg: dict) -> dict:
    pre = nemo_cfg["preprocessor"]
    enc = nemo_cfg["encoder"]
    dec = nemo_cfg["decoder"]
    joint = nemo_cfg["joint"]
    defaults = nemo_cfg.get("model_defaults", {})

    enc_hidden = defaults.get("enc_hidden", enc["d_model"])
    num_prompts = defaults.get("num_prompts", 128)

    return {
        "model_type": "nemotron_asr",
        "target": nemo_cfg.get("target", ""),
        "preprocessor": {
            "sample_rate": pre.get("sample_rate", 16000),
            "features": pre.get("features", 128),
            "n_fft": pre.get("n_fft", 512),
            "window_size": pre.get("window_size", 0.025),
            "window_stride": pre.get("window_stride", 0.01),
            "window": pre.get("window", "hann"),
            "preemph": (
                0.97 if pre.get("preemph", 0.97) is None else pre.get("preemph", 0.97)
            ),
            "dither": pre.get("dither", 1.0e-05),
            "normalize": str(pre.get("normalize", "NA")),
            "log_zero_guard_value": float(pre.get("log_zero_guard_value", 2.0**-24)),
            "pad_to": pre.get("pad_to", 0),
            "pad_value": pre.get("pad_value", 0.0),
        },
        "encoder": {
            "feat_in": enc["feat_in"],
            "n_layers": enc["n_layers"],
            "d_model": enc["d_model"],
            "n_heads": enc["n_heads"],
            "ff_expansion_factor": enc["ff_expansion_factor"],
            "subsampling_factor": enc["subsampling_factor"],
            "subsampling_conv_channels": enc["subsampling_conv_channels"],
            "conv_kernel_size": enc["conv_kernel_size"],
            "causal_downsampling": enc.get("causal_downsampling", True),
            "conv_context_size": enc.get("conv_context_size", "causal"),
            "conv_norm_type": enc.get("conv_norm_type", "layer_norm"),
            "self_attention_model": enc.get("self_attention_model", "rel_pos"),
            "att_context_style": enc.get("att_context_style", "chunked_limited"),
            "att_context_size": [
                list(c) for c in enc.get("att_context_size", [[56, 13]])
            ],
            "pos_emb_max_len": enc.get("pos_emb_max_len", 5000),
            "use_bias": enc.get("use_bias", False),
            "xscaling": enc.get("xscaling", False),
        },
        "prompt": {
            "num_prompts": num_prompts,
            "prompt_hidden": enc_hidden * 2,
            "prompt_dictionary": dict(defaults.get("prompt_dictionary", {})),
        },
        "decoder": {
            "pred_hidden": dec["prednet"]["pred_hidden"],
            "pred_rnn_layers": dec["prednet"]["pred_rnn_layers"],
            "vocab_size": dec["vocab_size"],
            "blank_as_pad": dec.get("blank_as_pad", True),
        },
        "joint": {
            "joint_hidden": joint["jointnet"]["joint_hidden"],
            "activation": joint["jointnet"].get("activation", "relu"),
            "encoder_hidden": joint["jointnet"]["encoder_hidden"],
            "pred_hidden": joint["jointnet"]["pred_hidden"],
            "num_classes": joint["num_classes"],
        },
        "vocabulary": list(joint["vocabulary"]),
        "default_language": "auto",
        "default_att_context_size": [56, 13],
        "max_symbols": nemo_cfg.get("decoding", {})
        .get("greedy", {})
        .get("max_symbols", 10),
    }


# ---------------------------------------------------------------------- weights
def _to_mx(t) -> mx.array:
    return mx.array(t.detach().to("cpu").float().numpy())


def _convert_weights(state_dict, dtype: mx.Dtype) -> dict:
    out: dict[str, mx.array] = {}
    lstm_bias: dict[str, list] = {}

    for key, tensor in state_dict.items():
        if key.startswith("preprocessor."):
            continue  # featurizer buffers — recomputed in MLX

        # LSTM: combine ih/hh biases and rename to nn.LSTM (Wx/Wh/bias) per layer.
        if ".dec_rnn.lstm." in key:
            base, suffix = key.split(".dec_rnn.lstm.")
            stem = f"{base}.dec_rnn.lstm"
            if suffix.startswith("weight_ih_l"):
                layer = suffix[len("weight_ih_l") :]
                out[f"{stem}.{layer}.Wx"] = _to_mx(tensor).astype(dtype)
            elif suffix.startswith("weight_hh_l"):
                layer = suffix[len("weight_hh_l") :]
                out[f"{stem}.{layer}.Wh"] = _to_mx(tensor).astype(dtype)
            elif suffix.startswith("bias_ih_l") or suffix.startswith("bias_hh_l"):
                layer = suffix.split("_l")[-1]
                lstm_bias.setdefault(f"{stem}.{layer}.bias", []).append(_to_mx(tensor))
            continue

        arr = _to_mx(tensor)
        if arr.ndim == 4:  # Conv2d: (O, I, kH, kW) -> (O, kH, kW, I)
            arr = arr.transpose(0, 2, 3, 1)
        elif arr.ndim == 3:  # Conv1d: (O, I, k) -> (O, k, I)
            arr = arr.transpose(0, 2, 1)
        out[key] = arr.astype(dtype)

    for key, biases in lstm_bias.items():
        out[key] = sum(biases).astype(dtype)

    return out


def _quantize(config: dict, weights: dict, q_bits: int, q_group_size: int):
    """Quantize the (Linear/Embedding) weights and update config in place.

    Conv/LayerNorm and any layer whose last dim isn't divisible by the group size
    are left in the base dtype — the standard mlx quantization predicate.
    """
    from mlx.utils import tree_flatten
    from mlx_lm.utils import quantize_model

    from .nemotron_asr import Model, ModelConfig

    model = Model(ModelConfig.from_dict(config))
    model.load_weights(list(weights.items()))

    def predicate(_path, module) -> bool:
        return (
            hasattr(module, "to_quantized")
            and hasattr(module, "weight")
            and module.weight.shape[-1] % q_group_size == 0
        )

    # quantize_model returns (quantized_model, updated_config).
    model, q_config = quantize_model(
        model, config, q_group_size, q_bits, quant_predicate=predicate
    )
    q_weights = dict(tree_flatten(model.parameters()))
    return q_config, q_weights


# ------------------------------------------------------------------------- main
def convert(
    nemo_path: str,
    mlx_path: str,
    dtype: str = "bfloat16",
    quantize: bool = False,
    q_bits: int = 4,
    q_group_size: int = 64,
) -> Path:
    import yaml

    mlx_dtype = getattr(mx, dtype)
    out_dir = Path(mlx_path).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    nemo_file = _locate_nemo(nemo_path)
    print(f"[INFO] Using .nemo: {nemo_file}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _extract_nemo(nemo_file, tmp)

        cfg_yaml = next(tmp.glob("*model_config.yaml"), None) or (
            tmp / "model_config.yaml"
        )
        with open(cfg_yaml) as f:
            nemo_cfg = yaml.safe_load(f)

        config = _build_config(nemo_cfg)

        # Tokenizer + vocab artifacts.
        for spm in tmp.glob("*tokenizer.model"):
            shutil.copy(spm, out_dir / "tokenizer.model")
        for vocab in tmp.glob("*vocab.txt"):
            shutil.copy(vocab, out_dir / "vocab.txt")

        ckpt = next(tmp.glob("*model_weights.ckpt"), None) or (
            tmp / "model_weights.ckpt"
        )
        import torch

        print("[INFO] Loading checkpoint…")
        state_dict = torch.load(ckpt, map_location="cpu", weights_only=False)
        if "state_dict" in state_dict and not hasattr(
            next(iter(state_dict.values())), "shape"
        ):
            state_dict = state_dict["state_dict"]

        print(f"[INFO] Converting {len(state_dict)} tensors → MLX ({dtype})")
        weights = _convert_weights(state_dict, mlx_dtype)

    if quantize:
        print(f"[INFO] Quantizing to {q_bits}-bit (group_size={q_group_size})")
        config, weights = _quantize(config, weights, q_bits, q_group_size)

    (out_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2)
    )
    print(f"[INFO] Wrote config.json ({len(config['vocabulary'])} vocab tokens)")
    mx.save_safetensors(str(out_dir / "model.safetensors"), weights)

    print(f"[INFO] Done. MLX model at: {out_dir}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Nemotron .nemo ASR to MLX")
    parser.add_argument(
        "--nemo-path",
        required=True,
        help="Local .nemo file or HuggingFace repo id (e.g. nvidia/nemotron-3.5-asr-streaming-0.6b)",
    )
    parser.add_argument("--mlx-path", required=True, help="Output directory")
    parser.add_argument(
        "--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"]
    )
    parser.add_argument(
        "-q",
        "--quantize",
        action="store_true",
        help="Quantize Linear/Embedding weights",
    )
    parser.add_argument("--q-bits", type=int, default=4, help="Quantization bits")
    parser.add_argument(
        "--q-group-size", type=int, default=64, help="Quantization group size"
    )
    args = parser.parse_args()
    convert(
        args.nemo_path,
        args.mlx_path,
        dtype=args.dtype,
        quantize=args.quantize,
        q_bits=args.q_bits,
        q_group_size=args.q_group_size,
    )


if __name__ == "__main__":
    main()
