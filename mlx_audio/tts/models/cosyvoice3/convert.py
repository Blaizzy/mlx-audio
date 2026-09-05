"""PyTorch -> MLX weight conversion for CosyVoice3.

CosyVoice3 ships three PyTorch checkpoints (no safetensors):
    llm.pt   flow.pt   hift.pt
plus a HF-format Qwen2 directory (the LLM backbone) referenced by qwen_pretrain_path.

This module loads those .pt files and remaps keys/shapes to the MLX param trees
in llm.py / flow.py / hift.py. It follows the idempotent shape-checking pattern
used across mlx-audio: only transpose when the shape actually needs it, so the
same sanitize works on raw-PyTorch and already-converted MLX weights.

Conv layout rules (PyTorch -> MLX):
  * Conv1d:          (out, in, k)      -> (out, k, in)          swapaxes(1, 2)
  * ConvTranspose1d: (in, out, k)      -> (out, k, in)          transpose(1, 2, 0)
  * weight_norm:     merge weight_g / weight_v (or parametrizations.weight.
                     original0/1) -> w = g * v / ||v||_(dim!=0)
  * drop *.num_batches_tracked.

Status: verified against the real checkpoint (see README) — convert_llm_weights
/convert_flow_weights/convert_hift_weights are the per-module key remaps used
by each sub-module's ``sanitize``; ``convert_cosyvoice3_assets`` below drives
the end-to-end conversion (raw checkpoint dir -> mlx-audio model dir).

Usage:
    python -m mlx_audio.tts.models.cosyvoice3.convert \
        --torch-dir /path/to/CosyVoice3 --out /path/to/mlx-cosyvoice3
"""

import re
from pathlib import Path
from typing import Dict, Union

import mlx.core as mx


# --------------------------------------------------------------------------- #
# mechanical transforms
# --------------------------------------------------------------------------- #
def conv1d_to_mlx(w: mx.array) -> mx.array:
    """(out, in, k) -> (out, k, in). No-op if already channels-last."""
    if w.ndim != 3:
        return w
    out, a, b = w.shape
    # PyTorch conv1d weight is (out, in, k); MLX wants (out, k, in).
    return mx.swapaxes(w, 1, 2)


def conv_transpose1d_to_mlx(w: mx.array) -> mx.array:
    """(in, out, k) -> (out, k, in)."""
    if w.ndim != 3:
        return w
    return mx.transpose(w, (1, 2, 0))


def fold_weight_norm(g: mx.array, v: mx.array) -> mx.array:
    """w = g * v / ||v|| computed over all dims except output channel (dim 0)."""
    axes = tuple(range(1, v.ndim))
    norm = mx.sqrt(mx.sum(v.astype(mx.float32) ** 2, axis=axes, keepdims=True))
    return (g.astype(mx.float32) * v.astype(mx.float32) / (norm + 1e-12)).astype(v.dtype)


def _merge_weight_norm(state: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """Collapse weight_norm parametrizations into plain .weight tensors."""
    out: Dict[str, mx.array] = {}
    consumed = set()
    for k in list(state.keys()):
        # new-style: <p>.parametrizations.weight.original0 (g) / original1 (v)
        if k.endswith(".parametrizations.weight.original0"):
            base = k[: -len(".parametrizations.weight.original0")]
            g = state[k]
            v = state.get(base + ".parametrizations.weight.original1")
            if v is not None:
                out[base + ".weight"] = fold_weight_norm(g, v)
                consumed.add(k)
                consumed.add(base + ".parametrizations.weight.original1")
        # old-style: <p>.weight_g / <p>.weight_v
        elif k.endswith(".weight_g"):
            base = k[: -len(".weight_g")]
            g = state[k]
            v = state.get(base + ".weight_v")
            if v is not None:
                out[base + ".weight"] = fold_weight_norm(g, v)
                consumed.add(k)
                consumed.add(base + ".weight_v")
    for k, val in state.items():
        if k in consumed or k.endswith(".num_batches_tracked"):
            continue
        if k not in out:
            out[k] = val
    return out


# --------------------------------------------------------------------------- #
# per-component conversion (key remap scaffolding)
# --------------------------------------------------------------------------- #
def convert_llm_weights(
    torch_state: Dict[str, mx.array], tie_word_embeddings: bool = True
) -> Dict[str, mx.array]:
    """llm.pt -> CosyVoice3LM tree.

    Reference keys (PyTorch): llm.model.model.* (Qwen2), llm.model.lm_head.*,
    llm_decoder.weight, speech_embedding.weight. Map llm.model.model.* ->
    llm.model.* (mlx_lm Qwen2).

    ``llm.model.lm_head.weight`` is dropped when ``tie_word_embeddings`` (the
    v3 default): it is then bit-for-bit identical to embed_tokens.weight
    (verified against a real checkpoint), and mlx_lm's Qwen2 only allocates a
    separate ``lm_head`` parameter when weights are *not* tied. When untied,
    it is kept and remapped to ``llm.lm_head.weight``.
    """
    out: Dict[str, mx.array] = {}
    for k, v in torch_state.items():
        if k == "llm.model.lm_head.weight":
            if tie_word_embeddings:
                continue
            out["llm.lm_head.weight"] = v
            continue
        nk = k
        if k.startswith("llm.model.model."):
            nk = "llm.model." + k[len("llm.model.model.") :]
        out[nk] = v
    return out


def _remap_dit_sequential_indices(key: str) -> str:
    """Collapse PyTorch nn.Sequential indices that skip non-parametric layers.

    The reference DiT stores a few blocks as ``nn.Sequential`` with an
    activation/dropout interleaved between the parametric layers, e.g.
    ``FeedForward.ff = Sequential(Sequential(Linear, GELU), Dropout, Linear)``.
    This module's MLX tree stores only the parametric layers in a plain list,
    so ``ff.ff.0.0.*`` -> ``ff.ff.0.*`` and ``ff.ff.2.*`` -> ``ff.ff.1.*``.
    Likewise ``conv_pos_embed.conv{1,2}.0.*`` -> ``conv_pos_embed.conv{1,2}.*``
    (Sequential(Conv1d, Mish)) and ``time_mlp.2.*`` -> ``time_mlp.1.*``
    (Sequential(Linear, SiLU, Linear)).
    """
    key = re.sub(r"(\.ff\.ff\.)0\.0\.", r"\g<1>0.", key)
    key = re.sub(r"(\.ff\.ff\.)2\.", r"\g<1>1.", key)
    key = re.sub(r"(\.conv_pos_embed\.conv[12]\.)0\.", r"\1", key)
    key = re.sub(r"(\.time_mlp\.)2\.", r"\g<1>1.", key)
    return key


def convert_flow_weights(torch_state: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """flow.pt -> CausalMaskedDiffWithDiT tree (incl. DiT estimator).

    Transpose Conv1d weights (input_embedding is an Embedding, left as-is;
    pre_lookahead conv1/conv2 and DiT conv layers need conv1d_to_mlx) and
    collapse the DiT's nn.Sequential index gaps (see
    ``_remap_dit_sequential_indices``).

    ``rotary_embed.inv_freq`` is dropped: it is a non-persistent buffer in the
    reference RotaryEmbedding (recomputed from theta/dim, not learned), and
    this module's ``RotaryEmbedding`` is a plain Python helper (not an
    ``nn.Module``) that recomputes the identical values on demand — verified
    against a real checkpoint to match bit-for-bit.
    """
    out: Dict[str, mx.array] = {}
    for k, v in torch_state.items():
        if k.endswith("rotary_embed.inv_freq"):
            continue
        nk = _remap_dit_sequential_indices(k)
        if v.ndim == 3 and (".conv" in nk or "dwconv" in nk or "pre_lookahead" in nk):
            v = conv1d_to_mlx(v)
        out[nk] = v
    return out


def convert_hift_weights(torch_state: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """hift.pt -> CausalHiFTGenerator tree. Fold weight_norm; transpose convs.

    Every conv in the causal generator (including ``ups.*``) is a plain
    ``torch.nn.Conv1d`` subclass — ``CausalConv1dUpsample`` upsamples via
    nearest-neighbor + a regular conv rather than ``ConvTranspose1d`` — so all
    ndim==3 weights use the same (out, in, k) -> (out, k, in) transpose.
    ``f0_predictor.condnet.{0,2,4,6,8}`` (PyTorch nn.Sequential interleaving
    ELU at the odd indices) collapse to this module's flat ``condnet`` list
    indices ``{0,1,2,3,4}``.
    """
    state = _merge_weight_norm(torch_state)
    out: Dict[str, mx.array] = {}
    for k, v in state.items():
        nk = re.sub(
            r"f0_predictor\.condnet\.([02468])\.",
            lambda m: f"f0_predictor.condnet.{int(m.group(1)) // 2}.",
            k,
        )
        if v.ndim == 3:
            v = conv1d_to_mlx(v)
        out[nk] = v
    return out


# --------------------------------------------------------------------------- #
# end-to-end asset conversion (mirrors codec/models/stepaudio2/convert.py)
# --------------------------------------------------------------------------- #
def load_torch_state(path: Union[str, Path]) -> Dict[str, mx.array]:
    import torch

    state = torch.load(str(path), map_location="cpu", weights_only=True)
    return {k: mx.array(v.detach().cpu().float().numpy()) for k, v in state.items()}


def convert_cosyvoice3_assets(
    input_dir: Union[str, Path], output_dir: Union[str, Path]
) -> None:
    """Convert a raw CosyVoice3 checkpoint dir into the mlx-audio layout.

    ``input_dir`` is the real checkpoint directory shipped by ModelScope/HF
    (``llm.pt``, ``flow.pt``, ``hift.pt``, ``CosyVoice-BlankEN/`` HF Qwen2
    dir, ``campplus.onnx``, ``speech_tokenizer_v3.onnx``). Produces, flat in
    ``output_dir`` (required by ``AutoTokenizer.from_pretrained`` and
    ``CosyVoice3FrontEnd._first_existing_optional``, neither of which search
    nested paths): ``config.json``, ``model.safetensors``, the Qwen2
    tokenizer files, and the onnx assets.
    """
    import json
    import shutil

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    weights: Dict[str, mx.array] = {}
    for prefix, filename in (("llm", "llm.pt"), ("flow", "flow.pt"), ("hift", "hift.pt")):
        state = load_torch_state(input_dir / filename)
        for k, v in state.items():
            weights[f"{prefix}.{k}"] = v

    # Save raw (unsanitized) weights: base_load_model calls model.sanitize()
    # itself at load time, and convert_flow_weights's Conv1d transpose is not
    # idempotent — sanitizing here too would double-transpose on load.
    mx.save_safetensors(str(output_dir / "model.safetensors"), weights)

    config = {"model_type": "cosyvoice3", "use_onnx_speech_tokenizer": True}
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    qwen_dir = input_dir / "CosyVoice-BlankEN"
    for name in ("tokenizer_config.json", "vocab.json", "merges.txt"):
        src = qwen_dir / name
        if src.exists():
            shutil.copyfile(src, output_dir / name)

    for name in ("campplus.onnx", "speech_tokenizer_v3.onnx"):
        src = input_dir / name
        if src.exists():
            shutil.copyfile(src, output_dir / name)


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torch-dir", required=True, help="raw checkpoint directory")
    parser.add_argument("--out", required=True, help="output mlx-audio model directory")
    args = parser.parse_args()
    convert_cosyvoice3_assets(args.torch_dir, args.out)


if __name__ == "__main__":
    main()
