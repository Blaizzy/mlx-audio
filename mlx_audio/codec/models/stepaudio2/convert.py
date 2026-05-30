import re
import shutil
from pathlib import Path
from typing import Dict

import mlx.core as mx
from mlx.utils import tree_flatten


def materialize_weight_norm(g: mx.array, v: mx.array) -> mx.array:
    axes = tuple(range(1, v.ndim))
    v_norm = mx.sqrt(mx.sum(v * v, axis=axes, keepdims=True))
    return g * v / (v_norm + 1e-12)


def _merge_weight_norm(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    pairs: dict[str, dict[str, mx.array]] = {}
    merged = {}
    for key, value in weights.items():
        if ".parametrizations.weight.original0" in key:
            base = key.replace(".parametrizations.weight.original0", ".weight")
            pairs.setdefault(base, {})["g"] = value
        elif ".parametrizations.weight.original1" in key:
            base = key.replace(".parametrizations.weight.original1", ".weight")
            pairs.setdefault(base, {})["v"] = value
        else:
            merged[key] = value
    for key, pair in pairs.items():
        if "g" in pair and "v" in pair:
            merged[key] = materialize_weight_norm(pair["g"], pair["v"])
    return merged


def load_torch_weights(path: str | Path) -> Dict[str, mx.array]:
    import torch

    state = torch.load(path, map_location="cpu", weights_only=True)
    return {k: mx.array(v.detach().cpu().numpy()) for k, v in state.items()}


def sanitize_flow_weights(model, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    curr = dict(tree_flatten(model.parameters()))
    sanitized = {}
    for key, value in _merge_weight_norm(weights).items():
        if "num_batches_tracked" in key:
            continue

        new_key = key
        new_key = new_key.replace(".embed.out.0.", ".embed.linear.")
        new_key = new_key.replace(".embed.out.1.", ".embed.norm.")
        new_key = new_key.replace(".up_embed.out.0.", ".up_embed.linear.")
        new_key = new_key.replace(".up_embed.out.1.", ".up_embed.norm.")

        if "weight" in new_key and value.ndim == 3:
            if new_key in curr and value.shape != curr[new_key].shape:
                value = mx.swapaxes(value, 1, 2)

        if new_key in curr:
            sanitized[new_key] = value
    for key in (
        "decoder.rand_noise",
        "encoder.embed.pos_enc.pe",
        "encoder.up_embed.pos_enc.pe",
    ):
        if key in curr and key not in sanitized:
            sanitized[key] = curr[key]
    return sanitized


def sanitize_hift_weights(model, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    curr = dict(tree_flatten(model.parameters()))
    sanitized = {}
    for key, value in _merge_weight_norm(weights).items():
        if "num_batches_tracked" in key:
            continue
        new_key = key.removeprefix("generator.")
        new_key = re.sub(
            r"f0_predictor\.condnet\.([02468])\.",
            lambda m: f"f0_predictor.condnet.{int(m.group(1)) // 2}.",
            new_key,
        )

        if "weight" in new_key and value.ndim == 3:
            if new_key in curr and value.shape != curr[new_key].shape:
                if new_key.startswith("ups."):
                    value = mx.transpose(value, (1, 2, 0))
                else:
                    value = mx.swapaxes(value, 1, 2)

        if new_key in curr:
            sanitized[new_key] = value
    if "stft_window" in curr and "stft_window" not in sanitized:
        sanitized["stft_window"] = curr["stft_window"]
    return sanitized


def _xvector_block_layer(key: str) -> str:
    match = re.match(r"xvector\.block(\d+)\.tdnnd(\d+)\.(.+)", key)
    if match is None:
        return key
    block = int(match.group(1)) - 1
    layer = int(match.group(2)) - 1
    return f"blocks.{block}.layers.{layer}.{match.group(3)}"


def _map_stepaudio2_campplus_key(key: str) -> str:
    key = _xvector_block_layer(key)
    key = re.sub(
        r"xvector\.transit(\d+)\.",
        lambda m: f"transits.{int(m.group(1)) - 1}.",
        key,
    )
    key = key.replace("xvector.tdnn.", "tdnn.")
    key = key.replace("xvector.dense.", "dense.")
    key = key.replace(".nonlinear1.batchnorm.", ".nonlinear1.0.")
    key = key.replace(".nonlinear.batchnorm.", ".nonlinear.0.")
    return key


def _conv_node_to_key(node_name: str) -> str | None:
    name = node_name.strip("/")
    if not name.endswith("/Conv"):
        return None
    parts = name[: -len("/Conv")].split("/")
    if parts[0] == "head":
        if parts[1] in {"conv1", "conv2"}:
            return f"head.{parts[1]}"
        if parts[1] in {"layer1", "layer2"}:
            layer = parts[2].split(".")[-1]
            module = parts[3]
            if module == "shortcut":
                return f"head.{parts[1]}.{layer}.shortcut.0"
            return f"head.{parts[1]}.{layer}.{module}"
    if parts[0] == "xvector":
        if parts[1] == "tdnn" and parts[2] == "linear":
            return "tdnn.linear"
        if parts[1].startswith("block"):
            block = int(parts[1].replace("block", "")) - 1
            layer = int(parts[2].replace("tdnnd", "")) - 1
            rest = ".".join(parts[3:])
            return f"blocks.{block}.layers.{layer}.{rest}"
        if parts[1].startswith("transit") and parts[2] == "linear":
            transit = int(parts[1].replace("transit", "")) - 1
            return f"transits.{transit}.linear"
        if parts[1] == "dense" and parts[2] == "linear":
            return "dense.linear"
    return None


def _bn_node_to_key(node_name: str) -> str | None:
    name = node_name.strip("/")
    if not name.endswith("/BatchNormalization"):
        return None
    parts = name[: -len("/BatchNormalization")].split("/")
    if parts[0] != "xvector":
        return None
    if parts[1].startswith("block"):
        block = int(parts[1].replace("block", "")) - 1
        layer = int(parts[2].replace("tdnnd", "")) - 1
        return f"blocks.{block}.layers.{layer}.nonlinear1.0"
    if parts[1].startswith("transit"):
        transit = int(parts[1].replace("transit", "")) - 1
        return f"transits.{transit}.nonlinear.0"
    if parts[1] == "dense":
        return "dense.nonlinear.0"
    return None


def sanitize_campplus_onnx_weights(model, onnx_path: str | Path) -> Dict[str, mx.array]:
    import onnx
    from onnx import numpy_helper

    graph = onnx.load(str(onnx_path)).graph
    raw = {i.name: mx.array(numpy_helper.to_array(i)) for i in graph.initializer}
    curr = dict(tree_flatten(model.parameters()))
    sanitized = {}

    for node in graph.node:
        if node.op_type == "Conv":
            base = _conv_node_to_key(node.name)
            if base is None:
                continue
            values = [raw[name] for name in node.input[1:] if name in raw]
            if not values:
                continue
            weight = values[0]
            key = f"{base}.weight"
            if key in curr and weight.shape != curr[key].shape:
                if weight.ndim == 4:
                    weight = mx.transpose(weight, (0, 2, 3, 1))
                elif weight.ndim == 3:
                    weight = mx.swapaxes(weight, 1, 2)
            if key in curr:
                sanitized[key] = weight
            bias_key = f"{base}.bias"
            if len(values) > 1 and bias_key in curr:
                sanitized[bias_key] = values[1]

        elif node.op_type == "BatchNormalization":
            base = _bn_node_to_key(node.name)
            if base is None:
                continue
            for input_name, suffix in zip(
                node.input[1:], ("weight", "bias", "running_mean", "running_var")
            ):
                key = f"{base}.{suffix}"
                if input_name in raw and key in curr:
                    sanitized[key] = raw[input_name]

    for key, value in raw.items():
        mapped = _map_stepaudio2_campplus_key(key)
        if mapped in curr and mapped not in sanitized:
            if "weight" in mapped and value.shape != curr[mapped].shape:
                if value.ndim == 4:
                    value = mx.transpose(value, (0, 2, 3, 1))
                elif value.ndim == 3:
                    value = mx.swapaxes(value, 1, 2)
            sanitized[mapped] = value

    missing = [key for key in curr if key not in sanitized]
    if missing:
        raise ValueError(
            "Missing CAMPPlus parameters after ONNX conversion: " f"{missing[:10]}"
        )
    return sanitized


def load_flow_weights(model, path: str | Path, strict: bool = True) -> None:
    path = Path(path)
    if path.suffix == ".safetensors":
        weights = mx.load(str(path), format="safetensors")
    else:
        weights = sanitize_flow_weights(model, load_torch_weights(path))
    model.load_weights(list(weights.items()), strict=strict)


def load_hift_weights(model, path: str | Path, strict: bool = True) -> None:
    path = Path(path)
    if path.suffix == ".safetensors":
        weights = mx.load(str(path), format="safetensors")
    else:
        weights = sanitize_hift_weights(model, load_torch_weights(path))
    model.load_weights(list(weights.items()), strict=strict)


def load_campplus_weights(model, path: str | Path, strict: bool = True) -> None:
    path = Path(path)
    if path.suffix == ".safetensors":
        weights = mx.load(str(path), format="safetensors")
    elif path.suffix == ".onnx":
        weights = sanitize_campplus_onnx_weights(model, path)
    else:
        raise ValueError(f"Unsupported CAMPPlus weight format: {path}")
    model.load_weights(list(weights.items()), strict=strict)


def convert_stepaudio2_assets(input_dir: str | Path, output_dir: str | Path) -> None:
    from .flow import CausalMaskedDiffWithXvec
    from .hift import StepAudio2HiFTGenerator

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (input_dir / "flow.yaml").exists():
        shutil.copyfile(input_dir / "flow.yaml", output_dir / "flow.yaml")

    flow = CausalMaskedDiffWithXvec()
    flow_weights = sanitize_flow_weights(
        flow, load_torch_weights(input_dir / "flow.pt")
    )
    mx.save_safetensors(str(output_dir / "flow.safetensors"), flow_weights)

    hift = StepAudio2HiFTGenerator()
    hift_weights = sanitize_hift_weights(
        hift, load_torch_weights(input_dir / "hift.pt")
    )
    mx.save_safetensors(str(output_dir / "hift.safetensors"), hift_weights)

    from .speaker import StepAudio2CAMPPlus

    speaker = StepAudio2CAMPPlus()
    speaker_weights = sanitize_campplus_onnx_weights(
        speaker, input_dir / "campplus.onnx"
    )
    mx.save_safetensors(str(output_dir / "campplus.safetensors"), speaker_weights)
