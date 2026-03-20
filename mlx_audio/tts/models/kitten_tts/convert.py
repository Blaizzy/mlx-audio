import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import mlx.core as mx
import numpy as np
import onnx
from huggingface_hub import hf_hub_download, snapshot_download
from mlx.utils import tree_flatten
from mlx_lm.utils import save_config, save_model
from onnx import helper, numpy_helper

from .kitten_tts import Model, ModelConfig


def _load_onnx(
    repo_id: str, token: str | None, revision: str | None
) -> Tuple[Path, dict]:
    snapshot_dir = snapshot_download(
        repo_id,
        revision=revision,
        allow_patterns=["*.onnx", "*.json", "*.npz", "README.md"],
        token=token,
    )
    snapshot_dir = Path(snapshot_dir)
    config_path = snapshot_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {snapshot_dir}")
    config = json.loads(config_path.read_text())
    model_file = config.get("model_file", "kitten_tts_nano_v0_8.onnx")
    onnx_path = snapshot_dir / model_file
    if not onnx_path.exists():
        onnx_path = Path(
            hf_hub_download(repo_id, model_file, token=token, revision=revision)
        )
    return onnx_path, config


def _init_map(model: onnx.ModelProto) -> Dict[str, np.ndarray]:
    return {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}


def _dequantize(init: Dict[str, np.ndarray], name: str) -> np.ndarray:
    arr = init[name]
    if not name.endswith("_quantized"):
        return arr
    scale_name = name.replace("_quantized", "_scale")
    zp_name = name.replace("_quantized", "_zero_point")
    scale = init.get(scale_name)
    zp = init.get(zp_name)
    if scale is None or zp is None:
        return arr
    scale = scale.astype(np.float32)
    zp = zp.astype(np.float32)
    arr_f = arr.astype(np.float32)
    if scale.ndim == 0:
        return (arr_f - zp) * scale
    if scale.ndim == 1 and arr_f.shape[0] == scale.shape[0]:
        shape = (scale.shape[0],) + (1,) * (arr_f.ndim - 1)
        scale = scale.reshape(shape)
        zp = zp.reshape(shape)
        return (arr_f - zp) * scale
    return (arr_f - zp) * scale


def _dequantize_onnx(model: onnx.ModelProto) -> onnx.ModelProto:
    """Replace quantized MatMulInteger/ConvInteger subgraphs with float ops.

    This folds weight dequantization into float initializers and removes
    the Cast/Mul scale chain. Activation quantization (DynamicQuantizeLinear)
    is bypassed by wiring the float input directly into the new op.
    """
    init_map = _init_map(model)
    init_tensors = {i.name: i for i in model.graph.initializer}

    node_by_output: Dict[str, onnx.NodeProto] = {}
    consumers: Dict[str, list[onnx.NodeProto]] = {}
    for node in model.graph.node:
        for out in node.output:
            node_by_output[out] = node
        for inp in node.input:
            consumers.setdefault(inp, []).append(node)

    remove_nodes: set[str] = set()
    remove_inits: set[str] = set()
    added_inits: Dict[str, onnx.TensorProto] = {}
    replacements: Dict[str, onnx.NodeProto] = {}

    def _float_weight_name(qname: str) -> str:
        return qname.replace("_quantized", "")

    def _ensure_dequant_weight(qname: str) -> str:
        base = _float_weight_name(qname)
        if base in init_tensors or base in added_inits:
            return base
        w = _dequantize(init_map, qname).astype(np.float32)
        added_inits[base] = numpy_helper.from_array(w, name=base)
        remove_inits.add(qname)
        remove_inits.add(qname.replace("_quantized", "_scale"))
        remove_inits.add(qname.replace("_quantized", "_zero_point"))
        return base

    def _get_single_consumer(output: str, op_type: str) -> onnx.NodeProto | None:
        for node in consumers.get(output, []):
            if node.op_type == op_type:
                return node
        return None

    for node in model.graph.node:
        if node.op_type not in {"MatMulInteger", "ConvInteger"}:
            continue

        if not node.input:
            continue

        dql = node_by_output.get(node.input[0])
        if dql is None or dql.op_type != "DynamicQuantizeLinear":
            continue

        cast = _get_single_consumer(node.output[0], "Cast")
        if cast is None or not cast.output:
            continue
        mul = _get_single_consumer(cast.output[0], "Mul")
        if mul is None or not mul.output:
            continue

        input_float = dql.input[0]
        weight_q = node.input[1]
        weight_f = _ensure_dequant_weight(weight_q)

        out_name = mul.output[0]

        if node.op_type == "MatMulInteger":
            new_node = helper.make_node(
                "MatMul",
                inputs=[input_float, weight_f],
                outputs=[out_name],
                name=node.name.replace("_quant", "_dequant"),
            )
        else:
            # ConvInteger attrs match Conv
            new_node = helper.make_node(
                "Conv",
                inputs=[input_float, weight_f],
                outputs=[out_name],
                name=node.name.replace("_quant", "_dequant"),
                **{a.name: helper.get_attribute_value(a) for a in node.attribute},
            )

        replacements[node.name] = new_node
        remove_nodes.update({node.name, cast.name, mul.name})

    new_nodes: list[onnx.NodeProto] = []
    for node in model.graph.node:
        if node.name in replacements:
            new_nodes.append(replacements[node.name])
            continue
        if node.name in remove_nodes:
            continue
        new_nodes.append(node)

    new_inits: list[onnx.TensorProto] = []
    for init in model.graph.initializer:
        if init.name in remove_inits:
            continue
        new_inits.append(init)
    new_inits.extend(added_inits.values())

    model.graph.ClearField("node")
    model.graph.node.extend(new_nodes)
    model.graph.ClearField("initializer")
    model.graph.initializer.extend(new_inits)
    return model


def _get_init(init: Dict[str, np.ndarray], name: str) -> np.ndarray:
    if name in init:
        return init[name]
    qname = f"{name}_quantized"
    if qname in init:
        return init[qname]
    raise KeyError(name)


def _collect_constants(model: onnx.ModelProto) -> set[int]:
    constants = set()
    for init in model.graph.initializer:
        arr = numpy_helper.to_array(init)
        if arr.size == 1 and np.issubdtype(arr.dtype, np.integer):
            constants.add(int(arr.item()))
    for node in model.graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    arr = numpy_helper.to_array(attr.t)
                    if arr.size == 1 and np.issubdtype(arr.dtype, np.integer):
                        constants.add(int(arr.item()))
    return constants


def _infer_num_hidden_layers(model: onnx.ModelProto) -> int:
    layer_ids = set()
    for node in model.graph.node:
        m = re.search(r"attention/query_(\d+)", node.name)
        if m:
            layer_ids.add(int(m.group(1)))
        elif "attention/query/" in node.name:
            layer_ids.add(0)
    if layer_ids:
        return max(layer_ids) + 1
    # Fallback: count query nodes (Add + MatMul per layer)
    query_nodes = [n for n in model.graph.node if "attention/query" in n.name]
    return max(1, len(query_nodes) // 2)


def _infer_attention_heads(hidden_size: int, constants: set[int]) -> int:
    for candidate in (64, 32, 16, 8):
        if hidden_size % candidate == 0 and candidate in constants:
            return hidden_size // candidate
    # Fallback to common ALBERT base
    for candidate in (12, 8, 6):
        if hidden_size % candidate == 0:
            return candidate
    return 1


def _infer_config(model: onnx.ModelProto, base_config: dict) -> dict:
    init = _init_map(model)

    word_emb = _get_init(init, "kmodel.bert.embeddings.word_embeddings.weight")
    n_token, embedding_size = word_emb.shape
    max_position_embeddings = _get_init(
        init, "kmodel.bert.embeddings.position_embeddings.weight"
    ).shape[0]

    hidden_size = init[
        "kmodel.bert.encoder.albert_layer_groups.0.albert_layers.0.attention.query.bias"
    ].shape[0]
    intermediate_size = init[
        "kmodel.bert.encoder.albert_layer_groups.0.albert_layers.0.ffn.bias"
    ].shape[0]

    constants = _collect_constants(model)
    num_hidden_layers = _infer_num_hidden_layers(model)
    num_attention_heads = _infer_attention_heads(hidden_size, constants)

    hidden_dim = init["kmodel.bert_encoder.bias"].shape[0]
    max_dur = init["kmodel.predictor.duration_proj.linear_layer.bias"].shape[0]
    max_conv_dim = _get_init(init, "kmodel.decoder.encode.conv1.weight").shape[0]
    asr_res_dim = _get_init(init, "kmodel.decoder.asr_res.0.weight").shape[0]
    text_encoder_kernel_size = _get_init(
        init, "kmodel.text_encoder.cnn.0.0.weight"
    ).shape[-1]

    # AdaIN fc weight gives style_dim (input dim), but ONNX may store transposed.
    style_dim = None
    for name, arr in init.items():
        if name.endswith(".norm1.fc.weight") and arr.ndim == 2:
            style_dim = int(min(arr.shape))
            break
    if style_dim is None:
        style_dim = 128

    # Generator params
    conv_post_w = _get_init(init, "kmodel.decoder.generator.conv_post.weight")
    gen_istft_n_fft = conv_post_w.shape[0] - 2

    gen_istft_hop_size = 5
    for node in model.graph.node:
        if (
            "stft.weight_forward_real" in " ".join(node.input)
            and node.op_type == "Conv"
        ):
            for attr in node.attribute:
                if attr.name == "strides":
                    gen_istft_hop_size = int(onnx.helper.get_attribute_value(attr)[0])
            break

    # Upsample params
    upsample_nodes = []
    for node in model.graph.node:
        if "/decoder/generator/ups." in node.name and node.op_type == "ConvTranspose":
            m = re.search(r"ups\.(\d+)", node.name)
            if m:
                idx = int(m.group(1))
                attrs = {
                    a.name: onnx.helper.get_attribute_value(a) for a in node.attribute
                }
                upsample_nodes.append(
                    (idx, int(attrs["strides"][0]), int(attrs["kernel_shape"][0]))
                )
    upsample_nodes.sort()
    upsample_rates = [u for _, u, _ in upsample_nodes]
    upsample_kernel_sizes = [k for _, _, k in upsample_nodes]

    upsample_initial_channel = _get_init(
        init, "kmodel.decoder.generator.ups.0.weight"
    ).shape[0]

    # Resblock params
    resblock_nodes = []
    for node in model.graph.node:
        m = re.search(r"resblocks\.(\d+)/convs1\.0/Conv", node.name)
        if m:
            resblock_nodes.append((int(m.group(1)), node))
    resblock_nodes.sort(key=lambda x: x[0])

    resblock_kernel_sizes = []
    if resblock_nodes:
        num_resblocks = len(resblock_nodes)
        num_upsamples = max(1, len(upsample_rates))
        num_kernels = max(1, num_resblocks // num_upsamples)
        for i in range(min(num_kernels, len(resblock_nodes))):
            _, node = resblock_nodes[i]
            attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
            if "kernel_shape" in attrs:
                resblock_kernel_sizes.append(int(attrs["kernel_shape"][0]))
    else:
        # Quantized graphs may not have Conv nodes; infer from weight shapes instead.
        resblock_weights = []
        for name in init:
            m = re.match(
                r"kmodel\.decoder\.generator\.resblocks\.(\d+)\.convs1\.0\.weight(_quantized)?$",
                name,
            )
            if m:
                resblock_weights.append((int(m.group(1)), name))
        resblock_weights.sort(key=lambda x: x[0])
        num_resblocks = len(resblock_weights)
        num_upsamples = max(1, len(upsample_rates))
        num_kernels = max(1, num_resblocks // num_upsamples)
        for i in range(min(num_kernels, len(resblock_weights))):
            _, wname = resblock_weights[i]
            w = init[wname]
            resblock_kernel_sizes.append(int(w.shape[-1]))

    # Dilations (fallback if not present)
    dilations = []
    for node in model.graph.node:
        if "resblocks.0/convs1" in node.name and node.op_type == "Conv":
            attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
            dilations.append(int(attrs.get("dilations", [1])[0]))
    dilations = dilations or [1, 3, 5]
    resblock_dilation_sizes = [dilations for _ in resblock_kernel_sizes]

    config = {
        "model_type": "kitten_tts",
        "hidden_dim": int(hidden_dim),
        "max_conv_dim": int(max_conv_dim),
        "max_dur": int(max_dur),
        "n_layer": int(
            len(
                {
                    int(m.group(1))
                    for n in init
                    if (m := re.search(r"kmodel\.text_encoder\.cnn\.(\d+)\.", n))
                }
            )
        ),
        "n_mels": 80,
        "n_token": int(n_token),
        "style_dim": int(style_dim),
        "text_encoder_kernel_size": int(text_encoder_kernel_size),
        "asr_res_dim": int(asr_res_dim),
        "decoder_out_dim": int(upsample_initial_channel),
        "sample_rate": int(base_config.get("sample_rate", 24000)),
        "voices_path": base_config.get("voices", "voices.npz"),
        "speed_priors": base_config.get("speed_priors", {}),
        "voice_aliases": base_config.get("voice_aliases", {}),
        "plbert": {
            "num_hidden_layers": int(num_hidden_layers),
            "num_attention_heads": int(num_attention_heads),
            "hidden_size": int(hidden_size),
            "intermediate_size": int(intermediate_size),
            "max_position_embeddings": int(max_position_embeddings),
            "embedding_size": int(embedding_size),
            "inner_group_num": 1,
            "num_hidden_groups": 1,
            "hidden_dropout_prob": 0.0,
            "attention_probs_dropout_prob": 0.0,
            "type_vocab_size": 2,
            "layer_norm_eps": 1e-12,
        },
        "istftnet": {
            "resblock_kernel_sizes": resblock_kernel_sizes,
            "upsample_rates": upsample_rates,
            "upsample_initial_channel": int(upsample_initial_channel),
            "resblock_dilation_sizes": resblock_dilation_sizes,
            "upsample_kernel_sizes": upsample_kernel_sizes,
            "gen_istft_n_fft": int(gen_istft_n_fft),
            "gen_istft_hop_size": int(gen_istft_hop_size),
        },
    }

    return config


def _infer_activation_quant_modules(model: onnx.ModelProto) -> List[str]:
    init = _init_map(model)
    modules: set[str] = set()

    for name in init:
        if not name.endswith("_quantized"):
            continue
        base = name.replace("_quantized", "")
        if not base.startswith("kmodel."):
            continue
        target = base.replace("kmodel.", "", 1)
        target = target.replace(".gamma", ".weight").replace(".beta", ".bias")
        if target.endswith((".weight", ".bias", ".weight_v", ".weight_g")):
            modules.add(target.rsplit(".", 1)[0])

    for node in model.graph.node:
        if node.op_type == "MatMulInteger":
            target = _map_matmul_target(node.name)
            if target:
                modules.add(target.rsplit(".", 1)[0])

    lstm_prefixes_quant = {
        "/text_encoder/lstm/LSTM_quant": "text_encoder.lstm",
        "/text_encoder/lstms.0/LSTM_quant": "predictor.text_encoder.lstms.0",
        "/text_encoder/lstms.2/LSTM_quant": "predictor.text_encoder.lstms.2",
        "/text_encoder/lstms.4/LSTM_quant": "predictor.text_encoder.lstms.4",
        "/lstm/LSTM_quant": "predictor.lstm",
        "/shared/LSTM_quant": "predictor.shared",
    }
    for node in model.graph.node:
        if node.op_type == "DynamicQuantizeLSTM" and node.name in lstm_prefixes_quant:
            modules.add(lstm_prefixes_quant[node.name])

    return sorted(modules)


def _map_matmul_target(node_name: str) -> str | None:
    if "/bert/encoder/embedding_hidden_mapping_in/MatMul" in node_name:
        return "bert.encoder.embedding_hidden_mapping_in.weight"
    if "/attention/query" in node_name:
        return (
            "bert.encoder.albert_layer_groups.0.albert_layers.0.attention.query.weight"
        )
    if "/attention/key" in node_name:
        return "bert.encoder.albert_layer_groups.0.albert_layers.0.attention.key.weight"
    if "/attention/value" in node_name:
        return (
            "bert.encoder.albert_layer_groups.0.albert_layers.0.attention.value.weight"
        )
    if "/attention/dense" in node_name:
        return (
            "bert.encoder.albert_layer_groups.0.albert_layers.0.attention.dense.weight"
        )
    if "/ffn_output" in node_name:
        return "bert.encoder.albert_layer_groups.0.albert_layers.0.ffn_output.weight"
    if "/ffn" in node_name:
        return "bert.encoder.albert_layer_groups.0.albert_layers.0.ffn.weight"
    if "/bert_encoder/MatMul" in node_name:
        return "bert_encoder.weight"
    if "/duration_proj/linear_layer/MatMul" in node_name:
        return "predictor.duration_proj.linear_layer.weight"
    if "/decoder/generator/m_source/l_linear/MatMul" in node_name:
        return "decoder.generator.m_source.l_linear.weight"
    return None


def _reorder_lstm_gates(arr: np.ndarray) -> np.ndarray:
    # ONNX LSTM gate order: [i, o, f, c]
    # MLX (PyTorch) gate order: [i, f, g(c), o]
    hidden = arr.shape[1] // 4
    reshaped = arr.reshape(arr.shape[0], 4, hidden, *arr.shape[2:])
    order = [0, 2, 3, 1]
    reshaped = reshaped[:, order, ...]
    return reshaped.reshape(arr.shape[0], 4 * hidden, *arr.shape[2:])


def _convert_lstm(
    node: onnx.NodeProto,
    init: Dict[str, np.ndarray],
    weights: Dict[str, mx.array],
    params: Dict[str, mx.array],
    prefix: str,
):
    w = _dequantize(init, node.input[1])
    r = _dequantize(init, node.input[2])
    b = init[node.input[3]]

    # Some quantized exports store weights as (num_dirs, input, 4*hidden)
    # or (num_dirs, hidden, 4*hidden). Normalize to ONNX LSTM layout.
    if r.ndim == 3:
        if r.shape[1] * 4 == r.shape[2]:
            r = r.transpose(0, 2, 1)
        hidden = r.shape[2]
    else:
        hidden = r.shape[1] // 4

    if w.ndim == 3:
        if w.shape[1] != 4 * hidden and w.shape[2] == 4 * hidden:
            w = w.transpose(0, 2, 1)

    w = _reorder_lstm_gates(w)
    r = _reorder_lstm_gates(r)
    b = b.reshape(b.shape[0], 2, -1)
    w_bias = _reorder_lstm_gates(b[:, 0, :])
    r_bias = _reorder_lstm_gates(b[:, 1, :])

    directions = ["forward", "backward"]
    for idx, direction in enumerate(directions):
        weights[f"{prefix}.Wx_{direction}"] = mx.array(w[idx])
        weights[f"{prefix}.Wh_{direction}"] = mx.array(r[idx])
        weights[f"{prefix}.bias_ih_{direction}"] = mx.array(w_bias[idx])
        weights[f"{prefix}.bias_hh_{direction}"] = mx.array(r_bias[idx])

    # Sanity check
    for name in (
        f"{prefix}.Wx_forward",
        f"{prefix}.Wh_forward",
        f"{prefix}.bias_ih_forward",
        f"{prefix}.bias_hh_forward",
        f"{prefix}.Wx_backward",
        f"{prefix}.Wh_backward",
        f"{prefix}.bias_ih_backward",
        f"{prefix}.bias_hh_backward",
    ):
        if name not in params:
            raise KeyError(f"Expected LSTM param missing: {name}")


def _weights_from_onnx(model: onnx.ModelProto, mlx_model: Model) -> Dict[str, mx.array]:
    init = _init_map(model)
    params = dict(tree_flatten(mlx_model.parameters()))
    weights: Dict[str, mx.array] = {}
    used = set()

    # LSTM mapping
    lstm_prefixes = {
        "/text_encoder/lstm/LSTM": "text_encoder.lstm",
        "/text_encoder/lstms.0/LSTM": "predictor.text_encoder.lstms.0",
        "/text_encoder/lstms.2/LSTM": "predictor.text_encoder.lstms.2",
        "/lstm/LSTM": "predictor.lstm",
        "/shared/LSTM": "predictor.shared",
    }
    lstm_prefixes_quant = {
        "/text_encoder/lstm/LSTM_quant": "text_encoder.lstm",
        "/text_encoder/lstms.0/LSTM_quant": "predictor.text_encoder.lstms.0",
        "/text_encoder/lstms.2/LSTM_quant": "predictor.text_encoder.lstms.2",
        "/text_encoder/lstms.4/LSTM_quant": "predictor.text_encoder.lstms.4",
        "/lstm/LSTM_quant": "predictor.lstm",
        "/shared/LSTM_quant": "predictor.shared",
    }

    for node in model.graph.node:
        if node.op_type in {"LSTM", "DynamicQuantizeLSTM"}:
            if node.name in lstm_prefixes:
                used.update(node.input[1:4])
                _convert_lstm(node, init, weights, params, lstm_prefixes[node.name])
            elif node.name in lstm_prefixes_quant:
                used.update(node.input[1:4])
                _convert_lstm(
                    node, init, weights, params, lstm_prefixes_quant[node.name]
                )
            else:
                continue

    # MatMul weights
    for node in model.graph.node:
        if node.op_type not in {"MatMul", "MatMulInteger"}:
            continue
        if len(node.input) < 2:
            continue
        if not node.input[1].startswith("onnx::MatMul"):
            continue
        if node.input[1] not in init:
            continue
        target = _map_matmul_target(node.name)
        if target is None:
            continue
        w = _dequantize(init, node.input[1]).T
        used.add(node.input[1])
        if target not in params:
            raise KeyError(f"Target param not found: {target}")
        expected = params[target].shape
        if w.shape != expected:
            raise ValueError(f"Shape mismatch for {target}: {w.shape} vs {expected}")
        weights[target] = mx.array(w)

    # Named initializers
    for name, arr in init.items():
        if name in used:
            continue
        if not name.startswith("kmodel."):
            continue
        if name.endswith("_scale") or name.endswith("_zero_point"):
            continue
        is_quant = name.endswith("_quantized")
        base_name = name.replace("_quantized", "") if is_quant else name
        target = base_name.replace("kmodel.", "", 1)
        target = target.replace(".gamma", ".weight").replace(".beta", ".bias")
        target = target.replace(".alpha1.", ".alpha1_").replace(".alpha2.", ".alpha2_")

        if target in params:
            expected = params[target].shape
            w = _dequantize(init, name) if is_quant else arr
            # AdaIN fc weights in ONNX are stored as (in, out). If square,
            # shape checks won't catch orientation, so transpose explicitly.
            if (
                w.ndim == 2
                and w.shape == expected
                and w.shape[0] == w.shape[1]
                and ".adain" in target
                and target.endswith(".fc.weight")
            ):
                w = w.T
            if w.shape != expected:
                if w.ndim == 3 and w.transpose(0, 2, 1).shape == expected:
                    w = w.transpose(0, 2, 1)
                elif w.T.shape == expected:
                    w = w.T
                else:
                    raise ValueError(
                        f"Shape mismatch for {target}: {w.shape} vs {expected}"
                    )
            weights[target] = mx.array(w)
            continue

        if target.endswith(".weight"):
            weight_v = target.replace(".weight", ".weight_v")
            weight_g = target.replace(".weight", ".weight_g")
            if weight_v in params and weight_g in params:
                w = _dequantize(init, name) if is_quant else arr
                expected = params[weight_v].shape
                if w.shape != expected:
                    if w.ndim == 3 and w.transpose(0, 2, 1).shape == expected:
                        w = w.transpose(0, 2, 1)
                    else:
                        raise ValueError(
                            f"Shape mismatch for {weight_v}: {w.shape} vs {expected}"
                        )
                g = np.linalg.norm(w, axis=tuple(range(1, w.ndim)), keepdims=True)
                weights[weight_v] = mx.array(w)
                weights[weight_g] = mx.array(g)
                continue

        # Ignore unknowns

    # Fill missing params with initialized values
    for name, value in params.items():
        if name not in weights:
            weights[name] = value

    return weights


def convert(
    hf_path: str,
    mlx_path: str,
    revision: str | None = None,
    token: str | None = None,
    dequantize: bool = False,
):
    onnx_path, base_config = _load_onnx(hf_path, token, revision)
    model = onnx.load(str(onnx_path))
    activation_quant_modules = _infer_activation_quant_modules(model)
    if dequantize and any(
        node.op_type in {"MatMulInteger", "ConvInteger"} for node in model.graph.node
    ):
        model = _dequantize_onnx(model)
    config_dict = _infer_config(model, base_config)
    if activation_quant_modules:
        config_dict["activation_quant_modules"] = activation_quant_modules

    config = ModelConfig.from_dict(config_dict)
    mlx_model = Model(config)
    mlx_model.eval()

    weights = _weights_from_onnx(model, mlx_model)
    mlx_model.load_weights(list(weights.items()), strict=True)

    mlx_path = Path(mlx_path)
    mlx_path.mkdir(parents=True, exist_ok=True)

    # Copy voices
    voices_path = Path(onnx_path).with_name(base_config.get("voices", "voices.npz"))
    if voices_path.exists():
        (mlx_path / voices_path.name).write_bytes(voices_path.read_bytes())

    save_model(mlx_path, mlx_model, donate_model=True)
    save_config(config_dict, config_path=mlx_path / "config.json")
    print(f"[INFO] Saved MLX model to {mlx_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-path", required=True)
    parser.add_argument("--mlx-path", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--token", default=None)
    parser.add_argument(
        "--dequantize",
        action="store_true",
        help="Dequantize integer MatMul/Conv subgraphs into float ops before conversion.",
    )
    args = parser.parse_args()

    convert(
        hf_path=args.hf_path,
        mlx_path=args.mlx_path,
        revision=args.revision,
        token=args.token,
        dequantize=args.dequantize,
    )


if __name__ == "__main__":
    main()
