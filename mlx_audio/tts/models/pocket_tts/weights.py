import re

from mlx.utils import tree_flatten


def _seanet_names(prefix, cfg, encoder):
    """Map PyTorch Sequential indices to the MLX SEANet modules."""
    names = {f"{prefix}.model.0": f"{prefix}.init_conv1d"}
    index = 1
    for layer in range(len(cfg.ratios)):
        dest = f"{prefix}.layers.{layer}"
        if not encoder:
            names[f"{prefix}.model.{index + 1}"] = f"{dest}.upsample"
            index += 2
        for residual in range(cfg.n_residual_layers):
            for block in range(2):
                names[f"{prefix}.model.{index}.block.{2 * block + 1}"] = (
                    f"{dest}.residuals.{residual}.block.{block}"
                )
            index += 1
        if encoder:
            names[f"{prefix}.model.{index + 1}"] = f"{dest}.downsample"
            index += 2
    names[f"{prefix}.model.{index + 1}"] = f"{prefix}.final_conv1d"
    return names


def map_weights(model, weights):
    """Normalize tensor names and convolution layouts for MLX inference."""
    if not any(
        ".decoder.model." in key or ".condition_provider." in key for key in weights
    ):
        return weights

    names = {}
    for kind in ("encoder", "decoder"):
        names.update(
            _seanet_names(f"mimi.{kind}", model.config.mimi.seanet, kind == "encoder")
        )
    names = sorted(names.items(), key=lambda item: -len(item[0]))
    target_keys = dict(tree_flatten(model.parameters()))
    result = {}
    for key, value in weights.items():
        if any(
            part in key
            for part in (".w_s_t.", ".quantizer.vq.", ".quantizer.logvar_proj.")
        ):
            continue
        key = key.replace(
            "flow_lm.condition_provider.conditioners.speaker_wavs.output_proj.weight",
            "speaker_proj_weight",
        )
        key = key.replace("flow_lm.speaker_proj_weight", "speaker_proj_weight")
        key = key.replace(
            "flow_lm.condition_provider.conditioners.transcript_in_segment.",
            "flow_lm.conditioner.",
        )
        for name in ("num_speakers", "duration_delta"):
            key = key.replace(
                f"flow_lm.condition_provider.conditioners.{name}.",
                f"flow_lm.{name}.",
            )
        key = key.replace("flow_lm.backbone.", "flow_lm.transformer.")
        key = key.replace("flow_lm.flow.", "flow_lm.flow_net.")
        key = key.replace("mimi.model.", "mimi.")
        for source, target in names:
            if key.startswith(source + "."):
                key = target + key[len(source) :]
                break
        key = key.replace(".in_proj_weight", ".in_proj.weight")
        key = re.sub(
            r"(\.(?:encoder|decoder)_transformer\.transformer\.layers\.\d+)\.(linear[12])\.",
            r"\1.gating.\2.",
            key,
        )
        if key not in target_keys:
            key = re.sub(r"\.(conv|convtr)\.(weight|bias)$", r".\1.\1.\2", key)
        if value.ndim == 3 and key.endswith(".weight"):
            if ".convtr." in key:
                # Torch [in, out/groups, kernel] -> MLX [out, kernel, in/groups].
                value = (
                    value.transpose(0, 2, 1)
                    if key.startswith("mimi.upsample.")
                    else value.transpose(1, 2, 0)
                )
            else:
                value = value.transpose(0, 2, 1)
        result[key] = value
    return result
