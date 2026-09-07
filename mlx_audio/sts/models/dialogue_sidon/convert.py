"""Convert the published torch.export archives to a standalone MLX model.

PyTorch is used only to deserialize tensor dictionaries, never at inference.
Example: python -m mlx_audio.sts.models.dialogue_sidon.convert \
    --mlx-path output/DialogueSidon-fp32 --dtype float32
"""

import argparse
import io
import json
import re
import zipfile
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from .config import DiffusionConfig, EncoderConfig, ModelConfig
from .model import Model

SOURCE_REPO = "sarulab-speech/DialogueSidon"
SOURCE_REVISION = "d43d7478402a5527136c6733c3f4359c37b312ab"
EXPORTS = ("ssl_encoder.pt2", "diffusion_head.pt2", "vae_decoder.pt2")


def load_export(path):
    """Read CPU tensors and graph metadata without executing an exported graph."""
    import torch

    with zipfile.ZipFile(path) as archive:
        members = archive.namelist()
        weights_path = next(n for n in members if n.endswith("/data/weights/model.pt"))
        graph_path = next(n for n in members if n.endswith("/models/model.json"))
        state = torch.load(
            io.BytesIO(archive.read(weights_path)),
            map_location="cpu",
            weights_only=True,
        )
        graph = json.loads(archive.read(graph_path))
    return {k: v.detach() for k, v in state.items()}, graph


def merge_lora(state, scale=0.25):
    """Merge PEFT's rank-64 / alpha-16 adapters in float32, checking coverage."""
    state = {k: v.detach().float() for k, v in state.items()}
    for key in list(state):
        if key.endswith(".lora_A.default.weight"):
            prefix = key.removesuffix(".lora_A.default.weight")
            a = state.pop(key)
            b = state.pop(prefix + ".lora_B.default.weight")
            base_key = prefix + ".base_layer.weight"
            state[base_key] = state[base_key] + scale * (b @ a)
    if any(".lora_" in k for k in state):
        raise ValueError("Unpaired or unsupported LoRA tensors in source checkpoint")
    return {k.replace(".base_layer.", "."): v for k, v in state.items()}


def convert_encoder_weights(state):
    converted = {}
    for key, value in merge_lora(state).items():
        prefix = "ssl_model.base_model.model."
        if key == prefix + "masked_spec_embed":
            continue  # Training-only SpecAugment vector, unused by exported inference.
        if key.startswith(prefix):
            key = key[len(prefix) :]
            key = key.removeprefix("encoder.")
            key = "encoder." + key
        if value.ndim == 3 and key.endswith(".weight"):
            value = value.permute(0, 2, 1)
        converted[key] = mx.array(value.contiguous().numpy())
    return converted


def convert_diffusion_weights(state):
    return {
        "diffusion_head."
        + re.sub(
            r"\.(mlp|adaLN_modulation)\.(\d+)\.", r".\1.layers.\2.", key
        ): mx.array(value.detach().float().numpy())
        for key, value in state.items()
    }


def convert_decoder_weights(state):
    converted = {}
    for key, value in state.items():
        if value.ndim == 3:
            transpose = bool(
                re.fullmatch(r"decoder\.model\.\d+\.block\.1\.weight_[gv]", key)
            )
            value = value.permute(1, 2, 0) if transpose else value.permute(0, 2, 1)
        key = re.sub(r"\.(model|block)\.(\d+)(?=\.)", r".\1.layers.\2", key)
        converted[key] = mx.array(value.detach().float().contiguous().numpy())
    return converted


def config_from_source(metadata, graphs, source_repo, revision):
    def shapes(graph):
        module = graph["graph_module"]
        values = module["graph"]["tensor_values"]
        return {
            spec["parameter"]["parameter_name"]: [
                d["as_int"] for d in values[spec["parameter"]["arg"]["name"]]["sizes"]
            ]
            for spec in module["signature"]["input_specs"]
            if "parameter" in spec
        }

    enc, dit, dec = map(shapes, graphs)
    root = "ssl_model.base_model.model."
    hidden = enc[root + "feature_projection.projection.weight"][0]
    layer_root = root + "encoder.layers.0."
    head_dim = enc[layer_root + "self_attn.distance_embedding.weight"][1]
    diffusion_dim = dit["latent_proj.weight"][0]
    # The export stores the head count in the q/k/v view, not in tensor weights.
    views = [
        n
        for n in graphs[1]["graph_module"]["graph"]["nodes"]
        if n["target"] == "torch.ops.aten.view.default"
    ]
    first_view = views[0]["outputs"][0]["as_tensor"]["name"]
    dit_heads = graphs[1]["graph_module"]["graph"]["tensor_values"][first_view][
        "sizes"
    ][2]["as_int"]
    scales = [
        arg["arg"]["as_float"]
        for node in graphs[0]["graph_module"]["graph"]["nodes"]
        if node["target"] == "torch.ops.aten.mul.Tensor"
        for arg in node["inputs"]
        if "as_float" in arg["arg"]
    ]
    lora_pairs = sum(".lora_A." in k for k in enc)
    if scales.count(0.25) != lora_pairs:
        raise ValueError(
            "Export does not have the expected alpha/rank=0.25 LoRA scaling"
        )
    ddpm = metadata["ddpm_config"]
    if ddpm["beta_schedule"] != "linear" or ddpm.get("rescale_betas_zero_snr", False):
        raise ValueError("Only the published linear-beta scheduler is supported")
    return ModelConfig(
        latent_dim=metadata["latent_dim"],
        sample_rate=metadata["sample_rate"],
        encoder=EncoderConfig(
            hidden_size=hidden,
            num_attention_heads=hidden // head_dim,
            intermediate_size=enc[
                layer_root + "ffn1.intermediate_dense.base_layer.weight"
            ][0],
            num_hidden_layers=len(
                {
                    re.search(r"encoder.layers.(\d+)", k).group(1)
                    for k in enc
                    if "encoder.layers." in k
                }
            ),
        ),
        diffusion=DiffusionConfig(
            hidden_size=diffusion_dim,
            num_heads=dit_heads,
            num_layers=len(
                {
                    re.search(r"blocks.(\d+)", k).group(1)
                    for k in dit
                    if k.startswith("blocks.")
                }
            ),
            ffn_ratio=dit["blocks.0.mlp.0.weight"][0] / diffusion_dim,
            frequency_embedding_size=dit["t_embedder.mlp.0.weight"][1],
            num_train_timesteps=ddpm["num_train_timesteps"],
            beta_start=ddpm["beta_start"],
            beta_end=ddpm["beta_end"],
            prediction_type=ddpm["prediction_type"],
        ),
        decoder_channels=dec["decoder.model.0.weight_v"][0],
        latent_norm_initialized=metadata["latent_norm_initialized"],
        latent_norm_mean=metadata["latent_norm_mean"],
        latent_norm_std=metadata["latent_norm_std"],
        source_repo=source_repo,
        source_revision=revision,
    )


def model_card(source_repo, revision, dtype):
    return f"""---
license: cc-by-nc-4.0
library_name: mlx-audio
tags:
- mlx
- audio
- audio-source-separation
base_model: {source_repo}
---

# DialogueSidon ({dtype}, MLX)

Separate a recording of two speakers into individual audio tracks on Apple
Silicon, with two mono outputs at 24 kHz. The model also restores degraded speech.
Original model: [{source_repo}](https://huggingface.co/{source_repo}/tree/{revision}).

Requires MLX Audio with DialogueSidon support.

Load this model using its Hugging Face repository ID or a local model directory:

```python
from mlx_audio.sts import load
from mlx_audio.audio_io import write

model = load("/path/to/this/model")
result = model.separate("dialogue.wav", num_steps=30, seed=0)
for i, speaker in enumerate(result.speakers, 1):
    write(f"speaker_{{i}}.wav", speaker, result.sample_rate)
```

The model supports exactly two speakers and processes recorded audio offline.
The tracks are numbered rather than labeled with speaker identities; a speaker
may switch tracks after a long silence. Because the model also restores speech,
adding the tracks together may not reproduce the input exactly.

Long recordings are processed automatically in 20-second chunks with 5-second
overlap. Adjust `chunk_seconds` and `overlap_seconds` to change these durations.
Use `chunk_seconds=None` to process an entire file at once with more memory.
Set `num_steps` to adjust the step count; fewer steps run faster but can change
output quality. Use `seed` to repeat a run with the same input, model, and settings.

The checkpoint retains **CC-BY-NC-4.0**. Original model by Wataru Nakata,
Yuki Saito, Kazuki Yamauchi, Emiru Tsunoo, and Hiroshi Saruwatari (SaruLab).
See the [original model card](https://huggingface.co/{source_repo}) and
[Sidon](https://github.com/sarulab-speech/Sidon).
"""


def convert(
    hf_path=SOURCE_REPO,
    mlx_path="output/DialogueSidon-fp32",
    dtype="float32",
    revision=SOURCE_REVISION,
):
    if dtype not in {"float32", "float16", "bfloat16"}:
        raise ValueError("dtype must be float32, float16, or bfloat16")
    source = Path(hf_path)
    if not source.is_dir():
        from huggingface_hub import snapshot_download

        source = Path(
            snapshot_download(
                hf_path, revision=revision, allow_patterns=[*EXPORTS, "metadata.json"]
            )
        )
    metadata = json.loads((source / "metadata.json").read_text())
    weights, graphs = {}, []
    for name, converter in zip(
        EXPORTS,
        (convert_encoder_weights, convert_diffusion_weights, convert_decoder_weights),
    ):
        state, graph = load_export(source / name)
        graphs.append(graph)
        weights.update(converter(state))
        print(f"Converted {name}: {len(state)} source tensors", flush=True)
        del state
    source_repo = hf_path if not Path(hf_path).is_dir() else SOURCE_REPO
    config = config_from_source(metadata, graphs, source_repo, revision)
    model = Model(config)
    model.load_weights(list(weights.items()), strict=True)
    parameters = dict(tree_flatten(model.parameters()))
    count = sum(v.size for v in parameters.values())
    parameters = {k: v.astype(getattr(mx, dtype)) for k, v in parameters.items()}
    mx.eval(parameters)
    destination = Path(mlx_path)
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.glob("*.safetensors")):
        raise FileExistsError(
            f"Refusing to overwrite existing weights in {destination}"
        )
    mx.save_safetensors(str(destination / "model.safetensors"), parameters)
    artifact_config = asdict(config)
    artifact_config["torch_dtype"] = dtype
    (destination / "config.json").write_text(
        json.dumps(artifact_config, indent=2) + "\n"
    )
    (destination / "README.md").write_text(model_card(source_repo, revision, dtype))
    print(f"Saved {len(parameters)} tensors ({count:,} parameters) to {destination}")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-path", default=SOURCE_REPO)
    parser.add_argument("--mlx-path", default="output/DialogueSidon-fp32")
    parser.add_argument("--revision", default=SOURCE_REVISION)
    parser.add_argument(
        "--dtype", choices=["float32", "float16", "bfloat16"], default="float32"
    )
    args = parser.parse_args()
    convert(args.hf_path, args.mlx_path, args.dtype, args.revision)


if __name__ == "__main__":
    main()
