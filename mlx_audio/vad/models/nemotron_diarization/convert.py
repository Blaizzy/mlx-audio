"""Convert Nemotron 3 Diarization from a local or Hugging Face NeMo archive.

PyTorch and PyYAML are conversion-only dependencies. No NeMo installation is
required and the archive is read without extracting files to disk.
"""

import argparse
import io
import json
import tarfile
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from .config import ModelConfig
from .nemotron_diarization import Model

SOURCE_REPO = "nvidia/Nemotron-3-Diarization"


def read_nemo(path):
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required to convert NeMo checkpoints. "
            "Install it with `pip install torch` and rerun the conversion."
        ) from exc

    import yaml

    with tarfile.open(path, "r:*") as archive:
        members = {}
        for member in archive.getmembers():
            name = Path(member.name).name
            if name in ("model_config.yaml", "model_weights.ckpt"):
                if not member.isfile() or name in members:
                    raise ValueError(f"Invalid or duplicate archive member: {name}")
                members[name] = member
        if set(members) != {"model_config.yaml", "model_weights.ckpt"}:
            raise ValueError(
                "NeMo archive must contain model_config.yaml and model_weights.ckpt"
            )
        config = yaml.safe_load(archive.extractfile(members["model_config.yaml"]))
        weights = torch.load(
            io.BytesIO(archive.extractfile(members["model_weights.ckpt"]).read()),
            map_location="cpu",
            weights_only=True,
        )
    return config, weights


def build_config(source, state_dict, dtype="float32"):
    enc, pp, mod = (
        source["encoder"],
        source["preprocessor"],
        source["sortformer_modules"],
    )
    supported = {
        "subsampling": "feature_stacking",
        "self_attention_model": "rope",
        "attn_mode": "full",
        "feat_out": -1,
    }
    for key, value in supported.items():
        if enc.get(key, value) != value:
            raise ValueError(f"Unsupported encoder {key}: {enc[key]}")
    if not source.get("high_resolution") or source.get("transformer_encoder"):
        raise ValueError(
            "Expected a high-resolution single-encoder Nemotron diarization model"
        )
    if pp.get("normalize", "NA") != "NA" or pp.get("window", "hann") != "hann":
        raise ValueError("Only unnormalized Hann-window log-mels are supported")
    for key, value in {
        "frame_splicing": 1,
        "exact_pad": False,
        "mag_power": 2.0,
        "log_zero_guard_type": "add",
        "log_zero_guard_value": 2**-24,
    }.items():
        if pp.get(key, value) != value:
            raise ValueError(f"Unsupported preprocessor {key}: {pp[key]}")
    if not mod.get("use_learnable_sil_emb", False):
        raise ValueError("Expected a learned silence embedding")
    cfg = ModelConfig()
    cfg.dtype = dtype
    cfg.num_speakers = mod["num_spks"]
    cfg.output_subsampling_factor = source.get("output_subsampling_factor", 1)
    for key in asdict(cfg.encoder_config):
        if key in enc:
            setattr(cfg.encoder_config, key, enc[key])
    for key in asdict(cfg.modules_config):
        if key in mod:
            setattr(cfg.modules_config, key, mod[key])
    cfg.modules_config.num_speakers = cfg.num_speakers
    cfg.modules_config.subsampling_factor = cfg.encoder_config.subsampling_factor
    cfg.modules_config.use_activity_head = any(
        k.startswith("sortformer_modules.activity_head.") for k in state_dict
    )
    # NVIDIA's recommended offline preset, rather than training chunk defaults.
    cfg.modules_config.chunk_len = 340
    cfg.modules_config.chunk_right_context = 40
    cfg.modules_config.fifo_len = 40
    cfg.modules_config.spkcache_update_period = 300
    proc = cfg.processor_config
    proc.feature_size = pp["features"]
    proc.sampling_rate = pp["sample_rate"]
    proc.n_fft = pp.get("n_fft", 512)
    proc.win_length = int(pp["window_size"] * proc.sampling_rate)
    proc.hop_length = int(pp["window_stride"] * proc.sampling_rate)
    proc.preemphasis = pp.get("preemph", 0.97)
    proc.pad_to = pp.get("pad_to", 16)
    cfg.__post_init__()
    return cfg


def convert_weights(state_dict, dtype="float32"):
    if dtype not in ("float32", "float16", "bfloat16"):
        raise ValueError(f"Unsupported dtype: {dtype}")
    converted = {}
    for key, tensor in state_dict.items():
        key = key.replace("preprocessor.featurizer.", "preprocessor.")
        key = key.replace(".ffn.net.0.", ".ffn.linear1.").replace(
            ".ffn.net.3.", ".ffn.linear2."
        )
        key = key.replace(".activity_head.0.", ".activity_head.layers.0.").replace(
            ".activity_head.1.", ".activity_head.layers.1."
        )
        array = mx.array(tensor.detach().cpu().float().numpy())
        if key == "sortformer_modules.subpixel_upsample.weight":
            array = array.transpose(0, 2, 1)
        # Feature extraction runs in float32, using the exact archived buffers.
        if not key.startswith("preprocessor."):
            array = array.astype(getattr(mx, dtype))
        if key in converted:
            raise ValueError(f"Duplicate converted key: {key}")
        converted[key] = array
    return converted


def validate_weights(model, weights):
    expected = dict(tree_flatten(model.parameters()))
    missing, extra = (
        sorted(expected.keys() - weights.keys()),
        sorted(weights.keys() - expected.keys()),
    )
    mismatched = [
        f"{k}: expected {expected[k].shape}, got {v.shape}"
        for k, v in weights.items()
        if k in expected and expected[k].shape != v.shape
    ]
    if missing or extra or mismatched:
        raise ValueError(
            f"Checkpoint mismatch:\nMissing: {missing}\nUnexpected: {extra}\nShapes: {mismatched}"
        )
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    mx.eval(model.parameters())


def model_card(model_id, source_id, config, quantization=None):
    precision = config.dtype
    if quantization:
        precision = (
            f"8-bit affine quantized (group size {quantization['group_size']}, "
            f"{config.dtype} activations and unquantized layers)"
        )
    return f"""---
license: openmdw-1.1
library_name: mlx-audio
pipeline_tag: voice-activity-detection
base_model: {source_id}
tags:
- mlx
- mlx-audio
- speaker-diarization
- streaming-sortformer
---

# Nemotron 3 Diarization (MLX)

Converted from [{source_id}](https://huggingface.co/{source_id}).
The weights retain the upstream OpenMDW 1.1 license.

This is a {precision} MLX port of the 31-layer RoPE Transformer with
feature stacking, learned AOSC silence embeddings and a subpixel output head.
It predicts up to eight speakers at 10 ms resolution from 16 kHz mono audio.
The default chunked inference uses NVIDIA's recommended 30.4 s input-buffer
preset (340 encoder frames plus 40 frames of right context).

Requires an mlx-audio version containing `nemotron_diarization` support.

```python
from mlx_audio.vad import load

model = load("{model_id}", strict=True)
result = model.generate("meeting.wav")
print(result.text)

# Incremental results retain speaker identities through the AOSC and FIFO.
for result in model.generate_stream("meeting.wav"):
    for segment in result.segments:
        print(segment.start, segment.end, segment.speaker)
```

For live PCM input, call `model.feed(chunk, state)` with a state from
`model.init_streaming_state()` and 16 kHz mono chunks. Flush the final partial
chunk and lookahead with `model.feed([], state, final=True)`. Timestamps are
absolute within the recording. Labels are generic arrival-order speaker IDs;
the model does not identify people. Overlapping speakers may be active together.

This port does not perform ASR or include NeMo's evaluation/postprocessing
pipeline. Threshold/minimum-duration/gap merging follow mlx-audio's diarization
API. Consult the [upstream model card](https://huggingface.co/{source_id})
for training data, evaluation and license information.
"""


def convert(
    nemo_path,
    mlx_path,
    dtype="float32",
    source_id=SOURCE_REPO,
    model_id=None,
    revision=None,
    quantize=False,
    group_size=64,
):
    path = Path(nemo_path)
    if not path.is_file():
        if path.suffix == ".nemo":
            raise FileNotFoundError(path)
        from huggingface_hub import hf_hub_download

        source_id = str(nemo_path)
        path = Path(
            hf_hub_download(
                source_id, source_id.rsplit("/", 1)[-1] + ".nemo", revision=revision
            )
        )
    source, tensors = read_nemo(path)
    config = build_config(source, tensors, dtype)
    weights = convert_weights(tensors, dtype)
    model = Model(config)
    validate_weights(model, weights)
    config_json = asdict(config)
    if quantize:
        from mlx_audio.lm.convert import quantize_model

        model, config_json = quantize_model(
            model, config_json, group_size=group_size, bits=8
        )
        weights = dict(tree_flatten(model.parameters()))
        mx.eval(weights)
    output = Path(mlx_path)
    output.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(output / "model.safetensors"), weights)
    config_json.update(source_repo=source_id, source_revision=revision)
    (output / "config.json").write_text(json.dumps(config_json, indent=2) + "\n")
    (output / "README.md").write_text(
        model_card(
            model_id or str(output),
            source_id,
            config,
            config_json.get("quantization"),
        )
    )
    print(f"Converted and strictly validated {len(weights)} tensors -> {output}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nemo-path", "--hf-path", dest="nemo_path", required=True)
    parser.add_argument("--mlx-path", "--output-dir", dest="mlx_path", required=True)
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="float32"
    )
    parser.add_argument(
        "--model-id",
        help="Future HF repo ID for the generated model card; does not upload",
    )
    parser.add_argument("--revision", help="Source Hugging Face revision")
    parser.add_argument(
        "--quantize",
        "-q",
        action="store_true",
        help="Quantize eligible linear weights to 8-bit affine MLX format",
    )
    parser.add_argument("--group-size", type=int, choices=(32, 64, 128), default=64)
    args = parser.parse_args()
    convert(**vars(args))


if __name__ == "__main__":
    main()
