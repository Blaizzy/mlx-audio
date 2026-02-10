"""Convert NVIDIA NeMo .nemo diarization models to MLX safetensors + config.json.

Usage:
    python -m mlx_audio.vad.models.sortformer.convert \
        --nemo-path nvidia/diar_streaming_sortformer_4spk-v2.1 \
        --output-dir ./sortformer-v2.1-mlx

    # Or from a local .nemo file:
    python -m mlx_audio.vad.models.sortformer.convert \
        --nemo-path /path/to/model.nemo \
        --output-dir ./sortformer-v2.1-mlx

    # Optionally upload to HuggingFace:
    python -m mlx_audio.vad.models.sortformer.convert \
        --nemo-path nvidia/diar_streaming_sortformer_4spk-v2.1 \
        --output-dir ./sortformer-v2.1-mlx \
        --upload <hf-repo-id>
"""

import argparse
import io
import json
import math
import tarfile
import tempfile
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

# ── Key remapping rules ──────────────────────────────────────────────
#
# NeMo module hierarchy → MLX key prefix
#   encoder.pre_encode.conv.N.*           → fc_encoder.subsampling.layers_N.*
#   encoder.pre_encode.out.*              → fc_encoder.subsampling.linear.*
#   encoder.layers.N.*                    → fc_encoder.layers.N.*
#   transformer_encoder.layers.N.*        → tf_encoder.layers.N.*
#   sortformer_modules.*                  → sortformer_modules.* (unchanged)
#
# FastConformer attention renames:
#   self_attn.linear_q      → self_attn.q_proj
#   self_attn.linear_k      → self_attn.k_proj
#   self_attn.linear_v      → self_attn.v_proj
#   self_attn.linear_out    → self_attn.o_proj
#   self_attn.linear_pos    → self_attn.relative_k_proj
#   self_attn.pos_bias_u    → self_attn.bias_u
#   self_attn.pos_bias_v    → self_attn.bias_v
#
# Conformer conv module:
#   conv.batch_norm.*       → conv.norm.*
#
# Conformer FFN: already matches (feed_forward1.linear1, etc.)
#
# Transformer encoder renames:
#   first_sub_layer.query_net       → self_attn.q_proj
#   first_sub_layer.key_net         → self_attn.k_proj
#   first_sub_layer.value_net       → self_attn.v_proj
#   first_sub_layer.out_projection  → self_attn.out_proj
#   second_sub_layer.dense_in       → fc1
#   second_sub_layer.dense_out      → fc2
#   layer_norm_1                    → self_attn_layer_norm
#   layer_norm_2                    → final_layer_norm

SKIP_KEYS = {
    "num_batches_tracked",
    "preprocessor",
}

# Conformer attention key renames
FC_ATTN_RENAMES = {
    "self_attn.linear_q.": "self_attn.q_proj.",
    "self_attn.linear_k.": "self_attn.k_proj.",
    "self_attn.linear_v.": "self_attn.v_proj.",
    "self_attn.linear_out.": "self_attn.o_proj.",
    "self_attn.linear_pos.": "self_attn.relative_k_proj.",
    "self_attn.pos_bias_u": "self_attn.bias_u",
    "self_attn.pos_bias_v": "self_attn.bias_v",
}

# Conformer conv batch_norm
FC_CONV_RENAMES = {
    "conv.batch_norm.": "conv.norm.",
}

# Transformer encoder renames
TF_RENAMES = {
    "first_sub_layer.query_net.": "self_attn.q_proj.",
    "first_sub_layer.key_net.": "self_attn.k_proj.",
    "first_sub_layer.value_net.": "self_attn.v_proj.",
    "first_sub_layer.out_projection.": "self_attn.out_proj.",
    "second_sub_layer.dense_in.": "fc1.",
    "second_sub_layer.dense_out.": "fc2.",
    "layer_norm_1.": "self_attn_layer_norm.",
    "layer_norm_2.": "final_layer_norm.",
}


def _apply_renames(key: str, renames: dict) -> str:
    """Apply string substitution rules to a key."""
    for old, new in renames.items():
        if old in key:
            key = key.replace(old, new)
    return key


def remap_key(nemo_key: str) -> str | None:
    """Map a NeMo state_dict key to its MLX equivalent. Returns None to skip."""
    # Skip unwanted keys
    if any(sk in nemo_key for sk in SKIP_KEYS):
        return None

    key = nemo_key

    # ── Subsampling ──
    if key.startswith("encoder.pre_encode.conv."):
        # encoder.pre_encode.conv.N.* → fc_encoder.subsampling.layers_N.*
        key = key.replace("encoder.pre_encode.conv.", "fc_encoder.subsampling.layers_")
        # "layers_0.weight" etc. — need to replace first dot after layer index
        # The key is now like "fc_encoder.subsampling.layers_0.weight"
        # which is already correct for our attribute naming
        return key

    if key.startswith("encoder.pre_encode.out."):
        return key.replace("encoder.pre_encode.out.", "fc_encoder.subsampling.linear.")

    # ── Conformer layers ──
    if key.startswith("encoder.layers."):
        key = key.replace("encoder.layers.", "fc_encoder.layers.")
        key = _apply_renames(key, FC_ATTN_RENAMES)
        key = _apply_renames(key, FC_CONV_RENAMES)
        return key

    # ── Transformer encoder ──
    if key.startswith("transformer_encoder."):
        key = key.replace("transformer_encoder.", "tf_encoder.")
        key = _apply_renames(key, TF_RENAMES)
        return key

    # ── Sortformer modules ── (no prefix change needed)
    if key.startswith("sortformer_modules."):
        return key

    # Unknown key — skip with warning
    return None


def _sinusoidal_embeddings(max_len: int, d_model: int) -> np.ndarray:
    """Generate sinusoidal positional embeddings (for TransformerEncoder)."""
    pe = np.zeros((max_len, d_model), dtype=np.float32)
    position = np.arange(0, max_len, dtype=np.float32)[:, None]
    div_term = np.exp(
        np.arange(0, d_model, 2, dtype=np.float32) * -(math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe


def convert_weights(state_dict: dict) -> Tuple[Dict[str, np.ndarray], list]:
    """Remap NeMo state dict keys and transpose conv weights.

    Returns:
        (mlx_weights, warnings) — dict of numpy arrays and list of warning strings
    """
    mlx_weights = {}
    warnings = []

    for nemo_key, tensor in state_dict.items():
        mlx_key = remap_key(nemo_key)
        if mlx_key is None:
            if not any(sk in nemo_key for sk in SKIP_KEYS):
                warnings.append(f"Skipped unknown key: {nemo_key}")
            continue

        arr = tensor.cpu().numpy()

        # Conv2d: PyTorch (O, I, H, W) → MLX (O, H, W, I)
        if "subsampling" in mlx_key and "weight" in mlx_key and "linear" not in mlx_key:
            if arr.ndim == 4:
                arr = arr.transpose(0, 2, 3, 1)

        # Conv1d: PyTorch (O, I, K) → MLX (O, K, I)
        if any(
            name in mlx_key
            for name in ["pointwise_conv1", "pointwise_conv2", "depthwise_conv"]
        ):
            if arr.ndim == 3 and "weight" in mlx_key:
                arr = arr.transpose(0, 2, 1)

        mlx_weights[mlx_key] = arr

    return mlx_weights, warnings


def build_config(yaml_cfg: dict) -> dict:
    """Build our config.json from NeMo's model_config.yaml."""
    enc = yaml_cfg.get("encoder", {})
    tf = yaml_cfg.get("transformer_encoder", {})
    sm = yaml_cfg.get("sortformer_modules", {})
    pp = yaml_cfg.get("preprocessor", {})

    sample_rate = pp.get("sample_rate", 16000)
    win_size = pp.get("window_size", 0.025)
    win_stride = pp.get("window_stride", 0.01)
    n_fft = pp.get("n_fft", 512)
    features = pp.get("features", 128)
    normalize = pp.get("normalize", "NA")

    # Detect if normalization is disabled
    use_aosc = normalize in ("NA", "None", None, "null")

    config = {
        "model_type": "sortformer",
        "num_speakers": sm.get("num_spks", 4),
        "ats_weight": yaml_cfg.get("ats_weight", 0.5),
        "pil_weight": yaml_cfg.get("pil_weight", 0.5),
        "dtype": "float32",
        "fc_encoder_config": {
            "model_type": "sortformer_fc_encoder",
            "hidden_size": enc.get("d_model", 512),
            "num_hidden_layers": enc.get("n_layers", 17),
            "num_attention_heads": enc.get("n_heads", 8),
            "num_key_value_heads": enc.get("n_heads", 8),
            "intermediate_size": enc.get("d_model", 512) * 4,
            "hidden_act": "silu",
            "num_mel_bins": features,
            "conv_kernel_size": enc.get("conv_kernel_size", 9),
            "subsampling_factor": enc.get("subsampling_factor", 8),
            "subsampling_conv_channels": enc.get("subsampling_conv_channels", 256),
            "subsampling_conv_kernel_size": 3,
            "subsampling_conv_stride": 2,
            "max_position_embeddings": 5000,
            "attention_bias": True,
            "scale_input": enc.get("xscaling", True),
        },
        "tf_encoder_config": {
            "model_type": "sortformer_tf_encoder",
            "d_model": tf.get("hidden_size", 192),
            "encoder_layers": tf.get("num_layers", 18),
            "encoder_attention_heads": tf.get("num_attention_heads", 8),
            "encoder_ffn_dim": tf.get("inner_size", 768),
            "activation_function": tf.get("hidden_act", "relu"),
            "max_source_positions": 1500,
            "k_proj_bias": True,  # NeMo v2.1 has bias on key_net
        },
        "modules_config": {
            "model_type": "sortformer_modules",
            "num_speakers": sm.get("num_spks", 4),
            "fc_d_model": sm.get("fc_d_model", 512),
            "tf_d_model": sm.get("tf_d_model", 192),
            "subsampling_factor": enc.get("subsampling_factor", 8),
            "chunk_len": sm.get("chunk_len", 188),
            "fifo_len": sm.get("fifo_len", 0),
            "spkcache_len": sm.get("spkcache_len", 188),
            "spkcache_update_period": sm.get("spkcache_update_period", 188),
            "chunk_left_context": sm.get("chunk_left_context", 1),
            "chunk_right_context": sm.get("chunk_right_context", 1),
            "spkcache_sil_frames_per_spk": sm.get("spkcache_sil_frames_per_spk", 3),
            "causal_attn_rc": sm.get("causal_attn_rc", 7),
            "scores_boost_latest": sm.get("scores_boost_latest", 0.05),
            "sil_threshold": sm.get("sil_threshold", 0.2),
            "pred_score_threshold": sm.get("pred_score_threshold", 0.25),
            "strong_boost_rate": sm.get("strong_boost_rate", 0.75),
            "weak_boost_rate": sm.get("weak_boost_rate", 1.5),
            "min_pos_scores_rate": sm.get("min_pos_scores_rate", 0.5),
            "max_index": sm.get("max_index", 99999),
            "use_aosc": use_aosc,
        },
        "processor_config": {
            "feature_size": features,
            "sampling_rate": sample_rate,
            "hop_length": int(win_stride * sample_rate),
            "n_fft": n_fft,
            "win_length": int(win_size * sample_rate),
            "preemphasis": pp.get("preemph", 0.97),
        },
    }

    return config


def download_nemo(model_id: str) -> Path:
    """Download a .nemo file from HuggingFace if model_id is a repo ID."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=model_id,
        filename=model_id.split("/")[-1] + ".nemo",
    )
    return Path(path)


def extract_nemo(nemo_path: Path, tmpdir: Path) -> Tuple[dict, dict]:
    """Extract model_config.yaml and model_weights.ckpt from .nemo tar.

    Returns:
        (yaml_config, pytorch_state_dict)
    """
    import torch
    import yaml

    with tarfile.open(nemo_path, "r") as tar:
        yaml_cfg = None
        state_dict = None

        for member in tar.getmembers():
            name = member.name.split("/")[-1]  # handle nested paths
            if name == "model_config.yaml":
                f = tar.extractfile(member)
                yaml_cfg = yaml.safe_load(f)
            elif name == "model_weights.ckpt":
                f = tar.extractfile(member)
                buf = io.BytesIO(f.read())
                state_dict = torch.load(buf, map_location="cpu", weights_only=True)

    if yaml_cfg is None:
        raise ValueError(f"model_config.yaml not found in {nemo_path}")
    if state_dict is None:
        raise ValueError(f"model_weights.ckpt not found in {nemo_path}")

    return yaml_cfg, state_dict


def save_safetensors(weights: Dict[str, np.ndarray], path: Path):
    """Save weights as safetensors."""
    import mlx.core as mx

    mx_weights = {k: mx.array(v) for k, v in weights.items()}
    mx.save_safetensors(str(path), mx_weights)


def main():
    parser = argparse.ArgumentParser(
        description="Convert NeMo .nemo diarization model to MLX safetensors"
    )
    parser.add_argument(
        "--nemo-path",
        type=str,
        required=True,
        help="Path to .nemo file or HuggingFace repo ID (e.g. nvidia/diar_streaming_sortformer_4spk-v2.1)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for safetensors + config.json",
    )
    parser.add_argument(
        "--upload",
        type=str,
        default=None,
        help="HuggingFace repo ID to upload the converted model to",
    )
    args = parser.parse_args()

    nemo_path = Path(args.nemo_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download if it's a HF repo ID
    if not nemo_path.exists():
        print(f"Downloading {args.nemo_path} from HuggingFace...")
        nemo_path = download_nemo(args.nemo_path)
        print(f"Downloaded to {nemo_path}")

    print("Extracting .nemo archive...")
    with tempfile.TemporaryDirectory() as tmpdir:
        yaml_cfg, state_dict = extract_nemo(nemo_path, Path(tmpdir))

    print(f"Found {len(state_dict)} weight tensors")

    # Build config
    config = build_config(yaml_cfg)
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {config_path}")

    # Convert weights
    print("Remapping and transposing weights...")
    mlx_weights, warnings = convert_weights(state_dict)
    for w in warnings:
        print(f"  WARNING: {w}")
    print(f"Converted {len(mlx_weights)} weight tensors")

    # Add sinusoidal positional embeddings for TransformerEncoder
    # (NeMo doesn't have them but our MLX model expects them)
    tf_cfg = config["tf_encoder_config"]
    max_pos = tf_cfg["max_source_positions"]
    d_model = tf_cfg["d_model"]
    pe = _sinusoidal_embeddings(max_pos, d_model)
    mlx_weights["tf_encoder.embed_positions.weight"] = pe
    print(f"Added sinusoidal positional embeddings ({max_pos}, {d_model})")

    # Save
    safetensors_path = output_dir / "model.safetensors"
    save_safetensors(mlx_weights, safetensors_path)
    print(f"Saved weights to {safetensors_path}")

    # Verify key count
    print(f"\nConversion complete: {len(mlx_weights)} tensors saved")

    # Upload
    if args.upload:
        _upload_to_hub(output_dir, args.upload, args.nemo_path)


def _generate_readme(upload_repo: str, source_id: str) -> str:
    """Generate a model card README following the mlx-audio format."""
    from mlx_audio.version import __version__

    return f"""\
---
library_name: mlx-audio
tags:
- mlx
- speaker-diarization
- speech
- voice-activity-detection
- streaming
- vad
- mlx-audio
base_model: {source_id}
---

# {upload_repo}

This model was converted to MLX format from [`{source_id}`](https://huggingface.co/{source_id}) using mlx-audio version **{__version__}**.

Refer to the [original model card](https://huggingface.co/{source_id}) for more details on the model.

## Use with mlx-audio

```bash
pip install -U mlx-audio
```

### Converting from NeMo

The original model is distributed as a `.nemo` archive. This repo contains the pre-converted MLX weights.

```bash
python -m mlx_audio.vad.models.sortformer.convert \\
    --nemo-path {source_id} \\
    --output-dir ./sortformer-v2.1-mlx
```

### Python Example — Streaming Inference (Recommended):

```python
from mlx_audio.vad import load

model = load("{upload_repo}")

for result in model.generate_stream("meeting.wav", chunk_duration=5.0, verbose=True):
    for seg in result.segments:
        print(f"Speaker {{seg.speaker}}: {{seg.start:.2f}}s - {{seg.end:.2f}}s")
```

### Python Example — Offline Inference:

```python
from mlx_audio.vad import load

model = load("{upload_repo}")
result = model.generate("meeting.wav", threshold=0.5, verbose=True)
print(result.text)
```

### Python Example — Real-time Microphone Streaming:

```python
from mlx_audio.vad import load

model = load("{upload_repo}")
state = model.init_streaming_state()

for chunk in mic_stream():  # your audio source
    result, state = model.feed(chunk, state, sample_rate=16000)
    for seg in result.segments:
        print(f"Speaker {{seg.speaker}}: {{seg.start:.2f}}s - {{seg.end:.2f}}s")
```

## Model Details

- **Architecture**: FastConformer (17 layers) + Transformer Encoder (18 layers) + Sortformer Modules
- **Mel bins**: 128
- **Max speakers**: 4
- **Streaming**: AOSC (Arrival-Order Speaker Cache) compression for intelligent long-range context
- **Input**: 16kHz mono audio
- **Output**: Per-frame speaker activity probabilities

### Key Streaming Features

- **Speaker Cache + FIFO** buffers for long-range and recent context
- **AOSC compression** scores frames by per-speaker log-likelihood ratio, boosting underrepresented speakers
- **Silence profiling** fills cache gaps with running-mean silence embeddings
- **Left/right context** for chunk boundary handling in file mode
"""


def _upload_to_hub(output_dir: Path, upload_repo: str, source_id: str):
    """Upload converted model with README to HuggingFace Hub."""
    from huggingface_hub import HfApi

    print(f"Generating README for {upload_repo}...")
    readme_content = _generate_readme(upload_repo, source_id)
    readme_path = output_dir / "README.md"
    with open(readme_path, "w") as f:
        f.write(readme_content)
    print(f"Saved README to {readme_path}")

    try:
        api = HfApi()
        api.create_repo(upload_repo, exist_ok=True)
        api.upload_folder(
            folder_path=str(output_dir),
            repo_id=upload_repo,
            commit_message="Upload MLX-converted sortformer model",
        )
        print(f"Uploaded to https://huggingface.co/{upload_repo}")
    except Exception as e:
        print(f"Upload failed: {e}")


if __name__ == "__main__":
    main()
