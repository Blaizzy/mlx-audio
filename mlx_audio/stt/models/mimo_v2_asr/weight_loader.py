"""
Weight loading utilities for MiMo models.

Converts HuggingFace/PyTorch weights to MLX format,
handling key remapping and tensor transpositions.
"""

from pathlib import Path
from typing import Dict, Optional

import mlx.core as mx
import numpy as np


def sanitize_audio_encoder_weights(
    hf_weights: Dict[str, mx.array],
) -> Dict[str, mx.array]:
    """
    Remap HuggingFace Audio Tokenizer weights to our MLX AudioEncoder keys.

    Key remappings:
      - Conv1d: transpose (out, in, k) → (out, k, in)
      - Attention: q_proj→query_proj, k_proj→key_proj, v_proj→value_proj
      - k_proj: HF has NO bias → insert zero bias for MLX
      - down_sample_layer.0 → down_sample (our model uses Conv1d directly)
      - Quantizer: skip cluster_size, embed_avg, inited (training-only buffers)
      - Conv1d: also need bias for conv1, conv2
      - LayerNorm: bias name is already .bias (same in both)

    Parameters
    ----------
    hf_weights : dict  — loaded from model.safetensors

    Returns
    -------
    dict — remapped MLX-compatible weights
    """
    out = {}

    for key, tensor in hf_weights.items():
        new_key = key
        new_tensor = tensor

        # 1. Skip decoder keys (we only need encoder)
        if key.startswith("decoder."):
            continue

        # 2. Strip "encoder." prefix — our MLX AudioEncoder is a top-level module,
        #    not nested under an `encoder` attribute like in MiMoAudioTokenizer.
        if new_key.startswith("encoder."):
            new_key = new_key[len("encoder."):]

        # 3. Conv1d weight layout:
        #    - raw HF tokenizer weights use (out, in, k)
        #    - our MLX tokenizer export already uses (out, k, in)
        #    Detect the raw HF layout by checking whether the trailing
        #    dimension looks like a kernel width.
        if new_key.endswith(".weight") and tensor.ndim == 3:
            if tensor.shape[-1] <= 8 and tensor.shape[1] > tensor.shape[-1]:
                new_tensor = tensor.transpose(0, 2, 1)

        # 4. down_sample_layer.0 → down_sample (we use a single Conv1d)
        if "down_sample_layer.0" in new_key:
            new_key = new_key.replace("down_sample_layer.0", "down_sample")

        # 5. Rename _codebook → codebook (MLX skips underscore-prefixed attributes)
        if "_codebook." in new_key:
            new_key = new_key.replace("_codebook.", "codebook.")

        # 6. Skip quantizer training-only buffers
        if any(x in new_key for x in (".cluster_size", ".embed_avg", ".inited")):
            continue

        out[new_key] = new_tensor

    # 6. HF k_proj has bias=False → our custom Attention has bias=True.
    #    Insert zero bias for k_proj.
    for key in list(out.keys()):
        if ".self_attn.k_proj.weight" in key:
            bias_key = key.replace(".weight", ".bias")
            if bias_key not in out:
                out[bias_key] = mx.zeros((out[key].shape[0],))

    return out


def sanitize_asr_weights(
    hf_weights: Dict[str, mx.array],
) -> Dict[str, mx.array]:
    """
    Remap HuggingFace MiMo-V2.5-ASR weights to MLX keys.

    Since we use Qwen2Model (inner class) directly, HF keys match our MLX keys
    without any remapping. This function is a no-op passthrough.
    """
    return dict(hf_weights)


def load_hf_weights(model_path: Path) -> Dict[str, mx.array]:
    """
    Load all safetensors from a HuggingFace model directory.

    Parameters
    ----------
    model_path : Path or str — directory containing *.safetensors files

    Returns
    -------
    dict — merged weights
    """
    import glob

    model_path = Path(model_path)
    weight_files = sorted(glob.glob(str(model_path / "*.safetensors")))
    if not weight_files:
        raise FileNotFoundError(f"No safetensors found in {model_path}")

    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf))

    return weights


def load_model_from_hf(
    hf_repo: str,
    local_dir: Optional[Path] = None,
) -> Dict[str, mx.array]:
    """
    Download (if needed) and load weights from a HuggingFace repo.

    Parameters
    ----------
    hf_repo : str — e.g. 'XiaomiMiMo/MiMo-Audio-Tokenizer'
    local_dir : Path, optional — pre-downloaded local directory

    Returns
    -------
    dict — merged weights
    """
    if local_dir and local_dir.exists():
        return load_hf_weights(local_dir)

    from huggingface_hub import snapshot_download

    path = snapshot_download(hf_repo)
    return load_hf_weights(Path(path))
