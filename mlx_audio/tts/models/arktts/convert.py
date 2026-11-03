"""Audio8 checkpoint hooks for ``python -m mlx_audio.convert``.

PyTorch is needed only to read the upstream codec.pth during conversion.
Converted checkpoints contain the complete model and codec as MLX safetensors.
"""

import json
import shutil
from pathlib import Path

import mlx.core as mx


def prepare_config(config, model_path):
    config = dict(config)
    config.pop("auto_map", None)
    config.pop("architectures", None)
    config["model_type"] = "arktts"
    config["library_name"] = "mlx-audio"
    config["codec_filename"] = "model.safetensors"
    config.setdefault("torch_dtype", config.get("dtype", "bfloat16"))
    return config


def load_source_weights(model_path):
    """Read the DualAR safetensors and, when necessary, the original codec."""
    model_path = Path(model_path)
    files = sorted(model_path.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No safetensors found in {model_path}")
    weights = {}
    for file in files:
        weights.update(mx.load(str(file)))
    if any(key.startswith("codec.") for key in weights):
        return weights

    config = json.loads((model_path / "config.json").read_text())
    codec_path = model_path / config.get("codec_filename", "codec.pth")
    if not codec_path.is_file():
        raise FileNotFoundError(f"Audio8 codec checkpoint not found: {codec_path}")
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "Converting Audio8's codec.pth requires PyTorch. Install torch for "
            "conversion; inference with the converted model uses only MLX."
        ) from exc

    state = torch.load(codec_path, map_location="cpu", weights_only=True)
    state = state.get("state_dict", state)
    if any(key.startswith("generator.") for key in state):
        state = {
            key.removeprefix("generator."): value
            for key, value in state.items()
            if key.startswith("generator.")
        }
    for key, value in state.items():
        if not key.endswith(("freqs_cis", "causal_mask")):
            weights[key] = mx.array(value.detach().cpu().float().numpy())
    return weights


def copy_supporting_files(source, dest):
    """Keep tokenizer data and upstream notices, without remote Python code."""
    source, dest = Path(source), Path(dest)
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "LICENSE",
        "NOTICE",
    ):
        file = source / name
        if file.is_file():
            shutil.copy2(file, dest / name)


def write_model_card(model_path, hf_path, repo_id, config):
    from huggingface_hub import ModelCard, ModelCardData

    # Preserve the checkpoint's own license/languages, including the distinct
    # Preview-0.1b license. A local source may be a downloaded HF snapshot.
    source_card = Path(hf_path) / "README.md"
    if source_card.is_file():
        card_data = ModelCard.load(str(source_card)).data.to_dict()
    elif not Path(hf_path).is_dir():
        card_data = ModelCard.load(hf_path).data.to_dict()
    else:
        card_data = {}
    source_id = str(hf_path)
    if Path(hf_path).exists():
        converted_source = (
            card_data.get("base_model")
            if card_data.get("library_name") == "mlx-audio"
            else None
        )
        source_id = (
            config.get("_name_or_path") or converted_source or Path(hf_path).name
        )
    tags = list(dict.fromkeys([*(card_data.get("tags") or []), "mlx", "mlx-audio"]))
    card_data.update(
        library_name="mlx-audio",
        pipeline_tag="text-to-speech",
        base_model=source_id,
        tags=tags,
    )
    text = (
        f"# {Path(repo_id).name}\n\n"
        f"MLX conversion of `{source_id}`. Audio8 generates mono 44.1 kHz speech "
        "with optional zero-shot voice cloning. The codec is included; runtime "
        "inference does not require PyTorch or remote model code.\n\n"
        "```python\nfrom mlx_audio.audio_io import write\n"
        "from mlx_audio.tts.utils import load\n\n"
        f'model = load("{repo_id}")\n'
        'result = next(model.generate(text="Welcome to Audio8 on Apple Silicon."))\n'
        'write("speech.wav", result.audio, result.sample_rate)\n'
        "```\n\n"
        'For voice cloning, pass `ref_audio="reference.wav"` and '
        '`ref_text="Exact transcript of the reference."` to `generate()`.\n\n'
        "```bash\npython -m mlx_audio.tts.generate \\\n"
        f"  --model {repo_id} \\\n"
        '  --text "Welcome to Audio8 on Apple Silicon."\n```\n\n'
        "The codec retains its source precision when converting or quantizing "
        "the DualAR model. Generation returns one complete waveform per call. "
        "Text, reference audio, and generated frames share a 2,048-position "
        "context budget.\n"
    )
    metadata = ModelCardData(**card_data).to_yaml()
    card = ModelCard(f"---\n{metadata}\n---\n\n{text}")
    card.save(Path(model_path) / "README.md")
