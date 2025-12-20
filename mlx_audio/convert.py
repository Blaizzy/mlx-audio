# Copyright © 2023-2024 Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)


import argparse
import glob
import importlib
import json
import logging
import shutil
from pathlib import Path
from textwrap import dedent
from typing import Optional

import mlx.core as mx
from huggingface_hub import snapshot_download
from mlx.utils import tree_flatten


# Auto-discover model types from directory structure
def _discover_model_types(domain: str) -> set:
    """Discover available model types by scanning the models directory."""
    models_dir = Path(__file__).parent / domain / "models"
    if not models_dir.exists():
        return set()
    return {
        d.name
        for d in models_dir.iterdir()
        if d.is_dir()
        and not d.name.startswith("_")
        and any((d / init).exists() for init in ["__init__.py", "__init__"])
    }


# Lazily computed model type sets
_tts_model_types = None
_stt_model_types = None


def get_tts_model_types() -> set:
    """Get the set of available TTS model types."""
    global _tts_model_types
    if _tts_model_types is None:
        _tts_model_types = _discover_model_types("tts")
    return _tts_model_types


def get_stt_model_types() -> set:
    """Get the set of available STT model types."""
    global _stt_model_types
    if _stt_model_types is None:
        _stt_model_types = _discover_model_types("stt")
    return _stt_model_types


# Aliases that map to actual model directories
MODEL_REMAPPING = {
    # TTS aliases
    "csm": "sesame",
    "voxcpm1.5": "voxcpm",
    "vibevoice_streaming": "vibevoice",
    # STT aliases
    "glm": "glmasr",
}

MODEL_CONVERSION_DTYPES = ["float16", "bfloat16", "float32"]
QUANT_RECIPES = ["mixed_2_6", "mixed_3_4", "mixed_3_6", "mixed_4_6"]


def get_model_path(path_or_hf_repo: str, revision: Optional[str] = None) -> Path:
    """
    Ensures the model is available locally. If the path does not exist locally,
    it is downloaded from the Hugging Face Hub.
    """
    model_path = Path(path_or_hf_repo)

    if not model_path.exists():
        model_path = Path(
            snapshot_download(
                path_or_hf_repo,
                revision=revision,
                allow_patterns=[
                    "*.json",
                    "*.safetensors",
                    "*.py",
                    "*.model",
                    "*.tiktoken",
                    "*.txt",
                    "*.jsonl",
                    "*.yaml",
                    "*.wav",
                    "*.pth",
                ],
            )
        )

    return model_path


def load_config(model_path: Path) -> dict:
    """Load model configuration from a path."""
    config_path = model_path / "config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    raise FileNotFoundError(f"Config not found at {model_path}")


def detect_model_domain(config: dict, model_path: Path) -> str:
    """Detect whether a model is TTS or STT based on its configuration."""
    model_type = config.get("model_type", "").lower()
    architectures = config.get("architectures", [])

    stt_types = get_stt_model_types() | set(MODEL_REMAPPING.keys())
    tts_types = get_tts_model_types() | set(MODEL_REMAPPING.keys())

    if model_type:
        normalized_type = MODEL_REMAPPING.get(model_type, model_type)
        if normalized_type in get_stt_model_types() or model_type in stt_types:
            return "stt"
        if normalized_type in get_tts_model_types() or model_type in tts_types:
            return "tts"

    for arch in architectures:
        arch_lower = arch.lower()
        if any(
            t in arch_lower for t in ["whisper", "wav2vec", "parakeet", "glm", "asr"]
        ):
            return "stt"
        if any(t in arch_lower for t in ["bark", "speech", "tts", "voice"]):
            return "tts"

    if "audio_encoder" in config or "whisper_config" in config:
        return "stt"

    path_str = str(model_path).lower()
    if any(t in path_str for t in ["whisper", "wav2vec", "parakeet", "asr", "stt"]):
        return "stt"

    return "tts"


def get_model_type(config: dict, model_path: Path, domain: str) -> str:
    """Determine the specific model type."""
    model_type = config.get("model_type", "").lower()

    if model_type:
        return MODEL_REMAPPING.get(model_type, model_type)

    # Infer from config or path
    if domain == "stt":
        if "whisper_config" in config or "audio_adapter" in config:
            return "glmasr"
        architectures = config.get("architectures", [])
        for arch in architectures:
            arch_lower = arch.lower()
            if "whisper" in arch_lower:
                return "whisper"
            if "wav2vec" in arch_lower:
                return "wav2vec"
            if "parakeet" in arch_lower:
                return "parakeet"
        return "whisper"

    # TTS - try to infer from path
    path_str = str(model_path).lower()
    for model in get_tts_model_types():
        if model in path_str:
            return MODEL_REMAPPING.get(model, model)

    return model_type or "unknown"


def get_model_class(model_type: str, domain: str):
    """Get the model class module for a given model type and domain."""
    module_path = f"mlx_audio.{domain}.models.{model_type}"
    try:
        return importlib.import_module(module_path)
    except ImportError:
        msg = f"Model type '{model_type}' not supported for {domain.upper()}."
        logging.error(msg)
        raise ValueError(msg)


def upload_to_hub(path: Path, upload_repo: str, hf_path: str, domain: str = "tts"):
    """Upload converted model to HuggingFace Hub."""
    from huggingface_hub import HfApi, ModelCard

    from mlx_audio.version import __version__

    print(f"[INFO] Uploading to {upload_repo}")

    # Define domain-specific tags
    tts_tags = ["text-to-speech", "speech", "speech generation", "voice cloning", "tts"]
    stt_tags = [
        "speech-to-text",
        "speech-to-speech",
        "speech",
        "speech generation",
        "stt",
    ]
    domain_tags = tts_tags if domain == "tts" else stt_tags

    try:
        card = ModelCard.load(hf_path)
        card.data.tags = ["mlx"] if card.data.tags is None else card.data.tags + ["mlx"]
        card.data.tags += domain_tags
        card.data.library_name = "mlx-audio"
    except Exception:
        card = ModelCard("")
        card.data.tags = ["mlx"] + domain_tags

    tts = dedent(
        f"""
        ### CLI Example:
        ```bash
            python -m mlx_audio.tts.generate --model {upload_repo} --text "Hello, this is a test."
        ```
        ### Python Example:
        ```python
            from mlx_audio.tts.utils import load_model
            from mlx_audio.tts.generate import generate_audio
            model = load_model("{upload_repo}")
            generate_audio(
                model=model, text="Hello, this is a test.",
                ref_audio="path_to_audio.wav",
                file_prefix="test_audio",
            )

        ```
    """
    )
    stt = dedent(
        f"""
        ### CLI Example:
        ```bash
            python -m mlx_audio.stt.generate --model {upload_repo} --audio "audio.wav"
        ```
        ### Python Example:
        ```python
            from mlx_audio.stt.utils import load_model
            from mlx_audio.stt.generate import generate_transcription
            model = load_model("{upload_repo}")
            transcription = generate_transcription(
                model=model,
                audio_path="path_to_audio.wav",
                output_path="path_to_output.txt",
                format="txt",
                verbose=True,
            )
            print(transcription.text)
        ```
    """
    )

    card.text = dedent(
        f"""
        # {upload_repo}
        This model was converted to MLX format from [`{hf_path}`](https://huggingface.co/{hf_path}) using mlx-audio version **{__version__}**.
        Refer to the [original model card](https://huggingface.co/{hf_path}) for more details on the model.

        ## Use with mlx-audio

        ```bash
        pip install -U mlx-audio
        ```
        """
    )
    if domain == "tts":
        card.text += tts
    elif domain == "stt":
        card.text += stt
    else:
        raise ValueError(f"Invalid domain: {domain}")
    card.save(path / "README.md")

    api = HfApi()
    api.create_repo(repo_id=upload_repo, exist_ok=True)
    api.upload_folder(
        folder_path=str(path),
        repo_id=upload_repo,
        repo_type="model",
    )
    print(f"[INFO] Upload complete! See https://huggingface.co/{upload_repo}")


def convert(
    hf_path: str,
    mlx_path: str = "mlx_model",
    quantize: bool = False,
    q_group_size: int = 64,
    q_bits: int = 4,
    dtype: str = None,
    upload_repo: str = None,
    revision: Optional[str] = None,
    dequantize: bool = False,
    quant_predicate: Optional[str] = None,
    model_domain: Optional[str] = None,
):
    """
    Convert a model from HuggingFace to MLX format.

    Automatically detects whether the model is TTS or STT and handles
    conversion appropriately.

    Args:
        hf_path: Path to the Hugging Face model or repo ID.
        mlx_path: Path to save the MLX model.
        quantize: Whether to quantize the model.
        q_group_size: Group size for quantization.
        q_bits: Bits per weight for quantization.
        dtype: Data type for weights (float16, bfloat16, float32).
        upload_repo: Hugging Face repo to upload the converted model.
        revision: Model revision to download.
        dequantize: Whether to dequantize a quantized model.
        trust_remote_code: Whether to trust remote code.
        quant_predicate: Mixed-bit quantization recipe.
        model_domain: Force model domain ("tts" or "stt"). Auto-detected if None.
    """
    from mlx_lm.utils import dequantize_model, quantize_model, save_config, save_model

    if quant_predicate:
        from mlx_lm.convert import mixed_quant_predicate_builder

    print(f"[INFO] Loading model from {hf_path}")
    model_path = get_model_path(hf_path, revision=revision)
    config = load_config(model_path)

    # Detect domain and model type
    if model_domain is None:
        model_domain = detect_model_domain(config, model_path)

    model_type = get_model_type(config, model_path, model_domain)
    print(f"\n[INFO] Model domain: {model_domain.upper()}, type: {model_type}")

    # Get model class
    model_class = get_model_class(model_type, model_domain)

    # Get model config
    model_config = (
        model_class.ModelConfig.from_dict(config)
        if hasattr(model_class, "ModelConfig")
        else config
    )

    # Handle model_path attribute if needed (e.g., for Spark)
    if hasattr(model_config, "model_path"):
        model_config.model_path = model_path

    # Load weights
    weights = {}
    weight_files = glob.glob(str(model_path / "*.safetensors"))
    if not weight_files:
        weight_files = glob.glob(str(model_path / "LLM" / "*.safetensors"))

    if not weight_files:
        raise FileNotFoundError(f"No safetensors found in {model_path}")

    for wf in weight_files:
        weights.update(mx.load(wf))

    # Instantiate model
    model = model_class.Model(model_config)

    # Sanitize weights
    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)

    # Load weights into model
    model.load_weights(list(weights.items()))

    # Get flattened weights
    weights = dict(tree_flatten(model.parameters()))

    # Convert dtype
    if dtype is None:
        dtype = config.get("torch_dtype", None)
    if dtype and dtype in MODEL_CONVERSION_DTYPES:
        print(f"[INFO] Converting to {dtype}")
        target_dtype = getattr(mx, dtype)
        weights = {k: v.astype(target_dtype) for k, v in weights.items()}

    if quantize and dequantize:
        raise ValueError("Choose either quantize or dequantize, not both.")

    # Build quantization predicate
    model_quant_predicate = getattr(model, "model_quant_predicate", lambda p, m: True)

    def base_quant_requirements(p, m):
        return (
            hasattr(m, "weight")
            and m.weight.shape[-1] % 64 == 0
            and hasattr(m, "to_quantized")
            and model_quant_predicate(p, m)
        )

    final_quant_predicate = base_quant_requirements
    if quant_predicate:
        mixed_predicate = mixed_quant_predicate_builder(quant_predicate, model)
        final_quant_predicate = lambda p, m: base_quant_requirements(
            p, m
        ) and mixed_predicate(p, m)

    if quantize:
        model.load_weights(list(weights.items()))
        weights, config = quantize_model(
            model, config, q_group_size, q_bits, quant_predicate=final_quant_predicate
        )

    if dequantize:
        print("[INFO] Dequantizing")
        model = dequantize_model(model)
        weights = dict(tree_flatten(model.parameters()))

    # Create output directory
    if isinstance(mlx_path, str):
        mlx_path = Path(mlx_path)
    mlx_path.mkdir(parents=True, exist_ok=True)

    # Save model weights
    save_model(mlx_path, model, donate_model=True)

    # Copy supporting files
    for pattern in [
        "*.py",
        "*.json",
        "*.yaml",
        "*.tiktoken",
        "*.model",
        "*.txt",
        "*.wav",
        "*.pt",
        "*.safetensors",
    ]:
        for file in glob.glob(str(model_path / pattern)):
            if Path(file).name == "model.safetensors.index.json":
                continue
            shutil.copy(file, mlx_path)

        # Check subdirectories
        for file in glob.glob(str(model_path / "**" / pattern), recursive=True):
            if Path(file).name == "model.safetensors.index.json":
                continue
            rel_path = Path(file).relative_to(model_path)
            dest_dir = mlx_path / rel_path.parent
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(file, dest_dir)

    # Save config
    config["model_type"] = model_type
    save_config(config, config_path=mlx_path / "config.json")

    print(f"[INFO] Conversion complete! Model saved to {mlx_path}")

    if upload_repo:
        upload_to_hub(mlx_path, upload_repo, hf_path, model_domain)


def configure_parser() -> argparse.ArgumentParser:
    """Configures and returns the argument parser."""
    parser = argparse.ArgumentParser(
        description="Convert Hugging Face model (TTS or STT) to MLX format"
    )

    parser.add_argument(
        "--hf-path",
        type=str,
        required=True,
        help="Path to the Hugging Face model or repo ID.",
    )
    parser.add_argument(
        "--mlx-path", type=str, default="mlx_model", help="Path to save the MLX model."
    )
    parser.add_argument(
        "-q", "--quantize", action="store_true", help="Generate a quantized model."
    )
    parser.add_argument(
        "--q-group-size", type=int, default=64, help="Group size for quantization."
    )
    parser.add_argument(
        "--q-bits", type=int, default=4, help="Bits per weight for quantization."
    )
    parser.add_argument(
        "--quant-predicate",
        choices=QUANT_RECIPES,
        type=str,
        help="Mixed-bit quantization recipe.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=MODEL_CONVERSION_DTYPES,
        default=None,
        help="Data type for weights.",
    )
    parser.add_argument(
        "--upload-repo",
        type=str,
        default=None,
        help="Hugging Face repo to upload the model to.",
    )
    parser.add_argument(
        "--revision", type=str, default=None, help="Model revision to download."
    )
    parser.add_argument(
        "-d", "--dequantize", action="store_true", help="Dequantize a quantized model."
    )
    parser.add_argument(
        "--model-domain",
        type=str,
        choices=["tts", "stt"],
        default=None,
        help="Force model domain.",
    )

    return parser


def main():
    parser = configure_parser()
    args = parser.parse_args()
    convert(**vars(args))


if __name__ == "__main__":
    main()
