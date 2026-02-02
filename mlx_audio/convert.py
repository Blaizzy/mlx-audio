# Copyright © 2023-2024 Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)


import argparse
import glob
import importlib
import json
import logging
import shutil
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path
from textwrap import dedent
from typing import Callable, Optional

import mlx.core as mx
from huggingface_hub import snapshot_download
from mlx.utils import tree_flatten

# Constants
MODEL_CONVERSION_DTYPES = ["float16", "bfloat16", "float32"]
QUANT_RECIPES = ["mixed_2_6", "mixed_3_4", "mixed_3_6", "mixed_4_6"]


class Domain(str, Enum):
    """Supported model domains."""

    TTS = "tts"
    STT = "stt"
    STS = "sts"


@dataclass
class DomainConfig:
    """Configuration for a specific domain."""

    name: str
    tags: list[str]
    cli_example: str
    python_example: str


# Domain-specific configurations for HuggingFace uploads
DOMAIN_CONFIGS = {
    Domain.TTS: DomainConfig(
        name="TTS",
        tags=["text-to-speech", "speech", "speech generation", "voice cloning", "tts"],
        cli_example='python -m mlx_audio.tts.generate --model {repo} --text "Hello, this is a test."',
        python_example="""
        from mlx_audio.tts.utils import load_model
        from mlx_audio.tts.generate import generate_audio

        model = load_model("{repo}")
        generate_audio(
            model=model,
            text="Hello, this is a test.",
            ref_audio="path_to_audio.wav",
            file_prefix="test_audio",
        )
        """,
    ),
    Domain.STT: DomainConfig(
        name="STT",
        tags=["speech-to-text", "speech", "transcription", "asr", "stt"],
        cli_example='python -m mlx_audio.stt.generate --model {repo} --audio "audio.wav"',
        python_example="""
        from mlx_audio.stt.utils import load_model
        from mlx_audio.stt.generate import generate_transcription

        model = load_model("{repo}")
        transcription = generate_transcription(
            model=model,
            audio_path="path_to_audio.wav",
            output_path="path_to_output.txt",
            format="txt",
            verbose=True,
        )
        print(transcription.text)
        """,
    ),
    Domain.STS: DomainConfig(
        name="STS",
        tags=[
            "speech-to-speech",
            "speech",
            "audio",
            "speech enhancement",
            "audio separation",
            "sts",
        ],
        cli_example='python -m mlx_audio.sts.generate --model {repo} --audio "audio.wav"',
        python_example="""
        from mlx_audio.sts.utils import load_model
        model = load_model("{repo}")
        # Usage depends on the specific STS model type
        # See model documentation for details
        """,
    ),
}


# Caches for dynamic discovery
_model_types_cache: dict[str, set[str]] = {}
_detection_hints_cache: dict[str, dict] = {}


def _discover_model_types(domain: str) -> set[str]:
    """Discover available model types by scanning the models directory."""
    models_dir = Path(__file__).parent / domain / "models"
    if not models_dir.exists():
        return set()

    return {
        d.name
        for d in models_dir.iterdir()
        if d.is_dir() and not d.name.startswith("_") and (d / "__init__.py").exists()
    }


def get_model_types(domain: Domain) -> set[str]:
    """Get the set of available model types for a domain."""
    domain_str = domain.value
    if domain_str not in _model_types_cache:
        _model_types_cache[domain_str] = _discover_model_types(domain_str)
    return _model_types_cache[domain_str]


def _get_config_keys(config_class) -> set[str]:
    """Extract field names from a config class (dataclass or regular class)."""
    if is_dataclass(config_class):
        return {f.name for f in fields(config_class)}
    return (
        set(vars(config_class).keys()) if hasattr(config_class, "__dict__") else set()
    )


def _discover_detection_hints(domain: str) -> dict:
    """
    Discover detection hints for all models in a domain.

    Each model can optionally define:
    - DETECTION_HINTS: dict with 'config_keys', 'architectures', 'path_patterns'
    - Or we infer from the ModelConfig class
    """
    hints = {
        "config_keys": {},  # model_type -> set of unique config keys
        "architectures": {},  # model_type -> set of architecture patterns
        "path_patterns": {},  # model_type -> set of path patterns
    }

    for model_type in get_model_types(Domain(domain)):
        module_path = f"mlx_audio.{domain}.models.{model_type}"
        try:
            module = importlib.import_module(module_path)

            # Check for explicit detection hints
            if hasattr(module, "DETECTION_HINTS"):
                model_hints = module.DETECTION_HINTS
                if "config_keys" in model_hints:
                    hints["config_keys"][model_type] = set(model_hints["config_keys"])
                if "architectures" in model_hints:
                    hints["architectures"][model_type] = set(
                        model_hints["architectures"]
                    )
                if "path_patterns" in model_hints:
                    hints["path_patterns"][model_type] = set(
                        model_hints["path_patterns"]
                    )
            else:
                # Infer from ModelConfig if available
                if hasattr(module, "ModelConfig"):
                    config_keys = _get_config_keys(module.ModelConfig)
                    hints["config_keys"][model_type] = config_keys

                # Use model_type as default path pattern
                hints["path_patterns"][model_type] = {
                    model_type,
                    model_type.replace("_", ""),
                }

        except ImportError:
            continue

    return hints


def get_detection_hints(domain: Domain) -> dict:
    """Get detection hints for a domain (cached)."""
    domain_str = domain.value
    if domain_str not in _detection_hints_cache:
        _detection_hints_cache[domain_str] = _discover_detection_hints(domain_str)
    return _detection_hints_cache[domain_str]


def get_model_path(path_or_hf_repo: str, revision: Optional[str] = None) -> Path:
    """
    Ensures the model is available locally.

    Downloads from HuggingFace Hub if the path doesn't exist locally.
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


def _match_by_model_type(model_type: str) -> Optional[Domain]:
    """Try to match a model_type string to a domain."""
    if not model_type:
        return None

    # Check each domain's known model types
    for domain in Domain:
        if model_type in get_model_types(domain):
            return domain

    return None


def _get_model_identifier(config: dict) -> str:
    """Get model identifier from config, checking model_type and name fields."""
    return config.get("model_type", "").lower() or config.get("name", "").lower()


def _match_by_config_keys(config: dict) -> Optional[tuple[Domain, str]]:
    """Try to match config keys to a domain and model type."""
    config_keys = set(config.keys())

    best_match = None
    best_score = 0

    for domain in Domain:
        hints = get_detection_hints(domain)
        for model_type, model_keys in hints.get("config_keys", {}).items():
            # Score by number of matching unique keys
            intersection = config_keys & model_keys
            # Weight by how unique the match is (intersection / total model keys)
            if model_keys:
                score = len(intersection) / len(model_keys)
                if score > best_score and score > 0.3:  # Require at least 30% match
                    best_score = score
                    best_match = (domain, model_type)

    return best_match


def _match_by_path(model_path: Path) -> Optional[tuple[Domain, str]]:
    """Try to match path patterns to a domain and model type."""
    path_str = str(model_path).lower()

    for domain in Domain:
        hints = get_detection_hints(domain)
        for model_type, patterns in hints.get("path_patterns", {}).items():
            if any(pattern in path_str for pattern in patterns):
                return (domain, model_type)

    return None


def detect_model_domain(config: dict, model_path: Path) -> Domain:
    """
    Detect whether a model is TTS, STT, or STS based on its configuration.

    Uses multiple heuristics in order of reliability:
    1. model_type or name field in config
    2. Config key matching
    3. Path pattern matching
    """
    model_identifier = _get_model_identifier(config)

    # 1. Path pattern matching
    match = _match_by_path(model_path)
    if match:
        return match[0]

    # 2. Direct model_type/name match
    domain = _match_by_model_type(model_identifier)
    if domain:
        return domain

    # 3. Config key matching
    match = _match_by_config_keys(config)
    if match:
        return match[0]

    # Default to TTS
    return Domain.TTS


def get_model_type(config: dict, model_path: Path, domain: Domain) -> str:
    """Determine the specific model type within a domain."""
    # Check both model_type and name fields
    model_type = config.get("model_type", "").lower()
    model_name = config.get("name", "").lower()

    # Direct match via config (model_type takes precedence)
    for candidate in [model_type, model_name]:
        if candidate and candidate in get_model_types(domain):
            return candidate

    # Try config key matching within domain
    hints = get_detection_hints(domain)
    config_keys = set(config.keys())

    best_match = None
    best_score = 0

    for mt, model_keys in hints.get("config_keys", {}).items():
        if model_keys:
            intersection = config_keys & model_keys
            score = len(intersection) / len(model_keys)
            if score > best_score:
                best_score = score
                best_match = mt

    if best_match and best_score > 0.3:
        return best_match

    # Try path matching within domain
    path_str = str(model_path).lower()
    for mt, patterns in hints.get("path_patterns", {}).items():
        if any(pattern in path_str for pattern in patterns):
            return mt

    # Fallback: return first available model type or "unknown"
    model_types = get_model_types(domain)
    return next(iter(model_types), "unknown") if model_types else "unknown"


def get_model_class(model_type: str, domain: Domain):
    """Get the model class module for a given model type and domain."""
    module_path = f"mlx_audio.{domain.value}.models.{model_type}"
    try:
        return importlib.import_module(module_path)
    except ImportError as e:
        msg = f"Model type '{model_type}' not supported for {domain.name}. Error: {e}"
        logging.error(msg)
        raise ValueError(msg)


def generate_readme_content(
    upload_repo: str, hf_path: str, domain: Domain
) -> tuple[list[str], str]:
    """Generate README content and tags for the model card."""
    from mlx_audio.version import __version__

    config = DOMAIN_CONFIGS[domain]

    tags = ["mlx"] + config.tags
    tags.append("mlx-audio")

    content = dedent(
        f"""\
        # {upload_repo}

        This model was converted to MLX format from [`{hf_path}`](https://huggingface.co/{hf_path}) using mlx-audio version **{__version__}**.

        Refer to the [original model card](https://huggingface.co/{hf_path}) for more details on the model.

        ## Use with mlx-audio

        ```bash
        pip install -U mlx-audio
        ```

        ### CLI Example:
        ```bash
        {config.cli_example.format(repo=upload_repo)}
        ```

        ### Python Example:
        ```python\
        {config.python_example.format(repo=upload_repo)}
        ```
        """
    )

    return tags, content


def upload_to_hub(path: Path, upload_repo: str, hf_path: str, domain: Domain):
    """Upload converted model to HuggingFace Hub."""
    from huggingface_hub import HfApi, ModelCard

    print(f"[INFO] Uploading to {upload_repo}")

    tags, readme_content = generate_readme_content(upload_repo, hf_path, domain)

    try:
        card = ModelCard.load(hf_path)
        card.data.tags = tags if card.data.tags is None else card.data.tags + tags
        card.data.library_name = "mlx-audio"
    except Exception:
        card = ModelCard("")
        card.data.tags = tags
        card.data.library_name = "mlx-audio"

    card.text = readme_content
    card.save(path / "README.md")

    api = HfApi()
    api.create_repo(repo_id=upload_repo, exist_ok=True)
    api.upload_folder(
        folder_path=str(path),
        repo_id=upload_repo,
        repo_type="model",
    )
    print(f"[INFO] Upload complete! See https://huggingface.co/{upload_repo}")


def build_quant_predicate(
    model, quant_predicate_name: Optional[str] = None
) -> Callable[[str, any], bool]:
    """Build the quantization predicate function."""
    model_quant_predicate = getattr(model, "model_quant_predicate", lambda p, m: True)

    def base_requirements(path: str, module) -> bool:
        return (
            hasattr(module, "weight")
            and module.weight.shape[-1] % 64 == 0
            and hasattr(module, "to_quantized")
            and model_quant_predicate(path, module)
        )

    if not quant_predicate_name:
        return base_requirements

    from mlx_lm.convert import mixed_quant_predicate_builder

    mixed_predicate = mixed_quant_predicate_builder(quant_predicate_name, model)
    return lambda p, m: base_requirements(p, m) and mixed_predicate(p, m)


def copy_model_files(source: Path, dest: Path):
    """Copy supporting files from source to destination."""
    patterns = [
        "*.py",
        "*.json",
        "*.yaml",
        "*.tiktoken",
        "*.model",
        "*.txt",
        "*.wav",
        "*.pt",
        "*.safetensors",
    ]

    for pattern in patterns:
        # Copy from root
        for file in glob.glob(str(source / pattern)):
            name = Path(file).name
            if name == "model.safetensors.index.json" or (
                name.startswith("model") and name.endswith(".safetensors")
            ):
                continue
            shutil.copy(file, dest)

        # Copy from subdirectories
        for file in glob.glob(str(source / "**" / pattern), recursive=True):
            rel_path = Path(file).relative_to(source)
            # Skip root-level files (already handled above)
            if len(rel_path.parts) <= 1:
                continue
            name = Path(file).name
            if name == "model.safetensors.index.json":
                continue
            dest_dir = dest / rel_path.parent
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(file, dest_dir)


def load_weights(model_path: Path) -> dict:
    """Load model weights from safetensors files."""
    weight_files = glob.glob(str(model_path / "*.safetensors"))

    if not weight_files:
        raise FileNotFoundError(f"No safetensors found in {model_path}")

    weights = {}
    for wf in weight_files:
        if "tokenizer" in wf:
            continue
        weights.update(mx.load(wf))

    return weights


def convert(
    hf_path: str,
    mlx_path: str = "mlx_model",
    quantize: bool = False,
    q_group_size: int = 64,
    q_bits: int = 4,
    dtype: Optional[str] = None,
    upload_repo: Optional[str] = None,
    revision: Optional[str] = None,
    dequantize: bool = False,
    quant_predicate: Optional[str] = None,
    model_domain: Optional[str] = None,
):
    """
    Convert a model from HuggingFace to MLX format.

    Automatically detects whether the model is TTS, STT, or STS and handles
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
        quant_predicate: Mixed-bit quantization recipe.
        model_domain: Force model domain ("tts", "stt", or "sts"). Auto-detected if None.
    """
    from mlx_lm.utils import dequantize_model, quantize_model, save_config, save_model

    if quantize and dequantize:
        raise ValueError("Choose either quantize or dequantize, not both.")

    print(f"[INFO] Loading model from {hf_path}")
    model_path = get_model_path(hf_path, revision=revision)
    config = load_config(model_path)

    # Detect domain and model type
    if model_domain is None:
        domain = detect_model_domain(config, model_path)
    else:
        domain = Domain(model_domain)

    model_type = get_model_type(config, model_path, domain)
    print(f"\n[INFO] Model domain: {domain.name}, type: {model_type}")

    # Get model class and instantiate
    model_class = get_model_class(model_type, domain)

    model_config = (
        model_class.ModelConfig.from_dict(config)
        if hasattr(model_class, "ModelConfig")
        else config
    )

    # Handle model_path attribute if needed
    if hasattr(model_config, "model_path"):
        model_config.model_path = model_path

    # Load and process weights
    weights = load_weights(model_path)
    model = model_class.Model(model_config)

    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)

    model.load_weights(list(weights.items()))
    weights = dict(tree_flatten(model.parameters()))

    # Convert dtype
    target_dtype = dtype or config.get("torch_dtype")
    if target_dtype and target_dtype in MODEL_CONVERSION_DTYPES:
        print(f"[INFO] Converting to {target_dtype}")
        mx_dtype = getattr(mx, target_dtype)
        weights = {k: v.astype(mx_dtype) for k, v in weights.items()}

    # Handle quantization/dequantization
    if quantize:
        final_predicate = build_quant_predicate(model, quant_predicate)
        model.load_weights(list(weights.items()))
        weights, config = quantize_model(
            model, config, q_group_size, q_bits, quant_predicate=final_predicate
        )

    if dequantize:
        print("[INFO] Dequantizing")
        model = dequantize_model(model)
        weights = dict(tree_flatten(model.parameters()))

    # Create output directory and copy files
    mlx_path = Path(mlx_path)
    mlx_path.mkdir(parents=True, exist_ok=True)
    copy_model_files(model_path, mlx_path)

    # Save model weights and config
    save_model(mlx_path, model, donate_model=True)
    config["model_type"] = model_type
    save_config(config, config_path=mlx_path / "config.json")

    print(f"[INFO] Conversion complete! Model saved to {mlx_path}")

    if upload_repo:
        upload_to_hub(mlx_path, upload_repo, hf_path, domain)


def configure_parser() -> argparse.ArgumentParser:
    """Configure and return the argument parser."""
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace model (TTS, STT, or STS) to MLX format"
    )

    parser.add_argument(
        "--hf-path",
        type=str,
        required=True,
        help="Path to the Hugging Face model or repo ID.",
    )
    parser.add_argument(
        "--mlx-path",
        type=str,
        default="mlx_model",
        help="Path to save the MLX model.",
    )
    parser.add_argument(
        "-q",
        "--quantize",
        action="store_true",
        help="Generate a quantized model.",
    )
    parser.add_argument(
        "--q-group-size",
        type=int,
        default=64,
        help="Group size for quantization.",
    )
    parser.add_argument(
        "--q-bits",
        type=int,
        default=4,
        help="Bits per weight for quantization.",
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
        "--revision",
        type=str,
        default=None,
        help="Model revision to download.",
    )
    parser.add_argument(
        "-d",
        "--dequantize",
        action="store_true",
        help="Dequantize a quantized model.",
    )
    parser.add_argument(
        "--model-domain",
        type=str,
        choices=["tts", "stt", "sts"],
        default=None,
        help="Force model domain (auto-detected if not specified).",
    )

    return parser


def main():
    parser = configure_parser()
    args = parser.parse_args()
    convert(**vars(args))


if __name__ == "__main__":
    main()
