"""Materialize Phonon-2/2.5 release archives as ordinary MLX checkpoints."""

import json
import shutil
import tempfile
from pathlib import Path

import mlx.core as mx
from huggingface_hub import snapshot_download

from mlx_audio.stt.models.phonon.transport import (
    _default_cache_root,
    _extract_archive,
    _safe_relative_path,
    _sha256_file,
)

from .container import CONTAINER_FORMAT, read_container

_MARKER = ".mlx_audio_phonon2.json"


def is_phonon2_model(config: dict) -> bool:
    if config.get("phonon2_modules") is not None:
        return False  # Already an MLX checkpoint.
    return config.get("model_type") == "parakeet_tdt_five_value" or (
        config.get("fermion", {}).get("container_format") == CONTAINER_FORMAT
    )


def prepare_model_path(
    model_path: Path,
    *,
    source: str | None = None,
    revision: str | None = None,
    force_download: bool = False,
) -> Path:
    """Validate and cache a release, or load an already unpacked container."""
    model_path = model_path.expanduser().resolve()
    outer = json.loads((model_path / "config.json").read_text())
    if not is_phonon2_model(outer):
        return model_path
    artifact = outer.get("artifact")
    if artifact is None:
        metadata = outer.get("fermion", {})
        container = model_path / str(
            _safe_relative_path(metadata.get("container", "model.fermion"))
        )
        digest = metadata.get("container_sha256")
        if not container.is_file():
            raise FileNotFoundError(f"Phonon container not found: {container}")
        actual = _sha256_file(container)
        if digest is not None and digest != actual:
            raise RuntimeError("SHA-256 mismatch for Phonon container")
        digest = actual
    else:
        filename = str(artifact.get("filename", ""))
        _safe_relative_path(filename)
        digest = str(artifact.get("sha256", ""))
        expected_bytes = int(artifact.get("unpacked_bytes", 0))
        if (
            not filename.endswith(".bps.tar.zst")
            or len(digest) != 64
            or expected_bytes <= 0
        ):
            raise RuntimeError("Invalid Phonon-2 artifact metadata")
        archive = model_path / filename
        if not archive.is_file() or (source is not None and force_download):
            repo_id = source or outer.get("model_id")
            if not repo_id:
                raise FileNotFoundError(f"Phonon archive not found: {archive}")
            downloaded = Path(
                snapshot_download(
                    str(repo_id),
                    revision=revision,
                    allow_patterns=[filename],
                    force_download=force_download,
                )
            )
            archive = downloaded / filename
        if _sha256_file(archive) != digest:
            raise RuntimeError("SHA-256 mismatch for Phonon archive")

    destination = _default_cache_root() / "phonon2" / digest
    if (
        (destination / _MARKER).is_file()
        and (destination / "config.json").is_file()
        and (destination / "model.safetensors").is_file()
    ):
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{digest[:12]}-", dir=destination.parent)
    )
    try:
        if artifact is not None:
            _extract_archive(archive, temporary, expected_bytes)
            config = json.loads((temporary / "config.json").read_text())
            container = temporary / str(
                _safe_relative_path(artifact.get("container", "model.fermion"))
            )
            container_sha = artifact.get("container_sha256")
            if container_sha is not None and _sha256_file(container) != container_sha:
                raise RuntimeError("SHA-256 mismatch for Phonon container")
        else:
            config = outer
        weights, modules = read_container(container)
        runtime_config = {
            k: config[k]
            for k in ("preprocessor", "encoder", "decoder", "joint", "decoding")
        }
        runtime_config.update(model_type="phonon2", phonon2_modules=modules)
        if "languages" in outer:
            runtime_config["languages"] = outer["languages"]
        mx.save_safetensors(str(temporary / "model.safetensors"), weights)
        (temporary / "config.json").write_text(
            json.dumps(runtime_config, indent=2) + "\n"
        )
        if artifact is not None:
            container.unlink()
        (temporary / _MARKER).write_text(json.dumps({"source_sha256": digest}) + "\n")
        if destination.exists():
            shutil.rmtree(destination)
        temporary.replace(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination
