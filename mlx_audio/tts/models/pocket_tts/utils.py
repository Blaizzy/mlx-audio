import hashlib
from pathlib import Path

import mlx.core as mx
import requests
from huggingface_hub import hf_hub_download

_voices_names = [
    "alba",
    "marius",
    "javert",
    "jean",
    "fantine",
    "cosette",
    "eponine",
    "azelma",
]
PREDEFINED_VOICES = {
    name: (
        "hf://kyutai/pocket-tts-without-voice-cloning/embeddings/"
        f"{name}.safetensors@d4fdd22ae8c8e1cb3634e150ebeff1dab2d16df3"
    )
    for name in _voices_names
}


def make_cache_directory() -> Path:
    cache_dir = Path.home() / ".cache" / "mlx_audio" / "pocket_tts"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def download_if_necessary(path: str) -> Path:
    if path.startswith("http://") or path.startswith("https://"):
        cache_dir = make_cache_directory()
        suffix = path.split(".")[-1]
        cached_file = (
            cache_dir / f"{hashlib.sha256(path.encode()).hexdigest()}.{suffix}"
        )
        if not cached_file.exists():
            response = requests.get(path, timeout=30)
            response.raise_for_status()
            cached_file.write_bytes(response.content)
        return cached_file
    if path.startswith("hf://"):
        path = path.removeprefix("hf://")
        parts = path.split("/")
        repo_id = "/".join(parts[:2])
        filename = "/".join(parts[2:])
        if "@" in filename:
            filename, revision = filename.split("@", 1)
        else:
            revision = None
        cached_file = hf_hub_download(
            repo_id=repo_id, filename=filename, revision=revision
        )
        return Path(cached_file)
    return Path(path)


def load_predefined_voice(voice_name: str) -> mx.array:
    if voice_name not in PREDEFINED_VOICES:
        raise ValueError(
            f"Predefined voice '{voice_name}' not found; "
            f"available voices are {list(PREDEFINED_VOICES)}."
        )
    voice_file = download_if_necessary(PREDEFINED_VOICES[voice_name])
    return mx.load(voice_file)["audio_prompt"]
