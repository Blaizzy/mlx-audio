from .config import EncoderConfig, ModelConfig, ProjectorConfig, TextConfig
from .granite_speech import Model


def copy_supporting_files(source, dest):
    """Copy runtime metadata and the optional BPE speculative-decoding head."""
    import shutil
    from pathlib import Path

    source, dest = Path(source), Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for pattern in ("*.json", "*.txt", "*.model", "*.jinja"):
        for path in source.glob(pattern):
            if path.name != "model.safetensors.index.json":
                shutil.copy(path, dest / path.name)
    head = source / "out_llm.safetensors"
    if head.exists():
        shutil.copy(head, dest / head.name)
    else:
        (dest / head.name).unlink(missing_ok=True)


def write_model_card(mlx_path, hf_path, upload_repo, config):
    """Document converted checkpoints with the same direct-load API."""
    from pathlib import Path

    metadata = "---\nlibrary_name: mlx\npipeline_tag: automatic-speech-recognition\n"
    if not Path(hf_path).exists():
        metadata += f"base_model: {hf_path}\n"
    content = (
        metadata + "---\n\n# Granite Speech for MLX\n\n"
        f"Converted from `{hf_path}` for speech recognition and translation "
        "with [MLX Audio](https://github.com/Blaizzy/mlx-audio).\n\n"
        "```python\nfrom mlx_audio.stt import load\n\n"
        f"model = load({upload_repo!r})\n"
        'print(model.generate("audio.wav").text)\n```\n\n'
        "Audio files are resampled to mono 16 kHz. Raw arrays must already "
        'contain mono 16 kHz samples. Use `language="fr"` for translation '
        "or `prompt=` for a custom instruction.\n\n"
        "The encoder and projector retain their floating-point weights. "
        "When `out_llm.safetensors` is present, use `speculative=True` for "
        "CTC-drafted, greedy-verified decoding. This requires temperature zero "
        "and no repetition penalty. Low-precision batching can produce small "
        "transcript differences.\n"
    )
    (Path(mlx_path) / "README.md").write_text(content, encoding="utf-8")


DETECTION_HINTS = {
    "config_keys": {"encoder_config", "projector_config", "audio_token_index"},
    "architectures": {
        "GraniteSpeechForConditionalGeneration",
        "GraniteSpeechPlusForConditionalGeneration",
    },
}

__all__ = [
    "DETECTION_HINTS",
    "EncoderConfig",
    "Model",
    "ModelConfig",
    "ProjectorConfig",
    "TextConfig",
    "copy_supporting_files",
    "write_model_card",
]
