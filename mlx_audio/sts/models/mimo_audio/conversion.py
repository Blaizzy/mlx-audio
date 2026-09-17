"""Runtime sidecars and model cards for converted MiMo checkpoints."""

import shutil
from pathlib import Path


def copy_supporting_files(source, destination):
    # The text tokenizer is self-contained. The audio tokenizer is a separate,
    # pinned dependency recorded in config.json, and may be overridden locally.
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
    ):
        path = Path(source) / name
        if path.is_file():
            shutil.copyfile(path, Path(destination) / name)


def write_model_card(destination, source, model_id, config):
    local_source = Path(source).exists()
    source_label = Path(source).name if local_source else source
    provenance = (
        f"Converted from local checkpoint `{source_label}`."
        if local_source
        else f"Converted from [{source}](https://huggingface.co/{source})."
    )
    base_model = "" if local_source else f"base_model: {source}\n"
    is_base = "base" in source_label.lower()
    call = (
        'task="completion", segments=["The capital of France is"], max_new_tokens=32'
        if is_base
        else 'task="tts", text="Hello from MLX Audio!"'
    )
    codec = config["audio_tokenizer_path"]
    revision = config.get("audio_tokenizer_revision")
    text = (
        "---\nlicense: mit\nlibrary_name: mlx-audio\n"
        + base_model
        + "tags:\n- mlx\n- audio\n- speech-to-speech\n- text-to-speech\n- automatic-speech-recognition\n---\n\n"
        "# MiMo-Audio for MLX\n\n"
        + provenance
        + "\n\nOffline, batch-one inference on Apple Silicon. Instruct supports TTS, ASR, "
        "audio understanding and voice dialogue; Base supports raw completion and "
        "few-shot speech tasks. Streaming is not implemented.\n\n"
        "```python\nfrom mlx_audio.sts import load\n"
        f"model = load({model_id!r}, strict=True)\n"
        f"result = model.generate({call})\n"
        "print(result.text)\n"
        "if result.audio is not None:\n"
        "    from mlx_audio.audio_io import write\n"
        "    write('output.wav', result.audio, result.sample_rate)\n```\n\n"
        f"Audio tasks load the separate `{codec}` checkpoint"
        + (f" at revision `{revision}`" if revision else "")
        + ". Set `audio_tokenizer_path` in `generate()` to use a local copy. "
        "Text-only tasks do not load it. Output audio is 24 kHz mono.\n\n"
        "Quantization applies to the global Qwen2 backbone and text head. "
        "The acoustic patch transformers, speech embeddings and speech heads remain "
        "unquantized; the codec is separate.\n\n"
        "See the [MiMo guide](https://github.com/Blaizzy/mlx-audio/blob/main/"
        "mlx_audio/sts/models/mimo_audio/README.md) for prompts and task examples. "
        "Reference implementation: [XiaomiMiMo/MiMo-Audio](https://github.com/XiaomiMiMo/MiMo-Audio) "
        "(Apache-2.0); model weights are MIT licensed.\n"
    )
    (Path(destination) / "README.md").write_text(text, encoding="utf-8")
