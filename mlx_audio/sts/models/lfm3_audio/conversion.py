"""Runtime sidecars and model card for the standard mlx_audio.convert command."""

import shutil
from pathlib import Path


def prepare_config(config, model_path):
    config = dict(config)
    config.pop("auto_map", None)
    config.pop("vision_config", None)
    return config


def copy_supporting_files(source, destination):
    source, destination = Path(source), Path(destination)
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "generation_config.json",
        "processor_config.json",
        "preprocessor_config.json",
    ):
        if (source / name).is_file():
            shutil.copy2(source / name, destination / name)
    for name in ("audio_tokenizer", "audio_detokenizer"):
        folder = source / name
        if folder.is_dir():
            target = destination / name
            target.mkdir(exist_ok=True)
            for file in folder.iterdir():
                if file.suffix in {".json", ".safetensors"}:
                    shutil.copy2(file, target / file.name)


def write_model_card(path, source, repo, config):
    # A local source path is useful in conversion logs but is not a Hub model ID.
    source = "LiquidAI/LFM3-Audio-2B" if Path(source).exists() else source
    Path(path, "README.md").write_text(f"""---
library_name: mlx-audio
base_model: {source}
license: lfm1.0
tags:
  - speech-to-speech
  - text-to-speech
  - automatic-speech-recognition
  - mlx
---

# LFM3-Audio for MLX Audio

```python
from mlx_audio.sts import load

model = load("{repo}")
print(model.generate("speech.wav", task="asr").text)
result = model.generate(text="Hello there!", task="tts")
```

```sh
python -m mlx_audio.sts.generate --model {repo} --task tts \\
  --text "Hello there!" --output-path speech.wav
```

Supports ASR, TTS, audio questions, and speech-to-speech chat. Output speech is
24 kHz. Use `language`, `voice`, `system_prompt`, and sampling options through
`model.generate()` or the CLI's `--gen-kwargs`.

Speech supports 18 languages; interleaved voice chat is English only. Each call
generates one conversation. LiveASR, waveform streaming, and voice cloning are
not supported. Earlier generated speech can be supplied as codec IDs.

The model weights are subject to the [LFM Open License v1.0](https://huggingface.co/{source}/blob/main/LICENSE).
""")
