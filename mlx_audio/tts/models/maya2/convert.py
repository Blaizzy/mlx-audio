import shutil
from dataclasses import asdict
from pathlib import Path

from .config import ModelConfig


def prepare_config(config, model_path):
    cfg = ModelConfig.from_dict({**config, "model_path": str(model_path)})
    result = {**config, **asdict(cfg)}
    for key in (
        "model_path",
        "model_config",
        "backbone",
        "loader",
        "speaker_encoder",
        "codec",
    ):
        result.pop(key, None)
    result["library_name"] = "mlx-audio"
    result["model_type"] = "maya2"
    return result


def copy_supporting_files(source, dest):
    source, dest = Path(source), Path(dest)
    for directory in ("backbone", "voices"):
        if (source / directory).is_dir():
            (dest / directory).mkdir(parents=True, exist_ok=True)
            for file in (source / directory).iterdir():
                if file.is_file() and file.suffix in {
                    ".json",
                    ".model",
                    ".txt",
                    ".jinja",
                }:
                    shutil.copy2(file, dest / directory / file.name)
    if (source / "LICENSE").is_file():
        shutil.copy2(source / "LICENSE", dest / "LICENSE")


def write_model_card(model_path, hf_path, repo_id, config):
    if Path(hf_path).exists():
        hf_path = "maya-research/maya2-native"
    card = (
        "---\nlibrary_name: mlx-audio\npipeline_tag: text-to-speech\n"
        f"license: apache-2.0\nbase_model: {hf_path}\ntags:\n- mlx\n---\n\n"
        f"# {repo_id}\n\n"
        f"Converted from [{hf_path}](https://huggingface.co/{hf_path}).\n\n"
        "```python\nfrom mlx_audio.tts.utils import load\n\n"
        f'model = load("{repo_id}")\n'
        'audio = next(model.generate("Hello from Maya.", voice="Ahana", lang_code="en")).audio\n'
        "```\n\n"
        "Maya2 generates mono 24 kHz speech in 12 languages with 20 built-in voices, "
        "streaming, and reference-audio voice cloning. Text normalization covers "
        "numbers, decimals, currency, dates/times, and common identifiers in all "
        "12 languages; pass normalize=False for text already prepared for speech. "
        "Mimi and ECAPA weights are downloaded separately when needed.\n\n"
        "## Licenses and attribution\n\n"
        "Maya Research's [Maya2 weights and reference implementation](https://github.com/MayaResearch/maya2-native) "
        "and [Qwen3](https://huggingface.co/Qwen/Qwen3-1.7B-Base) are Apache-2.0. "
        "[Kyutai Mimi](https://huggingface.co/kyutai/mimi) weights are CC-BY-4.0; "
        "[SpeechBrain ECAPA-TDNN](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) "
        "is Apache-2.0. Normalization rules and pronunciation tables are adapted "
        "from Maya Research's Apache-2.0 normalizer. Assamese number data includes "
        "[AI4Bhārat indic-numtowords](https://github.com/AI4Bharat/indic-numtowords), "
        "Copyright (c) 2022 AI4Bhārat, under the "
        "[MIT license](https://github.com/AI4Bharat/indic-numtowords/blob/master/LICENSE). "
        "Preserve the respective notices and attribution.\n"
    )
    (Path(model_path) / "README.md").write_text(card)
