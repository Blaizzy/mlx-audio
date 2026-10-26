"""Runtime assets and model cards for native Ming Tiny conversions."""

import shutil
from pathlib import Path

from mlx_audio.stt.models.ming_v3_asr.conversion import (
    copy_supporting_files as copy_tokenizer,
)
from mlx_audio.stt.models.ming_v3_asr.conversion import prepare_config

from .speaker import SPEAKER_SUBPATH, resolve_speaker_weights


def copy_supporting_files(source, destination):
    copy_tokenizer(source, destination)
    copy_speaker_weights(source, destination)


def copy_speaker_weights(source, destination):
    speaker = resolve_speaker_weights(Path(source) / SPEAKER_SUBPATH)
    target = Path(destination) / SPEAKER_SUBPATH
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(speaker, target)


def model_card(destination, source, model_id, title, domain, example, details):
    source = str(source)
    local = Path(source).exists()
    provenance = (
        f"Converted from local checkpoint `{Path(source).name}`."
        if local
        else f"Converted from [{source}](https://huggingface.co/{source})."
    )
    base = "" if local else f"base_model: {source}\n"
    text = (
        "---\nlicense: mit\nlibrary_name: mlx-audio\n"
        + base
        + "tags:\n- mlx\n- audio\n---\n\n"
        + f"# {title} for MLX\n\n{provenance}\n\n{details}\n\n"
        + f"```python\nfrom mlx_audio.{domain} import load\n"
        + f"model = load({model_id!r}, strict=True)\n{example}\n```\n\n"
        + "Offline inference on Apple Silicon using native MLX code. "
        + "The tokenizer is loaded without executing remote code.\n"
    )
    (Path(destination) / "README.md").write_text(text, encoding="utf-8")


def write_model_card(destination, source, model_id, config):
    precision = (
        "Only the language backbone is quantized; acoustic modules and MoE routers "
        "retain floating-point precision. "
        if config.get("quantization") or config.get("quantization_config")
        else "The language backbone and acoustic modules use BF16 weights. "
    )
    model_card(
        destination,
        source,
        model_id,
        "Ming-3.0 TTS Tiny",
        "tts",
        'result = next(model.generate("Hello from MLX Audio!", voice="US_James"))\n'
        "from mlx_audio.audio_io import write\n"
        'write("output.wav", result.audio, result.sample_rate)',
        "44.1 kHz speech with built-in voices, voice cloning and style controls. "
        + precision
        + "CAMPPlus speaker extraction runs natively "
        "in MLX using the bundled `speaker_encoder/campplus.safetensors`. "
        "Its weights retain FP32 precision. No ONNX packages are required. "
        "A 192-value `spk_emb` may also be supplied.",
    )
