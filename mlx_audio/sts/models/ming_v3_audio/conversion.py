"""Package Ming Audio's tokenizer and separately loaded speech decoder."""

import shutil
from pathlib import Path

from mlx_audio.stt.models.ming_v3_asr.conversion import TOKENIZER_FILES
from mlx_audio.stt.models.ming_v3_asr.conversion import (
    copy_supporting_files as copy_tokenizer,
)
from mlx_audio.stt.models.ming_v3_asr.conversion import prepare_config
from mlx_audio.tts.models.lingv3_tts.conversion import copy_speaker_weights, model_card


def copy_supporting_files(source, destination):
    copy_tokenizer(source, destination)
    for folder in ("talker", "talker/llm", "talker/vae"):
        for name in (
            *TOKENIZER_FILES,
            "config.json",
            "model.safetensors",
        ):
            path = Path(source) / folder / name
            if path.is_file():
                target = Path(destination) / folder / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
    copy_speaker_weights(Path(source) / "talker", Path(destination) / "talker")


def write_model_card(destination, source, model_id, config):
    precision = (
        "Only the thinker language backbone and text head are quantized; audio "
        "modules, routers and Talker remain floating-point. "
        if config.get("quantization") or config.get("quantization_config")
        else "The Thinker, audio encoder, Talker and audio VAE use BF16 weights. "
    )
    model_card(
        destination,
        source,
        model_id,
        "Ming-3.0 Audio Tiny",
        "sts",
        'result = model.generate("speech.wav", task="audio_understanding",\n'
        '                        text="Describe what you hear.")\nprint(result.text)',
        "English and Chinese audio understanding, transcription, text chat and "
        "spoken dialogue. `task='spoken_dialogue'` loads the bundled Talker lazily "
        "and produces 44.1 kHz audio. "
        + precision
        + "Speculative MTP decoding and streaming are not implemented. "
        "Reference speaker extraction uses the bundled native MLX CAMPPlus encoder. "
        "Its weights retain FP32 precision. No ONNX packages are required.",
    )
