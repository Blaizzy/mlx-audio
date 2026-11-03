from .config import ModelConfig
from .nemotron_transcribe import Model

DETECTION_HINTS = {
    "model_type_aliases": ["nemo_speechlm"],
    "architectures": [
        "NeMoSpeechLMForConditionalGeneration",
        "Nemotron3_5TranscribeForConditionalGeneration",
    ],
}


def copy_supporting_files(source, destination):
    from pathlib import Path

    from mlx_audio.convert import copy_model_files

    from .timestamps import embedded_tokenizer

    copy_model_files(source, destination)
    tokenizer_path = Path(destination) / "alignment_tokenizer"
    if (
        not (tokenizer_path / "tokenizer.json").exists()
        and not (tokenizer_path / "tokenizer.model").exists()
    ):
        proto = embedded_tokenizer(source)
        if proto is not None:
            tokenizer_path.mkdir(parents=True, exist_ok=True)
            (tokenizer_path / "tokenizer.model").write_bytes(proto)


__all__ = ["Model", "ModelConfig"]
