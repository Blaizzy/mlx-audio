from .config import ModelConfig
from .model import Model

DETECTION_HINTS = {
    "config_keys": {"model_cfg", "stft_cfg"},
    "path_patterns": {"nemotron-reuse", "nemotron_reuse", "nemotronlabs-re-use"},
}

__all__ = ["Model", "ModelConfig"]


def copy_supporting_files(source, dest):
    """Retain the upstream license without copying the CUDA inference scripts."""
    import shutil
    from pathlib import Path

    shutil.copyfile(Path(__file__).with_name("LICENSE"), dest / "LICENSE")


def write_model_card(output_path, source_repo, model_repo, config):
    (output_path / "README.md").write_text(
        "---\nlibrary_name: mlx-audio\nlicense: openmdw-1.1\n"
        "pipeline_tag: audio-to-audio\n"
        "base_model: nvidia/NVIDIA-NemotronLabs-RE-USE\n"
        "tags:\n- mlx\n- speech-enhancement\n---\n\n"
        "# NVIDIA NemotronLabs RE-USE for MLX\n\n"
        "Native MLX port of [NVIDIA NemotronLabs RE-USE]"
        "(https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE). "
        "Requires an mlx-audio version with `nemotron_reuse` support.\n\n"
        "```python\nfrom mlx_audio.sts import load\n"
        "from mlx_audio import audio_io\n\n"
        f"model = load({str(model_repo)!r}, strict=True)\n"
        "result = model.generate('noisy.wav')\n"
        "audio_io.write('enhanced.wav', result.audio, result.sample_rate)\n```\n\n"
        "Preserves the input sample rate (8–48 kHz). Uses five-second chunks "
        "with 50% overlap by default; set `chunk_seconds=None` for one pass. "
        "Set `target_sample_rate` for bandwidth extension. Multichannel audio "
        "is processed independently per channel. Inference uses MLX and Metal; "
        "PyTorch, NeMo, and CUDA are not required.\n\n"
        "For full float32 computation on M5, launch Python with "
        "`MLX_ENABLE_TF32=0`; MLX otherwise permits reduced-precision matmuls.\n\n"
        "FFT rounding near the -pi/+pi phase boundary can change predictions. "
        "Full float32 does not guarantee waveform parity across backends; "
        "CUDA reference parity has not yet been established.\n\n"
        "Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. "
        "Distributed under the included OpenMDW-1.1 license.\n",
        encoding="utf-8",
    )
