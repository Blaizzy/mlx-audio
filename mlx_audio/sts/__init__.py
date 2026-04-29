import importlib

from .utils import load, load_model

_LAZY_EXPORTS = {
    "DeepFilterNet2Config": (
        "mlx_audio.sts.models.deepfilternet",
        "DeepFilterNet2Config",
    ),
    "DeepFilterNet3Config": (
        "mlx_audio.sts.models.deepfilternet",
        "DeepFilterNet3Config",
    ),
    "DeepFilterNetConfig": (
        "mlx_audio.sts.models.deepfilternet",
        "DeepFilterNetConfig",
    ),
    "DeepFilterNetModel": ("mlx_audio.sts.models.deepfilternet", "DeepFilterNetModel"),
    "DeepFilterNetStreamer": (
        "mlx_audio.sts.models.deepfilternet",
        "DeepFilterNetStreamer",
    ),
    "DeepFilterNetStreamingConfig": (
        "mlx_audio.sts.models.deepfilternet",
        "DeepFilterNetStreamingConfig",
    ),
    "MossFormer2SE": ("mlx_audio.sts.models.mossformer2_se", "MossFormer2SE"),
    "MossFormer2SEConfig": (
        "mlx_audio.sts.models.mossformer2_se",
        "MossFormer2SEConfig",
    ),
    "MossFormer2SEModel": (
        "mlx_audio.sts.models.mossformer2_se",
        "MossFormer2SEModel",
    ),
    "Batch": ("mlx_audio.sts.models.sam_audio", "Batch"),
    "SAMAudio": ("mlx_audio.sts.models.sam_audio", "SAMAudio"),
    "SAMAudioConfig": ("mlx_audio.sts.models.sam_audio", "SAMAudioConfig"),
    "SAMAudioProcessor": ("mlx_audio.sts.models.sam_audio", "SAMAudioProcessor"),
    "SeparationResult": ("mlx_audio.sts.models.sam_audio", "SeparationResult"),
    "save_audio": ("mlx_audio.sts.models.sam_audio", "save_audio"),
}

__all__ = [
    "Batch",
    "DeepFilterNet2Config",
    "DeepFilterNet3Config",
    "DeepFilterNetConfig",
    "DeepFilterNetModel",
    "DeepFilterNetStreamer",
    "DeepFilterNetStreamingConfig",
    "MossFormer2SE",
    "MossFormer2SEConfig",
    "MossFormer2SEModel",
    "SAMAudio",
    "SAMAudioConfig",
    "SAMAudioProcessor",
    "SeparationResult",
    "VoicePipeline",
    "load",
    "load_model",
    "save_audio",
]


def __getattr__(name):
    if name == "VoicePipeline":
        try:
            from .voice_pipeline import VoicePipeline
        except ImportError:
            VoicePipeline = None
        globals()[name] = VoicePipeline
        return VoicePipeline

    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        value = getattr(importlib.import_module(module_name), attr_name)
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
